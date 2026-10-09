// Engine side of the NexTone ASIO driver (see drivers/win/asio/nextone_asio_shared.h).
#include "backends/win/asio_clients.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <thread>

#include "../../../../drivers/win/asio/nextone_asio_shared.h"
#include "util/log.h"
#include "util/platform.h"

namespace nt {

namespace {

int64_t qpc() {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return c.QuadPart;
}

bool processAlive(uint32_t pid) {
    HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, pid);
    if (!h) return false;
    const bool alive = WaitForSingleObject(h, 0) == WAIT_TIMEOUT;
    CloseHandle(h);
    return alive;
}

struct ClientMap {
    HANDLE handle = nullptr;
    NTASIO_CLIENT* client = nullptr;
    bool open(uint32_t pid, int slot) {
        wchar_t name[96];
        swprintf_s(name, NTASIO_CLIENT_NAME_FMT, pid, static_cast<unsigned>(slot));
        handle = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, name);
        if (!handle) return false;
        auto* head = static_cast<NTASIO_CLIENT*>(MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(NTASIO_CLIENT)));
        if (!head || head->magic != NTASIO_MAGIC) {
            if (head) UnmapViewOfFile(head);
            close();
            return false;
        }
        const size_t bytes = ntasio_client_bytes(head->inputs, head->outputs);
        UnmapViewOfFile(head);
        client = static_cast<NTASIO_CLIENT*>(MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, bytes));
        return client != nullptr;
    }
    void close() {
        if (client) UnmapViewOfFile(client);
        if (handle) CloseHandle(handle);
        client = nullptr;
        handle = nullptr;
    }
    ~ClientMap() { close(); }
};

// One direction of one DAW connection.
class AsioClientStream final : public DeviceStream {
public:
    AsioClientStream(bool dawOutputs, uint32_t pid, int slot, int channels)
        : dawOutputs_(dawOutputs), pid_(pid), slot_(slot), channels_(channels) {}
    ~AsioClientStream() override { stop(); }
    bool init(std::string& err) {
        if (!map_.open(pid_, slot_)) {
            err = "The program closed its NexTone ASIO connection";
            return false;
        }
        rate_ = map_.client->sampleRate > 0 ? map_.client->sampleRate : 48000.0;
        generation_ = map_.client->generation;
        return true;
    }
    double sampleRate() const override { return rate_; }
    int channels() const override { return channels_; }
    int periodFrames() const override {
        const int b = map_.client ? map_.client->bufferSize : 0;
        return b > 0 ? b : 256;
    }
    bool start(std::shared_ptr<CaptureBridge> cap, std::shared_ptr<RenderBridge> ren) override {
        cap_ = std::move(cap);
        ren_ = std::move(ren);
        if (cap_) cap_->markStarted();
        if (ren_) ren_->markStarted();
        running_ = true;
        thread_ = std::thread([this] { run(); });
        return true;
    }
    void stop() override {
        running_ = false;
        if (thread_.joinable()) thread_.join();
        if (cap_) cap_->markStopped();
        if (ren_) ren_->markStopped();
    }
    std::string error() const override { return failed_.load() ? error_ : std::string(); }

private:
    void run() {
        setAudioThreadPriority(dawOutputs_ ? "nt-asio-out" : "nt-asio-in");
        HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        if (!timer) timer = CreateWaitableTimerW(nullptr, FALSE, nullptr);
        NTASIO_CLIENT* c = map_.client;
        const uint32_t mask = NTASIO_RING_FRAMES - 1;
        std::vector<float> tmp;
        double lastFeed = nowSeconds();
        if (dawOutputs_) c->toEngine.readPos = c->toEngine.writePos;
        else c->fromEngine.writePos = c->fromEngine.readPos;
        int tick = 0;
        while (running_) {
            LARGE_INTEGER due;
            due.QuadPart = -15000;  // 1.5 ms
            SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
            WaitForSingleObject(timer, 50);
            const double now = nowSeconds();
            if (++tick % 200 == 0 && !processAlive(pid_)) {
                error_ = "The program using NexTone ASIO has closed";
                failed_ = true;
                break;
            }
            if (c->generation != generation_) {
                generation_ = c->generation;
                if (c->sampleRate != rate_) {
                    error_ = "restart: sample rate changed";
                    failed_ = true;
                    break;
                }
            }
            const bool running = c->running != 0;
            if (dawOutputs_) {
                NTASIO_RING& r = c->toEngine;
                const float* data = ntasio_to_engine(c);
                const int64_t w = r.writePos;
                int64_t rd = r.readPos;
                int64_t avail = w - rd;
                if (avail < 0 || avail > static_cast<int64_t>(NTASIO_RING_FRAMES)) {
                    rd = w;
                    avail = 0;
                }
                if (avail > 0 && cap_) {
                    tmp.resize(static_cast<size_t>(avail) * channels_);
                    for (int64_t i = 0; i < avail; ++i)
                        for (int ch = 0; ch < channels_; ++ch)
                            tmp[static_cast<size_t>(i) * channels_ + ch] =
                                ch < c->outputs ? data[((rd + i) & mask) * c->outputs + ch] : 0.0f;
                    cap_->deviceWrite(tmp.data(), static_cast<size_t>(avail), now);
                    r.readPos = rd + avail;
                    lastFeed = now;
                } else if (!running && cap_) {
                    const size_t n = static_cast<size_t>((now - lastFeed) * rate_);
                    if (n >= 64) {
                        cap_->deviceWriteSilence(n, now);
                        lastFeed += static_cast<double>(n) / rate_;
                    }
                } else if (now - lastFeed > 0.25) {
                    lastFeed = now;
                }
            } else {
                NTASIO_RING& r = c->fromEngine;
                float* data = ntasio_from_engine(c);
                const int64_t rd = r.readPos;
                int64_t w = r.writePos;
                if (!running) {
                    r.writePos = rd;
                    const size_t n = static_cast<size_t>((now - lastFeed) * rate_);
                    if (n >= 64 && ren_) {
                        tmp.resize(n * static_cast<size_t>(channels_));
                        ren_->deviceRead(tmp.data(), n, now);
                        lastFeed += static_cast<double>(n) / rate_;
                    }
                    continue;
                }
                if (w - rd < 0 || w - rd > static_cast<int64_t>(NTASIO_RING_FRAMES)) w = rd;
                const int64_t target = std::max<int64_t>(2 * static_cast<int64_t>(c->bufferSize), static_cast<int64_t>(rate_ * 0.006));
                const int64_t want = target - (w - rd);
                if (want > 0 && ren_) {
                    tmp.resize(static_cast<size_t>(want) * channels_);
                    ren_->deviceRead(tmp.data(), static_cast<size_t>(want), now);
                    for (int64_t i = 0; i < want; ++i)
                        for (int ch = 0; ch < c->inputs; ++ch)
                            data[((w + i) & mask) * c->inputs + ch] = ch < channels_ ? tmp[static_cast<size_t>(i) * channels_ + ch] : 0.0f;
                    MemoryBarrier();
                    r.writePos = w + want;
                }
                lastFeed = now;
            }
        }
        CloseHandle(timer);
        if (cap_) cap_->markStopped();
        if (ren_) ren_->markStopped();
    }

    bool dawOutputs_;
    uint32_t pid_;
    int slot_;
    int channels_;
    ClientMap map_;
    double rate_ = 48000;
    int64_t generation_ = 0;
    std::shared_ptr<CaptureBridge> cap_;
    std::shared_ptr<RenderBridge> ren_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> failed_{false};
    std::string error_;
};

}  // namespace

AsioClientBackend::AsioClientBackend() {
    HANDLE h = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(NTASIO_SERVER), NTASIO_SERVER_NAME);
    if (!h) {
        NT_WARN("NexTone ASIO: cannot create the client registry");
        return;
    }
    const bool existed = GetLastError() == ERROR_ALREADY_EXISTS;
    auto* s = static_cast<NTASIO_SERVER*>(MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(NTASIO_SERVER)));
    if (!s) {
        CloseHandle(h);
        return;
    }
    if (!existed || s->magic != NTASIO_MAGIC) std::memset(s, 0, sizeof(NTASIO_SERVER));
    s->version = NTASIO_VERSION;
    s->enginePid = GetCurrentProcessId();
    s->inputChannels = 8;
    s->outputChannels = 8;
    s->engineSampleRate = 48000.0;
    s->preferredBufferSize = 256;
    MemoryBarrier();
    s->magic = NTASIO_MAGIC;
    map_ = h;
    server_ = s;
}

AsioClientBackend::~AsioClientBackend() {
    if (server_) {
        server_->magic = 0;
        UnmapViewOfFile(server_);
    }
    if (map_) CloseHandle(static_cast<HANDLE>(map_));
}

void AsioClientBackend::setEngineInfo(double sampleRate, int blockSize, int inputs, int outputs) {
    if (!server_) return;
    server_->engineSampleRate = sampleRate;
    server_->preferredBufferSize = std::max(64, blockSize * 2);
    server_->inputChannels = std::clamp(inputs, 1, NTASIO_MAX_CHANNELS);
    server_->outputChannels = std::clamp(outputs, 1, NTASIO_MAX_CHANNELS);
}

bool AsioClientBackend::poll() {
    if (!server_) return false;
    server_->engineHeartbeat = qpc();
    std::string sig;
    for (int i = 0; i < NTASIO_MAX_CLIENTS; ++i) {
        NTASIO_SLOT& s = server_->slots[i];
        const int32_t st = s.state;
        if (st != NTASIO_FREE && s.pid && !processAlive(s.pid)) {
            // DAW crashed without releasing its slot.
            InterlockedExchange(reinterpret_cast<volatile LONG*>(&s.state), NTASIO_FREE);
            continue;
        }
        if (st == NTASIO_ACTIVE) sig += std::to_string(i) + ":" + std::to_string(s.pid) + ";";
    }
    if (sig == signature_) return false;
    signature_ = sig;
    return true;
}

std::vector<DeviceInfo> AsioClientBackend::enumerate() {
    std::vector<DeviceInfo> out;
    if (!server_) return out;
    std::map<std::string, int> seen;
    for (int i = 0; i < NTASIO_MAX_CLIENTS; ++i) {
        NTASIO_SLOT& s = server_->slots[i];
        if (s.state != NTASIO_ACTIVE) continue;
        ClientMap m;
        if (!m.open(s.pid, i)) continue;
        std::string exe = toUtf8(std::wstring(s.exe, wcsnlen(s.exe, 64)));
        std::transform(exe.begin(), exe.end(), exe.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        const int dup = seen[exe]++;
        const std::string key = dup ? exe + "#" + std::to_string(dup + 1) : exe;
        const std::string display = toUtf8(std::wstring(s.display, wcsnlen(s.display, 64)));
        for (int dir = 0; dir < 2; ++dir) {
            const bool dawOut = dir == 0;
            DeviceInfo d;
            d.kind = DeviceKind::AsioClient;
            d.input = dawOut;
            d.bin = dawOut ? Bin::AppOut : Bin::AppIn;
            d.nodeId = std::string(dawOut ? "asioclient:out:" : "asioclient:in:") + key;
            d.endpointId = std::to_string(s.pid) + ":" + std::to_string(i);
            d.name = display + " (ASIO)";
            d.subtitle = "NexTone ASIO";
            d.channels = dawOut ? m.client->outputs : m.client->inputs;
            d.sampleRate = m.client->sampleRate;
            for (int c = 0; c < d.channels; ++c) d.channelNames.push_back(std::to_string(c + 1));
            out.push_back(std::move(d));
        }
    }
    return out;
}

std::unique_ptr<DeviceStream> AsioClientBackend::open(const DeviceInfo& dev, const StreamRequest&, std::string& error) {
    const auto colon = dev.endpointId.find(':');
    if (colon == std::string::npos) {
        error = "Invalid ASIO client";
        return nullptr;
    }
    const uint32_t pid = static_cast<uint32_t>(std::stoul(dev.endpointId.substr(0, colon)));
    const int slot = std::stoi(dev.endpointId.substr(colon + 1));
    auto s = std::make_unique<AsioClientStream>(dev.input, pid, slot, dev.channels);
    if (!s->init(error)) return nullptr;
    return s;
}

}  // namespace nt
