#include "net/vban.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "util/log.h"
#include "util/platform.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
static constexpr socket_t kBadSocket = INVALID_SOCKET;
static void closeSocket(socket_t s) { closesocket(s); }
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
static constexpr socket_t kBadSocket = -1;
static void closeSocket(socket_t s) { ::close(s); }
#endif

namespace nt::vban {

namespace {
constexpr double kRates[] = {6000,  12000, 24000, 48000,  96000,  192000, 384000, 8000,   16000,  32000, 64000,
                             128000, 256000, 512000, 11025, 22050, 44100, 88200, 176400, 352800, 705600};
constexpr uint8_t kProtocolAudio = 0x00;
}  // namespace

int rateIndex(double rate) {
    for (int i = 0; i < static_cast<int>(std::size(kRates)); ++i)
        if (std::fabs(kRates[i] - rate) < 0.5) return i;
    return -1;
}
double rateFromIndex(int idx) { return idx >= 0 && idx < static_cast<int>(std::size(kRates)) ? kRates[idx] : 0.0; }

int bytesPerSample(DataType t) {
    switch (t) {
        case DataType::Int8: return 1;
        case DataType::Int16: return 2;
        case DataType::Int24: return 3;
        case DataType::Int32: return 4;
        case DataType::Float32: return 4;
        case DataType::Float64: return 8;
    }
    return 2;
}

int framesPerPacket(int channels, DataType t) {
    const int bytes = std::max(1, channels) * bytesPerSample(t);
    return std::clamp(static_cast<int>(kMaxPayload) / bytes, 1, kMaxSamplesPerPacket);
}

size_t encode(const Header& h, const float* in, uint8_t* out, size_t cap) {
    const int bps = bytesPerSample(h.type);
    const size_t payload = static_cast<size_t>(h.samples) * h.channels * bps;
    if (cap < kHeaderBytes + payload || payload > kMaxPayload) return 0;
    out[0] = 'V';
    out[1] = 'B';
    out[2] = 'A';
    out[3] = 'N';
    out[4] = static_cast<uint8_t>((rateIndex(h.sampleRate) & 0x1f) | kProtocolAudio);
    out[5] = static_cast<uint8_t>(h.samples - 1);
    out[6] = static_cast<uint8_t>(h.channels - 1);
    out[7] = static_cast<uint8_t>(static_cast<uint8_t>(h.type) & 0x07);  // codec PCM
    std::memset(out + 8, 0, 16);
    std::memcpy(out + 8, h.stream.data(), std::min<size_t>(16, h.stream.size()));
    std::memcpy(out + 24, &h.frame, 4);  // little endian on all supported platforms
    uint8_t* p = out + kHeaderBytes;
    const size_t n = static_cast<size_t>(h.samples) * h.channels;
    for (size_t i = 0; i < n; ++i) {
        const float v = std::clamp(in[i], -1.0f, 1.0f);
        switch (h.type) {
            case DataType::Int16: {
                const int16_t s = static_cast<int16_t>(std::lrint(v * 32767.0f));
                std::memcpy(p, &s, 2);
                break;
            }
            case DataType::Int24: {
                const int32_t s = static_cast<int32_t>(std::lrint(v * 8388607.0f));
                p[0] = static_cast<uint8_t>(s);
                p[1] = static_cast<uint8_t>(s >> 8);
                p[2] = static_cast<uint8_t>(s >> 16);
                break;
            }
            case DataType::Int32: {
                const int32_t s = static_cast<int32_t>(std::llrint(static_cast<double>(v) * 2147483647.0));
                std::memcpy(p, &s, 4);
                break;
            }
            case DataType::Float32: std::memcpy(p, &v, 4); break;
            case DataType::Float64: {
                const double d = v;
                std::memcpy(p, &d, 8);
                break;
            }
            case DataType::Int8: *p = static_cast<uint8_t>(static_cast<int8_t>(std::lrint(v * 127.0f))); break;
        }
        p += bps;
    }
    return kHeaderBytes + payload;
}

bool parse(const uint8_t* pkt, size_t len, Header& h, const uint8_t*& payload, size_t& payloadBytes) {
    if (len < kHeaderBytes || std::memcmp(pkt, "VBAN", 4) != 0) return false;
    if ((pkt[4] & 0xe0) != kProtocolAudio) return false;  // only the audio sub-protocol
    if ((pkt[7] & 0xf0) != 0) return false;               // only PCM (no codecs)
    h.sampleRate = rateFromIndex(pkt[4] & 0x1f);
    h.samples = pkt[5] + 1;
    h.channels = pkt[6] + 1;
    h.type = static_cast<DataType>(pkt[7] & 0x07);
    if (static_cast<uint8_t>(h.type) > 5 || h.sampleRate <= 0) return false;
    const char* name = reinterpret_cast<const char*>(pkt + 8);
    h.stream.assign(name, strnlen(name, 16));
    std::memcpy(&h.frame, pkt + 24, 4);
    payload = pkt + kHeaderBytes;
    payloadBytes = len - kHeaderBytes;
    return payloadBytes >= static_cast<size_t>(h.samples) * h.channels * bytesPerSample(h.type);
}

void decode(const Header& h, const uint8_t* payload, float* out, int outChannels) {
    const int bps = bytesPerSample(h.type);
    for (int f = 0; f < h.samples; ++f) {
        for (int c = 0; c < outChannels; ++c) {
            float v = 0.0f;
            if (c < h.channels) {
                const uint8_t* p = payload + (static_cast<size_t>(f) * h.channels + c) * bps;
                switch (h.type) {
                    case DataType::Int8: v = static_cast<int8_t>(*p) / 128.0f; break;
                    case DataType::Int16: {
                        int16_t s;
                        std::memcpy(&s, p, 2);
                        v = s / 32768.0f;
                        break;
                    }
                    case DataType::Int24: v = static_cast<float>((static_cast<int32_t>(p[2]) << 24) | (p[1] << 16) | (p[0] << 8)) / 2147483648.0f; break;
                    case DataType::Int32: {
                        int32_t s;
                        std::memcpy(&s, p, 4);
                        v = static_cast<float>(s / 2147483648.0);
                        break;
                    }
                    case DataType::Float32: std::memcpy(&v, p, 4); break;
                    case DataType::Float64: {
                        double d;
                        std::memcpy(&d, p, 8);
                        v = static_cast<float>(d);
                        break;
                    }
                }
            }
            out[static_cast<size_t>(f) * outChannels + c] = v;
        }
    }
}

// ============================================================================ networking
namespace {

struct SocketInit {
    SocketInit() {
#ifdef _WIN32
        WSADATA w;
        WSAStartup(MAKEWORD(2, 2), &w);
#endif
    }
};
void ensureSockets() { static SocketInit init; }

// One UDP socket per local port, shared by every VBAN input listening on it.
class Receiver {
public:
    using Sink = std::function<void(const Header&, const uint8_t*, uint32_t srcIp)>;
    static std::shared_ptr<Receiver> forPort(int port) {
        static std::mutex mu;
        static std::map<int, std::weak_ptr<Receiver>> all;
        std::lock_guard lk(mu);
        if (auto r = all[port].lock()) return r;
        auto r = std::shared_ptr<Receiver>(new Receiver(port));
        if (!r->ok()) return nullptr;
        all[port] = r;
        return r;
    }
    ~Receiver() {
        running_ = false;
        closeSocket(sock_);
        if (thread_.joinable()) thread_.join();
    }
    int subscribe(Sink s) {
        std::lock_guard lk(mu_);
        sinks_[++nextId_] = std::move(s);
        return nextId_;
    }
    void unsubscribe(int id) {
        std::lock_guard lk(mu_);
        sinks_.erase(id);
    }
    bool ok() const { return sock_ != kBadSocket; }

private:
    explicit Receiver(int port) {
        ensureSockets();
        sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock_ == kBadSocket) return;
        int yes = 1;
        setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
        int rcv = 1 << 20;
        setsockopt(sock_, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&rcv), sizeof(rcv));
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_ANY);
        a.sin_port = htons(static_cast<uint16_t>(port));
        if (bind(sock_, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) {
            NT_WARN("VBAN: UDP port %d is already in use", port);
            closeSocket(sock_);
            sock_ = kBadSocket;
            return;
        }
#ifdef _WIN32
        DWORD timeout = 200;
#else
        timeval timeout{0, 200000};
#endif
        setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout), sizeof(timeout));
        running_ = true;
        thread_ = std::thread([this] { run(); });
    }
    void run() {
        setAudioThreadPriority("nt-vban-rx");
        uint8_t buf[2048];
        while (running_) {
            sockaddr_in from{};
#ifdef _WIN32
            int flen = sizeof(from);
#else
            socklen_t flen = sizeof(from);
#endif
            const int n = recvfrom(sock_, reinterpret_cast<char*>(buf), sizeof(buf), 0, reinterpret_cast<sockaddr*>(&from), &flen);
            if (n <= 0) continue;
            Header h;
            const uint8_t* payload = nullptr;
            size_t pbytes = 0;
            if (!parse(buf, static_cast<size_t>(n), h, payload, pbytes)) continue;
            std::lock_guard lk(mu_);
            for (auto& [id, sink] : sinks_) sink(h, payload, ntohl(from.sin_addr.s_addr));
        }
    }
    socket_t sock_ = kBadSocket;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::mutex mu_;
    std::map<int, Sink> sinks_;
    int nextId_ = 0;
};

// Last sample rate seen per port/stream, so a reopened input matches the sender immediately.
std::mutex g_rateMu;
std::map<std::string, double> g_seenRate;

uint32_t parseIp(const std::string& host) {
    in_addr a{};
    if (host.empty() || inet_pton(AF_INET, host.c_str(), &a) != 1) return 0;
    return ntohl(a.s_addr);
}

// Network input: packets -> capture bridge.
class VbanInStream final : public DeviceStream {
public:
    VbanInStream(std::string stream, std::string host, int port, int channels)
        : stream_(std::move(stream)), filterIp_(parseIp(host)), port_(port), channels_(channels) {}
    ~VbanInStream() override { stop(); }
    double sampleRate() const override { return rate_; }
    int channels() const override { return channels_; }
    int periodFrames() const override { return 256; }
    double deviceLatencyMs() const override { return 0.0; }
    bool init(double engineRate, std::string& err) {
        rate_ = engineRate;
        {
            std::lock_guard lk(g_rateMu);
            if (auto it = g_seenRate.find(key()); it != g_seenRate.end()) rate_ = it->second;
        }
        rx_ = Receiver::forPort(port_);
        if (!rx_) {
            err = "Cannot listen on UDP port " + std::to_string(port_);
            return false;
        }
        return true;
    }
    bool start(std::shared_ptr<CaptureBridge> cap, std::shared_ptr<RenderBridge>) override {
        cap_ = std::move(cap);
        cap_->markStarted();
        tmp_.resize(static_cast<size_t>(kMaxSamplesPerPacket) * channels_);
        sub_ = rx_->subscribe([this](const Header& h, const uint8_t* payload, uint32_t ip) { onPacket(h, payload, ip); });
        // Keep the bridge clock alive with silence while nothing arrives.
        running_ = true;
        idle_ = std::thread([this] { idleLoop(); });
        return true;
    }
    void stop() override {
        if (rx_ && sub_) rx_->unsubscribe(sub_);
        sub_ = 0;
        running_ = false;
        if (idle_.joinable()) idle_.join();
        if (cap_) cap_->markStopped();
    }
    std::string error() const override { return failed_.load() ? error_ : std::string(); }

private:
    std::string key() const { return std::to_string(port_) + "/" + stream_; }
    void onPacket(const Header& h, const uint8_t* payload, uint32_t ip) {
        if (h.stream != stream_ || (filterIp_ && ip != filterIp_)) return;
        if (std::fabs(h.sampleRate - rate_) > 0.5) {
            // Sender changed rate: let the engine rebuild the bridge at the new rate.
            if (!failed_) {
                std::lock_guard lk(g_rateMu);
                g_seenRate[key()] = h.sampleRate;
                rate_ = h.sampleRate;
                error_ = "restart: VBAN sample rate changed";
                failed_ = true;
            }
            return;
        }
        decode(h, payload, tmp_.data(), channels_);
        const double now = nowSeconds();
        std::lock_guard lk(mu_);
        cap_->deviceWrite(tmp_.data(), static_cast<size_t>(h.samples), now);
        lastPacket_ = now;
        fed_ = now;
    }
    void idleLoop() {
        while (running_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            const double now = nowSeconds();
            std::lock_guard lk(mu_);
            if (now - lastPacket_ > 0.2) {
                if (fed_ < now - 0.2) fed_ = now - 0.005;
                const size_t n = static_cast<size_t>((now - fed_) * rate_);
                if (n >= 64) {
                    cap_->deviceWriteSilence(n, now);
                    fed_ += static_cast<double>(n) / rate_;
                }
            }
        }
    }
    std::string stream_;
    uint32_t filterIp_;
    int port_, channels_;
    double rate_ = 48000;
    std::shared_ptr<Receiver> rx_;
    int sub_ = 0;
    std::shared_ptr<CaptureBridge> cap_;
    std::vector<float> tmp_;
    std::mutex mu_;
    double lastPacket_ = 0, fed_ = 0;
    std::thread idle_;
    std::atomic<bool> running_{false};
    std::atomic<bool> failed_{false};
    std::string error_;
};

// Network output: render bridge -> packets, paced by a high-resolution timer.
class VbanOutStream final : public DeviceStream {
public:
    VbanOutStream(std::string stream, std::string host, int port, int channels)
        : stream_(std::move(stream)), host_(std::move(host)), port_(port), channels_(channels) {}
    ~VbanOutStream() override { stop(); }
    double sampleRate() const override { return rate_; }
    int channels() const override { return channels_; }
    int periodFrames() const override { return frames_; }
    bool init(double engineRate, std::string& err) {
        ensureSockets();
        rate_ = engineRate;
        if (rateIndex(rate_) < 0) rate_ = 48000;
        type_ = DataType::Int16;  // VoiceMeeter default; widely compatible
        frames_ = framesPerPacket(channels_, type_);
        sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock_ == kBadSocket) {
            err = "Cannot create a network socket";
            return false;
        }
        dest_.sin_family = AF_INET;
        dest_.sin_port = htons(static_cast<uint16_t>(port_));
        if (inet_pton(AF_INET, host_.c_str(), &dest_.sin_addr) != 1) {
            err = "Invalid destination address: " + host_;
            return false;
        }
        int yes = 1;
        setsockopt(sock_, SOL_SOCKET, SO_BROADCAST, reinterpret_cast<const char*>(&yes), sizeof(yes));
        return true;
    }
    bool start(std::shared_ptr<CaptureBridge>, std::shared_ptr<RenderBridge> ren) override {
        ren_ = std::move(ren);
        ren_->markStarted();
        running_ = true;
        thread_ = std::thread([this] { run(); });
        return true;
    }
    void stop() override {
        running_ = false;
        if (thread_.joinable()) thread_.join();
        if (ren_) ren_->markStopped();
        if (sock_ != kBadSocket) closeSocket(sock_);
        sock_ = kBadSocket;
    }

private:
    void run() {
        setAudioThreadPriority("nt-vban-tx");
        std::vector<float> frames(static_cast<size_t>(frames_) * channels_);
        uint8_t pkt[kHeaderBytes + kMaxPayload];
        Header h;
        h.sampleRate = rate_;
        h.samples = frames_;
        h.channels = channels_;
        h.type = type_;
        h.stream = stream_;
        const double period = frames_ / rate_;
        const double t0 = nowSeconds();
        uint64_t sent = 0;
        while (running_) {
            const double due = t0 + static_cast<double>(sent + 1) * period;
            double wait = due - nowSeconds();
            if (wait > 0.002) std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int64_t>((wait - 0.001) * 1e6)));
            while (nowSeconds() < due) std::this_thread::yield();
            ren_->deviceRead(frames.data(), static_cast<size_t>(frames_), nowSeconds());
            h.frame = static_cast<uint32_t>(sent);
            const size_t len = encode(h, frames.data(), pkt, sizeof(pkt));
            if (len) sendto(sock_, reinterpret_cast<const char*>(pkt), static_cast<int>(len), 0, reinterpret_cast<sockaddr*>(&dest_), sizeof(dest_));
            ++sent;
            if (nowSeconds() - due > 0.2) sent = static_cast<uint64_t>((nowSeconds() - t0) / period);  // resync after a stall
        }
    }
    std::string stream_, host_;
    int port_, channels_;
    double rate_ = 48000;
    DataType type_ = DataType::Int16;
    int frames_ = 256;
    socket_t sock_ = kBadSocket;
    sockaddr_in dest_{};
    std::shared_ptr<RenderBridge> ren_;
    std::thread thread_;
    std::atomic<bool> running_{false};
};

}  // namespace

VbanBackend::VbanBackend() { ensureSockets(); }
VbanBackend::~VbanBackend() = default;

std::unique_ptr<DeviceStream> VbanBackend::open(const DeviceInfo& dev, const StreamRequest& req, std::string& error) {
    const int port = req.port > 0 ? req.port : kDefaultPort;
    const std::string stream = req.streamName.empty() ? "Stream1" : req.streamName.substr(0, 16);
    const double rate = req.preferredRate > 0 ? req.preferredRate : 48000;
    if (dev.input) {
        auto s = std::make_unique<VbanInStream>(stream, req.host == "0.0.0.0" ? "" : req.host, port, dev.channels);
        if (!s->init(rate, error)) return nullptr;
        return s;
    }
    auto s = std::make_unique<VbanOutStream>(stream, req.host, port, dev.channels);
    if (!s->init(rate, error)) return nullptr;
    return s;
}

}  // namespace nt::vban
