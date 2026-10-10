// NexTone virtual audio driver backend (shared-memory rings).
#include "backends/win/vad.h"

#include <windows.h>

#include <mmdeviceapi.h>
#include <functiondiscoverykeys_devpkey.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "../../../../drivers/win/vad/include/nextone_vad_shared.h"
#include "util/log.h"
#include "util/platform.h"

namespace nt {

namespace {

constexpr int kMaxRings = 8;
constexpr double kPollSec = 0.002;

struct RingMap {
    HANDLE handle = nullptr;
    NTVAD_RING_HEADER* hdr = nullptr;
    uint8_t* data = nullptr;
    bool open(bool mic, int index) {
        wchar_t name[64];
        swprintf_s(name, L"Global\\NexToneVad%s%d", mic ? L"Mic" : L"Out", index);
        handle = OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, name);
        if (!handle) return false;
        void* v = MapViewOfFile(handle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, NTVAD_SECTION_BYTES);
        if (!v) {
            close();
            return false;
        }
        hdr = static_cast<NTVAD_RING_HEADER*>(v);
        data = static_cast<uint8_t*>(v) + NTVAD_HEADER_BYTES;
        if (hdr->magic != NTVAD_MAGIC || hdr->version != NTVAD_VERSION || (hdr->dataBytes & (hdr->dataBytes - 1)) != 0) {
            close();
            return false;
        }
        return true;
    }
    void close() {
        if (hdr) UnmapViewOfFile(hdr);
        if (handle) CloseHandle(handle);
        hdr = nullptr;
        data = nullptr;
        handle = nullptr;
    }
    ~RingMap() { close(); }
};

int64_t qpcNow() {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return c.QuadPart;
}

struct Fmt {
    int channels = 2, bits = 16, block = 4;
    bool isFloat = false;
    double rate = 48000;
};

Fmt readFmt(const NTVAD_RING_HEADER* h) {
    Fmt f;
    f.channels = static_cast<int>(h->channels ? h->channels : 2);
    f.bits = static_cast<int>(h->bitsPerSample ? h->bitsPerSample : 16);
    f.block = static_cast<int>(h->blockAlign ? h->blockAlign : f.channels * f.bits / 8);
    f.isFloat = h->isFloat != 0;
    f.rate = h->sampleRate ? h->sampleRate : 48000;
    return f;
}

float sampleToFloat(const uint8_t* p, const Fmt& f) {
    if (f.isFloat) {
        float v;
        std::memcpy(&v, p, 4);
        return v;
    }
    switch (f.bits) {
        case 16: {
            int16_t v;
            std::memcpy(&v, p, 2);
            return v * (1.0f / 32768.0f);
        }
        case 24: {
            const int32_t v = (static_cast<int32_t>(p[2]) << 24) | (p[1] << 16) | (p[0] << 8);
            return static_cast<float>(v) * (1.0f / 2147483648.0f);
        }
        default: {
            int32_t v;
            std::memcpy(&v, p, 4);
            return static_cast<float>(v) * (1.0f / 2147483648.0f);
        }
    }
}

void floatToSample(float x, uint8_t* p, const Fmt& f) {
    x = x > 1.0f ? 1.0f : x < -1.0f ? -1.0f : x;
    if (f.isFloat) {
        std::memcpy(p, &x, 4);
        return;
    }
    switch (f.bits) {
        case 16: {
            const int16_t v = static_cast<int16_t>(x * 32767.0f);
            std::memcpy(p, &v, 2);
            break;
        }
        case 24: {
            const int32_t v = static_cast<int32_t>(x * 8388607.0f);
            p[0] = static_cast<uint8_t>(v);
            p[1] = static_cast<uint8_t>(v >> 8);
            p[2] = static_cast<uint8_t>(v >> 16);
            break;
        }
        default: {
            const int32_t v = static_cast<int32_t>(static_cast<double>(x) * 2147483647.0);
            std::memcpy(p, &v, 4);
            break;
        }
    }
}

// Finds the Windows endpoint id for an endpoint with this description (e.g. "NexTone Output 1").
std::string findEndpointId(bool capture, const std::wstring& desc) {
    IMMDeviceEnumerator* en = nullptr;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                                reinterpret_cast<void**>(&en))))
        return {};
    std::string result;
    IMMDeviceCollection* coll = nullptr;
    if (SUCCEEDED(en->EnumAudioEndpoints(capture ? eCapture : eRender, DEVICE_STATE_ACTIVE, &coll))) {
        UINT n = 0;
        coll->GetCount(&n);
        for (UINT i = 0; i < n && result.empty(); ++i) {
            IMMDevice* d = nullptr;
            if (FAILED(coll->Item(i, &d))) continue;
            IPropertyStore* ps = nullptr;
            if (SUCCEEDED(d->OpenPropertyStore(STGM_READ, &ps))) {
                PROPVARIANT v;
                PropVariantInit(&v);
                if (SUCCEEDED(ps->GetValue(PKEY_Device_DeviceDesc, &v)) && v.vt == VT_LPWSTR && desc == v.pwszVal) {
                    LPWSTR id = nullptr;
                    if (SUCCEEDED(d->GetId(&id))) result = toUtf8(id);
                    CoTaskMemFree(id);
                }
                PropVariantClear(&v);
                ps->Release();
            }
            d->Release();
        }
        coll->Release();
    }
    en->Release();
    return result;
}

class VadStream final : public DeviceStream {
public:
    VadStream(bool mic, int index, int channels, std::string label) : mic_(mic), index_(index), channels_(channels), label_(std::move(label)) {}
    ~VadStream() override { stop(); }

    bool init(std::string& err) {
        if (!ring_.open(mic_, index_)) {
            err = "NexTone virtual device unavailable";
            return false;
        }
        fmt_ = readFmt(ring_.hdr);
        generation_ = ring_.hdr->formatGeneration;
        rate_ = fmt_.rate;
        return true;
    }
    double sampleRate() const override { return rate_; }
    int channels() const override { return channels_; }
    int periodFrames() const override { return static_cast<int>(rate_ * 0.010); }  // driver moves data every ~10 ms
    double deviceLatencyMs() const override { return 10.0; }
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
        setAudioThreadPriority(mic_ ? "nt-vad-mic" : "nt-vad-out");
        HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        if (!timer) timer = CreateWaitableTimerW(nullptr, FALSE, nullptr);
        NTVAD_RING_HEADER* h = ring_.hdr;
        const uint32_t mask = h->dataBytes - 1;
        std::vector<float> tmp;
        double lastFeed = nowSeconds();
        if (!mic_) h->readPos = h->writePos;  // start at "now"
        else h->writePos = h->readPos;
        while (running_) {
            LARGE_INTEGER due;
            due.QuadPart = -static_cast<LONGLONG>(kPollSec * 1e7);
            SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
            WaitForSingleObject(timer, 50);
            h->engineHeartbeat = qpcNow();
            const double now = nowSeconds();

            if (h->formatGeneration != generation_) {
                generation_ = h->formatGeneration;
                const Fmt nf = readFmt(h);
                if (nf.rate != rate_) {
                    // The bridges were built for the old rate: let the engine reopen us.
                    error_ = "restart: format changed";
                    failed_ = true;
                    break;
                }
                fmt_ = nf;
                if (!mic_) h->readPos = h->writePos;
                else h->writePos = h->readPos;
            }
            const bool active = h->streamActive != 0;
            const int rc = fmt_.channels;
            const int bps = fmt_.bits / 8;

            if (!mic_) {
                // Apps -> ring -> NexTone.
                const int64_t w = h->writePos;
                int64_t r = h->readPos;
                int64_t avail = w - r;
                if (avail < 0 || avail > static_cast<int64_t>(h->dataBytes)) {
                    r = w;
                    avail = 0;
                }
                const size_t frames = static_cast<size_t>(avail / fmt_.block);
                if (frames > 0 && cap_) {
                    tmp.resize(frames * static_cast<size_t>(channels_));
                    for (size_t i = 0; i < frames; ++i) {
                        const int64_t base = r + static_cast<int64_t>(i) * fmt_.block;
                        for (int c = 0; c < channels_; ++c) {
                            float v = 0;
                            if (c < rc) {
                                uint8_t s[4];
                                const int64_t pos = base + c * bps;
                                for (int b = 0; b < bps; ++b) s[b] = ring_.data[(pos + b) & mask];
                                v = sampleToFloat(s, fmt_);
                            }
                            tmp[i * channels_ + c] = v;
                        }
                    }
                    cap_->deviceWrite(tmp.data(), frames, now);
                    h->readPos = r + static_cast<int64_t>(frames) * fmt_.block;
                    lastFeed = now;
                } else if (!active && cap_) {
                    // Nobody is playing into this endpoint: keep the bridge's clock running with silence.
                    const size_t n = static_cast<size_t>((now - lastFeed) * rate_);
                    if (n >= 64) {
                        cap_->deviceWriteSilence(n, now);
                        lastFeed += static_cast<double>(n) / rate_;
                    }
                } else if (now - lastFeed > 0.25) {
                    lastFeed = now;  // stream active but stalled: don't burst later
                }
            } else {
                // NexTone -> ring -> apps recording the virtual microphone.
                const int64_t r = h->readPos;
                int64_t w = h->writePos;
                if (!active) {
                    // Keep the ring empty (lowest latency when an app starts recording) but keep
                    // pulling from the bridge so its clock stays locked.
                    h->writePos = r;
                    const size_t n = static_cast<size_t>((now - lastFeed) * rate_);
                    if (n >= 64 && ren_) {
                        tmp.resize(n * static_cast<size_t>(channels_));
                        ren_->deviceRead(tmp.data(), n, now);
                        lastFeed += static_cast<double>(n) / rate_;
                    }
                    continue;
                }
                if (w - r < 0 || w - r > static_cast<int64_t>(h->dataBytes)) w = r;
                const int64_t fillFrames = (w - r) / fmt_.block;
                const int64_t target = static_cast<int64_t>(rate_ * 0.020);  // driver consumes in ~10 ms chunks
                const int64_t want = target - fillFrames;
                if (want > 0 && ren_) {
                    const size_t n = static_cast<size_t>(want);
                    tmp.resize(n * static_cast<size_t>(channels_));
                    ren_->deviceRead(tmp.data(), n, now);
                    for (size_t i = 0; i < n; ++i) {
                        const int64_t base = w + static_cast<int64_t>(i) * fmt_.block;
                        for (int c = 0; c < rc; ++c) {
                            const float v = c < channels_ ? tmp[i * channels_ + c] : 0.0f;
                            uint8_t s[4];
                            floatToSample(v, s, fmt_);
                            const int64_t pos = base + c * bps;
                            for (int b = 0; b < bps; ++b) ring_.data[(pos + b) & mask] = s[b];
                        }
                    }
                    MemoryBarrier();
                    h->writePos = w + static_cast<int64_t>(n) * fmt_.block;
                }
                lastFeed = now;
            }
        }
        CloseHandle(timer);
        if (cap_) cap_->markStopped();
        if (ren_) ren_->markStopped();
    }

    bool mic_;
    int index_;
    int channels_;
    std::string label_;
    RingMap ring_;
    Fmt fmt_;
    long generation_ = 0;
    double rate_ = 48000;
    std::shared_ptr<CaptureBridge> cap_;
    std::shared_ptr<RenderBridge> ren_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> failed_{false};
    std::string error_;
};

}  // namespace

bool VadBackend::driverPresent() {
    RingMap m;
    return m.open(false, 1) || m.open(true, 1);
}

// The driver can come up after the engine (slow boot, driver install, device re-enabled):
// report a change whenever the set of rings changes, so the virtual devices appear on their own.
bool VadBackend::poll() {
    int present = 0;
    for (int mic = 0; mic < 2; ++mic)
        for (int i = 1; i <= kMaxRings; ++i) {
            RingMap m;
            if (m.open(mic != 0, i)) present |= 1 << (mic * kMaxRings + i - 1);
        }
    const bool changed = present != lastPresent_;
    lastPresent_ = present;
    return changed;
}

std::vector<DeviceInfo> VadBackend::enumerate() {
    std::vector<DeviceInfo> out;
    for (int mic = 0; mic < 2; ++mic) {
        for (int i = 1; i <= kMaxRings; ++i) {
            RingMap m;
            if (!m.open(mic != 0, i)) continue;
            DeviceInfo d;
            d.kind = DeviceKind::Vad;
            d.input = mic == 0;  // outputs are sources for NexTone, mics are destinations
            d.bin = mic ? Bin::VirtualIn : Bin::VirtualOut;
            d.nodeId = std::string(mic ? "vad:in:" : "vad:out:") + std::to_string(i);
            std::wstring wname(m.hdr->name, wcsnlen(m.hdr->name, 64));
            if (wname.empty()) wname = (mic ? L"NexTone Mic " : L"NexTone Output ") + std::to_wstring(i);
            d.name = toUtf8(wname);
            d.subtitle = "NexTone virtual device";
            const Fmt f = readFmt(m.hdr);
            d.channels = std::clamp(f.channels, 1, 8);
            d.sampleRate = f.rate;
            d.channelNames = d.channels == 2 ? std::vector<std::string>{"L", "R"} : std::vector<std::string>{};
            if (d.channelNames.empty())
                for (int c = 0; c < d.channels; ++c) d.channelNames.push_back(std::to_string(c + 1));
            d.appEndpointId = findEndpointId(mic != 0, wname);  // used for per-app redirection
            out.push_back(std::move(d));
        }
    }
    return out;
}

std::unique_ptr<DeviceStream> VadBackend::open(const DeviceInfo& dev, const StreamRequest&, std::string& error) {
    const bool mic = dev.bin == Bin::VirtualIn || (dev.kind == DeviceKind::Vad && !dev.input);
    const auto colon = dev.nodeId.rfind(':');
    const int index = colon == std::string::npos ? 1 : std::atoi(dev.nodeId.c_str() + colon + 1);
    auto s = std::make_unique<VadStream>(mic, index, dev.channels, dev.name);
    if (!s->init(error)) return nullptr;
    return s;
}

}  // namespace nt
