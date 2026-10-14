// Engine-side ASIO host: loads one real ASIO driver and bridges its channels.
#include "backends/win/asio_host.h"

#include <windows.h>
#include <objbase.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <thread>

#include "asiosys.h"
#include "asio.h"
#include "iasiodrv.h"
#include "util/log.h"
#include "util/platform.h"

namespace ts {

namespace {

struct DriverReg {
    std::string name;
    CLSID clsid{};
};

std::vector<DriverReg> listDrivers() {
    std::vector<DriverReg> out;
    HKEY asio;
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\ASIO", 0, KEY_READ, &asio) != ERROR_SUCCESS) return out;
    wchar_t name[256];
    for (DWORD i = 0;; ++i) {
        DWORD len = 256;
        if (RegEnumKeyExW(asio, i, name, &len, nullptr, nullptr, nullptr, nullptr) != ERROR_SUCCESS) break;
        if (_wcsicmp(name, L"ToneSphere ASIO") == 0) continue;  // never host ourselves
        wchar_t clsid[128];
        DWORD clen = sizeof(clsid);
        if (RegGetValueW(asio, name, L"CLSID", RRF_RT_REG_SZ, nullptr, clsid, &clen) != ERROR_SUCCESS) continue;
        DriverReg d;
        d.name = toUtf8(name);
        if (SUCCEEDED(CLSIDFromString(clsid, &d.clsid))) out.push_back(d);
    }
    RegCloseKey(asio);
    return out;
}

// ---- sample conversion -------------------------------------------------------------
float toFloat(const void* buf, long i, ASIOSampleType t) {
    const auto* b = static_cast<const uint8_t*>(buf);
    switch (t) {
        case ASIOSTFloat32LSB: return reinterpret_cast<const float*>(b)[i];
        case ASIOSTFloat64LSB: return static_cast<float>(reinterpret_cast<const double*>(b)[i]);
        case ASIOSTInt16LSB: return reinterpret_cast<const int16_t*>(b)[i] * (1.0f / 32768.0f);
        case ASIOSTInt24LSB: {
            const uint8_t* p = b + i * 3;
            return static_cast<float>((static_cast<int32_t>(p[2]) << 24) | (p[1] << 16) | (p[0] << 8)) * (1.0f / 2147483648.0f);
        }
        case ASIOSTInt32LSB: return static_cast<float>(reinterpret_cast<const int32_t*>(b)[i]) * (1.0f / 2147483648.0f);
        case ASIOSTInt32LSB16: return static_cast<float>(reinterpret_cast<const int32_t*>(b)[i]) * (1.0f / 32768.0f);
        case ASIOSTInt32LSB18: return static_cast<float>(reinterpret_cast<const int32_t*>(b)[i]) * (1.0f / 131072.0f);
        case ASIOSTInt32LSB20: return static_cast<float>(reinterpret_cast<const int32_t*>(b)[i]) * (1.0f / 524288.0f);
        case ASIOSTInt32LSB24: return static_cast<float>(reinterpret_cast<const int32_t*>(b)[i]) * (1.0f / 8388608.0f);
        default: return 0.0f;
    }
}

void fromFloat(void* buf, long i, ASIOSampleType t, float v) {
    v = v > 1.0f ? 1.0f : v < -1.0f ? -1.0f : v;
    auto* b = static_cast<uint8_t*>(buf);
    switch (t) {
        case ASIOSTFloat32LSB: reinterpret_cast<float*>(b)[i] = v; break;
        case ASIOSTFloat64LSB: reinterpret_cast<double*>(b)[i] = v; break;
        case ASIOSTInt16LSB: reinterpret_cast<int16_t*>(b)[i] = static_cast<int16_t>(v * 32767.0f); break;
        case ASIOSTInt24LSB: {
            const int32_t x = static_cast<int32_t>(v * 8388607.0f);
            uint8_t* p = b + i * 3;
            p[0] = static_cast<uint8_t>(x);
            p[1] = static_cast<uint8_t>(x >> 8);
            p[2] = static_cast<uint8_t>(x >> 16);
            break;
        }
        case ASIOSTInt32LSB: reinterpret_cast<int32_t*>(b)[i] = static_cast<int32_t>(static_cast<double>(v) * 2147483647.0); break;
        case ASIOSTInt32LSB16: reinterpret_cast<int32_t*>(b)[i] = static_cast<int32_t>(v * 32767.0f); break;
        case ASIOSTInt32LSB18: reinterpret_cast<int32_t*>(b)[i] = static_cast<int32_t>(v * 131071.0f); break;
        case ASIOSTInt32LSB20: reinterpret_cast<int32_t*>(b)[i] = static_cast<int32_t>(v * 524287.0f); break;
        case ASIOSTInt32LSB24: reinterpret_cast<int32_t*>(b)[i] = static_cast<int32_t>(v * 8388607.0f); break;
        default: break;
    }
}

int bytesPerSample(ASIOSampleType t) {
    switch (t) {
        case ASIOSTInt16LSB: return 2;
        case ASIOSTInt24LSB: return 3;
        case ASIOSTFloat64LSB: return 8;
        default: return 4;
    }
}

// ---- STA worker thread ------------------------------------------------------------
class StaThread {
public:
    StaThread() {
        thread_ = std::thread([this] { run(); });
        std::unique_lock lk(mu_);
        cv_.wait(lk, [this] { return ready_; });
    }
    ~StaThread() {
        {
            std::lock_guard lk(mu_);
            quit_ = true;
        }
        PostThreadMessageW(tid_, WM_NULL, 0, 0);
        if (thread_.joinable()) thread_.join();
    }
    HWND hwnd() const { return hwnd_; }
    // Runs fn on the STA thread and waits (bounded) for it.
    bool call(std::function<void()> fn, DWORD timeoutMs = 15000) {
        auto task = std::make_shared<std::packaged_task<void()>>(std::move(fn));
        auto fut = task->get_future();
        {
            std::lock_guard lk(mu_);
            tasks_.push_back([task] { (*task)(); });
        }
        PostThreadMessageW(tid_, WM_NULL, 0, 0);
        return fut.wait_for(std::chrono::milliseconds(timeoutMs)) == std::future_status::ready;
    }

private:
    void run() {
        setThreadName("ts-asio-sta");
        CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
        tid_ = GetCurrentThreadId();
        MSG m;
        PeekMessageW(&m, nullptr, WM_USER, WM_USER, PM_NOREMOVE);  // create the queue
        hwnd_ = CreateWindowExW(0, L"STATIC", L"ToneSphereAsioHost", 0, 0, 0, 0, 0, HWND_MESSAGE, nullptr, nullptr, nullptr);
        {
            std::lock_guard lk(mu_);
            ready_ = true;
        }
        cv_.notify_all();
        for (;;) {
            MsgWaitForMultipleObjects(0, nullptr, FALSE, 100, QS_ALLINPUT);
            while (PeekMessageW(&m, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&m);
                DispatchMessageW(&m);
            }
            std::deque<std::function<void()>> todo;
            {
                std::lock_guard lk(mu_);
                todo.swap(tasks_);
                if (quit_ && todo.empty()) break;
            }
            for (auto& t : todo) t();
        }
        if (hwnd_) DestroyWindow(hwnd_);
        CoUninitialize();
    }
    std::thread thread_;
    DWORD tid_ = 0;
    HWND hwnd_ = nullptr;
    std::mutex mu_;
    std::condition_variable cv_;
    bool ready_ = false, quit_ = false;
    std::deque<std::function<void()>> tasks_;
};

// ---- the loaded driver (one per process) --------------------------------------------
class AsioDevice {
public:
    static AsioDevice& get() {
        static AsioDevice d;
        return d;
    }

    StaThread& sta() {
        if (!sta_) sta_ = std::make_unique<StaThread>();
        return *sta_;
    }

    // STA thread only.
    bool loadOnSta(const DriverReg& reg, double wantRate, std::string& err) {
        if (drv_ && loaded_ == reg.name) return true;
        unloadOnSta();
        IASIO* d = nullptr;
        if (FAILED(CoCreateInstance(reg.clsid, nullptr, CLSCTX_INPROC_SERVER, reg.clsid, reinterpret_cast<void**>(&d))) || !d) {
            err = "Cannot load the ASIO driver";
            return false;
        }
        if (!d->init(sta().hwnd())) {
            char msg[128] = {};
            d->getErrorMessage(msg);
            err = std::string("ASIO driver failed to start: ") + msg;
            d->Release();
            return false;
        }
        long ins = 0, outs = 0;
        d->getChannels(&ins, &outs);
        long mn = 0, mx = 0, pref = 0, gran = 0;
        d->getBufferSize(&mn, &mx, &pref, &gran);
        if (wantRate > 0 && d->canSampleRate(wantRate) == ASE_OK) d->setSampleRate(wantRate);
        ASIOSampleRate rate = 0;
        d->getSampleRate(&rate);
        drv_ = d;
        loaded_ = reg.name;
        inputs_ = std::clamp<long>(ins, 0, kMaxChannels);
        outputs_ = std::clamp<long>(outs, 0, kMaxChannels);
        bufferSize_ = pref > 0 ? pref : 512;
        rate_ = rate > 0 ? rate : 48000.0;
        bufs_.clear();
        for (long c = 0; c < inputs_; ++c) bufs_.push_back({ASIOTrue, c, {nullptr, nullptr}});
        for (long c = 0; c < outputs_; ++c) bufs_.push_back({ASIOFalse, c, {nullptr, nullptr}});
        static ASIOCallbacks cb{&AsioDevice::onBufferSwitch, &AsioDevice::onRateChange, &AsioDevice::onMessage, &AsioDevice::onBufferSwitchTimeInfo};
        if (bufs_.empty() || d->createBuffers(bufs_.data(), static_cast<long>(bufs_.size()), bufferSize_, &cb) != ASE_OK) {
            err = "The ASIO driver refused the buffer setup";
            unloadOnSta();
            return false;
        }
        types_.clear();
        inNames_.clear();
        outNames_.clear();
        for (auto& b : bufs_) {
            ASIOChannelInfo ci{};
            ci.channel = b.channelNum;
            ci.isInput = b.isInput;
            d->getChannelInfo(&ci);
            types_.push_back(ci.type);
            (b.isInput ? inNames_ : outNames_).push_back(ci.name[0] ? ci.name : std::to_string(b.channelNum + 1));
        }
        inTmp_.assign(static_cast<size_t>(bufferSize_) * std::max<long>(1, inputs_), 0.0f);
        outTmp_.assign(static_cast<size_t>(bufferSize_) * std::max<long>(1, outputs_), 0.0f);
        outputReady_ = d->outputReady() == ASE_OK;
        long il = 0, ol = 0;
        d->getLatencies(&il, &ol);
        inLatencyMs_ = il / rate_ * 1000.0;
        outLatencyMs_ = ol / rate_ * 1000.0;
        reset_ = false;
        TS_INFO("ASIO: loaded %s (%ld in, %ld out, %.0f Hz, buffer %ld)", reg.name.c_str(), inputs_, outputs_, rate_, bufferSize_);
        return true;
    }

    void unloadOnSta() {
        if (!drv_) return;
        if (started_) drv_->stop();
        started_ = false;
        drv_->disposeBuffers();
        drv_->Release();
        drv_ = nullptr;
        TS_INFO("ASIO: unloaded %s", loaded_.c_str());
        loaded_.clear();
    }

    void attach(bool input, CaptureBridge* cap, RenderBridge* ren) {
        if (input) cap_.store(cap);
        else ren_.store(ren);
        if (!started_ && drv_) {
            sta().call([this] {
                if (drv_ && !started_ && drv_->start() == ASE_OK) started_ = true;
            });
        }
    }
    void detach(bool input) {
        if (input) cap_.store(nullptr);
        else ren_.store(nullptr);
        while (inCallback_.load() > 0) std::this_thread::yield();
        if (!cap_.load() && !ren_.load()) sta().call([this] { unloadOnSta(); });
    }

    std::string loaded_;
    long inputs_ = 0, outputs_ = 0, bufferSize_ = 0;
    double rate_ = 48000.0, inLatencyMs_ = 0, outLatencyMs_ = 0;
    std::vector<std::string> inNames_, outNames_;
    std::atomic<bool> reset_{false};
    std::map<std::string, std::pair<int, int>> knownChannels_;  // driver -> (in, out) seen before
    IASIO* drv_ = nullptr;

private:
    void process(long index) {
        inCallback_.fetch_add(1);
        CaptureBridge* cap = cap_.load();
        RenderBridge* ren = ren_.load();
        const double now = nowSeconds();
        const long n = bufferSize_;
        if (cap && inputs_ > 0) {
            for (size_t k = 0; k < bufs_.size(); ++k) {
                if (!bufs_[k].isInput) continue;
                const long ch = bufs_[k].channelNum;
                for (long i = 0; i < n; ++i) inTmp_[static_cast<size_t>(i) * inputs_ + ch] = toFloat(bufs_[k].buffers[index], i, types_[k]);
            }
            cap->deviceWrite(inTmp_.data(), static_cast<size_t>(n), now);
        }
        if (outputs_ > 0) {
            if (ren) ren->deviceRead(outTmp_.data(), static_cast<size_t>(n), now);
            else std::fill(outTmp_.begin(), outTmp_.end(), 0.0f);
            for (size_t k = 0; k < bufs_.size(); ++k) {
                if (bufs_[k].isInput) continue;
                const long ch = bufs_[k].channelNum;
                for (long i = 0; i < n; ++i) fromFloat(bufs_[k].buffers[index], i, types_[k], outTmp_[static_cast<size_t>(i) * outputs_ + ch]);
            }
        }
        if (outputReady_ && drv_) drv_->outputReady();
        inCallback_.fetch_sub(1);
    }

    static void onBufferSwitch(long index, ASIOBool) { get().process(index); }
    static ASIOTime* onBufferSwitchTimeInfo(ASIOTime* t, long index, ASIOBool) {
        get().process(index);
        return t;
    }
    static void onRateChange(ASIOSampleRate) { get().reset_ = true; }
    static long onMessage(long selector, long value, void*, double*) {
        switch (selector) {
            case kAsioSelectorSupported:
                return value == kAsioResetRequest || value == kAsioEngineVersion || value == kAsioResyncRequest ||
                       value == kAsioLatenciesChanged || value == kAsioSupportsTimeInfo || value == kAsioBufferSizeChange;
            case kAsioEngineVersion: return 2;
            case kAsioResetRequest:
            case kAsioBufferSizeChange:
                get().reset_ = true;
                return 1;
            case kAsioResyncRequest:
            case kAsioLatenciesChanged: return 1;
            case kAsioSupportsTimeInfo: return 1;
            default: return 0;
        }
    }

    std::unique_ptr<StaThread> sta_;
    std::vector<ASIOBufferInfo> bufs_;
    std::vector<ASIOSampleType> types_;
    std::vector<float> inTmp_, outTmp_;
    bool started_ = false;
    bool outputReady_ = false;
    std::atomic<CaptureBridge*> cap_{nullptr};
    std::atomic<RenderBridge*> ren_{nullptr};
    std::atomic<int> inCallback_{0};
};

class AsioHostStream final : public DeviceStream {
public:
    explicit AsioHostStream(bool input) : input_(input) {}
    ~AsioHostStream() override { stop(); }
    double sampleRate() const override { return AsioDevice::get().rate_; }
    int channels() const override {
        const auto& d = AsioDevice::get();
        return std::max<int>(1, static_cast<int>(input_ ? d.inputs_ : d.outputs_));
    }
    int periodFrames() const override { return static_cast<int>(AsioDevice::get().bufferSize_); }
    double deviceLatencyMs() const override { return input_ ? AsioDevice::get().inLatencyMs_ : AsioDevice::get().outLatencyMs_; }
    bool start(std::shared_ptr<CaptureBridge> cap, std::shared_ptr<RenderBridge> ren) override {
        cap_ = std::move(cap);
        ren_ = std::move(ren);
        if (cap_) cap_->markStarted();
        if (ren_) ren_->markStarted();
        AsioDevice::get().attach(input_, cap_.get(), ren_.get());
        started_ = true;
        return true;
    }
    void stop() override {
        if (!started_) return;
        started_ = false;
        AsioDevice::get().detach(input_);
        if (cap_) cap_->markStopped();
        if (ren_) ren_->markStopped();
    }
    std::string error() const override { return AsioDevice::get().reset_.load() ? "restart: ASIO driver reset" : std::string(); }

private:
    bool input_;
    bool started_ = false;
    std::shared_ptr<CaptureBridge> cap_;
    std::shared_ptr<RenderBridge> ren_;
};

double g_preferredRate = 48000.0;

}  // namespace

AsioHostBackend::AsioHostBackend() = default;
AsioHostBackend::~AsioHostBackend() {
    auto& d = AsioDevice::get();
    if (d.drv_) d.sta().call([&d] { d.unloadOnSta(); });
}

void AsioHostBackend::setPreferred(double sampleRate, int) { g_preferredRate = sampleRate; }

std::vector<DeviceInfo> AsioHostBackend::enumerate() {
    std::vector<DeviceInfo> out;
    auto& dev = AsioDevice::get();
    for (const DriverReg& r : listDrivers()) {
        int ins = 2, outs = 2;
        std::vector<std::string> inNames, outNames;
        if (dev.loaded_ == r.name) {
            ins = static_cast<int>(dev.inputs_);
            outs = static_cast<int>(dev.outputs_);
            inNames = dev.inNames_;
            outNames = dev.outNames_;
            dev.knownChannels_[r.name] = {ins, outs};
        } else if (auto it = dev.knownChannels_.find(r.name); it != dev.knownChannels_.end()) {
            ins = it->second.first;
            outs = it->second.second;
        }
        for (int dir = 0; dir < 2; ++dir) {
            const bool input = dir == 0;
            const int ch = input ? ins : outs;
            if (ch <= 0) continue;
            DeviceInfo d;
            d.kind = DeviceKind::Asio;
            d.input = input;
            d.bin = input ? Bin::PhysicalIn : Bin::PhysicalOut;
            d.nodeId = std::string(input ? "asio:in:" : "asio:out:") + r.name;
            d.endpointId = r.name;
            d.name = r.name + (input ? " inputs" : " outputs");
            d.subtitle = "ASIO";
            d.channels = ch;
            d.channelNames = input ? inNames : outNames;
            if (static_cast<int>(d.channelNames.size()) != ch) {
                d.channelNames.clear();
                for (int c = 0; c < ch; ++c) d.channelNames.push_back(std::to_string(c + 1));
            }
            d.hasControlPanel = true;
            out.push_back(std::move(d));
        }
    }
    return out;
}

std::unique_ptr<DeviceStream> AsioHostBackend::open(const DeviceInfo& dev, const StreamRequest&, std::string& error) {
    auto& d = AsioDevice::get();
    DriverReg reg;
    for (const DriverReg& r : listDrivers())
        if (r.name == dev.endpointId) reg = r;
    if (reg.name.empty()) {
        error = "ASIO driver not installed";
        return nullptr;
    }
    if (!d.loaded_.empty() && d.loaded_ != reg.name) {
        error = "Another ASIO device (" + d.loaded_ + ") is in use. Only one ASIO device can be used at a time.";
        return nullptr;
    }
    bool ok = false;
    std::string err;
    if (!d.sta().call([&] { ok = d.loadOnSta(reg, g_preferredRate, err); })) err = "The ASIO driver stopped responding";
    if (!ok) {
        error = err;
        return nullptr;
    }
    d.knownChannels_[reg.name] = {static_cast<int>(d.inputs_), static_cast<int>(d.outputs_)};
    if ((dev.input ? d.inputs_ : d.outputs_) <= 0) {
        error = dev.input ? "This ASIO device has no inputs" : "This ASIO device has no outputs";
        return nullptr;
    }
    return std::make_unique<AsioHostStream>(dev.input);
}

bool AsioHostBackend::poll() {
    // Channel counts become known when a driver loads: refresh the device list once.
    auto& d = AsioDevice::get();
    static std::string lastSig;
    const std::string sig = d.loaded_ + ":" + std::to_string(d.inputs_) + ":" + std::to_string(d.outputs_);
    if (sig == lastSig) return false;
    lastSig = sig;
    return true;
}

bool AsioHostBackend::controlPanel(const std::string& driverName, std::string& error) {
    auto& d = AsioDevice::get();
    if (d.loaded_ != driverName || !d.drv_) {
        error = "Connect the ASIO device first, then open its control panel.";
        return false;
    }
    d.sta().call([&d] { d.drv_->controlPanel(); }, 2000);
    return true;
}

}  // namespace ts
