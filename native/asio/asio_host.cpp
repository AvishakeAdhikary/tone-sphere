// ToneSphere ASIO host.
//
// Copyright (C) 2026 Neural Nexus Studios. GPLv3 — see LICENSE in this directory; this
// file uses the Steinberg ASIO SDK under its GPLv3 option.
//
// Threads. ASIO drivers are in-process COM objects, and many assume that every control
// call — init, createBuffers, start, stop, disposeBuffers, Release — arrives on the one
// thread that created them, and that the thread pumps messages. So each loaded driver
// gets a dedicated STA thread with a hidden window and a message loop, and every control
// call is marshalled onto it. The driver calls bufferSwitch on a thread of its own: that
// thread is the engine's audio thread for as long as the stream runs.
//
// The audio callback converts the driver's buffers to float, runs the engine through its
// C ABI, and converts back. It never locks, allocates, logs or calls into Python. Driver
// requests that need real work (a reset, a latency change) are flagged there and serviced
// by the control plane, which sees them in the stream status.
#include "tonesphere_asio.h"

#include <windows.h>
// WIN32_LEAN_AND_MEAN (set for the whole build) leaves out objbase.h, which is where the
// SDK's `interface IASIO` keyword and CoInitialize come from.
#include <objbase.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <avrt.h>

// The SDK's headers: asiosys.h picks the platform, asio.h the types and callbacks,
// iasiodrv.h the COM interface.
#include "asiosys.h"
#include "asio.h"
#include "iasiodrv.h"

#include "convert.h"

namespace {

thread_local std::string t_error;

void set_error(std::string message) { t_error = std::move(message); }

const char* asio_error_name(ASIOError e) {
    switch (e) {
        case ASE_OK: return "ok";
        case ASE_SUCCESS: return "success";
        case ASE_NotPresent: return "hardware input or output is not present or available";
        case ASE_HWMalfunction: return "hardware is malfunctioning";
        case ASE_InvalidParameter: return "invalid input parameter";
        case ASE_InvalidMode: return "hardware is in a bad mode or used in a bad mode";
        case ASE_SPNotAdvancing: return "hardware is not running when sample position is inquired";
        case ASE_NoClock: return "sample clock or rate cannot be determined or is not present";
        case ASE_NoMemory: return "not enough memory for completing the request";
        default: return "unknown ASIO error";
    }
}

std::string describe(IASIO* driver, const char* what, ASIOError e) {
    std::string s = std::string(what) + ": " + asio_error_name(e) + " (" + std::to_string(e) + ")";
    if (driver) {
        char message[128] = {};
        driver->getErrorMessage(message);
        if (message[0]) s += "; the driver says: " + std::string(message);
    }
    return s;
}

// ---- Sample types --------------------------------------------------------------------------

bool is_supported(ASIOSampleType t) {
    switch (t) {
        case ASIOSTInt16LSB: case ASIOSTInt24LSB: case ASIOSTInt32LSB: case ASIOSTFloat32LSB:
        case ASIOSTFloat64LSB: case ASIOSTInt32LSB16: case ASIOSTInt32LSB18: case ASIOSTInt32LSB20:
        case ASIOSTInt32LSB24:
            return true;
        default:
            // Big-endian (MSB) types and DSD are not produced by Windows drivers in practice;
            // a driver that offers one is refused with its type named, not misread.
            return false;
    }
}

uint32_t bytes_of(ASIOSampleType t) {
    switch (t) {
        case ASIOSTInt16LSB: return 2;
        case ASIOSTInt24LSB: return 3;
        case ASIOSTFloat64LSB: return 8;
        default: return 4;
    }
}

uint32_t bits_of(ASIOSampleType t) {
    switch (t) {
        case ASIOSTInt16LSB: case ASIOSTInt32LSB16: return 16;
        case ASIOSTInt32LSB18: return 18;
        case ASIOSTInt32LSB20: return 20;
        case ASIOSTInt24LSB: case ASIOSTInt32LSB24: return 24;
        case ASIOSTFloat64LSB: return 64;
        default: return 32;
    }
}

// Int32LSBnn: a 32-bit container whose value is an nn-bit sample in the low bits.
void to_float(ASIOSampleType t, const void* src, float* dst, size_t n) noexcept {
    using ts::SampleFormat;
    switch (t) {
        case ASIOSTInt16LSB: ts::to_float(SampleFormat::Int16, src, dst, n); return;
        case ASIOSTInt24LSB: ts::to_float(SampleFormat::Int24, src, dst, n); return;
        case ASIOSTInt32LSB: ts::to_float(SampleFormat::Int32, src, dst, n); return;
        case ASIOSTFloat32LSB: ts::to_float(SampleFormat::Float32, src, dst, n); return;
        case ASIOSTFloat64LSB: ts::to_float(SampleFormat::Float64, src, dst, n); return;
        default: {
            const double scale = 1.0 / std::ldexp(1.0, static_cast<int>(bits_of(t)) - 1);
            const auto* p = static_cast<const int32_t*>(src);
            for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(p[i] * scale);
        }
    }
}

void from_float(ASIOSampleType t, const float* src, void* dst, size_t n) noexcept {
    using ts::SampleFormat;
    switch (t) {
        case ASIOSTInt16LSB: ts::from_float(SampleFormat::Int16, src, dst, n); return;
        case ASIOSTInt24LSB: ts::from_float(SampleFormat::Int24, src, dst, n); return;
        case ASIOSTInt32LSB: ts::from_float(SampleFormat::Int32, src, dst, n); return;
        case ASIOSTFloat32LSB: ts::from_float(SampleFormat::Float32, src, dst, n); return;
        case ASIOSTFloat64LSB: ts::from_float(SampleFormat::Float64, src, dst, n); return;
        default: {
            const double full = std::ldexp(1.0, static_cast<int>(bits_of(t)) - 1);
            auto* p = static_cast<int32_t*>(dst);
            for (size_t i = 0; i < n; ++i) {
                const double v = std::nearbyint(std::fmin(1.0, std::fmax(-1.0, static_cast<double>(src[i]))) * full);
                p[i] = static_cast<int32_t>(std::fmin(full - 1.0, std::fmax(-full, v)));
            }
        }
    }
}

// ---- Registry ------------------------------------------------------------------------------

std::string reg_string(HKEY key, const char* value) {
    char buffer[512] = {};
    DWORD size = sizeof(buffer) - 1;
    DWORD type = 0;
    if (RegQueryValueExA(key, value, nullptr, &type, reinterpret_cast<BYTE*>(buffer), &size) != ERROR_SUCCESS ||
        (type != REG_SZ && type != REG_EXPAND_SZ))
        return {};
    return buffer;
}

std::vector<ts_asio_driver> list_drivers() {
    std::vector<ts_asio_driver> drivers;
    HKEY root = nullptr;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SOFTWARE\\ASIO", 0, KEY_READ, &root) != ERROR_SUCCESS) return drivers;
    for (DWORD i = 0;; ++i) {
        char name[256] = {};
        DWORD length = sizeof(name);
        if (RegEnumKeyExA(root, i, name, &length, nullptr, nullptr, nullptr, nullptr) != ERROR_SUCCESS) break;
        HKEY sub = nullptr;
        if (RegOpenKeyExA(root, name, 0, KEY_READ, &sub) != ERROR_SUCCESS) continue;
        ts_asio_driver d{};
        std::snprintf(d.name, sizeof(d.name), "%s", name);
        std::snprintf(d.clsid, sizeof(d.clsid), "%s", reg_string(sub, "CLSID").c_str());
        std::snprintf(d.description, sizeof(d.description), "%s", reg_string(sub, "Description").c_str());
        RegCloseKey(sub);
        const std::string server = std::string("CLSID\\") + d.clsid + "\\InprocServer32";
        HKEY inproc = nullptr;
        if (RegOpenKeyExA(HKEY_CLASSES_ROOT, server.c_str(), 0, KEY_READ, &inproc) == ERROR_SUCCESS) {
            char path[MAX_PATH] = {};
            DWORD size = sizeof(path) - 1;
            if (RegQueryValueExA(inproc, nullptr, nullptr, nullptr, reinterpret_cast<BYTE*>(path), &size) == ERROR_SUCCESS) {
                char expanded[MAX_PATH] = {};
                ExpandEnvironmentStringsA(path, expanded, MAX_PATH);
                std::snprintf(d.dll_path, sizeof(d.dll_path), "%s", expanded);
                d.dll_present = GetFileAttributesA(expanded) != INVALID_FILE_ATTRIBUTES ? 1u : 0u;
            }
            RegCloseKey(inproc);
        }
        drivers.push_back(d);
    }
    RegCloseKey(root);
    return drivers;
}

// ---- The driver's own thread -----------------------------------------------------------------

class StaThread {
public:
    StaThread() {
        wake_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        ready_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        thread_ = std::thread([this] { run(); });
        WaitForSingleObject(ready_, INFINITE);
    }
    ~StaThread() {
        call([this] { quit_ = true; });
        thread_.join();
        CloseHandle(wake_);
        CloseHandle(ready_);
    }

    HWND window() const { return window_; }

    // Run `fn` on this thread and wait for it, or for `timeout_ms`. Control plane only.
    // Returns false if `fn` had not returned by then: it still owns this thread, which must
    // then be abandoned (never destroyed), since it is inside the driver.
    bool call(std::function<void()> fn, DWORD timeout_ms = INFINITE) {
        auto task = std::make_shared<std::function<void()>>(std::move(fn));
        HANDLE done = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back([task, done] {
                (*task)();
                SetEvent(done);
            });
        }
        SetEvent(wake_);
        if (WaitForSingleObject(done, timeout_ms) != WAIT_OBJECT_0) return false;  // `done` goes with the thread
        CloseHandle(done);
        return true;
    }

private:
    void run() {
        CoInitialize(nullptr);
        WNDCLASSW wc{};
        wc.lpfnWndProc = DefWindowProcW;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"ToneSphereAsioHost";
        RegisterClassW(&wc);
        // Hidden, but a real top-level window rather than a message-only one: some drivers
        // parent their control panel to it.
        window_ = CreateWindowExW(0, wc.lpszClassName, L"ToneSphere ASIO", WS_OVERLAPPED, 0, 0, 0, 0, nullptr, nullptr,
                                  wc.hInstance, nullptr);
        SetEvent(ready_);
        while (!quit_) {
            MsgWaitForMultipleObjects(1, &wake_, FALSE, INFINITE, QS_ALLINPUT);
            MSG msg;
            while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            std::deque<std::function<void()>> work;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                work.swap(queue_);
            }
            for (auto& fn : work) fn();
        }
        if (window_) DestroyWindow(window_);
        CoUninitialize();
    }

    std::thread thread_;
    HANDLE wake_ = nullptr;
    HANDLE ready_ = nullptr;
    HWND window_ = nullptr;
    std::mutex mutex_;
    std::deque<std::function<void()>> queue_;
    bool quit_ = false;
};

// Loads a driver by registry name on `thread`. Returns nullptr and sets the error on failure.
IASIO* load(StaThread& thread, const char* name) {
    const auto drivers = list_drivers();
    const ts_asio_driver* found = nullptr;
    for (const auto& d : drivers)
        if (std::strcmp(d.name, name) == 0) found = &d;
    if (!found) {
        set_error(std::string("no ASIO driver named \"") + name + "\" is registered under HKLM\\SOFTWARE\\ASIO");
        return nullptr;
    }
    if (!found->dll_present) {
        set_error(std::string("ASIO driver \"") + name + "\" is registered but its DLL is missing: " + found->dll_path);
        return nullptr;
    }
    CLSID clsid{};
    wchar_t wide[64] = {};
    MultiByteToWideChar(CP_ACP, 0, found->clsid, -1, wide, 64);
    if (FAILED(CLSIDFromString(wide, &clsid))) {
        set_error(std::string("ASIO driver \"") + name + "\" has a malformed CLSID: " + found->clsid);
        return nullptr;
    }

    IASIO* driver = nullptr;
    std::string error;
    thread.call([&] {
        // An ASIO driver exposes IASIO under its own CLSID, not a published interface ID;
        // that is how every ASIO host loads one.
        const HRESULT hr = CoCreateInstance(clsid, nullptr, CLSCTX_INPROC_SERVER, clsid, reinterpret_cast<void**>(&driver));
        if (FAILED(hr)) {
            char code[16];
            std::snprintf(code, sizeof(code), "0x%08lX", static_cast<unsigned long>(hr));
            error = std::string("could not load ASIO driver \"") + name + "\" (" + code + ")";
            driver = nullptr;
            return;
        }
        if (!driver->init(thread.window())) {
            error = describe(driver, "the driver refused to initialise", ASE_NotPresent);
            driver->Release();
            driver = nullptr;
        }
    });
    if (!driver) set_error(error);
    return driver;
}

// ---- The running stream --------------------------------------------------------------------

struct AsioStream;
std::atomic<AsioStream*> g_active{nullptr};

// How long a driver's stop/disposeBuffers/Release may take before it is given up on.
constexpr DWORD kDriverStopTimeoutMs = 5000;

// Set once a driver has been abandoned inside its own stop(). ASIO drivers are in-process
// and often single-instance: loading one again beside the stuck one took the process down
// (0xC0000409), so none is loaded again until the process restarts.
std::atomic<bool> g_abandoned{false};
std::string g_abandoned_name;  // written once, before g_abandoned is set

bool refuse_after_abandonment() {
    if (!g_abandoned.load(std::memory_order_acquire)) return false;
    set_error("the ASIO driver \"" + g_abandoned_name + "\" hung in stop() and was abandoned in this process; " +
              "restart ToneSphere to use ASIO again");
    return true;
}

struct AsioStream {
    ts_engine* engine = nullptr;
    std::unique_ptr<StaThread> thread;
    IASIO* driver = nullptr;
    ts_asio_config config{};
    std::string driver_name;
    long buffer_frames = 0;
    double sample_rate = 0;
    uint32_t max_block = 0;
    bool post_output = false;
    long input_latency = 0;
    long output_latency = 0;

    std::vector<ASIOBufferInfo> buffers;  // inputs first, then outputs
    std::vector<ASIOSampleType> types;
    ASIOCallbacks callbacks{};

    std::unique_ptr<float[]> in_interleaved;
    std::unique_ptr<float[]> out_interleaved;
    std::unique_ptr<float[]> planar;
    ts_port_buffer in_port{};
    ts_port_buffer out_port{};

    std::atomic<uint32_t> state{TS_STREAM_STATE_STARTING};
    std::atomic<uint64_t> blocks{0};
    std::atomic<uint64_t> overloads{0};
    std::atomic<uint64_t> resyncs{0};
    std::atomic<uint32_t> reset_requested{0};
    std::atomic<uint32_t> latencies_changed{0};
    std::atomic<double> rate_changed_to{0.0};
    std::atomic<bool> abandoned{false};
    bool stopped = false;

    void process(long index) noexcept {
        thread_local bool registered = false;
        if (!registered) {
            // Most drivers call back on a thread already at real-time priority; joining
            // MMCSS costs nothing when that is so and matters when it is not.
            DWORD task = 0;
            AvSetMmThreadCharacteristicsW(L"Pro Audio", &task);
            registered = true;
        }
        const uint32_t frames = static_cast<uint32_t>(buffer_frames);
        const uint32_t nin = config.input_count;
        const uint32_t nout = config.output_count;

        for (uint32_t c = 0; c < nin; ++c) {
            to_float(types[c], buffers[c].buffers[index], planar.get(), frames);
            for (uint32_t i = 0; i < frames; ++i) in_interleaved[static_cast<size_t>(i) * nin + c] = planar[i];
        }

        // Equal pieces, not full blocks plus a short remainder: the engine judges load per
        // block against that block's own period, and a remainder's tiny period would make
        // a block's fixed cost look like an overload.
        const uint32_t pieces = (frames + max_block - 1) / max_block;
        const uint32_t block = (frames + pieces - 1) / pieces;
        for (uint32_t done = 0; done < frames;) {
            const uint32_t n = frames - done < block ? frames - done : block;
            in_port.data = in_interleaved.get() + static_cast<size_t>(done) * nin;
            out_port.data = out_interleaved.get() + static_cast<size_t>(done) * nout;
            ts_engine_run_block(engine, nin ? &in_port : nullptr, nin ? 1 : 0, nout ? &out_port : nullptr, nout ? 1 : 0, n);
            done += n;
        }

        for (uint32_t c = 0; c < nout; ++c) {
            for (uint32_t i = 0; i < frames; ++i) planar[i] = out_interleaved[static_cast<size_t>(i) * nout + c];
            from_float(types[nin + c], planar.get(), buffers[nin + c].buffers[index], frames);
        }
        if (post_output) driver->outputReady();
        blocks.fetch_add(1, std::memory_order_relaxed);
    }

    void stop() {
        if (stopped) return;
        stopped = true;
        // A driver that never returns from stop() must not take the control plane with it:
        // ASIO4ALL on a VM's virtual device, after asking for a reset, spun in stop() for
        // good. Past the timeout the driver, its thread and this stream are abandoned —
        // leaked, never freed, since the thread is still inside the driver and holds `this` —
        // and the stream reports why.
        const bool returned = thread->call([this] {
            driver->stop();
            driver->disposeBuffers();
            driver->Release();
            driver = nullptr;
        }, kDriverStopTimeoutMs);
        g_active.store(nullptr);
        ts_engine_set_backend_running(engine, 0);
        if (!returned) {
            g_abandoned_name = driver_name;
            g_abandoned.store(true, std::memory_order_release);
            abandoned.store(true);
            (void)thread.release();
            state.store(TS_STREAM_STATE_FAILED);
            return;
        }
        state.store(TS_STREAM_STATE_STOPPED);
    }

    int32_t status(ts_stream_status* out, int32_t capacity) {
        int32_t n = 0;
        auto fill = [&](uint32_t node, uint32_t kind, uint32_t channels, long latency, ASIOSampleType type) {
            if (n >= capacity) return;
            ts_stream_status& s = out[n++];
            std::memset(&s, 0, sizeof(s));
            s.node_id = node;
            s.kind = kind;
            s.state = state.load();
            s.is_master = 1;
            s.share_mode = TS_SHARE_EXCLUSIVE;  // an ASIO driver is always exclusive to its host
            s.sample_rate = static_cast<uint32_t>(sample_rate);
            s.channels = channels;
            s.bits = bytes_of(type) * 8;
            s.valid_bits = bits_of(type);
            s.is_float = (type == ASIOSTFloat32LSB || type == ASIOSTFloat64LSB) ? 1 : 0;
            s.buffer_frames = static_cast<uint32_t>(buffer_frames);
            s.period_frames = static_cast<uint32_t>(buffer_frames);
            s.stream_latency_hns = sample_rate > 0 ? static_cast<int64_t>(latency * 1e7 / sample_rate) : 0;
            s.frames = blocks.load() * static_cast<uint64_t>(buffer_frames);
            s.glitches = overloads.load() + resyncs.load();
            s.drift_ratio = 1.0;
            std::string message = "ASIO: " + driver_name;
            if (abandoned.load())
                message += "; the driver did not return from stop() within " +
                           std::to_string(kDriverStopTimeoutMs / 1000) + " s and was abandoned, its thread still in it";
            if (reset_requested.load()) message += "; the driver requested a reset: restart the stream";
            if (latencies_changed.load()) message += "; the driver reports its latencies changed";
            if (const double r = rate_changed_to.load(); r > 0)
                message += "; the driver's sample rate changed to " + std::to_string(static_cast<int>(r)) + " Hz";
            std::snprintf(s.error, sizeof(s.error), "%s", message.c_str());
        };
        if (config.input_count) fill(config.input_node, TS_STREAM_CAPTURE, config.input_count, input_latency, types[0]);
        if (config.output_count)
            fill(config.output_node, TS_STREAM_RENDER, config.output_count, output_latency, types[config.input_count]);
        return n;
    }
};

// ASIO's callbacks carry no context: they reach the one active stream through g_active.
void on_buffer_switch(long index, ASIOBool) {
    if (AsioStream* s = g_active.load(std::memory_order_acquire)) s->process(index);
}

ASIOTime* on_buffer_switch_time_info(ASIOTime* time, long index, ASIOBool) {
    if (AsioStream* s = g_active.load(std::memory_order_acquire)) s->process(index);
    return time;
}

void on_sample_rate_changed(ASIOSampleRate rate) {
    if (AsioStream* s = g_active.load(std::memory_order_acquire)) s->rate_changed_to.store(rate);
}

long on_message(long selector, long value, void*, double*) {
    AsioStream* s = g_active.load(std::memory_order_acquire);
    switch (selector) {
        case kAsioSelectorSupported:
            return (value == kAsioResetRequest || value == kAsioEngineVersion || value == kAsioResyncRequest ||
                    value == kAsioLatenciesChanged || value == kAsioSupportsTimeInfo || value == kAsioOverload ||
                    value == kAsioBufferSizeChange) ? 1 : 0;
        case kAsioEngineVersion:
            return 2;
        case kAsioResetRequest:
        case kAsioBufferSizeChange:
            // Never reset from inside a driver callback: flag it, and let the control plane
            // stop and restart the stream on its own thread.
            if (s) s->reset_requested.store(1);
            return 1;
        case kAsioResyncRequest:
            if (s) s->resyncs.fetch_add(1);
            return 1;
        case kAsioLatenciesChanged:
            if (s) s->latencies_changed.store(1);
            return 1;
        case kAsioSupportsTimeInfo:
            return 1;
        case kAsioOverload:
            if (s) {
                s->overloads.fetch_add(1);
                ts_engine_add_xruns(s->engine, 1);
            }
            return 1;
        default:
            return 0;
    }
}

void op_stop(void* context) { static_cast<AsioStream*>(context)->stop(); }
int32_t op_status(void* context, ts_stream_status* out, int32_t capacity) {
    return static_cast<AsioStream*>(context)->status(out, capacity);
}
void op_destroy(void* context) {
    auto* s = static_cast<AsioStream*>(context);
    if (!s->abandoned.load()) delete s;
}

}  // namespace

extern "C" {

TS_ASIO_API int32_t ts_asio_abi_version(void) { return TS_ASIO_ABI_VERSION; }

TS_ASIO_API int32_t ts_asio_last_error(char* buffer, int32_t capacity) {
    if (!buffer || capacity <= 0) return 0;
    const int32_t n = static_cast<int32_t>(t_error.size()) < capacity - 1 ? static_cast<int32_t>(t_error.size()) : capacity - 1;
    std::memcpy(buffer, t_error.data(), static_cast<size_t>(n));
    buffer[n] = '\0';
    return n;
}

TS_ASIO_API int32_t ts_asio_list(ts_asio_driver* out, int32_t capacity) {
    if (capacity < 0 || (capacity && !out)) return TS_ERR_INVALID;
    const auto drivers = list_drivers();
    for (int32_t i = 0; i < capacity && i < static_cast<int32_t>(drivers.size()); ++i) out[i] = drivers[i];
    return static_cast<int32_t>(drivers.size());
}

TS_ASIO_API ts_result ts_asio_query(const char* name, ts_asio_info* out) {
    if (!name || !out) return TS_ERR_INVALID;
    if (refuse_after_abandonment()) return TS_ERR_STATE;
    if (g_active.load()) {
        set_error("an ASIO stream is running; ASIO allows one driver per process");
        return TS_ERR_STATE;
    }
    std::memset(out, 0, sizeof(*out));
    StaThread thread;
    IASIO* driver = load(thread, name);
    if (!driver) return TS_ERR_BACKEND;
    thread.call([&] {
        driver->getDriverName(out->driver_name);
        out->driver_version = driver->getDriverVersion();
        long in = 0, outs = 0;
        driver->getChannels(&in, &outs);
        out->inputs = in;
        out->outputs = outs;
        long mn = 0, mx = 0, pref = 0, gran = 0;
        driver->getBufferSize(&mn, &mx, &pref, &gran);
        out->min_buffer = mn;
        out->max_buffer = mx;
        out->preferred_buffer = pref;
        out->granularity = gran;
        driver->getSampleRate(&out->current_sample_rate);
        for (int i = 0; i < TS_ASIO_RATE_COUNT; ++i)
            if (driver->canSampleRate(TS_ASIO_RATES[i]) == ASE_OK) out->rates_supported |= 1u << i;
        long il = 0, ol = 0;
        driver->getLatencies(&il, &ol);
        out->input_latency = il;
        out->output_latency = ol;
        out->post_output = driver->outputReady() == ASE_OK ? 1 : 0;
        for (int dir = 0; dir < 2; ++dir) {
            const long count = dir == 0 ? in : outs;
            ts_asio_channel* channels = dir == 0 ? out->input_channels : out->output_channels;
            for (long c = 0; c < count && c < TS_ASIO_MAX_CHANNELS; ++c) {
                ASIOChannelInfo info{};
                info.channel = c;
                info.isInput = dir == 0 ? ASIOTrue : ASIOFalse;
                if (driver->getChannelInfo(&info) != ASE_OK) continue;
                std::snprintf(channels[c].name, sizeof(channels[c].name), "%s", info.name);
                channels[c].sample_type = info.type;
                channels[c].group = info.channelGroup;
                channels[c].supported = is_supported(info.type) ? 1u : 0u;
            }
        }
        driver->Release();
    });
    return TS_OK;
}

TS_ASIO_API ts_result ts_asio_start(ts_engine* engine, const ts_asio_config* config) {
    if (!engine || !config) return TS_ERR_INVALID;
    if (refuse_after_abandonment()) return TS_ERR_STATE;
    if (config->input_count > TS_ASIO_MAX_CHANNELS || config->output_count > TS_ASIO_MAX_CHANNELS ||
        (config->input_count == 0 && config->output_count == 0)) {
        set_error("an ASIO stream needs between 1 and 64 input or output channels");
        return TS_ERR_INVALID;
    }
    if ((config->input_count && !config->input_node) || (config->output_count && !config->output_node)) {
        set_error("channels were requested without an engine node to carry them");
        return TS_ERR_INVALID;
    }
    AsioStream* expected = nullptr;
    auto stream = std::make_unique<AsioStream>();
    if (!g_active.compare_exchange_strong(expected, stream.get())) {
        set_error("an ASIO stream is already running; ASIO allows one driver per process");
        return TS_ERR_STATE;
    }
    // Held from here until the stream stops, so a second start is refused throughout.
    auto give_back = [&] { g_active.store(nullptr); };

    stream->engine = engine;
    stream->config = *config;
    stream->driver_name = config->driver;
    stream->max_block = ts_engine_max_block(engine);
    stream->sample_rate = ts_engine_sample_rate(engine);
    stream->thread = std::make_unique<StaThread>();
    stream->driver = load(*stream->thread, config->driver);
    if (!stream->driver) {
        give_back();
        return TS_ERR_BACKEND;
    }

    std::string error;
    AsioStream* s = stream.get();
    s->thread->call([&] {
        IASIO* d = s->driver;
        long in = 0, outs = 0;
        if (ASIOError e = d->getChannels(&in, &outs); e != ASE_OK) { error = describe(d, "reading the channel count", e); return; }
        for (uint32_t i = 0; i < config->input_count; ++i)
            if (config->inputs[i] >= static_cast<uint32_t>(in)) { error = "input channel " + std::to_string(config->inputs[i]) + " does not exist (the driver has " + std::to_string(in) + ")"; return; }
        for (uint32_t i = 0; i < config->output_count; ++i)
            if (config->outputs[i] >= static_cast<uint32_t>(outs)) { error = "output channel " + std::to_string(config->outputs[i]) + " does not exist (the driver has " + std::to_string(outs) + ")"; return; }

        // The engine runs at one rate, fixed when it was created; the driver must match it.
        if (d->canSampleRate(s->sample_rate) != ASE_OK) {
            error = "the driver cannot run at " + std::to_string(static_cast<int>(s->sample_rate)) + " Hz";
            return;
        }
        ASIOSampleRate current = 0;
        d->getSampleRate(&current);
        if (current != s->sample_rate) {
            if (ASIOError e = d->setSampleRate(s->sample_rate); e != ASE_OK) { error = describe(d, "setting the sample rate", e); return; }
        }

        long mn = 0, mx = 0, pref = 0, gran = 0;
        if (ASIOError e = d->getBufferSize(&mn, &mx, &pref, &gran); e != ASE_OK) { error = describe(d, "reading buffer sizes", e); return; }
        long size = config->buffer_frames ? static_cast<long>(config->buffer_frames) : pref;
        bool valid = size >= mn && size <= mx;
        if (valid && gran == -1) valid = (size & (size - 1)) == 0;
        else if (valid && gran > 0) valid = (size - mn) % gran == 0;
        if (!valid) {
            error = "buffer size " + std::to_string(size) + " is not one the driver allows (min " + std::to_string(mn) +
                    ", max " + std::to_string(mx) + ", preferred " + std::to_string(pref) + ", granularity " +
                    std::to_string(gran) + ")";
            return;
        }
        s->buffer_frames = size;

        const uint32_t total = config->input_count + config->output_count;
        s->buffers.resize(total);
        s->types.resize(total);
        for (uint32_t i = 0; i < total; ++i) {
            const bool input = i < config->input_count;
            const uint32_t channel = input ? config->inputs[i] : config->outputs[i - config->input_count];
            s->buffers[i].isInput = input ? ASIOTrue : ASIOFalse;
            s->buffers[i].channelNum = static_cast<long>(channel);
            ASIOChannelInfo info{};
            info.channel = static_cast<long>(channel);
            info.isInput = s->buffers[i].isInput;
            if (ASIOError e = d->getChannelInfo(&info); e != ASE_OK) { error = describe(d, "reading channel info", e); return; }
            if (!is_supported(info.type)) {
                error = std::string(input ? "input" : "output") + " channel " + std::to_string(channel) +
                        " uses ASIO sample type " + std::to_string(info.type) + ", which ToneSphere cannot convert";
                return;
            }
            s->types[i] = info.type;
        }

        s->callbacks.bufferSwitch = on_buffer_switch;
        s->callbacks.sampleRateDidChange = on_sample_rate_changed;
        s->callbacks.asioMessage = on_message;
        s->callbacks.bufferSwitchTimeInfo = on_buffer_switch_time_info;
        if (ASIOError e = d->createBuffers(s->buffers.data(), static_cast<long>(total), size, &s->callbacks); e != ASE_OK) {
            error = describe(d, "creating buffers", e);
            return;
        }

        s->in_interleaved.reset(new float[static_cast<size_t>(size) * (config->input_count ? config->input_count : 1)]());
        s->out_interleaved.reset(new float[static_cast<size_t>(size) * (config->output_count ? config->output_count : 1)]());
        s->planar.reset(new float[static_cast<size_t>(size)]());
        s->in_port = ts_port_buffer{config->input_node, config->input_count, s->in_interleaved.get()};
        s->out_port = ts_port_buffer{config->output_node, config->output_count, s->out_interleaved.get()};
        // Both halves of every output silent before the first switch: whatever the driver
        // plays before the host has written anything must not be old memory.
        for (uint32_t i = config->input_count; i < total; ++i)
            for (int half = 0; half < 2; ++half)
                std::memset(s->buffers[i].buffers[half], 0, static_cast<size_t>(size) * bytes_of(s->types[i]));

        d->getLatencies(&s->input_latency, &s->output_latency);
        s->post_output = d->outputReady() == ASE_OK;

        ts_engine_set_backend_running(s->engine, 1);
        if (ASIOError e = d->start(); e != ASE_OK) {
            ts_engine_set_backend_running(s->engine, 0);
            error = describe(d, "starting the driver", e);
            d->disposeBuffers();
            return;
        }
        s->state.store(TS_STREAM_STATE_RUNNING);
    });

    if (!error.empty()) {
        s->thread->call([s] {
            if (s->driver) s->driver->Release();
            s->driver = nullptr;
        });
        give_back();
        set_error(error);
        return TS_ERR_BACKEND;
    }

    ts_backend_ops ops{stream.get(), op_stop, op_status, op_destroy};
    const ts_result attached = ts_engine_attach_backend(engine, &ops);
    if (attached != TS_OK) {
        stream->stop();
        set_error("the engine refused the backend (is another backend running?)");
        return attached;
    }
    stream.release();  // owned by the engine now; op_destroy frees it
    return TS_OK;
}

TS_ASIO_API ts_result ts_asio_convert_in(int32_t type, const void* src, float* dst, uint32_t samples) {
    if (!src || !dst || !is_supported(type)) return TS_ERR_INVALID;
    to_float(type, src, dst, samples);
    return TS_OK;
}

TS_ASIO_API ts_result ts_asio_convert_out(int32_t type, const float* src, void* dst, uint32_t samples) {
    if (!src || !dst || !is_supported(type)) return TS_ERR_INVALID;
    from_float(type, src, dst, samples);
    return TS_OK;
}

}  // extern "C"
