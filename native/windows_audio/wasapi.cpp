// WASAPI endpoint enumeration, device notifications, and the event-driven device backend.
//
// Threads. Every stream has its own thread, joined to the MTA and registered with MMCSS
// ("Pro Audio"). The master stream's thread is the engine's audio thread: on each device
// event it asks the engine for exactly as many frames as the device wants, in blocks of at
// most max_block. Every other stream (a "satellite") runs on its own device clock and meets
// the master through a FrameRing, read through a DriftResampler on the consuming side, so
// two clocks that disagree by a few ppm neither drift apart nor block each other.
//
// Nothing on any stream thread allocates, locks or logs once the stream is running.
// Failures are recorded as an HRESULT plus a message in the stream's status, where the
// control plane reads them.
#include "wasapi.h"

#include <windows.h>
// clang-format off
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <audioclientactivationparams.h>
#include <functiondiscoverykeys_devpkey.h>
#include <avrt.h>
#include <ksmedia.h>
// clang-format on
#include <propkeydef.h>

#include <algorithm>
#include <cstring>
#include <cwchar>

#include "convert.h"
#include "engine.h"
#include "resampler.h"
#include "rt_alloc.h"

namespace ts::wasapi {

namespace {

template <typename T>
void release(T*& p) {
    if (p) {
        p->Release();
        p = nullptr;
    }
}

struct ComScope {
    HRESULT hr;
    ComScope() : hr(CoInitializeEx(nullptr, COINIT_MULTITHREADED)) {}
    ~ComScope() {
        if (SUCCEEDED(hr)) CoUninitialize();
    }
    // A thread that is already an STA (Qt's GUI thread is one) cannot join the MTA, but
    // MMDevice and IAudioClient work from an STA too, so that is not a failure.
    bool ok() const { return SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE; }
};

std::string narrow(const wchar_t* w) {
    if (!w) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

std::string describe(HRESULT hr) {
    struct Name { HRESULT hr; const char* text; };
    static const Name names[] = {
        {AUDCLNT_E_DEVICE_INVALIDATED, "the device was removed or disabled"},
        {AUDCLNT_E_DEVICE_IN_USE, "another application holds the device in exclusive mode"},
        {AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED, "exclusive mode is disabled for this device in Windows sound settings"},
        {AUDCLNT_E_UNSUPPORTED_FORMAT, "the device does not support the requested format"},
        {AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED, "the requested buffer is not aligned to the device"},
        {AUDCLNT_E_BUFFER_SIZE_ERROR, "the requested buffer size is outside what the device allows"},
        {AUDCLNT_E_CPUUSAGE_EXCEEDED, "the audio engine exceeded its CPU budget"},
        {AUDCLNT_E_SERVICE_NOT_RUNNING, "the Windows Audio service is not running"},
        {AUDCLNT_E_ENDPOINT_CREATE_FAILED, "the endpoint could not be created"},
        {AUDCLNT_E_UNSUPPORTED_FORMAT, "unsupported format"},
        {E_ACCESSDENIED, "access denied (check Windows privacy settings for microphone access)"},
        {E_NOTFOUND, "no such endpoint"},
        {static_cast<HRESULT>(0x80070490), "no such endpoint"},
    };
    for (const auto& n : names)
        if (n.hr == hr) return n.text;
    char* text = nullptr;
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr,
                   static_cast<DWORD>(hr), 0, reinterpret_cast<char*>(&text), 0, nullptr);
    std::string s = text ? text : "unknown error";
    if (text) LocalFree(text);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == '.')) s.pop_back();
    return s;
}

std::string fmt_hr(const std::string& what, HRESULT hr) {
    char code[16];
    std::snprintf(code, sizeof(code), "0x%08lX", static_cast<unsigned long>(hr));
    return what + ": " + describe(hr) + " (" + code + ")";
}

void copy_w(uint16_t* dst, size_t capacity, const wchar_t* src) {
    size_t i = 0;
    if (src)
        for (; i + 1 < capacity && src[i]; ++i) dst[i] = static_cast<uint16_t>(src[i]);
    dst[i] = 0;
}

// PKEY_Devices_AudioDevice_RawProcessingSupported, from devpkey.h's audio section; spelled
// out because the header that defines it drags in the whole device-property catalogue.
const PROPERTYKEY kRawProcessingSupported = {{0x8943b373, 0x388c, 0x4395, {0xb5, 0x57, 0xbc, 0x6d, 0xba, 0xff, 0xaf, 0xdb}}, 2};

bool raw_supported(IMMDevice* device) {
    IPropertyStore* props = nullptr;
    if (FAILED(device->OpenPropertyStore(STGM_READ, &props))) return false;
    PROPVARIANT v;
    PropVariantInit(&v);
    const bool supported = SUCCEEDED(props->GetValue(kRawProcessingSupported, &v)) && v.vt == VT_BOOL && v.boolVal;
    PropVariantClear(&v);
    props->Release();
    return supported;
}

const GUID kSubtypeFloat = KSDATAFORMAT_SUBTYPE_IEEE_FLOAT;
const GUID kSubtypePcm = KSDATAFORMAT_SUBTYPE_PCM;

bool is_float_format(const WAVEFORMATEX* f) {
    if (f->wFormatTag == WAVE_FORMAT_IEEE_FLOAT) return true;
    if (f->wFormatTag == WAVE_FORMAT_EXTENSIBLE)
        return IsEqualGUID(reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(f)->SubFormat, kSubtypeFloat);
    return false;
}

uint32_t valid_bits_of(const WAVEFORMATEX* f) {
    if (f->wFormatTag == WAVE_FORMAT_EXTENSIBLE)
        return reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(f)->Samples.wValidBitsPerSample;
    return f->wBitsPerSample;
}

bool sample_format_of(const WAVEFORMATEX* f, SampleFormat& out) {
    const bool flt = is_float_format(f);
    switch (f->wBitsPerSample) {
        case 16: if (flt) return false; out = SampleFormat::Int16; return true;
        case 24: if (flt) return false; out = SampleFormat::Int24; return true;
        case 32: out = flt ? SampleFormat::Float32 : SampleFormat::Int32; return true;
        case 64: if (!flt) return false; out = SampleFormat::Float64; return true;
    }
    return false;
}

DWORD default_mask(uint32_t channels) {
    switch (channels) {
        case 1: return SPEAKER_FRONT_CENTER;
        case 2: return SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT;
        case 4: return KSAUDIO_SPEAKER_QUAD;
        case 6: return KSAUDIO_SPEAKER_5POINT1;
        case 8: return KSAUDIO_SPEAKER_7POINT1_SURROUND;
        default: return 0;
    }
}

WAVEFORMATEXTENSIBLE make_format(uint32_t rate, uint32_t channels, uint32_t container_bits, uint32_t valid_bits,
                                 bool flt, DWORD mask) {
    WAVEFORMATEXTENSIBLE f{};
    f.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
    f.Format.nChannels = static_cast<WORD>(channels);
    f.Format.nSamplesPerSec = rate;
    f.Format.wBitsPerSample = static_cast<WORD>(container_bits);
    f.Format.nBlockAlign = static_cast<WORD>(channels * container_bits / 8);
    f.Format.nAvgBytesPerSec = rate * f.Format.nBlockAlign;
    f.Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX);
    f.Samples.wValidBitsPerSample = static_cast<WORD>(valid_bits);
    f.dwChannelMask = mask ? mask : default_mask(channels);
    f.SubFormat = flt ? kSubtypeFloat : kSubtypePcm;
    return f;
}

HRESULT open_device(const wchar_t* id, IMMDevice** device) {
    IMMDeviceEnumerator* enumerator = nullptr;
    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                                  reinterpret_cast<void**>(&enumerator));
    if (FAILED(hr)) return hr;
    hr = enumerator->GetDevice(id, device);
    enumerator->Release();
    return hr;
}

// ---- Process loopback activation -------------------------------------------------------

class ActivationHandler final : public IActivateAudioInterfaceCompletionHandler, public IAgileObject {
public:
    ActivationHandler() : done_(CreateEventW(nullptr, TRUE, FALSE, nullptr)) {}
    ~ActivationHandler() { CloseHandle(done_); }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IActivateAudioInterfaceCompletionHandler)) {
            *out = static_cast<IActivateAudioInterfaceCompletionHandler*>(this);
        } else if (riid == __uuidof(IAgileObject)) {
            // Without IAgileObject, activation fails with E_ILLEGAL_METHOD_CALL.
            *out = static_cast<IAgileObject*>(this);
        } else {
            *out = nullptr;
            return E_NOINTERFACE;
        }
        AddRef();
        return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return InterlockedIncrement(&refs_); }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG n = InterlockedDecrement(&refs_);
        if (n == 0) delete this;
        return n;
    }
    HRESULT STDMETHODCALLTYPE ActivateCompleted(IActivateAudioInterfaceAsyncOperation* op) override {
        IUnknown* unknown = nullptr;
        HRESULT activated = E_FAIL;
        result_ = op->GetActivateResult(&activated, &unknown);
        if (SUCCEEDED(result_)) result_ = activated;
        if (SUCCEEDED(result_) && unknown) result_ = unknown->QueryInterface(__uuidof(IAudioClient), reinterpret_cast<void**>(&client_));
        if (unknown) unknown->Release();
        SetEvent(done_);
        return S_OK;
    }

    HANDLE done() const { return done_; }
    HRESULT result() const { return result_; }
    IAudioClient* take() {
        IAudioClient* c = client_;
        client_ = nullptr;
        return c;
    }

private:
    LONG refs_ = 1;
    HANDLE done_;
    HRESULT result_ = E_PENDING;
    IAudioClient* client_ = nullptr;
};

HRESULT activate_process_loopback(uint32_t pid, bool include_tree, IAudioClient** client) {
    AUDIOCLIENT_ACTIVATION_PARAMS params{};
    params.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK;
    params.ProcessLoopbackParams.TargetProcessId = pid;
    params.ProcessLoopbackParams.ProcessLoopbackMode =
        include_tree ? PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE : PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE;
    PROPVARIANT variant{};
    variant.vt = VT_BLOB;
    variant.blob.cbSize = sizeof(params);
    variant.blob.pBlobData = reinterpret_cast<BYTE*>(&params);

    auto* handler = new ActivationHandler();
    IActivateAudioInterfaceAsyncOperation* op = nullptr;
    HRESULT hr = ActivateAudioInterfaceAsync(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, __uuidof(IAudioClient), &variant,
                                             handler, &op);
    if (SUCCEEDED(hr)) {
        hr = WaitForSingleObject(handler->done(), 5000) == WAIT_OBJECT_0 ? handler->result() : HRESULT_FROM_WIN32(ERROR_TIMEOUT);
        if (SUCCEEDED(hr)) *client = handler->take();
    }
    if (op) op->Release();
    handler->Release();
    return hr;
}

// ---- Streams ----------------------------------------------------------------------------

struct Stream {
    ts_stream_desc desc{};
    std::wstring device_id;
    bool master = false;
    bool render = false;

    IAudioClient* client = nullptr;
    IAudioRenderClient* render_client = nullptr;
    IAudioCaptureClient* capture_client = nullptr;
    HANDLE event = nullptr;
    bool polled = false;  // loopback: Windows signals no event while nothing plays, so poll

    SampleFormat format = SampleFormat::Float32;
    uint32_t channels = 0;
    uint32_t rate = 0;
    uint32_t bits = 0;
    uint32_t valid_bits = 0;
    bool is_float = false;
    uint32_t block_align = 0;
    uint32_t buffer_frames = 0;
    uint32_t period_frames = 0;
    bool exclusive = false;
    bool raw = false;
    bool raw_capable = false;
    REFERENCE_TIME latency = 0;

    // Satellite side of the clock boundary.
    std::unique_ptr<FrameRing> ring;
    std::unique_ptr<DriftResampler> resampler;
    std::unique_ptr<float[]> scratch;   // interleaved float, max_block * channels
    std::unique_ptr<float[]> convert;   // device-format staging for satellites, buffer_frames * channels

    std::thread thread;
    HANDLE ready = nullptr;
    std::atomic<uint32_t> state{TS_STREAM_STATE_STARTING};
    std::atomic<int32_t> hresult{0};
    // `error` is written only by fail(), and published by its release-store of FAILED;
    // status() reads it only after seeing FAILED. `note` is written only before the
    // stream signals `ready`, so it is stable by the time anyone can read it.
    std::string error;
    std::string note;

    std::atomic<uint64_t> frames{0};
    std::atomic<uint64_t> glitches{0};
    std::atomic<uint64_t> underruns{0};
    std::atomic<uint64_t> overruns{0};
    std::atomic<double> drift{1.0};

    ~Stream() {
        if (ready) CloseHandle(ready);
        if (event) CloseHandle(event);
    }

    void fail(const std::string& what, HRESULT hr) {
        error = fmt_hr(what, hr);
        hresult.store(hr);
        state.store(TS_STREAM_STATE_FAILED, std::memory_order_release);
    }
};

void add(std::atomic<uint64_t>& a, uint64_t n) { a.fetch_add(n, std::memory_order_relaxed); }

// The engine block to cut a device buffer into: equal pieces, never full blocks plus a
// short remainder. Load is judged per block against that block's own period, and a
// 144-frame exclusive period cut 128 + 16 would give the 16-frame piece a 0.33 ms budget
// that the fixed cost of a block alone can exceed, reporting an overload that never
// happened (measured on the AI-04: 152 % for a period that finished in 2 % of its time).
uint32_t even_block(uint32_t frames, uint32_t max_block) {
    const uint32_t pieces = (frames + max_block - 1) / max_block;
    return pieces ? (frames + pieces - 1) / pieces : 0;
}

}  // namespace

class Backend final : public DeviceBackend {
public:
    Backend(Engine& engine) : engine_(engine), stop_(CreateEventW(nullptr, TRUE, FALSE, nullptr)),
                              go_(CreateEventW(nullptr, TRUE, FALSE, nullptr)) {}
    ~Backend() override {
        stop();
        CloseHandle(stop_);
        CloseHandle(go_);
    }

    bool start(const ts_stream_desc* descs, uint32_t count, uint32_t master, std::string& error);
    void stop() override;
    int32_t status(ts_stream_status* out, int32_t capacity) override;

private:
    void run(Stream& s);
    HRESULT initialize(Stream& s);
    HRESULT initialize_exclusive(Stream& s, IMMDevice* device);
    HRESULT initialize_shared(Stream& s, bool loopback);
    void master_render(Stream& s);
    void master_capture(Stream& s);
    void satellite_render(Stream& s);
    void satellite_capture(Stream& s);
    void run_engine(uint32_t frames, const float* master_input, float* master_output) noexcept;
    bool wait(Stream& s);
    void teardown(Stream& s);

    Engine& engine_;
    std::vector<std::unique_ptr<Stream>> streams_;
    Stream* master_ = nullptr;
    HANDLE stop_;
    HANDLE go_;
    bool stopped_ = false;

    // Bound once at start, read only by the master thread.
    std::vector<ts_port_buffer> inputs_;
    std::vector<ts_port_buffer> outputs_;
    std::vector<Stream*> capture_satellites_;
    std::vector<Stream*> render_satellites_;
    std::unique_ptr<float[]> master_scratch_;
};

bool Backend::start(const ts_stream_desc* descs, uint32_t count, uint32_t master, std::string& error) {
    const uint32_t max_block = engine_.max_block();
    for (uint32_t i = 0; i < count; ++i) {
        auto s = std::make_unique<Stream>();
        s->desc = descs[i];
        s->master = i == master;
        s->render = descs[i].kind == TS_STREAM_RENDER;
        s->channels = descs[i].channels;
        for (size_t k = 0; k < TS_DEVICE_ID_CHARS && descs[i].device_id[k]; ++k)
            s->device_id.push_back(static_cast<wchar_t>(descs[i].device_id[k]));
        s->ready = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        s->scratch.reset(new float[static_cast<size_t>(max_block) * s->channels]());
        streams_.push_back(std::move(s));
    }
    master_ = streams_[master].get();
    master_scratch_.reset(new float[static_cast<size_t>(max_block) * master_->channels]());

    for (auto& s : streams_) s->thread = std::thread([this, stream = s.get()] { run(*stream); });

    // Control thread: waiting here is allowed. Every stream reports once, opened or not.
    for (auto& s : streams_) {
        if (WaitForSingleObject(s->ready, 10000) != WAIT_OBJECT_0 && s->state.load() == TS_STREAM_STATE_STARTING)
            s->fail("opening the device", HRESULT_FROM_WIN32(ERROR_TIMEOUT));
    }
    if (master_->state.load() == TS_STREAM_STATE_FAILED) {
        error = "master stream: " + master_->error;
        stop();
        return false;
    }

    for (auto& s : streams_) {
        if (s.get() == master_ || s->state.load() == TS_STREAM_STATE_FAILED) continue;
        // Two device periods of cushion (or two blocks, whichever is larger) absorb the
        // jitter between the two clocks' wake-ups; the ring holds four times that.
        const uint32_t cushion = 2 * std::max(max_block, std::max(s->period_frames, master_->period_frames));
        s->ring = std::make_unique<FrameRing>(cushion * 4, s->channels);
        s->resampler = std::make_unique<DriftResampler>(s->channels, std::max(max_block, s->buffer_frames), cushion);
        (s->render ? render_satellites_ : capture_satellites_).push_back(s.get());
    }

    for (Stream* s : capture_satellites_) inputs_.push_back(ts_port_buffer{s->desc.node_id, s->channels, s->scratch.get()});
    if (!master_->render) inputs_.push_back(ts_port_buffer{master_->desc.node_id, master_->channels, master_scratch_.get()});
    if (master_->render) outputs_.push_back(ts_port_buffer{master_->desc.node_id, master_->channels, master_scratch_.get()});
    for (Stream* s : render_satellites_) outputs_.push_back(ts_port_buffer{s->desc.node_id, s->channels, s->scratch.get()});

    engine_.set_backend_running(true);
    SetEvent(go_);
    return true;
}

void Backend::stop() {
    if (stopped_) return;
    stopped_ = true;
    SetEvent(stop_);
    SetEvent(go_);
    for (auto& s : streams_)
        if (s->thread.joinable()) s->thread.join();
    engine_.set_backend_running(false);
}

int32_t Backend::status(ts_stream_status* out, int32_t capacity) {
    int32_t n = 0;
    for (auto& s : streams_) {
        if (n >= capacity) break;
        ts_stream_status& st = out[n++];
        std::memset(&st, 0, sizeof(st));
        st.node_id = s->desc.node_id;
        st.kind = s->desc.kind;
        st.state = s->state.load(std::memory_order_acquire);
        st.is_master = s->master ? 1 : 0;
        st.share_mode = s->exclusive ? TS_SHARE_EXCLUSIVE : TS_SHARE_SHARED;
        st.hresult = s->hresult.load();
        st.sample_rate = s->rate;
        st.channels = s->channels;
        st.bits = s->bits;
        st.valid_bits = s->valid_bits;
        st.is_float = s->is_float ? 1 : 0;
        st.buffer_frames = s->buffer_frames;
        st.period_frames = s->period_frames;
        st.stream_latency_hns = s->latency;
        st.frames = s->frames.load(std::memory_order_relaxed);
        st.glitches = s->glitches.load(std::memory_order_relaxed);
        st.underruns = s->underruns.load(std::memory_order_relaxed);
        st.overruns = s->overruns.load(std::memory_order_relaxed);
        st.drift_ratio = s->drift.load(std::memory_order_relaxed);
        st.ring_fill = s->ring ? s->ring->available() : 0;
        st.raw = s->raw ? 1 : 0;
        const std::string& text = st.state == TS_STREAM_STATE_FAILED ? s->error : s->note;
        std::snprintf(st.error, sizeof(st.error), "%s", text.c_str());
    }
    return n;
}

void Backend::run(Stream& s) {
    ComScope com;
    if (!com.ok()) {
        s.fail("joining COM", com.hr);
        SetEvent(s.ready);
        return;
    }

    const HRESULT hr = initialize(s);
    if (FAILED(hr)) {
        if (s.state.load() != TS_STREAM_STATE_FAILED) s.fail("opening the stream", hr);
        teardown(s);
        SetEvent(s.ready);
        return;
    }
    SetEvent(s.ready);

    HANDLE handles[2] = {go_, stop_};
    WaitForMultipleObjects(2, handles, FALSE, INFINITE);
    if (WaitForSingleObject(stop_, 0) == WAIT_OBJECT_0 || s.state.load() == TS_STREAM_STATE_FAILED) {
        teardown(s);
        return;
    }

    DWORD task_index = 0;
    HANDLE mmcss = AvSetMmThreadCharacteristicsW(L"Pro Audio", &task_index);

    if (s.render) {
        // Pre-fill the device buffer with silence so the first period does not underrun.
        BYTE* data = nullptr;
        UINT32 padding = 0;
        s.client->GetCurrentPadding(&padding);
        const UINT32 fill = s.buffer_frames - padding;
        if (fill && SUCCEEDED(s.render_client->GetBuffer(fill, &data)))
            s.render_client->ReleaseBuffer(fill, AUDCLNT_BUFFERFLAGS_SILENT);
    }

    const HRESULT started = s.client->Start();
    if (FAILED(started)) {
        s.fail("starting the stream", started);
    } else {
        s.state.store(TS_STREAM_STATE_RUNNING, std::memory_order_release);
        if (s.master)
            s.render ? master_render(s) : master_capture(s);
        else
            s.render ? satellite_render(s) : satellite_capture(s);
        s.client->Stop();
        if (s.state.load() == TS_STREAM_STATE_RUNNING) s.state.store(TS_STREAM_STATE_STOPPED);
    }

    if (mmcss) AvRevertMmThreadCharacteristics(mmcss);
    teardown(s);
}

HRESULT Backend::initialize(Stream& s) {
    const bool loopback = s.desc.kind == TS_STREAM_LOOPBACK || s.desc.kind == TS_STREAM_PROCESS_LOOPBACK;
    s.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    if (s.desc.kind == TS_STREAM_PROCESS_LOOPBACK) {
        HRESULT hr = activate_process_loopback(s.desc.process_id, (s.desc.flags & TS_STREAM_FLAG_INCLUDE_TREE) != 0,
                                               &s.client);
        if (FAILED(hr)) return s.fail("activating process loopback", hr), hr;
        return initialize_shared(s, true);
    }

    IMMDevice* device = nullptr;
    HRESULT hr = open_device(s.device_id.c_str(), &device);
    if (FAILED(hr)) return s.fail("opening the endpoint", hr), hr;
    s.raw_capable = !loopback && raw_supported(device);

    auto activate = [&] {
        release(s.client);
        return device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&s.client));
    };
    hr = activate();
    if (FAILED(hr)) {
        device->Release();
        return s.fail("activating the audio client", hr), hr;
    }

    if (!loopback && s.desc.share_mode == TS_SHARE_EXCLUSIVE) {
        hr = initialize_exclusive(s, device);
        if (FAILED(hr) && (s.desc.flags & TS_STREAM_FLAG_ALLOW_SHARED_FALLBACK)) {
            // A visible fallback: status reports share_mode = shared, and why exclusive
            // was refused stays in `error` for the UI to show beside it.
            const std::string refused = fmt_hr("exclusive mode refused", hr);
            s.state.store(TS_STREAM_STATE_STARTING);
            s.hresult.store(0);
            hr = activate();
            if (SUCCEEDED(hr)) hr = initialize_shared(s, false);
            if (SUCCEEDED(hr)) s.note = refused + "; running shared";
        }
    } else {
        hr = initialize_shared(s, loopback);
    }
    device->Release();
    return hr;
}

HRESULT Backend::initialize_exclusive(Stream& s, IMMDevice* device) {
    const uint32_t rate = engine_.sample_rate();
    WAVEFORMATEX* mix = nullptr;
    DWORD mask = 0;
    if (SUCCEEDED(s.client->GetMixFormat(&mix)) && mix) {
        if (mix->wFormatTag == WAVE_FORMAT_EXTENSIBLE && mix->nChannels == s.channels)
            mask = reinterpret_cast<WAVEFORMATEXTENSIBLE*>(mix)->dwChannelMask;
        CoTaskMemFree(mix);
    }

    // Best first. Exclusive mode talks to the driver directly, so the format must be one
    // the hardware takes natively; there is no converter in between.
    struct Candidate { uint32_t container, valid; bool flt; };
    const Candidate candidates[] = {{32, 32, true}, {32, 24, false}, {32, 32, false}, {24, 24, false}, {16, 16, false}};
    WAVEFORMATEXTENSIBLE chosen{};
    bool found = false;
    for (const auto& c : candidates) {
        WAVEFORMATEXTENSIBLE f = make_format(rate, s.channels, c.container, c.valid, c.flt, mask);
        if (s.client->IsFormatSupported(AUDCLNT_SHAREMODE_EXCLUSIVE, &f.Format, nullptr) == S_OK) {
            chosen = f;
            found = true;
            break;
        }
    }
    if (!found) {
        s.fail("no exclusive-mode format at " + std::to_string(rate) + " Hz and " + std::to_string(s.channels) +
                   " channels", AUDCLNT_E_UNSUPPORTED_FORMAT);
        return AUDCLNT_E_UNSUPPORTED_FORMAT;
    }

    REFERENCE_TIME default_period = 0, min_period = 0;
    s.client->GetDevicePeriod(&default_period, &min_period);
    const uint32_t wanted_frames = s.desc.period_frames ? s.desc.period_frames : engine_.max_block();
    REFERENCE_TIME period = static_cast<REFERENCE_TIME>((10'000'000.0 * wanted_frames) / rate + 0.5);
    period = std::max(period, min_period);

    HRESULT hr = s.client->Initialize(AUDCLNT_SHAREMODE_EXCLUSIVE, AUDCLNT_STREAMFLAGS_EVENTCALLBACK, period, period,
                                      &chosen.Format, nullptr);
    if (hr == AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED) {
        // The documented recovery: ask what the device would align to, and start again with
        // a fresh client at exactly that period.
        UINT32 aligned = 0;
        s.client->GetBufferSize(&aligned);
        period = static_cast<REFERENCE_TIME>((10'000'000.0 * aligned) / rate + 0.5);
        release(s.client);
        hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&s.client));
        if (SUCCEEDED(hr))
            hr = s.client->Initialize(AUDCLNT_SHAREMODE_EXCLUSIVE, AUDCLNT_STREAMFLAGS_EVENTCALLBACK, period, period,
                                      &chosen.Format, nullptr);
    }
    if (FAILED(hr)) return s.fail("initialising exclusive mode", hr), hr;

    s.exclusive = true;
    s.rate = rate;
    s.bits = chosen.Format.wBitsPerSample;
    s.valid_bits = chosen.Samples.wValidBitsPerSample;
    s.is_float = IsEqualGUID(chosen.SubFormat, kSubtypeFloat) != 0;
    sample_format_of(&chosen.Format, s.format);
    s.block_align = chosen.Format.nBlockAlign;
    s.client->GetBufferSize(&s.buffer_frames);
    // Event-driven exclusive mode delivers one buffer per event.
    s.period_frames = s.buffer_frames;
    s.client->GetStreamLatency(&s.latency);
    hr = s.client->SetEventHandle(s.event);
    if (FAILED(hr)) return s.fail("setting the event handle", hr), hr;
    hr = s.render ? s.client->GetService(__uuidof(IAudioRenderClient), reinterpret_cast<void**>(&s.render_client))
                  : s.client->GetService(__uuidof(IAudioCaptureClient), reinterpret_cast<void**>(&s.capture_client));
    if (FAILED(hr)) return s.fail("getting the render/capture service", hr), hr;
    if (!s.master) s.convert.reset(new float[static_cast<size_t>(s.buffer_frames) * s.channels]());
    return S_OK;
}

HRESULT Backend::initialize_shared(Stream& s, bool loopback) {
    const uint32_t rate = engine_.sample_rate();
    HRESULT hr;
    DWORD flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK;
    if (loopback) flags |= AUDCLNT_STREAMFLAGS_LOOPBACK;

    // Raw mode must be set before Initialize. Where the endpoint does not support it the
    // request is simply not honoured, so it is reported from the endpoint's own claim.
    if (!loopback && (s.desc.flags & TS_STREAM_FLAG_RAW)) {
        IAudioClient2* client2 = nullptr;
        if (SUCCEEDED(s.client->QueryInterface(__uuidof(IAudioClient2), reinterpret_cast<void**>(&client2)))) {
            AudioClientProperties props{};
            props.cbSize = sizeof(props);
            props.eCategory = AudioCategory_Media;
            props.Options = AUDCLNT_STREAMOPTIONS_RAW;
            s.raw = SUCCEEDED(client2->SetClientProperties(&props)) && s.raw_capable;
            client2->Release();
        }
    }

    // Low-latency shared mode (IAudioClient3) needs the mix format exactly, so it applies
    // only when the device already runs float32 at the engine rate and width.
    bool low_latency = false;
    if (!loopback) {
        IAudioClient3* client3 = nullptr;
        WAVEFORMATEX* mix = nullptr;
        if (SUCCEEDED(s.client->QueryInterface(__uuidof(IAudioClient3), reinterpret_cast<void**>(&client3))) &&
            SUCCEEDED(client3->GetMixFormat(&mix)) && mix && mix->nSamplesPerSec == rate &&
            mix->nChannels == s.channels && is_float_format(mix) && mix->wBitsPerSample == 32) {
            UINT32 def = 0, fundamental = 0, min_p = 0, max_p = 0;
            if (SUCCEEDED(client3->GetSharedModeEnginePeriod(mix, &def, &fundamental, &min_p, &max_p)) && fundamental) {
                const UINT32 target = s.desc.period_frames ? s.desc.period_frames : engine_.max_block();
                const UINT32 wanted = std::clamp<UINT32>(((target + fundamental - 1) / fundamental) * fundamental,
                                                         min_p, max_p);
                hr = client3->InitializeSharedAudioStream(AUDCLNT_STREAMFLAGS_EVENTCALLBACK, wanted, mix, nullptr);
                if (SUCCEEDED(hr)) {
                    low_latency = true;
                    s.period_frames = wanted;
                }
            }
        }
        if (mix) CoTaskMemFree(mix);
        release(client3);
    }

    WAVEFORMATEXTENSIBLE f = make_format(rate, s.channels, 32, 32, true, 0);
    if (!low_latency) {
        // Ask Windows to convert: the engine runs at one rate and width, whatever the
        // device's mix format is.
        hr = s.client->Initialize(AUDCLNT_SHAREMODE_SHARED,
                                  flags | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
                                  loopback ? 200000 : 0, 0, &f.Format, nullptr);
        if (FAILED(hr)) return s.fail("initialising shared mode", hr), hr;
        REFERENCE_TIME def = 0, min_p = 0;
        s.client->GetDevicePeriod(&def, &min_p);
        s.period_frames = static_cast<uint32_t>((def * rate + 5'000'000) / 10'000'000);
    }

    s.exclusive = false;
    s.rate = rate;
    s.bits = 32;
    s.valid_bits = 32;
    s.is_float = true;
    s.format = SampleFormat::Float32;
    s.block_align = s.channels * 4;
    s.client->GetBufferSize(&s.buffer_frames);
    s.client->GetStreamLatency(&s.latency);
    s.polled = loopback;
    hr = s.client->SetEventHandle(s.event);
    if (FAILED(hr)) return s.fail("setting the event handle", hr), hr;
    hr = s.render ? s.client->GetService(__uuidof(IAudioRenderClient), reinterpret_cast<void**>(&s.render_client))
                  : s.client->GetService(__uuidof(IAudioCaptureClient), reinterpret_cast<void**>(&s.capture_client));
    if (FAILED(hr)) return s.fail("getting the render/capture service", hr), hr;
    if (!s.master) s.convert.reset(new float[static_cast<size_t>(s.buffer_frames) * s.channels]());
    return S_OK;
}

void Backend::teardown(Stream& s) {
    release(s.render_client);
    release(s.capture_client);
    release(s.client);
}

// Wait for the stream's next device event. False means stop (or a dead device).
bool Backend::wait(Stream& s) {
    HANDLE handles[2] = {s.event, stop_};
    // Loopback streams get no event while nothing plays; poll them every 10 ms instead of
    // stalling. Everything else waits up to two seconds, then treats silence as a fault.
    const DWORD r = WaitForMultipleObjects(2, handles, FALSE, s.polled ? 10 : 2000);
    if (r == WAIT_OBJECT_0 + 1) return false;
    if (r == WAIT_TIMEOUT && !s.polled) {
        s.fail("the device stopped delivering events", HRESULT_FROM_WIN32(ERROR_TIMEOUT));
        return false;
    }
    return true;
}

void Backend::run_engine(uint32_t frames, const float* master_input, float* master_output) noexcept {
    for (Stream* s : capture_satellites_) {
        const uint32_t missing = s->resampler->process(*s->ring, s->scratch.get(), frames);
        if (missing) add(s->underruns, missing);
        s->drift.store(s->resampler->ratio(), std::memory_order_relaxed);
    }
    if (master_input)
        std::memcpy(master_scratch_.get(), master_input, sizeof(float) * frames * master_->channels);

    engine_.run_block(inputs_.data(), static_cast<uint32_t>(inputs_.size()), outputs_.data(),
                      static_cast<uint32_t>(outputs_.size()), frames);

    if (master_output)
        std::memcpy(master_output, master_scratch_.get(), sizeof(float) * frames * master_->channels);
    for (Stream* s : render_satellites_) {
        const uint32_t written = s->ring->write(s->scratch.get(), frames);
        if (written < frames) add(s->overruns, frames - written);
    }
}

void Backend::master_render(Stream& s) {
    AudioThreadScope scope;
    const uint32_t max_block = engine_.max_block();
    float* staging = s.scratch.get();  // float output before conversion to the device format
    bool first = true;
    while (wait(s)) {
        UINT32 available = s.buffer_frames;
        if (!s.exclusive) {
            UINT32 padding = 0;
            const HRESULT hr = s.client->GetCurrentPadding(&padding);
            if (FAILED(hr)) { s.fail("reading the render position", hr); break; }
            // Woken with nothing queued: the device already played through everything we
            // gave it, i.e. it ran dry — a dropout the listener heard.
            if (padding == 0 && !first) {
                add(s.glitches, 1);
                engine_.add_xruns(1);
            }
            available = s.buffer_frames - padding;
        }
        first = false;
        if (available == 0) continue;

        BYTE* data = nullptr;
        HRESULT hr = s.render_client->GetBuffer(available, &data);
        if (FAILED(hr)) { s.fail("getting the render buffer", hr); break; }
        const uint32_t block = even_block(available, max_block);
        for (UINT32 done = 0; done < available;) {
            const uint32_t n = std::min<uint32_t>(block, available - done);
            run_engine(n, nullptr, staging);
            from_float(s.format, staging, data + done * s.block_align, static_cast<size_t>(n) * s.channels);
            done += n;
        }
        hr = s.render_client->ReleaseBuffer(available, 0);
        if (FAILED(hr)) { s.fail("releasing the render buffer", hr); break; }
        add(s.frames, available);
    }
}

void Backend::master_capture(Stream& s) {
    AudioThreadScope scope;
    const uint32_t max_block = engine_.max_block();
    float* staging = s.scratch.get();
    while (wait(s)) {
        for (;;) {
            UINT32 packet = 0;
            HRESULT hr = s.capture_client->GetNextPacketSize(&packet);
            if (FAILED(hr)) { s.fail("reading the capture position", hr); return; }
            if (packet == 0) break;
            BYTE* data = nullptr;
            UINT32 frames = 0;
            DWORD flags = 0;
            hr = s.capture_client->GetBuffer(&data, &frames, &flags, nullptr, nullptr);
            if (FAILED(hr)) { s.fail("getting the capture buffer", hr); return; }
            if (flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY) {
                add(s.glitches, 1);
                engine_.add_xruns(1);
            }
            const uint32_t block = even_block(frames, max_block);
            for (UINT32 done = 0; done < frames;) {
                const uint32_t n = std::min<uint32_t>(block, frames - done);
                if (flags & AUDCLNT_BUFFERFLAGS_SILENT)
                    std::fill(staging, staging + static_cast<size_t>(n) * s.channels, 0.0f);
                else
                    to_float(s.format, data + done * s.block_align, staging, static_cast<size_t>(n) * s.channels);
                run_engine(n, staging, nullptr);
                done += n;
            }
            s.capture_client->ReleaseBuffer(frames);
            add(s.frames, frames);
        }
    }
}

void Backend::satellite_capture(Stream& s) {
    AudioThreadScope scope;
    float* staging = s.convert.get();
    while (wait(s)) {
        for (;;) {
            UINT32 packet = 0;
            HRESULT hr = s.capture_client->GetNextPacketSize(&packet);
            if (FAILED(hr)) { s.fail("reading the capture position", hr); return; }
            if (packet == 0) break;
            BYTE* data = nullptr;
            UINT32 frames = 0;
            DWORD flags = 0;
            hr = s.capture_client->GetBuffer(&data, &frames, &flags, nullptr, nullptr);
            if (FAILED(hr)) { s.fail("getting the capture buffer", hr); return; }
            if (flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY) add(s.glitches, 1);
            for (UINT32 done = 0; done < frames;) {
                const uint32_t n = std::min<uint32_t>(s.buffer_frames, frames - done);
                if (flags & AUDCLNT_BUFFERFLAGS_SILENT)
                    std::fill(staging, staging + static_cast<size_t>(n) * s.channels, 0.0f);
                else
                    to_float(s.format, data + done * s.block_align, staging, static_cast<size_t>(n) * s.channels);
                const uint32_t written = s.ring->write(staging, n);
                if (written < n) add(s.overruns, n - written);
                done += n;
            }
            s.capture_client->ReleaseBuffer(frames);
            add(s.frames, frames);
        }
    }
}

void Backend::satellite_render(Stream& s) {
    AudioThreadScope scope;
    float* staging = s.convert.get();
    bool first = true;
    while (wait(s)) {
        UINT32 available = s.buffer_frames;
        if (!s.exclusive) {
            UINT32 padding = 0;
            const HRESULT hr = s.client->GetCurrentPadding(&padding);
            if (FAILED(hr)) { s.fail("reading the render position", hr); return; }
            if (padding == 0 && !first) add(s.glitches, 1);
            available = s.buffer_frames - padding;
        }
        first = false;
        if (available == 0) continue;
        BYTE* data = nullptr;
        HRESULT hr = s.render_client->GetBuffer(available, &data);
        if (FAILED(hr)) { s.fail("getting the render buffer", hr); return; }
        const uint32_t missing = s.resampler->process(*s.ring, staging, available);
        if (missing) add(s.underruns, missing);
        s.drift.store(s.resampler->ratio(), std::memory_order_relaxed);
        from_float(s.format, staging, data, static_cast<size_t>(available) * s.channels);
        s.render_client->ReleaseBuffer(available, 0);
        add(s.frames, available);
    }
}

// ---- Public entry points -----------------------------------------------------------------

std::unique_ptr<DeviceBackend> start(Engine& engine, const ts_stream_desc* streams, uint32_t count, uint32_t master,
                                     std::string& error) {
    if (!streams || count == 0) { error = "no streams"; return nullptr; }
    if (master >= count) { error = "master index out of range"; return nullptr; }
    if (streams[master].kind != TS_STREAM_RENDER && streams[master].kind != TS_STREAM_CAPTURE) {
        error = "the master stream must be a render or capture endpoint: a loopback stream has no clock of its own";
        return nullptr;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const ts_stream_desc& d = streams[i];
        if (d.kind < TS_STREAM_RENDER || d.kind > TS_STREAM_PROCESS_LOOPBACK) { error = "unknown stream kind"; return nullptr; }
        if (d.channels < 1 || d.channels > TS_MAX_CHANNELS) { error = "stream channel count out of range"; return nullptr; }
        if (d.kind != TS_STREAM_PROCESS_LOOPBACK && d.device_id[0] == 0) { error = "stream has no device id"; return nullptr; }
        for (uint32_t j = 0; j < i; ++j)
            if (streams[j].node_id == d.node_id) { error = "two streams bound to one node"; return nullptr; }
    }
    auto backend = std::make_unique<Backend>(engine);
    if (!backend->start(streams, count, master, error)) return nullptr;
    return backend;
}

int32_t enumerate(ts_device_info* out, int32_t capacity, std::string& error) {
    ComScope com;
    if (!com.ok()) { error = fmt_hr("joining COM", com.hr); return TS_ERR_BACKEND; }

    IMMDeviceEnumerator* enumerator = nullptr;
    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                                  reinterpret_cast<void**>(&enumerator));
    if (FAILED(hr)) { error = fmt_hr("creating the device enumerator", hr); return TS_ERR_BACKEND; }

    std::wstring defaults[2][3];
    const ERole roles[3] = {eConsole, eMultimedia, eCommunications};
    for (int flow = 0; flow < 2; ++flow)
        for (int r = 0; r < 3; ++r) {
            IMMDevice* d = nullptr;
            if (SUCCEEDED(enumerator->GetDefaultAudioEndpoint(flow == 0 ? eRender : eCapture, roles[r], &d))) {
                LPWSTR id = nullptr;
                if (SUCCEEDED(d->GetId(&id))) {
                    defaults[flow][r] = id;
                    CoTaskMemFree(id);
                }
                d->Release();
            }
        }

    IMMDeviceCollection* collection = nullptr;
    hr = enumerator->EnumAudioEndpoints(eAll, DEVICE_STATE_ACTIVE, &collection);
    if (FAILED(hr)) {
        enumerator->Release();
        error = fmt_hr("enumerating endpoints", hr);
        return TS_ERR_BACKEND;
    }
    UINT count = 0;
    collection->GetCount(&count);
    int32_t n = 0;
    for (UINT i = 0; i < count; ++i) {
        IMMDevice* device = nullptr;
        if (FAILED(collection->Item(i, &device))) continue;
        if (n < capacity && out) {
            ts_device_info& info = out[n];
            std::memset(&info, 0, sizeof(info));
            IMMEndpoint* endpoint = nullptr;
            if (SUCCEEDED(device->QueryInterface(__uuidof(IMMEndpoint), reinterpret_cast<void**>(&endpoint)))) {
                EDataFlow flow = eRender;
                endpoint->GetDataFlow(&flow);
                info.flow = flow == eRender ? TS_FLOW_RENDER : TS_FLOW_CAPTURE;
                endpoint->Release();
            }
            LPWSTR id = nullptr;
            if (SUCCEEDED(device->GetId(&id))) {
                copy_w(info.id, TS_DEVICE_ID_CHARS, id);
                const int flow_index = info.flow == TS_FLOW_RENDER ? 0 : 1;
                for (int r = 0; r < 3; ++r)
                    if (defaults[flow_index][r] == id) info.default_roles |= 1u << r;
                CoTaskMemFree(id);
            }
            DWORD state = 0;
            device->GetState(&state);
            info.state = state;
            IPropertyStore* props = nullptr;
            if (SUCCEEDED(device->OpenPropertyStore(STGM_READ, &props))) {
                PROPVARIANT name;
                PropVariantInit(&name);
                if (SUCCEEDED(props->GetValue(PKEY_Device_FriendlyName, &name)) && name.vt == VT_LPWSTR)
                    copy_w(info.name, TS_DEVICE_NAME_CHARS, name.pwszVal);
                PropVariantClear(&name);
                props->Release();
            }
            IAudioClient* client = nullptr;
            if (SUCCEEDED(device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&client)))) {
                WAVEFORMATEX* mix = nullptr;
                if (SUCCEEDED(client->GetMixFormat(&mix)) && mix) {
                    info.mix_channels = mix->nChannels;
                    info.mix_sample_rate = mix->nSamplesPerSec;
                    info.mix_bits = mix->wBitsPerSample;
                    info.mix_is_float = is_float_format(mix) ? 1 : 0;
                    IAudioClient3* client3 = nullptr;
                    if (SUCCEEDED(client->QueryInterface(__uuidof(IAudioClient3), reinterpret_cast<void**>(&client3)))) {
                        UINT32 d = 0, f = 0, mn = 0, mx = 0;
                        if (SUCCEEDED(client3->GetSharedModeEnginePeriod(mix, &d, &f, &mn, &mx))) {
                            info.shared_default_period_frames = d;
                            info.shared_fundamental_period_frames = f;
                            info.shared_min_period_frames = mn;
                            info.shared_max_period_frames = mx;
                        }
                        client3->Release();
                    }
                    CoTaskMemFree(mix);
                }
                REFERENCE_TIME d = 0, mn = 0;
                if (SUCCEEDED(client->GetDevicePeriod(&d, &mn))) {
                    info.default_period_hns = d;
                    info.min_period_hns = mn;
                }
                client->Release();
            }
            info.raw_supported = raw_supported(device) ? 1 : 0;
        }
        ++n;
        device->Release();
    }
    collection->Release();
    enumerator->Release();
    return n;
}

// ---- Device notifications ------------------------------------------------------------------

namespace {

class NotificationClient final : public IMMNotificationClient {
public:
    explicit NotificationClient(DeviceWatcher& watcher) : watcher_(watcher) {}

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IMMNotificationClient)) {
            *out = static_cast<IMMNotificationClient*>(this);
            AddRef();
            return S_OK;
        }
        *out = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return InterlockedIncrement(&refs_); }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG n = InterlockedDecrement(&refs_);
        if (n == 0) delete this;
        return n;
    }

    HRESULT STDMETHODCALLTYPE OnDeviceStateChanged(LPCWSTR id, DWORD state) override {
        return push(TS_DEVICE_EVENT_STATE_CHANGED, id, 0, 0, state);
    }
    HRESULT STDMETHODCALLTYPE OnDeviceAdded(LPCWSTR id) override { return push(TS_DEVICE_EVENT_ADDED, id, 0, 0, 0); }
    HRESULT STDMETHODCALLTYPE OnDeviceRemoved(LPCWSTR id) override { return push(TS_DEVICE_EVENT_REMOVED, id, 0, 0, 0); }
    HRESULT STDMETHODCALLTYPE OnDefaultDeviceChanged(EDataFlow flow, ERole role, LPCWSTR id) override {
        const uint32_t r = role == eConsole ? TS_ROLE_CONSOLE : role == eMultimedia ? TS_ROLE_MULTIMEDIA : TS_ROLE_COMMUNICATIONS;
        return push(TS_DEVICE_EVENT_DEFAULT_CHANGED, id, flow == eRender ? TS_FLOW_RENDER : TS_FLOW_CAPTURE, r, 0);
    }
    HRESULT STDMETHODCALLTYPE OnPropertyValueChanged(LPCWSTR, const PROPERTYKEY) override { return S_OK; }

private:
    HRESULT push(uint32_t kind, LPCWSTR id, uint32_t flow, uint32_t role, uint32_t state) {
        ts_device_event e{};
        e.kind = kind;
        e.flow = flow;
        e.role = role;
        e.state = state;
        copy_w(e.id, TS_DEVICE_ID_CHARS, id);
        watcher_.push(e);
        return S_OK;
    }

    LONG refs_ = 1;
    DeviceWatcher& watcher_;
};

}  // namespace

DeviceWatcher::~DeviceWatcher() { stop(); }

bool DeviceWatcher::start(std::string& error) {
    if (client_) return true;
    // The enumerator and its registration outlive this call; create them from an MTA
    // thread of our own so the calling thread's apartment (possibly Qt's STA) is irrelevant.
    HRESULT result = E_FAIL;
    std::thread([&] {
        ComScope com;
        IMMDeviceEnumerator* enumerator = nullptr;
        result = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                                  reinterpret_cast<void**>(&enumerator));
        if (FAILED(result)) return;
        auto* client = new NotificationClient(*this);
        result = enumerator->RegisterEndpointNotificationCallback(client);
        if (FAILED(result)) {
            client->Release();
            enumerator->Release();
            return;
        }
        enumerator_ = enumerator;
        client_ = client;
    }).join();
    if (FAILED(result)) {
        error = fmt_hr("registering for device notifications", result);
        return false;
    }
    return true;
}

void DeviceWatcher::stop() {
    if (!client_) return;
    std::thread([&] {
        ComScope com;
        auto* enumerator = static_cast<IMMDeviceEnumerator*>(enumerator_);
        auto* client = static_cast<NotificationClient*>(client_);
        enumerator->UnregisterEndpointNotificationCallback(client);
        client->Release();
        enumerator->Release();
    }).join();
    client_ = nullptr;
    enumerator_ = nullptr;
}

void DeviceWatcher::push(const ts_device_event& event) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.size() >= 1024) {
        ++dropped_;
        return;
    }
    if (dropped_) {
        ts_device_event lost{};
        lost.kind = TS_DEVICE_EVENT_LOST;
        lost.state = dropped_;
        queue_.push_back(lost);
        dropped_ = 0;
    }
    queue_.push_back(event);
}

int32_t DeviceWatcher::poll(ts_device_event* out, int32_t capacity) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int32_t n = std::min<int32_t>(capacity, static_cast<int32_t>(queue_.size()));
    std::copy(queue_.begin(), queue_.begin() + n, out);
    queue_.erase(queue_.begin(), queue_.begin() + n);
    return n;
}

}  // namespace ts::wasapi
