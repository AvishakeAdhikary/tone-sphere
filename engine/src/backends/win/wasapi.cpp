// WASAPI backend (Windows audio endpoints).
#include <initguid.h>  // instantiate PKEY_* / GUIDs used below in this TU

#include "backends/win/wasapi.h"

#include <audioclient.h>
#include <mmdeviceapi.h>  // must precede functiondiscoverykeys (DEFINE_PROPERTYKEY)
#include <functiondiscoverykeys_devpkey.h>
#include <ksmedia.h>
#include <mmreg.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <vector>

#include "util/log.h"
#include "util/platform.h"

#pragma comment(lib, "ole32.lib")

namespace ts {

namespace {

template <class T>
void safeRelease(T*& p) {
    if (p) {
        p->Release();
        p = nullptr;
    }
}

std::wstring propString(IPropertyStore* ps, const PROPERTYKEY& key) {
    PROPVARIANT v;
    PropVariantInit(&v);
    std::wstring out;
    if (SUCCEEDED(ps->GetValue(key, &v)) && v.vt == VT_LPWSTR && v.pwszVal) out = v.pwszVal;
    PropVariantClear(&v);
    return out;
}

bool deviceFormat(IPropertyStore* ps, int& channels, double& rate, DWORD& mask) {
    PROPVARIANT v;
    PropVariantInit(&v);
    bool ok = false;
    if (SUCCEEDED(ps->GetValue(PKEY_AudioEngine_DeviceFormat, &v)) && v.vt == VT_BLOB && v.blob.cbSize >= sizeof(WAVEFORMATEX)) {
        auto* wf = reinterpret_cast<WAVEFORMATEX*>(v.blob.pBlobData);
        channels = wf->nChannels;
        rate = wf->nSamplesPerSec;
        mask = 0;
        if (wf->wFormatTag == WAVE_FORMAT_EXTENSIBLE && v.blob.cbSize >= sizeof(WAVEFORMATEXTENSIBLE))
            mask = reinterpret_cast<WAVEFORMATEXTENSIBLE*>(wf)->dwChannelMask;
        ok = true;
    }
    PropVariantClear(&v);
    return ok;
}

std::vector<std::string> channelNamesFor(int channels, DWORD mask) {
    static const std::pair<DWORD, const char*> kSpk[] = {
        {SPEAKER_FRONT_LEFT, "L"},        {SPEAKER_FRONT_RIGHT, "R"},       {SPEAKER_FRONT_CENTER, "C"},
        {SPEAKER_LOW_FREQUENCY, "LFE"},    {SPEAKER_BACK_LEFT, "BL"},        {SPEAKER_BACK_RIGHT, "BR"},
        {SPEAKER_FRONT_LEFT_OF_CENTER, "FLC"}, {SPEAKER_FRONT_RIGHT_OF_CENTER, "FRC"}, {SPEAKER_BACK_CENTER, "BC"},
        {SPEAKER_SIDE_LEFT, "SL"},        {SPEAKER_SIDE_RIGHT, "SR"},
    };
    std::vector<std::string> names;
    if (channels == 1) return {"Mono"};
    if (mask) {
        for (auto& [bit, n] : kSpk)
            if (mask & bit) names.emplace_back(n);
    }
    if (static_cast<int>(names.size()) != channels) {
        names.clear();
        if (channels == 2) return {"L", "R"};
        for (int i = 0; i < channels; ++i) names.push_back(std::to_string(i + 1));
    }
    return names;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

DWORD defaultMask(int ch) {
    switch (ch) {
        case 1: return SPEAKER_FRONT_CENTER;
        case 2: return KSAUDIO_SPEAKER_STEREO;
        case 4: return KSAUDIO_SPEAKER_QUAD;
        case 6: return KSAUDIO_SPEAKER_5POINT1;
        case 8: return KSAUDIO_SPEAKER_7POINT1_SURROUND;
        default: return 0;
    }
}

WAVEFORMATEXTENSIBLE makeFormat(int channels, double rate, SampleFormat sf, DWORD mask) {
    WAVEFORMATEXTENSIBLE f{};
    f.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
    f.Format.nChannels = static_cast<WORD>(channels);
    f.Format.nSamplesPerSec = static_cast<DWORD>(rate);
    const int container = sf == SampleFormat::Int16 ? 16 : sf == SampleFormat::Int24 ? 24 : 32;
    f.Format.wBitsPerSample = static_cast<WORD>(container);
    f.Format.nBlockAlign = static_cast<WORD>(channels * container / 8);
    f.Format.nAvgBytesPerSec = f.Format.nBlockAlign * f.Format.nSamplesPerSec;
    f.Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX);
    f.Samples.wValidBitsPerSample = static_cast<WORD>(sf == SampleFormat::Int32 ? 24 : container);
    f.dwChannelMask = mask ? mask : defaultMask(channels);
    f.SubFormat = sf == SampleFormat::Float32 ? KSDATAFORMAT_SUBTYPE_IEEE_FLOAT : KSDATAFORMAT_SUBTYPE_PCM;
    return f;
}

bool parseFormat(const WAVEFORMATEX* wf, SampleFormat& sf, int& bytes) {
    bool isFloat = wf->wFormatTag == WAVE_FORMAT_IEEE_FLOAT;
    bool isPcm = wf->wFormatTag == WAVE_FORMAT_PCM;
    if (wf->wFormatTag == WAVE_FORMAT_EXTENSIBLE) {
        auto* ext = reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(wf);
        isFloat = IsEqualGUID(ext->SubFormat, KSDATAFORMAT_SUBTYPE_IEEE_FLOAT);
        isPcm = IsEqualGUID(ext->SubFormat, KSDATAFORMAT_SUBTYPE_PCM);
    }
    bytes = wf->wBitsPerSample / 8;
    if (isFloat && wf->wBitsPerSample == 32) { sf = SampleFormat::Float32; return true; }
    if (isPcm && wf->wBitsPerSample == 16) { sf = SampleFormat::Int16; return true; }
    if (isPcm && wf->wBitsPerSample == 24) { sf = SampleFormat::Int24; return true; }
    if (isPcm && wf->wBitsPerSample == 32) { sf = SampleFormat::Int32; return true; }
    return false;
}

void toFloat(const BYTE* src, float* dst, size_t samples, SampleFormat sf) {
    switch (sf) {
        case SampleFormat::Float32: std::memcpy(dst, src, samples * 4); break;
        case SampleFormat::Int16: {
            auto* s = reinterpret_cast<const int16_t*>(src);
            for (size_t i = 0; i < samples; ++i) dst[i] = s[i] * (1.0f / 32768.0f);
            break;
        }
        case SampleFormat::Int24:
            for (size_t i = 0; i < samples; ++i) {
                const int32_t v = (static_cast<int32_t>(src[i * 3 + 2]) << 24) | (src[i * 3 + 1] << 16) | (src[i * 3] << 8);
                dst[i] = static_cast<float>(v) * (1.0f / 2147483648.0f);
            }
            break;
        case SampleFormat::Int32: {
            auto* s = reinterpret_cast<const int32_t*>(src);
            for (size_t i = 0; i < samples; ++i) dst[i] = static_cast<float>(s[i]) * (1.0f / 2147483648.0f);
            break;
        }
    }
}

void fromFloat(const float* src, BYTE* dst, size_t samples, SampleFormat sf) {
    auto clamp1 = [](float v) { return v > 1.0f ? 1.0f : v < -1.0f ? -1.0f : v; };
    switch (sf) {
        case SampleFormat::Float32: std::memcpy(dst, src, samples * 4); break;
        case SampleFormat::Int16: {
            auto* d = reinterpret_cast<int16_t*>(dst);
            for (size_t i = 0; i < samples; ++i) d[i] = static_cast<int16_t>(clamp1(src[i]) * 32767.0f);
            break;
        }
        case SampleFormat::Int24:
            for (size_t i = 0; i < samples; ++i) {
                const int32_t v = static_cast<int32_t>(clamp1(src[i]) * 8388607.0f);
                dst[i * 3] = static_cast<BYTE>(v & 0xff);
                dst[i * 3 + 1] = static_cast<BYTE>((v >> 8) & 0xff);
                dst[i * 3 + 2] = static_cast<BYTE>((v >> 16) & 0xff);
            }
            break;
        case SampleFormat::Int32: {
            auto* d = reinterpret_cast<int32_t*>(dst);
            for (size_t i = 0; i < samples; ++i) d[i] = static_cast<int32_t>(static_cast<double>(clamp1(src[i])) * 2147483647.0);
            break;
        }
    }
}

}  // namespace

// ------------------------------------------------------------------ notifications
class WasapiNotifier final : public IMMNotificationClient {
public:
    explicit WasapiNotifier(WasapiBackend* b) : backend_(b) {}
    ULONG STDMETHODCALLTYPE AddRef() override { return ++ref_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG r = --ref_;
        if (r == 0) delete this;
        return r;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IMMNotificationClient)) {
            *ppv = static_cast<IMMNotificationClient*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    HRESULT STDMETHODCALLTYPE OnDeviceStateChanged(LPCWSTR, DWORD) override { fire(); return S_OK; }
    HRESULT STDMETHODCALLTYPE OnDeviceAdded(LPCWSTR) override { fire(); return S_OK; }
    HRESULT STDMETHODCALLTYPE OnDeviceRemoved(LPCWSTR) override { fire(); return S_OK; }
    HRESULT STDMETHODCALLTYPE OnDefaultDeviceChanged(EDataFlow, ERole, LPCWSTR) override { fire(); return S_OK; }
    HRESULT STDMETHODCALLTYPE OnPropertyValueChanged(LPCWSTR, const PROPERTYKEY key) override {
        if (key == PKEY_AudioEngine_DeviceFormat || key == PKEY_Device_FriendlyName) fire();
        return S_OK;
    }
    void detach() { backend_ = nullptr; }

private:
    void fire() {
        if (backend_ && backend_->onDevicesChanged) backend_->onDevicesChanged();
    }
    std::atomic<ULONG> ref_{1};
    WasapiBackend* backend_;
};

// ------------------------------------------------------------------ backend
WasapiBackend::WasapiBackend() {
    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                                  reinterpret_cast<void**>(&enumerator_));
    if (FAILED(hr)) {
        TS_ERROR("WASAPI: cannot create device enumerator (0x%08lx)", hr);
        return;
    }
    notifier_ = new WasapiNotifier(this);
    enumerator_->RegisterEndpointNotificationCallback(notifier_);
}

WasapiBackend::~WasapiBackend() {
    if (enumerator_ && notifier_) enumerator_->UnregisterEndpointNotificationCallback(notifier_);
    if (notifier_) {
        notifier_->detach();
        notifier_->Release();
    }
    safeRelease(enumerator_);
}

bool WasapiBackend::isToneSphereEndpoint(const std::wstring& iface) {
    return iface.find(L"ToneSphere") != std::wstring::npos;
}

bool WasapiBackend::isVirtualCable(const std::wstring& iface, const std::wstring& name) {
    const std::string i = lower(toUtf8(iface)), n = lower(toUtf8(name));
    return i.find("vb-audio") != std::string::npos || i.find("voicemeeter") != std::string::npos ||
           i.find("virtual audio cable") != std::string::npos || i.find("virtual cable") != std::string::npos ||
           n.find("cable input") != std::string::npos || n.find("cable output") != std::string::npos;
}

std::vector<DeviceInfo> WasapiBackend::enumerate() {
    std::vector<DeviceInfo> out;
    if (!enumerator_) return out;

    std::wstring defIn, defOut, defInComm, defOutComm;
    auto defaultId = [&](EDataFlow f, ERole r) {
        IMMDevice* d = nullptr;
        std::wstring id;
        if (SUCCEEDED(enumerator_->GetDefaultAudioEndpoint(f, r, &d))) {
            LPWSTR s = nullptr;
            if (SUCCEEDED(d->GetId(&s))) id = s;
            CoTaskMemFree(s);
            d->Release();
        }
        return id;
    };
    defIn = defaultId(eCapture, eConsole);
    defOut = defaultId(eRender, eConsole);
    defInComm = defaultId(eCapture, eCommunications);
    defOutComm = defaultId(eRender, eCommunications);

    struct Ep {
        std::wstring id, desc, iface;
        bool capture;
        int ch;
        double rate;
        DWORD mask;
    };
    std::vector<Ep> eps;
    IMMDeviceCollection* coll = nullptr;
    if (FAILED(enumerator_->EnumAudioEndpoints(eAll, DEVICE_STATE_ACTIVE, &coll))) return out;
    UINT count = 0;
    coll->GetCount(&count);
    for (UINT i = 0; i < count; ++i) {
        IMMDevice* dev = nullptr;
        if (FAILED(coll->Item(i, &dev))) continue;
        Ep e{};
        LPWSTR id = nullptr;
        if (SUCCEEDED(dev->GetId(&id))) e.id = id;
        CoTaskMemFree(id);
        IMMEndpoint* ep = nullptr;
        EDataFlow flow = eRender;
        if (SUCCEEDED(dev->QueryInterface(__uuidof(IMMEndpoint), reinterpret_cast<void**>(&ep)))) {
            ep->GetDataFlow(&flow);
            ep->Release();
        }
        e.capture = flow == eCapture;
        IPropertyStore* ps = nullptr;
        if (SUCCEEDED(dev->OpenPropertyStore(STGM_READ, &ps))) {
            e.desc = propString(ps, PKEY_Device_DeviceDesc);
            e.iface = propString(ps, PKEY_DeviceInterface_FriendlyName);
            if (e.desc.empty()) e.desc = propString(ps, PKEY_Device_FriendlyName);
            if (!deviceFormat(ps, e.ch, e.rate, e.mask)) {
                e.ch = 2;
                e.rate = 48000;
                e.mask = 0;
            }
            ps->Release();
        }
        dev->Release();
        eps.push_back(std::move(e));
    }
    coll->Release();

    // Group virtual-cable endpoints into render/capture pairs by interface name.
    std::map<std::wstring, std::pair<const Ep*, const Ep*>> cables;  // iface -> (render, capture)
    for (const Ep& e : eps) {
        if (isToneSphereEndpoint(e.iface)) continue;  // provided by the VAD backend
        if (compat_ && isVirtualCable(e.iface, e.desc)) {
            auto& pair = cables[e.iface];
            (e.capture ? pair.second : pair.first) = &e;
            continue;
        }
        DeviceInfo d;
        d.kind = DeviceKind::Wasapi;
        d.input = e.capture;
        d.bin = e.capture ? Bin::PhysicalIn : Bin::PhysicalOut;
        d.endpointId = toUtf8(e.id);
        d.nodeId = std::string(e.capture ? "wasapi:in:" : "wasapi:out:") + d.endpointId;
        d.name = toUtf8(e.desc);
        d.subtitle = toUtf8(e.iface);
        d.channels = e.ch;
        d.sampleRate = e.rate;
        d.channelNames = channelNamesFor(e.ch, e.mask);
        d.isDefault = e.id == (e.capture ? defIn : defOut);
        d.isDefaultComms = e.id == (e.capture ? defInComm : defOutComm);
        d.supportsExclusive = true;
        out.push_back(std::move(d));
    }
    for (auto& [iface, pair] : cables) {
        const Ep* r = pair.first;
        const Ep* c = pair.second;
        if (!r || !c) continue;
        // Virtual output: apps play into the render side; ToneSphere records the capture side.
        DeviceInfo vo;
        vo.kind = DeviceKind::Compat;
        vo.bin = Bin::VirtualOut;
        vo.input = true;
        vo.endpointId = toUtf8(c->id);
        vo.nodeId = "compat:out:" + toUtf8(r->id);
        vo.appEndpointId = toUtf8(r->id);
        vo.name = toUtf8(r->desc);
        vo.subtitle = toUtf8(iface);
        vo.channels = c->ch;
        vo.sampleRate = c->rate;
        vo.channelNames = channelNamesFor(c->ch, c->mask);
        vo.isDefault = r->id == defOut;
        out.push_back(vo);
        // Virtual input: ToneSphere plays into the render side; apps record the capture side.
        DeviceInfo vi;
        vi.kind = DeviceKind::Compat;
        vi.bin = Bin::VirtualIn;
        vi.input = false;
        vi.endpointId = toUtf8(r->id);
        vi.nodeId = "compat:in:" + toUtf8(c->id);
        vi.appEndpointId = toUtf8(c->id);
        vi.name = toUtf8(c->desc);
        vi.subtitle = toUtf8(iface);
        vi.channels = r->ch;
        vi.sampleRate = r->rate;
        vi.channelNames = channelNamesFor(r->ch, r->mask);
        vi.isDefault = c->id == defIn;
        out.push_back(vi);
    }
    return out;
}

std::unique_ptr<DeviceStream> WasapiBackend::open(const DeviceInfo& dev, const StreamRequest& req, std::string& error) {
    if (!enumerator_) {
        error = "Windows audio is not available";
        return nullptr;
    }
    IMMDevice* d = nullptr;
    const std::wstring wid = toWide(dev.endpointId);
    HRESULT hr = enumerator_->GetDevice(wid.c_str(), &d);
    if (FAILED(hr)) {
        error = "Device not found";
        return nullptr;
    }
    IAudioClient* client = nullptr;
    hr = d->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&client));
    if (FAILED(hr)) {
        d->Release();
        error = "Cannot open device";
        return nullptr;
    }
    auto s = std::make_unique<WasapiStream>(client, dev.input, dev.name);
    bool ok = false;
    if (req.exclusive && dev.kind == DeviceKind::Wasapi) {
        ok = s->initExclusive(req.bufferMs, req.preferredRate, dev.channels, error);
        if (!ok) {
            TS_WARN("%s: exclusive mode failed (%s), using shared mode", dev.name.c_str(), error.c_str());
            client = nullptr;
            hr = d->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&client));
            if (SUCCEEDED(hr)) {
                s = std::make_unique<WasapiStream>(client, dev.input, dev.name);
                ok = s->initShared(req.bufferMs, false, nullptr, error);
            }
        }
    } else {
        ok = s->initShared(req.bufferMs, false, nullptr, error);
    }
    d->Release();
    if (!ok) return nullptr;
    return s;
}

// ------------------------------------------------------------------ stream
WasapiStream::WasapiStream(IAudioClient* client, bool capture, std::string label)
    : client_(client), capture_(capture), label_(std::move(label)) {}

WasapiStream::~WasapiStream() {
    stop();
    safeRelease(client_);
    if (event_) CloseHandle(event_);
}

bool WasapiStream::initShared(double bufferMs, bool loopback, const WAVEFORMATEX* forced, std::string& err) {
    WAVEFORMATEX* mix = nullptr;
    WAVEFORMATEXTENSIBLE fmt{};
    if (forced) {
        std::memcpy(&fmt, forced, forced->wFormatTag == WAVE_FORMAT_EXTENSIBLE ? sizeof(WAVEFORMATEXTENSIBLE) : sizeof(WAVEFORMATEX));
    } else {
        if (FAILED(client_->GetMixFormat(&mix))) {
            err = "Cannot read the device format";
            return false;
        }
        DWORD mask = mix->wFormatTag == WAVE_FORMAT_EXTENSIBLE ? reinterpret_cast<WAVEFORMATEXTENSIBLE*>(mix)->dwChannelMask : 0;
        fmt = makeFormat(mix->nChannels, mix->nSamplesPerSec, SampleFormat::Float32, mask);
    }
    // Process loopback only accepts LOOPBACK | EVENTCALLBACK | AUTOCONVERTPCM.
    DWORD flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK | (loopback ? AUDCLNT_STREAMFLAGS_LOOPBACK : AUDCLNT_STREAMFLAGS_NOPERSIST);
    HRESULT hr = E_FAIL;

    // Low-latency shared mode (IAudioClient3) when a small buffer is requested.
    if (!loopback && !forced && bufferMs < 10.0 && mix) {
        SampleFormat msf;
        int mb;
        IAudioClient3* c3 = nullptr;
        if (parseFormat(mix, msf, mb) && msf == SampleFormat::Float32 &&
            SUCCEEDED(client_->QueryInterface(__uuidof(IAudioClient3), reinterpret_cast<void**>(&c3)))) {
            UINT32 def = 0, fund = 0, mn = 0, mx = 0;
            if (SUCCEEDED(c3->GetSharedModeEnginePeriod(mix, &def, &fund, &mn, &mx)) && fund > 0) {
                const UINT32 want = static_cast<UINT32>(bufferMs * mix->nSamplesPerSec / 1000.0);
                UINT32 p = std::max(mn, ((want + fund - 1) / fund) * fund);
                p = std::min(p, mx);
                hr = c3->InitializeSharedAudioStream(AUDCLNT_STREAMFLAGS_EVENTCALLBACK, p, mix, nullptr);
                if (SUCCEEDED(hr)) std::memcpy(&fmt, mix, mix->wFormatTag == WAVE_FORMAT_EXTENSIBLE ? sizeof(WAVEFORMATEXTENSIBLE) : sizeof(WAVEFORMATEX));
            }
            c3->Release();
        }
    }
    if (FAILED(hr)) {
        flags |= AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | (loopback ? 0 : AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY);
        const REFERENCE_TIME hns = static_cast<REFERENCE_TIME>(std::max(bufferMs, 3.0) * 10000.0);
        hr = client_->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, hns, 0, &fmt.Format, nullptr);
    }
    if (mix) CoTaskMemFree(mix);
    if (FAILED(hr)) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "Cannot start the device (0x%08lx)", hr);
        err = buf;
        return false;
    }
    exclusive_ = false;
    if (!parseFormat(&fmt.Format, fmt_, bytesPerSample_)) {
        err = "Unsupported sample format";
        return false;
    }
    rate_ = fmt.Format.nSamplesPerSec;
    channels_ = fmt.Format.nChannels;
    client_->GetBufferSize(&bufferFrames_);
    REFERENCE_TIME def = 0, mn = 0;
    if (FAILED(client_->GetDevicePeriod(&def, &mn)) || def <= 0) def = 100000;  // process loopback: assume 10 ms
    period_ = std::max(1, static_cast<int>(static_cast<double>(def) * 1e-7 * rate_ + 0.5));
    REFERENCE_TIME lat = 0;
    client_->GetStreamLatency(&lat);
    latencyMs_ = static_cast<double>(lat) / 10000.0;
    event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    client_->SetEventHandle(event_);
    return true;
}

bool WasapiStream::initExclusive(double bufferMs, double preferredRate, int preferredChannels, std::string& err) {
    WAVEFORMATEX* mix = nullptr;
    if (FAILED(client_->GetMixFormat(&mix))) {
        err = "Cannot read the device format";
        return false;
    }
    const double rate = preferredRate > 0 ? preferredRate : mix->nSamplesPerSec;
    const int ch = preferredChannels > 0 ? preferredChannels : mix->nChannels;
    const DWORD mask = mix->wFormatTag == WAVE_FORMAT_EXTENSIBLE ? reinterpret_cast<WAVEFORMATEXTENSIBLE*>(mix)->dwChannelMask : 0;
    CoTaskMemFree(mix);

    WAVEFORMATEXTENSIBLE fmt{};
    bool found = false;
    for (double r : {rate, 48000.0, 44100.0, 96000.0}) {
        for (SampleFormat sf : {SampleFormat::Float32, SampleFormat::Int32, SampleFormat::Int24, SampleFormat::Int16}) {
            WAVEFORMATEXTENSIBLE f = makeFormat(ch, r, sf, mask);
            if (client_->IsFormatSupported(AUDCLNT_SHAREMODE_EXCLUSIVE, &f.Format, nullptr) == S_OK) {
                fmt = f;
                found = true;
                break;
            }
        }
        if (found) break;
    }
    if (!found) {
        err = "No exclusive-mode format supported";
        return false;
    }
    REFERENCE_TIME def = 0, mn = 0;
    client_->GetDevicePeriod(&def, &mn);
    REFERENCE_TIME period = std::max(mn, static_cast<REFERENCE_TIME>(bufferMs * 10000.0));
    HRESULT hr = client_->Initialize(AUDCLNT_SHAREMODE_EXCLUSIVE, AUDCLNT_STREAMFLAGS_EVENTCALLBACK, period, period, &fmt.Format, nullptr);
    if (hr == AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED) {
        UINT32 frames = 0;
        client_->GetBufferSize(&frames);
        err = "buffer alignment";  // caller falls back to shared mode; aligned retry needs a fresh client
        return false;
    }
    if (FAILED(hr)) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "exclusive init failed (0x%08lx)", hr);
        err = buf;
        return false;
    }
    exclusive_ = true;
    parseFormat(&fmt.Format, fmt_, bytesPerSample_);
    rate_ = fmt.Format.nSamplesPerSec;
    channels_ = fmt.Format.nChannels;
    client_->GetBufferSize(&bufferFrames_);
    period_ = static_cast<int>(bufferFrames_);
    latencyMs_ = static_cast<double>(period) / 10000.0;
    event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    client_->SetEventHandle(event_);
    return true;
}

bool WasapiStream::start(std::shared_ptr<CaptureBridge> capture, std::shared_ptr<RenderBridge> render) {
    cap_ = std::move(capture);
    ren_ = std::move(render);
    if (!client_ || !event_) return false;
    if (!capture_) {
        // Pre-fill with silence (required for exclusive event-driven rendering).
        IAudioRenderClient* rc = nullptr;
        if (SUCCEEDED(client_->GetService(__uuidof(IAudioRenderClient), reinterpret_cast<void**>(&rc)))) {
            BYTE* data = nullptr;
            if (SUCCEEDED(rc->GetBuffer(bufferFrames_, &data))) rc->ReleaseBuffer(bufferFrames_, AUDCLNT_BUFFERFLAGS_SILENT);
            rc->Release();
        }
    }
    if (cap_) cap_->markStarted();
    if (ren_) ren_->markStarted();
    running_ = true;
    thread_ = std::thread([this] { run(); });
    return true;
}

void WasapiStream::stop() {
    running_ = false;
    if (thread_.joinable()) {
        if (event_) SetEvent(event_);
        thread_.join();
    }
    if (cap_) cap_->markStopped();
    if (ren_) ren_->markStopped();
}

std::string WasapiStream::error() const { return hasError_.load() ? error_ : std::string(); }

void WasapiStream::run() {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    setAudioThreadPriority(capture_ ? "ts-wasapi-in" : "ts-wasapi-out");
    if (FAILED(client_->Start())) {
        error_ = "Device could not start";
        hasError_ = true;
    } else {
        if (capture_)
            captureLoop();
        else
            renderLoop();
        client_->Stop();
    }
    if (cap_) cap_->markStopped();
    if (ren_) ren_->markStopped();
    CoUninitialize();
}

void WasapiStream::captureLoop() {
    IAudioCaptureClient* cc = nullptr;
    if (FAILED(client_->GetService(__uuidof(IAudioCaptureClient), reinterpret_cast<void**>(&cc)))) {
        error_ = "No capture service";
        hasError_ = true;
        return;
    }
    std::vector<float> tmp(static_cast<size_t>(std::max<unsigned>(bufferFrames_, 4096)) * channels_);
    while (running_) {
        const DWORD w = WaitForSingleObject(event_, 500);
        if (!running_) break;
        if (w != WAIT_OBJECT_0) continue;
        for (;;) {
            UINT32 packet = 0;
            HRESULT hr = cc->GetNextPacketSize(&packet);
            if (FAILED(hr)) {
                error_ = hr == AUDCLNT_E_DEVICE_INVALIDATED ? "Device was removed" : "Capture error";
                hasError_ = true;
                running_ = false;
                break;
            }
            if (packet == 0) break;
            BYTE* data = nullptr;
            UINT32 frames = 0;
            DWORD flags = 0;
            hr = cc->GetBuffer(&data, &frames, &flags, nullptr, nullptr);
            if (FAILED(hr)) break;
            // Arrival time (not the capture-position timestamp, which can be much older): the bridge
            // models data becoming available continuously from this moment.
            const double t = nowSeconds();
            if (cap_) {
                if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
                    cap_->deviceWriteSilence(frames, t);
                } else {
                    const size_t samples = static_cast<size_t>(frames) * channels_;
                    if (tmp.size() < samples) tmp.resize(samples);
                    toFloat(data, tmp.data(), samples, fmt_);
                    cap_->deviceWrite(tmp.data(), frames, t);
                }
            }
            cc->ReleaseBuffer(frames);
        }
    }
    cc->Release();
}

void WasapiStream::renderLoop() {
    IAudioRenderClient* rc = nullptr;
    if (FAILED(client_->GetService(__uuidof(IAudioRenderClient), reinterpret_cast<void**>(&rc)))) {
        error_ = "No render service";
        hasError_ = true;
        return;
    }
    std::vector<float> tmp(static_cast<size_t>(bufferFrames_) * channels_);
    // Periods kept queued in WASAPI (shared mode). Starts low for latency and grows when the
    // device is found empty (a late wake-up under load, or a device that consumes in bursts).
    UINT32 queuePeriods = 3;
    bool primed = false;
    while (running_) {
        const DWORD w = WaitForSingleObject(event_, 500);
        if (!running_) break;
        if (w != WAIT_OBJECT_0) continue;
        UINT32 frames = bufferFrames_;
        if (!exclusive_) {
            UINT32 padding = 0;
            HRESULT hr = client_->GetCurrentPadding(&padding);
            if (FAILED(hr)) {
                error_ = hr == AUDCLNT_E_DEVICE_INVALIDATED ? "Device was removed" : "Playback error";
                hasError_ = true;
                break;
            }
            if (primed && padding == 0) {
                glitches_.fetch_add(1, std::memory_order_relaxed);
                if (queuePeriods < 4) ++queuePeriods;
            }
            // Keep a few periods queued in WASAPI (not the whole buffer): consumption from our
            // ring stays regular, which keeps drift estimation clean and latency low.
            const UINT32 want = std::min<UINT32>(bufferFrames_, queuePeriods * static_cast<UINT32>(period_));
            frames = padding >= want ? 0 : std::min(bufferFrames_ - padding, want - padding);
        }
        if (frames == 0) continue;
        BYTE* data = nullptr;
        HRESULT hr = rc->GetBuffer(frames, &data);
        if (FAILED(hr)) {
            if (hr == AUDCLNT_E_DEVICE_INVALIDATED) {
                error_ = "Device was removed";
                hasError_ = true;
                break;
            }
            continue;
        }
        const size_t samples = static_cast<size_t>(frames) * channels_;
        if (ren_) {
            ren_->deviceRead(tmp.data(), frames, nowSeconds());
            fromFloat(tmp.data(), data, samples, fmt_);
            rc->ReleaseBuffer(frames, 0);
            primed = true;
        } else {
            rc->ReleaseBuffer(frames, AUDCLNT_BUFFERFLAGS_SILENT);
        }
    }
    rc->Release();
}

}  // namespace ts
