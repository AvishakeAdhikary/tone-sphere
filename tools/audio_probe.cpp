// audio-probe: plays or records a Windows audio endpoint and analyses a test tone.
//   audio-probe list
//   audio-probe capture <name-fragment> <seconds>          -> JSON analysis of recorded tone
//   audio-probe play <name-fragment> <freqHz> <dBFS> <seconds>
// Used by automated loopback tests: frequency, level, dropouts (silence gaps) and
// discontinuities (clicks) are measured with a sine predictor.
#include <windows.h>

#include <audioclient.h>
#include <mmdeviceapi.h>
#include <functiondiscoverykeys_devpkey.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#pragma comment(lib, "ole32.lib")

namespace {

std::string utf8(const wchar_t* w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    return s;
}

IMMDevice* findDevice(EDataFlow flow, const std::string& frag, std::string& name) {
    IMMDeviceEnumerator* en = nullptr;
    CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), reinterpret_cast<void**>(&en));
    IMMDeviceCollection* coll = nullptr;
    en->EnumAudioEndpoints(flow, DEVICE_STATE_ACTIVE, &coll);
    UINT n = 0;
    coll->GetCount(&n);
    IMMDevice* found = nullptr;
    for (UINT i = 0; i < n && !found; ++i) {
        IMMDevice* d = nullptr;
        coll->Item(i, &d);
        IPropertyStore* ps = nullptr;
        d->OpenPropertyStore(STGM_READ, &ps);
        PROPVARIANT v;
        PropVariantInit(&v);
        ps->GetValue(PKEY_Device_FriendlyName, &v);
        std::string fn = v.pwszVal ? utf8(v.pwszVal) : "";
        PropVariantClear(&v);
        ps->Release();
        if (frag.empty() || lower(fn).find(lower(frag)) != std::string::npos) {
            found = d;
            name = fn;
        } else {
            d->Release();
        }
    }
    coll->Release();
    en->Release();
    return found;
}

void list() {
    for (EDataFlow f : {eRender, eCapture}) {
        IMMDeviceEnumerator* en = nullptr;
        CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator), reinterpret_cast<void**>(&en));
        IMMDeviceCollection* coll = nullptr;
        en->EnumAudioEndpoints(f, DEVICE_STATE_ACTIVE, &coll);
        UINT n = 0;
        coll->GetCount(&n);
        for (UINT i = 0; i < n; ++i) {
            IMMDevice* d = nullptr;
            coll->Item(i, &d);
            IPropertyStore* ps = nullptr;
            d->OpenPropertyStore(STGM_READ, &ps);
            PROPVARIANT v;
            PropVariantInit(&v);
            ps->GetValue(PKEY_Device_FriendlyName, &v);
            std::printf("%s  %s\n", f == eRender ? "render " : "capture", v.pwszVal ? utf8(v.pwszVal).c_str() : "?");
            PropVariantClear(&v);
            ps->Release();
            d->Release();
        }
        coll->Release();
        en->Release();
    }
}

// Shared-mode float32 stream with the device mix format.
struct Stream {
    IAudioClient* client = nullptr;
    WAVEFORMATEX* fmt = nullptr;
    UINT32 bufferFrames = 0;
    bool open(IMMDevice* d) {
        if (FAILED(d->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&client)))) return false;
        client->GetMixFormat(&fmt);
        WAVEFORMATEXTENSIBLE f{};
        f.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
        f.Format.nChannels = fmt->nChannels;
        f.Format.nSamplesPerSec = fmt->nSamplesPerSec;
        f.Format.wBitsPerSample = 32;
        f.Format.nBlockAlign = static_cast<WORD>(4 * fmt->nChannels);
        f.Format.nAvgBytesPerSec = f.Format.nBlockAlign * f.Format.nSamplesPerSec;
        f.Format.cbSize = 22;
        f.Samples.wValidBitsPerSample = 32;
        f.dwChannelMask = 0;
        f.SubFormat = {0x00000003, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};  // IEEE float
        HRESULT hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
                                        2000000, 0, &f.Format, nullptr);
        if (FAILED(hr)) return false;
        client->GetBufferSize(&bufferFrames);
        return true;
    }
};

int capture(const std::string& frag, double seconds) {
    std::string name;
    IMMDevice* d = findDevice(eCapture, frag, name);
    if (!d) {
        std::printf("{\"error\":\"capture device not found\"}\n");
        return 2;
    }
    Stream s;
    if (!s.open(d)) {
        std::printf("{\"error\":\"cannot open\"}\n");
        return 2;
    }
    const int ch = s.fmt->nChannels;
    const double rate = s.fmt->nSamplesPerSec;
    IAudioCaptureClient* cc = nullptr;
    s.client->GetService(__uuidof(IAudioCaptureClient), reinterpret_cast<void**>(&cc));
    std::vector<float> mono;
    size_t discontinuities = 0;  // the capture side itself lost data (not the signal's fault)
    s.client->Start();
    const ULONGLONG end = GetTickCount64() + static_cast<ULONGLONG>(seconds * 1000);
    while (GetTickCount64() < end) {
        Sleep(5);
        UINT32 packet = 0;
        while (SUCCEEDED(cc->GetNextPacketSize(&packet)) && packet) {
            BYTE* data;
            UINT32 frames;
            DWORD flags;
            cc->GetBuffer(&data, &frames, &flags, nullptr, nullptr);
            if ((flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY) && !mono.empty()) ++discontinuities;
            const float* f = reinterpret_cast<const float*>(data);
            for (UINT32 i = 0; i < frames; ++i) mono.push_back(flags & AUDCLNT_BUFFERFLAGS_SILENT ? 0.0f : f[i * ch]);
            cc->ReleaseBuffer(frames);
        }
    }
    s.client->Stop();

    // Skip the first 0.5 s (stream start-up) and analyse the rest.
    const size_t skip = static_cast<size_t>(rate * 0.5);
    double peak = 0, sum = 0;
    size_t n = 0, crossings = 0;
    for (size_t i = skip; i < mono.size(); ++i) {
        peak = std::max(peak, static_cast<double>(std::fabs(mono[i])));
        sum += static_cast<double>(mono[i]) * mono[i];
        ++n;
        if (i > skip && mono[i - 1] < 0 && mono[i] >= 0) ++crossings;
    }
    const double rms = n ? std::sqrt(sum / n) : 0;
    const double freq = n ? crossings * rate / n : 0;
    // Sine predictor: x[n+1] = 2cos(w)x[n] - x[n-1]; large residual = click/discontinuity.
    const double w = 2 * 3.14159265358979 * freq / rate;
    const double k = 2 * std::cos(w);
    size_t clicks = 0, gaps = 0, silentRun = 0;
    double firstGapAt = -1;
    const double thr = std::max(0.02, peak * 0.08);
    for (size_t i = skip + 2; i < mono.size(); ++i) {
        const double pred = k * mono[i - 1] - mono[i - 2];
        if (std::fabs(mono[i] - pred) > thr) ++clicks;
        if (std::fabs(mono[i]) < 1e-5) {
            if (++silentRun == static_cast<size_t>(rate * 0.002)) {  // >2 ms of digital silence
                if (firstGapAt < 0) firstGapAt = i / rate;
                ++gaps;
            }
        } else {
            silentRun = 0;
        }
    }
    std::printf("{\"device\":\"%s\",\"rate\":%.0f,\"seconds\":%.2f,\"freqHz\":%.1f,\"rmsDb\":%.2f,\"peakDb\":%.2f,\"clicks\":%zu,\"gaps\":%zu,\"firstGapAt\":%.3f,\"discontinuities\":%zu}\n",
                name.c_str(), rate, n / rate, freq, rms > 0 ? 20 * std::log10(rms) : -200.0, peak > 0 ? 20 * std::log10(peak) : -200.0,
                clicks, gaps, firstGapAt, discontinuities);
    return 0;
}

int play(const std::string& frag, double freq, double db, double seconds) {
    std::string name;
    IMMDevice* d = findDevice(eRender, frag, name);
    if (!d) {
        std::printf("{\"error\":\"render device not found\"}\n");
        return 2;
    }
    Stream s;
    if (!s.open(d)) return 2;
    const int ch = s.fmt->nChannels;
    const double rate = s.fmt->nSamplesPerSec;
    IAudioRenderClient* rc = nullptr;
    s.client->GetService(__uuidof(IAudioRenderClient), reinterpret_cast<void**>(&rc));
    const double amp = std::pow(10.0, db / 20.0);
    double phase = 0;
    const double inc = 2 * 3.14159265358979 * freq / rate;
    s.client->Start();
    const ULONGLONG end = GetTickCount64() + static_cast<ULONGLONG>(seconds * 1000);
    while (GetTickCount64() < end) {
        Sleep(5);
        UINT32 pad = 0;
        s.client->GetCurrentPadding(&pad);
        const UINT32 frames = s.bufferFrames - pad;
        if (!frames) continue;
        BYTE* data;
        if (FAILED(rc->GetBuffer(frames, &data))) continue;
        float* f = reinterpret_cast<float*>(data);
        for (UINT32 i = 0; i < frames; ++i) {
            const float v = static_cast<float>(amp * std::sin(phase));
            phase += inc;
            for (int c = 0; c < ch; ++c) f[i * ch + c] = v;
        }
        rc->ReleaseBuffer(frames, 0);
    }
    Sleep(200);
    s.client->Stop();
    std::printf("{\"played\":\"%s\"}\n", name.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    SetConsoleOutputCP(CP_UTF8);
    const std::string cmd = argc > 1 ? argv[1] : "list";
    if (cmd == "list") {
        list();
        return 0;
    }
    if (cmd == "capture" && argc >= 4) return capture(argv[2], std::atof(argv[3]));
    if (cmd == "play" && argc >= 6) return play(argv[2], std::atof(argv[3]), std::atof(argv[4]), std::atof(argv[5]));
    std::fprintf(stderr, "usage: audio-probe list | capture <name> <sec> | play <name> <freq> <dB> <sec>\n");
    return 1;
}
