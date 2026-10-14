// asio-test-host: minimal ASIO host for automated driver tests.
//   asio-test-host list
//   asio-test-host run "<driver name>" <seconds> [toneHz] [toneDb] [bufferSize]
// Plays a sine on outputs 1-2 (toneHz 0 = silence) and analyses inputs 1-2.
// Prints a JSON report: buffer switches, timing, input frequency/level/clicks.
#include <windows.h>
#include <objbase.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "asiosys.h"
#include "asio.h"
#include "iasiodrv.h"

namespace {

struct DriverEntry {
    std::string name;
    CLSID clsid;
};

std::vector<DriverEntry> listDrivers() {
    std::vector<DriverEntry> out;
    HKEY asio;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SOFTWARE\\ASIO", 0, KEY_READ, &asio) != ERROR_SUCCESS) return out;
    char name[256];
    for (DWORD i = 0;; ++i) {
        DWORD len = sizeof(name);
        if (RegEnumKeyExA(asio, i, name, &len, nullptr, nullptr, nullptr, nullptr) != ERROR_SUCCESS) break;
        char clsid[128];
        DWORD clen = sizeof(clsid);
        if (RegGetValueA(asio, name, "CLSID", RRF_RT_REG_SZ, nullptr, clsid, &clen) != ERROR_SUCCESS) continue;
        wchar_t wclsid[128];
        MultiByteToWideChar(CP_ACP, 0, clsid, -1, wclsid, 128);
        DriverEntry e;
        e.name = name;
        if (SUCCEEDED(CLSIDFromString(wclsid, &e.clsid))) out.push_back(e);
    }
    RegCloseKey(asio);
    return out;
}

IASIO* g_drv = nullptr;
std::vector<ASIOBufferInfo> g_bufs;
std::vector<ASIOChannelInfo> g_info;
long g_bufSize = 0;
double g_rate = 48000;
double g_toneHz = 1000, g_toneAmp = 0.25, g_phase = 0;
std::vector<float> g_in;  // captured input channel 1
std::atomic<long> g_switches{0};
std::vector<double> g_intervals;
LARGE_INTEGER g_freq, g_last{};
bool g_timeInfo = false;

float readSample(void* buf, long i, ASIOSampleType t) {
    switch (t) {
        case ASIOSTFloat32LSB: return static_cast<float*>(buf)[i];
        case ASIOSTInt32LSB: return static_cast<float>(static_cast<int32_t*>(buf)[i]) / 2147483648.0f;
        case ASIOSTInt16LSB: return static_cast<float>(static_cast<int16_t*>(buf)[i]) / 32768.0f;
        case ASIOSTInt24LSB: {
            auto* p = static_cast<uint8_t*>(buf) + i * 3;
            return static_cast<float>((static_cast<int32_t>(p[2]) << 24) | (p[1] << 16) | (p[0] << 8)) / 2147483648.0f;
        }
        default: return 0;
    }
}

void writeSample(void* buf, long i, ASIOSampleType t, float v) {
    switch (t) {
        case ASIOSTFloat32LSB: static_cast<float*>(buf)[i] = v; break;
        case ASIOSTInt32LSB: static_cast<int32_t*>(buf)[i] = static_cast<int32_t>(v * 2147483647.0); break;
        case ASIOSTInt16LSB: static_cast<int16_t*>(buf)[i] = static_cast<int16_t>(v * 32767.0f); break;
        case ASIOSTInt24LSB: {
            const int32_t x = static_cast<int32_t>(v * 8388607.0f);
            auto* p = static_cast<uint8_t*>(buf) + i * 3;
            p[0] = static_cast<uint8_t>(x);
            p[1] = static_cast<uint8_t>(x >> 8);
            p[2] = static_cast<uint8_t>(x >> 16);
            break;
        }
        default: break;
    }
}

void process(long index) {
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    if (g_last.QuadPart) g_intervals.push_back(static_cast<double>(now.QuadPart - g_last.QuadPart) / g_freq.QuadPart * 1000.0);
    g_last = now;
    const double inc = 2 * 3.14159265358979 * g_toneHz / g_rate;
    std::vector<float> tone(g_bufSize);
    for (long i = 0; i < g_bufSize; ++i) {
        tone[i] = g_toneHz > 0 ? static_cast<float>(g_toneAmp * std::sin(g_phase)) : 0.0f;
        g_phase += inc;
    }
    g_phase = std::fmod(g_phase, 2 * 3.14159265358979);
    for (size_t k = 0; k < g_bufs.size(); ++k) {
        auto& b = g_bufs[k];
        const ASIOSampleType t = g_info[k].type;
        if (b.isInput) {
            if (b.channelNum == 0)
                for (long i = 0; i < g_bufSize; ++i) g_in.push_back(readSample(b.buffers[index], i, t));
        } else {
            for (long i = 0; i < g_bufSize; ++i) writeSample(b.buffers[index], i, t, tone[i]);
        }
    }
    ++g_switches;
}

void bufferSwitch(long index, ASIOBool) { process(index); }
ASIOTime* bufferSwitchTimeInfo(ASIOTime* t, long index, ASIOBool) {
    process(index);
    return t;
}
void sampleRateChanged(ASIOSampleRate r) { g_rate = r; }
long asioMessage(long selector, long value, void*, double*) {
    switch (selector) {
        case kAsioSelectorSupported:
            return value == kAsioEngineVersion || value == kAsioSupportsTimeInfo || value == kAsioResetRequest;
        case kAsioEngineVersion: return 2;
        case kAsioSupportsTimeInfo: return 1;
        default: return 0;
    }
}

}  // namespace

int main(int argc, char** argv) {
    CoInitialize(nullptr);
    QueryPerformanceFrequency(&g_freq);
    const std::string cmd = argc > 1 ? argv[1] : "list";
    auto drivers = listDrivers();
    if (cmd == "list") {
        for (auto& d : drivers) std::printf("%s\n", d.name.c_str());
        return 0;
    }
    if (cmd != "run" || argc < 4) {
        std::fprintf(stderr, "usage: asio-test-host list | run <driver> <seconds> [toneHz] [toneDb] [bufferSize]\n");
        return 1;
    }
    const std::string want = argv[2];
    const double seconds = std::atof(argv[3]);
    g_toneHz = argc > 4 ? std::atof(argv[4]) : 1000;
    g_toneAmp = std::pow(10.0, (argc > 5 ? std::atof(argv[5]) : -12.0) / 20.0);
    const long requestedBuf = argc > 6 ? std::atol(argv[6]) : 0;
    auto it = std::find_if(drivers.begin(), drivers.end(), [&](auto& d) { return d.name == want; });
    if (it == drivers.end()) {
        std::printf("{\"error\":\"driver not found\"}\n");
        return 2;
    }
    if (FAILED(CoCreateInstance(it->clsid, nullptr, CLSCTX_INPROC_SERVER, it->clsid, reinterpret_cast<void**>(&g_drv)))) {
        std::printf("{\"error\":\"CoCreateInstance failed\"}\n");
        return 2;
    }
    if (!g_drv->init(GetConsoleWindow())) {
        char msg[128] = {};
        g_drv->getErrorMessage(msg);
        std::printf("{\"error\":\"init failed: %s\"}\n", msg);
        return 2;
    }
    long ins = 0, outs = 0, mn = 0, mx = 0, pref = 0, gran = 0;
    g_drv->getChannels(&ins, &outs);
    g_drv->getBufferSize(&mn, &mx, &pref, &gran);
    g_drv->getSampleRate(&g_rate);
    g_bufSize = requestedBuf ? requestedBuf : pref;
    for (long c = 0; c < std::min(2L, ins); ++c) g_bufs.push_back({ASIOTrue, c, {nullptr, nullptr}});
    for (long c = 0; c < std::min(2L, outs); ++c) g_bufs.push_back({ASIOFalse, c, {nullptr, nullptr}});
    static ASIOCallbacks cb{bufferSwitch, sampleRateChanged, asioMessage, bufferSwitchTimeInfo};
    if (g_drv->createBuffers(g_bufs.data(), static_cast<long>(g_bufs.size()), g_bufSize, &cb) != ASE_OK) {
        std::printf("{\"error\":\"createBuffers failed\"}\n");
        return 2;
    }
    for (auto& b : g_bufs) {
        ASIOChannelInfo ci{};
        ci.channel = b.channelNum;
        ci.isInput = b.isInput;
        g_drv->getChannelInfo(&ci);
        g_info.push_back(ci);
    }
    long inLat = 0, outLat = 0;
    g_drv->getLatencies(&inLat, &outLat);
    g_in.reserve(static_cast<size_t>(seconds * g_rate) + 8192);
    g_drv->start();
    const ULONGLONG end = GetTickCount64() + static_cast<ULONGLONG>(seconds * 1000);
    while (GetTickCount64() < end) {
        MSG m;
        while (PeekMessageW(&m, nullptr, 0, 0, PM_REMOVE)) DispatchMessageW(&m);
        Sleep(10);
    }
    g_drv->stop();
    g_drv->disposeBuffers();
    g_drv->Release();

    // Analyse input channel 1 (skip the first second).
    const size_t skip = std::min(g_in.size(), static_cast<size_t>(g_rate));
    double peak = 0, sum = 0;
    size_t n = 0, crossings = 0;
    for (size_t i = skip; i < g_in.size(); ++i) {
        peak = std::max(peak, static_cast<double>(std::fabs(g_in[i])));
        sum += static_cast<double>(g_in[i]) * g_in[i];
        if (i > skip && g_in[i - 1] < 0 && g_in[i] >= 0) ++crossings;
        ++n;
    }
    const double freq = n ? crossings * g_rate / n : 0;
    const double k = 2 * std::cos(2 * 3.14159265358979 * freq / g_rate);
    size_t clicks = 0;
    const double thr = std::max(0.02, peak * 0.08);
    for (size_t i = skip + 2; i < g_in.size(); ++i)
        if (std::fabs(g_in[i] - (k * g_in[i - 1] - g_in[i - 2])) > thr) ++clicks;
    double maxInt = 0, sumInt = 0;
    for (double v : g_intervals) {
        maxInt = std::max(maxInt, v);
        sumInt += v;
    }
    const double expectSwitches = seconds * g_rate / g_bufSize;
    std::printf(
        "{\"driver\":\"%s\",\"inputs\":%ld,\"outputs\":%ld,\"rate\":%.0f,\"buffer\":%ld,\"inLatency\":%ld,\"outLatency\":%ld,"
        "\"switches\":%ld,\"expectedSwitches\":%.0f,\"avgIntervalMs\":%.3f,\"maxIntervalMs\":%.3f,"
        "\"inFreqHz\":%.1f,\"inPeakDb\":%.2f,\"inRmsDb\":%.2f,\"inClicks\":%zu,\"sampleType\":%ld}\n",
        want.c_str(), ins, outs, g_rate, g_bufSize, inLat, outLat, g_switches.load(), expectSwitches,
        g_intervals.empty() ? 0.0 : sumInt / g_intervals.size(), maxInt, freq, peak > 0 ? 20 * std::log10(peak) : -200.0,
        n ? 10 * std::log10(sum / n + 1e-20) : -200.0, clicks, g_info.empty() ? -1L : static_cast<long>(g_info[0].type));
    return 0;
}
