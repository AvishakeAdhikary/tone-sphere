// tonesphere-plugscan: loads VST3 modules in a separate process for the engine.
//   tonesphere-plugscan <module.vst3>                      # JSON report of the module's effects
//   tonesphere-plugscan --list [extra folders...]          # modules found in the VST3 folders
//   tonesphere-plugscan --probe <module.vst3> <classId>    # instantiate + process a test signal
// The JSON report is printed on its own last line (plugins sometimes print to stdout too).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "plugins/vst3_host.h"

#ifdef _WIN32
#include <windows.h>
#include <objbase.h>
#include <shellapi.h>
#endif

using nlohmann::json;
using namespace ts;

namespace {

void emit(const json& j) {
    std::fflush(stdout);
    std::printf("\n%s\n", j.dump().c_str());
    std::fflush(stdout);
}

double rmsDb(const std::vector<float>& v) {
    double s = 0;
    for (float x : v) s += double(x) * x;
    return 10.0 * std::log10(s / std::max<size_t>(v.size(), 1) + 1e-20);
}

json probe(const std::string& path, const std::string& cid) {
    auto fx = createVst3Fx(path, cid);
    constexpr int kBlock = 128, kBlocks = 400;
    constexpr double kRate = 48000;
    fx->prepare(kRate, kBlock, 2);
    std::vector<float> l(kBlock), r(kBlock), in, out;
    float* bufs[2] = {l.data(), r.data()};
    double ph = 0;
    for (int b = 0; b < kBlocks; ++b) {
        for (int i = 0; i < kBlock; ++i) {
            l[i] = r[i] = 0.25f * static_cast<float>(std::sin(ph));
            ph += 2 * 3.14159265358979 * 1000.0 / kRate;
        }
        if (b >= kBlocks / 2) in.insert(in.end(), l.begin(), l.end());
        fx->process(bufs, 2, kBlock);
        if (b >= kBlocks / 2) out.insert(out.end(), l.begin(), l.end());
    }
    const std::string state = fx->saveState();
    fx->loadState(state);
    const std::string state2 = fx->saveState();
    return {{"ok", fx->status().empty()},
            {"name", fx->name()},
            {"inDb", rmsDb(in)},
            {"outDb", rmsDb(out)},
            {"stateBytes", state.size()},
            {"stateRoundTrip", state == state2},
            {"hasEditor", fx->hasEditor()},
            {"status", fx->status()}};
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    // Arguments as UTF-8 (paths can contain any characters).
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    std::vector<std::string> args;
    for (int i = 1; i < wargc; ++i) {
        const int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s(n > 0 ? n - 1 : 0, '\0');
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), n, nullptr, nullptr);
        args.push_back(s);
    }
    LocalFree(wargv);
#else
    std::vector<std::string> args(argv + 1, argv + argc);
#endif
    (void)argc;
    (void)argv;
    if (args.empty()) {
        std::fprintf(stderr, "usage: tonesphere-plugscan <module.vst3> | --list [folders...] | --probe <module> <classId>\n");
        return 2;
    }
    try {
        if (args[0] == "--list") {
            emit({{"ok", true}, {"modules", standardVst3Modules({args.begin() + 1, args.end()})}});
        } else if (args[0] == "--probe" && args.size() >= 3) {
            emit(probe(args[1], args[2]));
        } else {
            emit(describeVst3Module(args[0]));
        }
    } catch (const std::exception& e) {
        emit({{"ok", false}, {"error", e.what()}});
        return 1;
    }
    std::fflush(stdout);
    // Some plugins misbehave on unload; the OS cleans up anyway.
#ifdef _WIN32
    TerminateProcess(GetCurrentProcess(), 0);
#endif
    return 0;
}
