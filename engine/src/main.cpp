// nextone-engine: the always-running NexTone audio engine.
//   nextone-engine [--background] [--port N] [--console]
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "core/engine.h"
#ifdef _WIN32
#include "backends/win/hotkeys.h"
#endif
#include "ipc/ws_server.h"
#include "util/log.h"
#include "util/platform.h"
#include "util/ui_thread.h"

#ifdef _WIN32
#include <windows.h>
#include <appmodel.h>
#include <objbase.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

using nlohmann::json;
using namespace nt;

namespace {

std::atomic<bool> g_quit{false};
#ifdef _WIN32
DWORD g_mainThread = 0;
#endif

int currentPid() {
#ifdef _WIN32
    return static_cast<int>(GetCurrentProcessId());
#else
    return static_cast<int>(getpid());
#endif
}

void requestQuit() {
    g_quit = true;
#ifdef _WIN32
    if (g_mainThread) PostThreadMessageW(g_mainThread, WM_QUIT, 0, 0);
#endif
}

#ifdef _WIN32
BOOL WINAPI consoleHandler(DWORD) {
    requestQuit();
    return TRUE;
}

// Hidden window: receives power (resume) and session-end notifications.
LRESULT CALLBACK hiddenProc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    static Engine* engine = nullptr;
    if (msg == WM_CREATE) engine = static_cast<Engine*>(reinterpret_cast<CREATESTRUCTW*>(lp)->lpCreateParams);
    if (msg == WM_POWERBROADCAST && (wp == PBT_APMRESUMEAUTOMATIC || wp == PBT_APMRESUMESUSPEND) && engine) {
        NT_INFO("System resumed: re-opening devices");
        // Never block the main thread on the engine (plugins run work on it while holding the engine lock).
        std::thread([e = engine] {
            try {
                e->call("devices.refresh", json::object());
            } catch (...) {
            }
        }).detach();
    }
    if (msg == WM_ENDSESSION && wp) requestQuit();
    if (msg == WM_QUERYENDSESSION) return TRUE;
    return DefWindowProcW(h, msg, wp, lp);
}
#endif

struct ComScope {
    ComScope() {
#ifdef _WIN32
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
#endif
    }
    ~ComScope() {
#ifdef _WIN32
        CoUninitialize();
#endif
    }
};

}  // namespace

int main(int argc, char** argv) {
    bool console = false;
    int port = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--console")) console = true;
        else if (!std::strcmp(argv[i], "--port") && i + 1 < argc) port = std::atoi(argv[++i]);
    }

#ifdef _WIN32
    // One engine per user session.
    HANDLE single = CreateMutexW(nullptr, TRUE, L"Local\\NexToneEngine");
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        std::fprintf(stderr, "NexTone engine is already running\n");
        return 0;
    }
    if (console && !GetConsoleWindow()) {
        AllocConsole();
        FILE* f = nullptr;
        freopen_s(&f, "CONOUT$", "w", stderr);
    }
    g_mainThread = GetCurrentThreadId();
    ui::init();  // plugin windows and plugin calls run on this thread (queued until the message loop runs)
    SetConsoleCtrlHandler(consoleHandler, TRUE);
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    // Never let a background audio app throttle us.
    PROCESS_POWER_THROTTLING_STATE pt{};
    pt.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    pt.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
    pt.StateMask = 0;
    SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pt, sizeof(pt));
    SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
    // Store build: the sign-in startup task runs the engine with no arguments (the app and
    // the ASIO driver always pass some). Bring up the app in the tray as well.
    UINT32 pkgLen = 0;
    if (argc == 1 && GetCurrentPackageFullName(&pkgLen, nullptr) == ERROR_INSUFFICIENT_BUFFER) {
        wchar_t self[MAX_PATH];
        std::wstring dir(self, GetModuleFileNameW(nullptr, self, MAX_PATH));
        dir = dir.substr(0, dir.find_last_of(L'\\'));  // ...\app\resources\engine
        dir = dir.substr(0, dir.find_last_of(L'\\'));  // ...\app\resources
        dir = dir.substr(0, dir.find_last_of(L'\\'));  // ...\app
        std::wstring cmd = L"\"" + dir + L"\\NexTone.exe\" --hidden";
        STARTUPINFOW si{sizeof(si)};
        PROCESS_INFORMATION pi{};
        if (CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
        }
    }
#else
    // One engine per user: an exclusive lock next to the discovery file.
    {
        const std::string disc = discoveryPath();
        const std::string lockPath = disc.substr(0, disc.find_last_of('/')) + "/engine.lock";
        makeDirs(disc.substr(0, disc.find_last_of('/')));
        const int lockFd = ::open(lockPath.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (lockFd < 0 || ::flock(lockFd, LOCK_EX | LOCK_NB) != 0) {
            std::fprintf(stderr, "NexTone engine is already running\n");
            return 0;
        }
        // The descriptor stays open (and locked) for the life of the process.
    }
    std::signal(SIGINT, [](int) { requestQuit(); });
    std::signal(SIGTERM, [](int) { requestQuit(); });
#endif

    log::init(dataDir() + "/engine.log", console);
    NT_INFO("NexTone engine starting (pid %d)", currentPid());

    Engine engine;
    WsServer server;
    std::mutex stateMu;
    std::condition_variable stateCv;
    std::atomic<bool> stateDirty{true};

    engine.onStateChanged = [&] {
        stateDirty = true;
        stateCv.notify_one();
    };
    engine.onShutdownRequested = [] { requestQuit(); };
    log::setSink([&](log::Level lvl, const std::string& msg) {
        json ev = {{"event", "log"}, {"data", {{"level", lvl == log::Level::Error ? "error" : "warn"}, {"message", msg}}}};
        server.broadcastText(ev.dump());
    });

    server.onConnect = [&](const std::shared_ptr<WsServer::Client>& c) {
        static thread_local ComScope com;
        json ev = {{"event", "state"}, {"data", engine.state()}};
        server.sendText(c, ev.dump(-1, ' ', false, json::error_handler_t::replace));
    };
    server.onMessage = [&](const std::shared_ptr<WsServer::Client>& c, const std::string& text) {
        static thread_local ComScope com;
        json req;
        try {
            req = json::parse(text);
        } catch (...) {
            return;
        }
        json resp = {{"id", req.value("id", 0)}};
        try {
            json result = engine.call(req.value("method", ""), req.contains("params") ? req["params"] : json::object());
            resp["result"] = result;
        } catch (const std::exception& e) {
            resp["error"] = {{"code", -32000}, {"message", e.what()}};
        }
        server.sendText(c, resp.dump(-1, ' ', false, json::error_handler_t::replace));
    };

    engine.start();

    const std::string token = randomHex(16);
    const int actualPort = server.start(token, port);
    if (!actualPort) {
        NT_ERROR("Cannot open the local control port");
        return 1;
    }
    const json discovery = {{"port", actualPort},
                            {"token", token},
                            {"pid", currentPid()},
                            {"version", engine.version()},
                            {"protocol", 1}};
    writeFileAtomic(discoveryPath(), discovery.dump(2));
    NT_INFO("Control port 127.0.0.1:%d", actualPort);

    // State broadcaster (<= 20 Hz) and meter broadcaster (~30 Hz).
    std::thread stateThread([&] {
        setThreadName("nt-ipc-state");
        ComScope com;
        while (!g_quit) {
            std::unique_lock lk(stateMu);
            stateCv.wait_for(lk, std::chrono::milliseconds(500), [&] { return stateDirty.load() || g_quit.load(); });
            lk.unlock();
            if (g_quit) break;
            if (stateDirty.exchange(false) && server.clientCount()) {
                json ev = {{"event", "state"}, {"data", engine.state()}};
                server.broadcastText(ev.dump(-1, ' ', false, json::error_handler_t::replace));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });
    std::thread meterThread([&] {
        setThreadName("nt-ipc-meters");
        std::vector<float> vals;
        std::vector<uint8_t> frame;
        while (!g_quit) {
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
            if (!server.clientCount()) continue;
            engine.pollExternalMeters();
            const uint32_t layout = engine.meters(vals);
            const uint32_t magic = 0x314D544E, count = static_cast<uint32_t>(vals.size());
            frame.resize(12 + vals.size() * 4);
            std::memcpy(frame.data(), &magic, 4);
            std::memcpy(frame.data() + 4, &layout, 4);
            std::memcpy(frame.data() + 8, &count, 4);
            if (count) std::memcpy(frame.data() + 12, vals.data(), vals.size() * 4);
            server.broadcastBinary(frame.data(), frame.size());
        }
    });

#ifdef _WIN32
    WNDCLASSW wc{};
    wc.lpfnWndProc = hiddenProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"NexToneEngineHidden";
    RegisterClassW(&wc);
    HWND hidden = CreateWindowExW(0, wc.lpszClassName, L"NexTone Engine", 0, 0, 0, 0, 0, nullptr, nullptr, wc.hInstance, &engine);
    Hotkeys hotkeys(engine);
    hotkeys.install();  // needs this thread's message loop
    MSG msg;
    while (!g_quit && GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    ui::stop();
    if (hidden) DestroyWindow(hidden);
#else
    while (!g_quit) std::this_thread::sleep_for(std::chrono::milliseconds(200));
#endif

    NT_INFO("Shutting down");
    g_quit = true;
    stateCv.notify_all();
    stateThread.join();
    meterThread.join();
    server.stop();
    engine.shutdown();
    std::remove(discoveryPath().c_str());
#ifdef _WIN32
    if (single) {
        ReleaseMutex(single);
        CloseHandle(single);
    }
    CoUninitialize();
#endif
    return 0;
}
