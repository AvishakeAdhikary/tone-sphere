#include "util/ui_thread.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#endif

namespace nt::ui {

namespace {
std::atomic<bool> g_running{false};
#ifdef _WIN32
DWORD g_thread = 0;
HWND g_wnd = nullptr;
constexpr UINT kRunMsg = WM_APP + 17;

struct Task {
    std::function<void()> fn;
    std::promise<void> done;
    std::atomic<bool> claimed{false};  // whoever claims it runs it (main thread or, after stop, the waiter)

    void execute() {
        try {
            fn();
            done.set_value();
        } catch (...) {
            done.set_exception(std::current_exception());
        }
    }
};

LRESULT CALLBACK proc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == kRunMsg) {
        std::unique_ptr<std::shared_ptr<Task>> t(reinterpret_cast<std::shared_ptr<Task>*>(lp));
        if (!(*t)->claimed.exchange(true)) (*t)->execute();
        return 0;
    }
    return DefWindowProcW(h, msg, wp, lp);
}
#endif
}  // namespace

void init() {
#ifdef _WIN32
    g_thread = GetCurrentThreadId();
    WNDCLASSW wc{};
    wc.lpfnWndProc = proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"NexToneUiThread";
    RegisterClassW(&wc);
    g_wnd = CreateWindowExW(0, wc.lpszClassName, L"", 0, 0, 0, 0, 0, HWND_MESSAGE, nullptr, wc.hInstance, nullptr);
    g_running = g_wnd != nullptr;
#endif
}

void stop() {
    g_running = false;
#ifdef _WIN32
    MSG m;
    while (g_wnd && PeekMessageW(&m, g_wnd, kRunMsg, kRunMsg, PM_REMOVE)) DispatchMessageW(&m);
#endif
}

bool isMainThread() {
#ifdef _WIN32
    return GetCurrentThreadId() == g_thread;
#else
    return true;
#endif
}

void run(const std::function<void()>& fn) {
#ifdef _WIN32
    if (!g_running || isMainThread()) {
        fn();
        return;
    }
    auto t = std::make_shared<Task>();
    t->fn = fn;
    auto fut = t->done.get_future();
    auto* msg = new std::shared_ptr<Task>(t);
    if (!PostMessageW(g_wnd, kRunMsg, 0, reinterpret_cast<LPARAM>(msg))) {
        delete msg;
        fn();
        return;
    }
    while (fut.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
        if (!g_running && !t->claimed.exchange(true)) {
            t->execute();
            break;
        }
    }
    fut.get();
#else
    fn();
#endif
}

}  // namespace nt::ui
