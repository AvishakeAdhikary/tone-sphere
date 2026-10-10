#include "backends/win/hotkeys.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "core/engine.h"
#include "util/log.h"
#include "util/platform.h"

namespace nt {

namespace {

struct Binding {
    unsigned vk = 0;
    bool ctrl = false, alt = false, shift = false, win = false;
    std::string action;
};

std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

}  // namespace

bool Hotkeys::parse(const std::string& accel, unsigned& vk, bool& ctrl, bool& alt, bool& shift, bool& win) {
    vk = 0;
    ctrl = alt = shift = win = false;
    size_t start = 0;
    while (start <= accel.size()) {
        size_t plus = accel.find('+', start);
        // A trailing "+" key (e.g. "Ctrl++") is the plus key itself.
        if (plus == start && plus + 1 == accel.size()) plus = std::string::npos;
        std::string tok = accel.substr(start, plus == std::string::npos ? std::string::npos : plus - start);
        const std::string u = upper(tok);
        if (u == "CTRL" || u == "CONTROL") ctrl = true;
        else if (u == "ALT") alt = true;
        else if (u == "SHIFT") shift = true;
        else if (u == "WIN" || u == "META" || u == "SUPER") win = true;
        else if (!u.empty()) {
            if (u.size() == 1 && ((u[0] >= 'A' && u[0] <= 'Z') || (u[0] >= '0' && u[0] <= '9'))) vk = static_cast<unsigned>(u[0]);
            else if (u.size() >= 2 && u[0] == 'F' && std::isdigit(static_cast<unsigned char>(u[1]))) {
                const int n = std::atoi(u.c_str() + 1);
                if (n >= 1 && n <= 24) vk = VK_F1 + static_cast<unsigned>(n - 1);
            } else if (u == "SPACE" || u == " ") vk = VK_SPACE;
            else if (u == "ENTER" || u == "RETURN") vk = VK_RETURN;
            else if (u == "TAB") vk = VK_TAB;
            else if (u == "ESC" || u == "ESCAPE") vk = VK_ESCAPE;
            else if (u == "ARROWUP" || u == "UP") vk = VK_UP;
            else if (u == "ARROWDOWN" || u == "DOWN") vk = VK_DOWN;
            else if (u == "ARROWLEFT" || u == "LEFT") vk = VK_LEFT;
            else if (u == "ARROWRIGHT" || u == "RIGHT") vk = VK_RIGHT;
            else if (u == "HOME") vk = VK_HOME;
            else if (u == "END") vk = VK_END;
            else if (u == "PAGEUP") vk = VK_PRIOR;
            else if (u == "PAGEDOWN") vk = VK_NEXT;
            else if (u == "INSERT") vk = VK_INSERT;
            else if (u == "DELETE") vk = VK_DELETE;
            else if (u == "PAUSE") vk = VK_PAUSE;
            else if (u == "SCROLLLOCK") vk = VK_SCROLL;
            else if (u == "CAPSLOCK") vk = VK_CAPITAL;
            else if (u == "`" || u == "BACKQUOTE") vk = VK_OEM_3;
            else if (u == "-") vk = VK_OEM_MINUS;
            else if (u == "=" || u == "+") vk = VK_OEM_PLUS;
            else if (u == "[") vk = VK_OEM_4;
            else if (u == "]") vk = VK_OEM_6;
            else if (u == ";") vk = VK_OEM_1;
            else if (u == "'") vk = VK_OEM_7;
            else if (u == ",") vk = VK_OEM_COMMA;
            else if (u == ".") vk = VK_OEM_PERIOD;
            else if (u == "/") vk = VK_OEM_2;
            else if (u == "\\") vk = VK_OEM_5;
            else if (u == "MEDIAPLAYPAUSE") vk = VK_MEDIA_PLAY_PAUSE;
            else return false;
        }
        if (plus == std::string::npos) break;
        start = plus + 1;
    }
    return vk != 0;
}

struct Hotkeys::Impl {
    Engine& engine;
    HHOOK hook = nullptr;
    std::mutex mu;
    std::vector<Binding> bindings;
    std::set<unsigned> down;  // keys currently held (to ignore auto-repeat)
    std::deque<std::string> queue;
    std::condition_variable cv;
    std::thread worker;
    std::atomic<bool> running{true};
    std::string signature;

    explicit Impl(Engine& e) : engine(e) {
        worker = std::thread([this] { work(); });
    }
    ~Impl() {
        running = false;
        cv.notify_all();
        if (worker.joinable()) worker.join();
        if (hook) UnhookWindowsHookEx(hook);
    }

    static Impl*& instance() {
        static Impl* p = nullptr;
        return p;
    }

    static LRESULT CALLBACK proc(int code, WPARAM wp, LPARAM lp) {
        Impl* self = instance();
        if (code == HC_ACTION && self) {
            const auto* k = reinterpret_cast<KBDLLHOOKSTRUCT*>(lp);
            const bool isDown = wp == WM_KEYDOWN || wp == WM_SYSKEYDOWN;
            const bool isUp = wp == WM_KEYUP || wp == WM_SYSKEYUP;
            if (isDown || isUp) self->onKey(k->vkCode, isDown);
        }
        return CallNextHookEx(nullptr, code, wp, lp);
    }

    void onKey(unsigned vk, bool isDown) {
        std::lock_guard lk(mu);
        if (isDown) {
            if (down.count(vk)) return;  // auto-repeat
            down.insert(vk);
            const bool ctrl = GetAsyncKeyState(VK_CONTROL) & 0x8000, alt = GetAsyncKeyState(VK_MENU) & 0x8000,
                       shift = GetAsyncKeyState(VK_SHIFT) & 0x8000,
                       win = (GetAsyncKeyState(VK_LWIN) | GetAsyncKeyState(VK_RWIN)) & 0x8000;
            for (const Binding& b : bindings) {
                if (b.vk == vk && b.ctrl == ctrl && b.alt == alt && b.shift == shift && b.win == win) {
                    queue.push_back(b.action == "push-to-talk" ? "ptt-down" : b.action);
                    cv.notify_one();
                }
            }
        } else {
            down.erase(vk);
            for (const Binding& b : bindings) {
                if (b.vk == vk && b.action == "push-to-talk") {
                    queue.push_back("ptt-up");
                    cv.notify_one();
                }
            }
        }
    }

    void reload() {
        json st;
        try {
            st = engine.state();
        } catch (...) {
            return;
        }
        std::vector<Binding> next;
        std::string sig;
        auto add = [&](const std::string& accel, const std::string& action) {
            Binding b;
            if (accel.empty() || !Hotkeys::parse(accel, b.vk, b.ctrl, b.alt, b.shift, b.win)) return;
            b.action = action;
            next.push_back(b);
            sig += accel + "=" + action + ";";
        };
        for (auto& [action, accel] : st["settings"]["hotkeys"].items())
            if (accel.is_string()) add(accel.get<std::string>(), action);
        for (auto& sc : st["scenes"])
            if (sc.contains("hotkey")) add(sc["hotkey"].get<std::string>(), "scene:" + sc["id"].get<std::string>());
        std::lock_guard lk(mu);
        if (sig != signature) {
            signature = sig;
            bindings = std::move(next);
            NT_INFO("Shortcuts: %zu active", bindings.size());
        }
    }

    void setMuted(const json& st, const char* bin, bool muted) {
        for (auto& n : st["nodes"])
            if (n["bin"] == bin && n["kind"] != "tone" && n["state"] != "unavailable")
                engine.call("node.update", {{"id", n["id"]}, {"muted", muted}});
    }

    void run(const std::string& a) {
        try {
            const json st = engine.state();
            if (a == "mute-mic" || a == "mute-all") {
                const char* bin = a == "mute-mic" ? "physical-in" : "physical-out";
                bool anyOpen = false;
                for (auto& n : st["nodes"])
                    if (n["bin"] == bin && n["kind"] != "tone" && !n["muted"].get<bool>()) anyOpen = true;
                setMuted(st, bin, anyOpen);
            } else if (a == "ptt-down") {
                setMuted(st, "physical-in", false);
            } else if (a == "ptt-up") {
                setMuted(st, "physical-in", true);
            } else if (a == "save-replay") {
                engine.call("recorder.saveReplay", json::object());
            } else if (a == "toggle-record") {
                if (st["recorder"]["recording"].get<bool>()) {
                    engine.call("recorder.stop", json::object());
                } else {
                    json ids = json::array();
                    for (auto& n : st["nodes"])
                        if (n["bin"] == "bus") ids.push_back(n["id"]);
                    if (ids.empty())
                        for (auto& n : st["nodes"])
                            if (n["bin"] == "physical-out" && n["state"] != "unavailable") ids.push_back(n["id"]);
                    engine.call("recorder.start", {{"nodeIds", ids}, {"format", "wav"}});
                }
            } else if (a == "next-scene") {
                const auto& scenes = st["scenes"];
                if (!scenes.empty()) {
                    size_t idx = 0;
                    const std::string active = st.value("activeScene", "");
                    for (size_t i = 0; i < scenes.size(); ++i)
                        if (scenes[i]["id"] == active) idx = (i + 1) % scenes.size();
                    engine.call("scene.load", {{"id", scenes[idx]["id"]}});
                }
            } else if (a.rfind("scene:", 0) == 0) {
                engine.call("scene.load", {{"id", a.substr(6)}});
            }
        } catch (const std::exception& e) {
            NT_WARN("Shortcut \"%s\": %s", a.c_str(), e.what());
        }
    }

    void work() {
        setThreadName("nt-hotkeys");
        double lastReload = 0;
        while (running) {
            std::string action;
            {
                std::unique_lock lk(mu);
                cv.wait_for(lk, std::chrono::milliseconds(500), [this] { return !queue.empty() || !running; });
                if (!running) break;
                if (!queue.empty()) {
                    action = queue.front();
                    queue.pop_front();
                }
            }
            if (!action.empty()) run(action);
            if (nowSeconds() - lastReload > 2.0) {
                lastReload = nowSeconds();
                reload();
            }
        }
    }
};

Hotkeys::Hotkeys(Engine& engine) : impl_(std::make_unique<Impl>(engine)) {}

Hotkeys::~Hotkeys() {
    if (Impl::instance() == impl_.get()) Impl::instance() = nullptr;
}

void Hotkeys::install() {
    Impl::instance() = impl_.get();
    impl_->hook = SetWindowsHookExW(WH_KEYBOARD_LL, &Impl::proc, GetModuleHandleW(nullptr), 0);
    if (!impl_->hook) NT_WARN("Keyboard shortcuts are unavailable (hook failed)");
}

}  // namespace nt
