#pragma once
// System-wide keyboard shortcuts (work while other apps/games have focus).
// Actions: mute-mic, push-to-talk (hold), mute-all, save-replay, toggle-record,
// next-scene, and per-scene shortcuts ("scene:<id>").
#include <memory>
#include <string>

namespace ts {

class Engine;

class Hotkeys {
public:
    explicit Hotkeys(Engine& engine);
    ~Hotkeys();
    /// Installs the keyboard hook; call on a thread that runs a message loop.
    void install();
    /// Parses "Ctrl+Shift+M" style accelerators (exposed for tests).
    static bool parse(const std::string& accel, unsigned& vk, bool& ctrl, bool& alt, bool& shift, bool& win);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ts
