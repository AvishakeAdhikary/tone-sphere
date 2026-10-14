#pragma once
// Runs work on the engine's main (UI) thread. Plugins must be created, edited and destroyed
// on a thread with a message loop, and their editor windows live there too.
// The main thread must never wait for the engine mutex, or run() could deadlock.
#include <functional>

namespace ts::ui {

/// Call once on the main thread before its message loop starts.
void init();
/// Call when the main message loop has stopped; later run() calls execute inline.
void stop();
/// Runs `fn` on the main thread and waits for it (inline when already on it or not running).
/// Exceptions thrown by `fn` are rethrown in the caller.
void run(const std::function<void()>& fn);
/// True on the main thread.
bool isMainThread();

}  // namespace ts::ui
