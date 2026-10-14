#pragma once
// Minimal thread-safe logger: writes to a rolling file and keeps recent warnings for the UI.
#include <cstdarg>
#include <functional>
#include <string>

namespace ts::log {

enum class Level { Debug, Info, Warn, Error };

void init(const std::string& filePath, bool alsoStderr);
void write(Level level, const char* fmt, ...);
/// Called for Warn/Error messages so the IPC layer can forward them to the UI.
void setSink(std::function<void(Level, const std::string&)> sink);

}  // namespace ts::log

#define TS_DEBUG(...) ::ts::log::write(::ts::log::Level::Debug, __VA_ARGS__)
#define TS_INFO(...) ::ts::log::write(::ts::log::Level::Info, __VA_ARGS__)
#define TS_WARN(...) ::ts::log::write(::ts::log::Level::Warn, __VA_ARGS__)
#define TS_ERROR(...) ::ts::log::write(::ts::log::Level::Error, __VA_ARGS__)
