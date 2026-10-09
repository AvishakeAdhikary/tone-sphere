#pragma once
// Minimal thread-safe logger: writes to a rolling file and keeps recent warnings for the UI.
#include <cstdarg>
#include <functional>
#include <string>

namespace nt::log {

enum class Level { Debug, Info, Warn, Error };

void init(const std::string& filePath, bool alsoStderr);
void write(Level level, const char* fmt, ...);
/// Called for Warn/Error messages so the IPC layer can forward them to the UI.
void setSink(std::function<void(Level, const std::string&)> sink);

}  // namespace nt::log

#define NT_DEBUG(...) ::nt::log::write(::nt::log::Level::Debug, __VA_ARGS__)
#define NT_INFO(...) ::nt::log::write(::nt::log::Level::Info, __VA_ARGS__)
#define NT_WARN(...) ::nt::log::write(::nt::log::Level::Warn, __VA_ARGS__)
#define NT_ERROR(...) ::nt::log::write(::nt::log::Level::Error, __VA_ARGS__)
