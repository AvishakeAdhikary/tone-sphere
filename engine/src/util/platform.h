#pragma once
// Small cross-platform helpers: paths, time, ids, string conversion.
#include <cstdint>
#include <string>

namespace nt {

/// Monotonic seconds (QueryPerformanceCounter / CLOCK_MONOTONIC).
double nowSeconds();

/// Per-user data directory (created on demand): %LOCALAPPDATA%\NexTone, ~/.local/share/nextone, ...
std::string dataDir();
/// Discovery file path (see shared/protocol/PROTOCOL.md).
std::string discoveryPath();
/// Default recordings folder (Music\NexTone).
std::string defaultRecordFolder();

/// Random hex string with `bytes` bytes of entropy.
std::string randomHex(size_t bytes);
/// Short unique id, e.g. "r_1a2b3c4d".
std::string newId(const char* prefix);

bool makeDirs(const std::string& path);
bool writeFileAtomic(const std::string& path, const std::string& content);
bool readFile(const std::string& path, std::string& out);

#ifdef _WIN32
std::string toUtf8(const wchar_t* w);
std::string toUtf8(const std::wstring& w);
std::wstring toWide(const std::string& s);
#endif

/// Raises the calling thread to real-time audio priority (MMCSS "Pro Audio" / SCHED_FIFO when allowed).
void setAudioThreadPriority(const char* name);
/// Names the current thread for debuggers.
void setThreadName(const char* name);

}  // namespace nt
