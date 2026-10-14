#include "util/log.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <mutex>

namespace nt::log {

namespace {
std::mutex g_mutex;
FILE* g_file = nullptr;
bool g_stderr = true;
std::function<void(Level, const std::string&)> g_sink;
size_t g_written = 0;
std::string g_path;

const char* name(Level l) {
    switch (l) {
        case Level::Debug: return "DEBUG";
        case Level::Info: return "INFO ";
        case Level::Warn: return "WARN ";
        case Level::Error: return "ERROR";
    }
    return "?";
}
}  // namespace

void init(const std::string& filePath, bool alsoStderr) {
    std::lock_guard lock(g_mutex);
    g_stderr = alsoStderr;
    g_path = filePath;
    if (!filePath.empty()) {
        // Keep one previous log for diagnostics.
        std::string old = filePath + ".old";
        std::remove(old.c_str());
        std::rename(filePath.c_str(), old.c_str());
        g_file = std::fopen(filePath.c_str(), "w");
    }
}

void setSink(std::function<void(Level, const std::string&)> sink) {
    std::lock_guard lock(g_mutex);
    g_sink = std::move(sink);
}

void write(Level level, const char* fmt, ...) {
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);

    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char line[2200];
    std::snprintf(line, sizeof(line), "%02d:%02d:%02d.%03d %s %s\n", tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(ms),
                  name(level), msg);

    std::function<void(Level, const std::string&)> sink;
    {
        std::lock_guard lock(g_mutex);
        if (g_file) {
            g_written += std::fputs(line, g_file) >= 0 ? std::char_traits<char>::length(line) : 0;
            std::fflush(g_file);
            if (g_written > 8u * 1024 * 1024) {  // cap log size
                std::fclose(g_file);
                g_file = std::fopen(g_path.c_str(), "w");
                g_written = 0;
            }
        }
        if (g_stderr) std::fputs(line, stderr);
        if (level >= Level::Warn) sink = g_sink;
    }
    if (sink) sink(level, msg);
}

}  // namespace nt::log
