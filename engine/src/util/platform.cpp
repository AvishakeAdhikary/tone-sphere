#include "util/platform.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#include <avrt.h>
#include <shlobj.h>
#pragma comment(lib, "avrt.lib")
#else
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace nt {

double nowSeconds() {
#ifdef _WIN32
    static const double inv = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return 1.0 / static_cast<double>(f.QuadPart);
    }();
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return static_cast<double>(c.QuadPart) * inv;
#else
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
#endif
}

#ifdef _WIN32
std::string toUtf8(const wchar_t* w) {
    if (!w) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return {};
    std::string s(static_cast<size_t>(n - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}
std::string toUtf8(const std::wstring& w) { return toUtf8(w.c_str()); }
std::wstring toWide(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(static_cast<size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    return w;
}

static std::string knownFolder(REFKNOWNFOLDERID id) {
    PWSTR p = nullptr;
    std::string out;
    if (SUCCEEDED(SHGetKnownFolderPath(id, KF_FLAG_CREATE, nullptr, &p))) out = toUtf8(p);
    CoTaskMemFree(p);
    return out;
}
#endif

std::string dataDir() {
#ifdef _WIN32
    std::string base = knownFolder(FOLDERID_LocalAppData);
    std::string d = base + "\\NexTone";
#elif defined(__APPLE__)
    std::string d = std::string(getenv("HOME")) + "/Library/Application Support/NexTone";
#else
    const char* xdg = getenv("XDG_DATA_HOME");
    std::string d = xdg ? std::string(xdg) + "/nextone" : std::string(getenv("HOME")) + "/.local/share/nextone";
#endif
    makeDirs(d);
    return d;
}

std::string discoveryPath() {
    if (const char* env = getenv("NEXTONE_DISCOVERY")) return env;
#if defined(_WIN32) || defined(__APPLE__)
    return dataDir() + (
#ifdef _WIN32
                           "\\engine.json"
#else
                           "/engine.json"
#endif
                       );
#else
    const char* rt = getenv("XDG_RUNTIME_DIR");
    std::string d = rt ? std::string(rt) + "/nextone" : std::string(getenv("HOME")) + "/.local/state/nextone";
    makeDirs(d);
    return d + "/engine.json";
#endif
}

std::string defaultRecordFolder() {
#ifdef _WIN32
    std::string m = knownFolder(FOLDERID_Music);
    return (m.empty() ? dataDir() : m) + "\\NexTone";
#else
    return std::string(getenv("HOME")) + "/Music/NexTone";
#endif
}

std::string randomHex(size_t bytes) {
    static thread_local std::mt19937_64 rng{std::random_device{}() ^
                                            static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count())};
    static const char* hex = "0123456789abcdef";
    std::string s;
    s.reserve(bytes * 2);
    for (size_t i = 0; i < bytes; ++i) {
        const auto v = static_cast<unsigned>(rng() & 0xff);
        s.push_back(hex[v >> 4]);
        s.push_back(hex[v & 15]);
    }
    return s;
}

std::string newId(const char* prefix) { return std::string(prefix) + "_" + randomHex(4); }

bool makeDirs(const std::string& path) {
    std::error_code ec;
    fs::create_directories(fs::u8path(path), ec);
    return !ec;
}

bool writeFileAtomic(const std::string& path, const std::string& content) {
    const fs::path p = fs::u8path(path);
    const fs::path tmp = fs::u8path(path + ".tmp");
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write(content.data(), static_cast<std::streamsize>(content.size()));
        if (!f) return false;
    }
    std::error_code ec;
    fs::rename(tmp, p, ec);
    if (ec) {
        fs::remove(p, ec);
        fs::rename(tmp, p, ec);
    }
    return !ec;
}

bool readFile(const std::string& path, std::string& out) {
    std::ifstream f(fs::u8path(path), std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

void setAudioThreadPriority(const char* name) {
    setThreadName(name);
#ifdef _WIN32
    DWORD idx = 0;
    if (!AvSetMmThreadCharacteristicsW(L"Pro Audio", &idx)) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
#else
    sched_param sp{};
    sp.sched_priority = 70;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0) {
        // Not permitted (no rtkit / limits): fall back to a high nice value silently.
    }
#endif
}

void setThreadName(const char* name) {
#ifdef _WIN32
    const std::wstring w = toWide(name);
    SetThreadDescription(GetCurrentThread(), w.c_str());
#elif defined(__linux__)
    pthread_setname_np(pthread_self(), std::string(name).substr(0, 15).c_str());
#endif
}

}  // namespace nt
