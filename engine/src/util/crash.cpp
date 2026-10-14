#include "util/crash.h"

#include <cstdio>
#include <cstring>
#include <ctime>

#include "util/platform.h"

#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#else
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#endif

namespace ts {

namespace {
// Filled in at install time: a crashing process must not allocate.
char g_dir[1024] = {};
char g_version[64] = {};

#ifdef _WIN32
wchar_t g_wdir[1024] = {};

LONG WINAPI onCrash(EXCEPTION_POINTERS* ep) {
    static volatile LONG once = 0;
    if (InterlockedExchange(&once, 1)) return EXCEPTION_CONTINUE_SEARCH;
    SYSTEMTIME t;
    GetLocalTime(&t);
    wchar_t stamp[64];
    swprintf(stamp, 64, L"crash-%04u%02u%02u-%02u%02u%02u", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
    wchar_t path[1200];

    // Readable summary first (it is what the app shows), then the minidump.
    const DWORD code = ep && ep->ExceptionRecord ? ep->ExceptionRecord->ExceptionCode : 0;
    void* addr = ep && ep->ExceptionRecord ? ep->ExceptionRecord->ExceptionAddress : nullptr;
    HMODULE mod = nullptr;
    wchar_t modName[MAX_PATH] = L"?";
    if (addr && GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, static_cast<LPCWSTR>(addr), &mod))
        GetModuleFileNameW(mod, modName, MAX_PATH);
    swprintf(path, 1200, L"%s\\%s.txt", g_wdir, stamp);
    HANDLE f = CreateFileW(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f != INVALID_HANDLE_VALUE) {
        char text[2048];
        const int n = std::snprintf(text, sizeof(text),
                                    "ToneSphere engine %s stopped unexpectedly.\r\n"
                                    "Fault: 0x%08lX at %p\r\nModule: %ls (offset 0x%llX)\r\nThread: %lu\r\n"
                                    "A minidump with the same name was written next to this file.\r\n",
                                    g_version, code, addr, modName,
                                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(addr) - reinterpret_cast<uintptr_t>(mod)),
                                    GetCurrentThreadId());
        DWORD w = 0;
        WriteFile(f, text, static_cast<DWORD>(n > 0 ? n : 0), &w, nullptr);
        CloseHandle(f);
    }
    swprintf(path, 1200, L"%s\\%s.dmp", g_wdir, stamp);
    HANDLE d = CreateFileW(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (d != INVALID_HANDLE_VALUE) {
        MINIDUMP_EXCEPTION_INFORMATION mei{GetCurrentThreadId(), ep, FALSE};
        MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), d,
                          static_cast<MINIDUMP_TYPE>(MiniDumpWithThreadInfo | MiniDumpWithIndirectlyReferencedMemory), ep ? &mei : nullptr, nullptr,
                          nullptr);
        CloseHandle(d);
    }
    return EXCEPTION_EXECUTE_HANDLER;  // end the process (the app restarts the engine)
}

void onPurecall() {
    RaiseException(0xE0000001, EXCEPTION_NONCONTINUABLE, 0, nullptr);
}
void onInvalidParameter(const wchar_t*, const wchar_t*, const wchar_t*, unsigned, uintptr_t) {
    RaiseException(0xE0000002, EXCEPTION_NONCONTINUABLE, 0, nullptr);
}
#else
void onSignal(int sig) {
    char path[1200];
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    localtime_r(&now, &tm);
    std::snprintf(path, sizeof(path), "%s/crash-%04d%02d%02d-%02d%02d%02d.txt", g_dir, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour,
                  tm.tm_min, tm.tm_sec);
    const int fd = ::open(path, O_CREAT | O_WRONLY | O_TRUNC, 0600);
    if (fd >= 0) {
        char text[512];
        const int n = std::snprintf(text, sizeof(text), "ToneSphere engine %s stopped unexpectedly.\nSignal: %d\n", g_version, sig);
        if (n > 0) (void)!::write(fd, text, static_cast<size_t>(n));
        ::close(fd);
    }
    ::signal(sig, SIG_DFL);
    ::raise(sig);
}
#endif
}  // namespace

void installCrashHandler(const std::string& dir, const char* version) {
    makeDirs(dir);
    std::snprintf(g_dir, sizeof(g_dir), "%s", dir.c_str());
    std::snprintf(g_version, sizeof(g_version), "%s", version);
#ifdef _WIN32
    const std::wstring w = toWide(dir);
    wcsncpy_s(g_wdir, w.c_str(), _TRUNCATE);
    LoadLibraryW(L"dbghelp.dll");  // loaded now, not while crashing
    SetUnhandledExceptionFilter(onCrash);
    _set_purecall_handler(onPurecall);
    _set_invalid_parameter_handler(onInvalidParameter);
#else
    for (int sig : {SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT}) ::signal(sig, onSignal);
#endif
}

}  // namespace ts
