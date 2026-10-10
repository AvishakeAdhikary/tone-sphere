// nextone-setup: installs/removes the NexTone Driver Pack components (run elevated).
//   nextone-setup status
//   nextone-setup install-asio  <dir> [--engine <path>] [--ui <path>]
//   nextone-setup uninstall-asio <dir>
//   nextone-setup install-vad   <dir>      (dir contains nextonevad.inf/.sys/.cat [+ NexToneTest.cer])
//   nextone-setup uninstall-vad
// Prints one JSON line with the result; exit code 0 on success.
#include <windows.h>

#include <cfgmgr32.h>
#include <newdev.h>
#include <setupapi.h>
#include <wincrypt.h>
#include <winternl.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "newdev.lib")
#pragma comment(lib, "crypt32.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "ntdll.lib")

namespace {

const wchar_t* kHwid = L"ROOT\\NexToneVad";

std::string utf8(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(n - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), n, nullptr, nullptr);
    return s;
}

std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}

int result(bool ok, const std::string& message, const std::string& extra = "") {
    std::printf("{\"ok\":%s,\"message\":\"%s\"%s%s}\n", ok ? "true" : "false", esc(message).c_str(), extra.empty() ? "" : ",", extra.c_str());
    return ok ? 0 : 1;
}

std::string lastError(const char* what) {
    const DWORD e = GetLastError();
    wchar_t buf[512] = {};
    FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, e, 0, buf, 512, nullptr);
    char out[700];
    std::snprintf(out, sizeof(out), "%s failed (0x%08lx): %s", what, e, utf8(buf).c_str());
    return out;
}

bool isAdmin() {
    BOOL admin = FALSE;
    PSID group = nullptr;
    SID_IDENTIFIER_AUTHORITY nt = SECURITY_NT_AUTHORITY;
    if (AllocateAndInitializeSid(&nt, 2, SECURITY_BUILTIN_DOMAIN_RID, DOMAIN_ALIAS_RID_ADMINS, 0, 0, 0, 0, 0, 0, &group)) {
        CheckTokenMembership(nullptr, group, &admin);
        FreeSid(group);
    }
    return admin != FALSE;
}

bool testSigningEnabled() {
    struct {
        ULONG Length;
        ULONG CodeIntegrityOptions;
    } ci{sizeof(ci), 0};
    using Fn = NTSTATUS(NTAPI*)(ULONG, PVOID, ULONG, PULONG);
    auto q = reinterpret_cast<Fn>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQuerySystemInformation"));
    if (!q) return false;
    if (q(103 /* SystemCodeIntegrityInformation */, &ci, sizeof(ci), nullptr) != 0) return false;
    return (ci.CodeIntegrityOptions & 0x02) != 0;  // CODEINTEGRITY_OPTION_TESTSIGN
}

// Calls fn(devInfoSet, devInfoData) for every device whose hardware IDs contain kHwid.
template <class F>
int forEachVadDevice(F fn) {
    HDEVINFO set = SetupDiGetClassDevsW(nullptr, nullptr, nullptr, DIGCF_ALLCLASSES);
    if (set == INVALID_HANDLE_VALUE) return 0;
    int count = 0;
    SP_DEVINFO_DATA d{sizeof(d)};
    for (DWORD i = 0; SetupDiEnumDeviceInfo(set, i, &d); ++i) {
        wchar_t ids[1024] = {};
        if (!SetupDiGetDeviceRegistryPropertyW(set, &d, SPDRP_HARDWAREID, nullptr, reinterpret_cast<PBYTE>(ids), sizeof(ids) - 4, nullptr))
            continue;
        for (const wchar_t* p = ids; *p; p += wcslen(p) + 1) {
            if (_wcsicmp(p, kHwid) == 0) {
                ++count;
                fn(set, d);
                break;
            }
        }
    }
    SetupDiDestroyDeviceInfoList(set);
    return count;
}

bool asioRegistered() {
    HKEY k;
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\ASIO\\NexTone ASIO", 0, KEY_READ | KEY_WOW64_64KEY, &k) != ERROR_SUCCESS) return false;
    RegCloseKey(k);
    return true;
}

bool importCert(const std::wstring& path, const wchar_t* store) {
    HANDLE f = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (f == INVALID_HANDLE_VALUE) return false;
    std::vector<BYTE> data(GetFileSize(f, nullptr));
    DWORD read = 0;
    ReadFile(f, data.data(), static_cast<DWORD>(data.size()), &read, nullptr);
    CloseHandle(f);
    HCERTSTORE s = CertOpenStore(CERT_STORE_PROV_SYSTEM_W, 0, 0, CERT_SYSTEM_STORE_LOCAL_MACHINE, store);
    if (!s) return false;
    const BOOL ok = CertAddEncodedCertificateToStore(s, X509_ASN_ENCODING | PKCS_7_ASN_ENCODING, data.data(), read, CERT_STORE_ADD_REPLACE_EXISTING, nullptr);
    CertCloseStore(s, 0);
    return ok != FALSE;
}

bool registerDll(const std::wstring& path, bool unregister) {
    // 32-bit DLLs must be registered by the 32-bit regsvr32.
    HMODULE m = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (m) {
        using Fn = HRESULT(__stdcall*)();
        auto fn = reinterpret_cast<Fn>(GetProcAddress(m, unregister ? "DllUnregisterServer" : "DllRegisterServer"));
        const bool ok = fn && SUCCEEDED(fn());
        FreeLibrary(m);
        return ok;
    }
    wchar_t sysWow[MAX_PATH];
    GetSystemWow64DirectoryW(sysWow, MAX_PATH);
    std::wstring cmd = L"\"" + std::wstring(sysWow) + L"\\regsvr32.exe\" /s " + (unregister ? L"/u " : L"") + L"\"" + path + L"\"";
    STARTUPINFOW si{sizeof(si)};
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) return false;
    WaitForSingleObject(pi.hProcess, 30000);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return code == 0;
}

void setPath(const wchar_t* name, const std::wstring& value) {
    HKEY k;
    if (RegCreateKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\NexTone", 0, nullptr, 0, KEY_WRITE | KEY_WOW64_64KEY, nullptr, &k, nullptr) != ERROR_SUCCESS)
        return;
    RegSetValueExW(k, name, 0, REG_SZ, reinterpret_cast<const BYTE*>(value.c_str()), static_cast<DWORD>((value.size() + 1) * 2));
    RegCloseKey(k);
}

int installAsio(const std::wstring& dir, const std::wstring& engine, const std::wstring& ui) {
    const bool a = registerDll(dir + L"\\NexToneASIO64.dll", false);
    const bool b = registerDll(dir + L"\\NexToneASIO32.dll", false);
    if (!engine.empty()) setPath(L"EnginePath", engine);
    if (!ui.empty()) setPath(L"UiPath", ui);
    return result(a, a ? (b ? "ASIO driver registered (64- and 32-bit)" : "ASIO driver registered (64-bit)") : "Could not register the ASIO driver");
}

int uninstallAsio(const std::wstring& dir) {
    registerDll(dir + L"\\NexToneASIO64.dll", true);
    registerDll(dir + L"\\NexToneASIO32.dll", true);
    RegDeleteKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\NexTone", KEY_WOW64_64KEY, 0);
    return result(true, "ASIO driver removed");
}

int installVad(const std::wstring& dir) {
    const std::wstring inf = dir + L"\\nextonevad.inf";
    if (GetFileAttributesW(inf.c_str()) == INVALID_FILE_ATTRIBUTES) return result(false, "nextonevad.inf not found");
    const std::wstring cer = dir + L"\\NexToneTest.cer";
    const bool testCert = GetFileAttributesW(cer.c_str()) != INVALID_FILE_ATTRIBUTES;
    if (testCert) {
        if (!testSigningEnabled())
            return result(false, "This build of the virtual audio driver is test-signed and needs Windows test-signing mode.",
                          "\"needsTestSigning\":true");
        importCert(cer, L"Root");
        importCert(cer, L"TrustedPublisher");
    }
    if (forEachVadDevice([](HDEVINFO, SP_DEVINFO_DATA&) {}) == 0) {
        GUID cls;
        wchar_t className[MAX_CLASS_NAME_LEN];
        if (!SetupDiGetINFClassW(inf.c_str(), &cls, className, MAX_CLASS_NAME_LEN, nullptr)) return result(false, lastError("Reading the INF"));
        HDEVINFO set = SetupDiCreateDeviceInfoList(&cls, nullptr);
        SP_DEVINFO_DATA d{sizeof(d)};
        if (!SetupDiCreateDeviceInfoW(set, className, &cls, nullptr, nullptr, DICD_GENERATE_ID, &d)) {
            SetupDiDestroyDeviceInfoList(set);
            return result(false, lastError("Creating the device"));
        }
        const wchar_t hw[] = L"ROOT\\NexToneVad\0";
        SetupDiSetDeviceRegistryPropertyW(set, &d, SPDRP_HARDWAREID, reinterpret_cast<const BYTE*>(hw), sizeof(hw));
        if (!SetupDiCallClassInstaller(DIF_REGISTERDEVICE, set, &d)) {
            SetupDiDestroyDeviceInfoList(set);
            return result(false, lastError("Registering the device"));
        }
        SetupDiDestroyDeviceInfoList(set);
    }
    BOOL reboot = FALSE;
    if (!UpdateDriverForPlugAndPlayDevicesW(nullptr, kHwid, inf.c_str(), INSTALLFLAG_FORCE, &reboot)) return result(false, lastError("Installing the driver"));
    return result(true, reboot ? "Virtual audio driver installed (restart required)" : "Virtual audio driver installed",
                  reboot ? "\"reboot\":true" : "\"reboot\":false");
}

int uninstallVad() {
    BOOL reboot = FALSE;
    const int n = forEachVadDevice([&](HDEVINFO set, SP_DEVINFO_DATA& d) {
        BOOL r = FALSE;
        DiUninstallDevice(nullptr, set, &d, 0, &r);
        reboot |= r;
    });
    // Remove the package from the driver store: find the published oemNN.inf that is ours.
    wchar_t win[MAX_PATH];
    GetWindowsDirectoryW(win, MAX_PATH);
    const std::wstring infDir = std::wstring(win) + L"\\INF\\";
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW((infDir + L"oem*.inf").c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            HANDLE f = CreateFileW((infDir + fd.cFileName).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
            if (f == INVALID_HANDLE_VALUE) continue;
            std::vector<char> data(std::min<DWORD>(fd.nFileSizeLow, 1 << 20));
            DWORD read = 0;
            ReadFile(f, data.data(), static_cast<DWORD>(data.size()), &read, nullptr);
            CloseHandle(f);
            // INF files are UTF-16 or ANSI: search for the service binary name in both encodings.
            const std::string a(data.data(), read);
            std::wstring w(reinterpret_cast<const wchar_t*>(data.data()), read / 2);
            if (a.find("nextonevad.sys") != std::string::npos || w.find(L"nextonevad.sys") != std::wstring::npos)
                SetupUninstallOEMInfW(fd.cFileName, SUOI_FORCEDELETE, nullptr);
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }
    return result(true, n ? "Virtual audio driver removed" : "Virtual audio driver was not installed", reboot ? "\"reboot\":true" : "\"reboot\":false");
}

int status() {
    const int devices = forEachVadDevice([](HDEVINFO, SP_DEVINFO_DATA&) {});
    char extra[256];
    std::snprintf(extra, sizeof(extra), "\"admin\":%s,\"testSigning\":%s,\"vadDevice\":%s,\"asio\":%s", isAdmin() ? "true" : "false",
                  testSigningEnabled() ? "true" : "false", devices ? "true" : "false", asioRegistered() ? "true" : "false");
    return result(true, "status", extra);
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    const std::wstring cmd = argc > 1 ? argv[1] : L"status";
    std::wstring dir = argc > 2 ? argv[2] : L"";
    std::wstring engine, ui;
    for (int i = 3; i + 1 < argc; ++i) {
        if (!wcscmp(argv[i], L"--engine")) engine = argv[++i];
        else if (!wcscmp(argv[i], L"--ui")) ui = argv[++i];
    }
    if (cmd == L"status") return status();
    if (!isAdmin()) return result(false, "Administrator rights are required");
    if (cmd == L"install-asio") return installAsio(dir, engine, ui);
    if (cmd == L"uninstall-asio") return uninstallAsio(dir);
    if (cmd == L"install-vad") return installVad(dir);
    if (cmd == L"uninstall-vad") return uninstallVad();
    return result(false, "unknown command");
}
