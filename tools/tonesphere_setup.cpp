// tonesphere-setup: installs/removes the ToneSphere drivers and VB-CABLE (run elevated).
//   tonesphere-setup status
//   tonesphere-setup install-asio  <dir> [--engine <path>] [--ui <path>]
//   tonesphere-setup uninstall-asio <dir>
//   tonesphere-setup install-vad   <dir>      (dir contains tonespherevad.inf/.sys/.cat [+ ToneSphereTest.cer])
//   tonesphere-setup uninstall-vad
//   tonesphere-setup install-vbcable <dir>    (dir holds the official VB-CABLE pack, unzipped)
//   tonesphere-setup install-all   <dir> [--engine <path>] [--ui <path>] [--vbcable <dir>]
//                                  (ASIO from <dir>; the virtual audio driver from <dir>\vad when bundled)
//   tonesphere-setup verify-signature <file>  (is it VB-Audio's signed VB-CABLE setup? no admin needed)
//   ... [--result <file>]                     (also append each result line to <file>)
// Prints one JSON line per step; exit code 0 on success.
#include <windows.h>

#include <cfgmgr32.h>
#include <newdev.h>
#include <setupapi.h>
#include <softpub.h>
#include <wincrypt.h>
#include <wintrust.h>
#include <winternl.h>

#include <algorithm>
#include <cwctype>
#include <cstdio>
#include <string>
#include <vector>

#pragma comment(lib, "setupapi.lib")
#pragma comment(lib, "newdev.lib")
#pragma comment(lib, "crypt32.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "ntdll.lib")
#pragma comment(lib, "wintrust.lib")

namespace {

const wchar_t* kHwid = L"ROOT\\ToneSphereVad";
const wchar_t* kVbCableHwid = L"VBAudioVACWDM";  // from VB-CABLE's vbMmeCable64_win10.inf
FILE* resultLog = nullptr;                         // --result <file>

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
    char line[2048];
    std::snprintf(line, sizeof(line), "{\"ok\":%s,\"message\":\"%s\"%s%s}\n", ok ? "true" : "false", esc(message).c_str(), extra.empty() ? "" : ",",
                  extra.c_str());
    std::fputs(line, stdout);
    if (resultLog) {
        std::fputs(line, resultLog);
        std::fflush(resultLog);
    }
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

// Calls fn(devInfoSet, devInfoData) for every device whose hardware IDs contain hwid.
template <class F>
int forEachDevice(const wchar_t* hwid, F fn) {
    HDEVINFO set = SetupDiGetClassDevsW(nullptr, nullptr, nullptr, DIGCF_ALLCLASSES);
    if (set == INVALID_HANDLE_VALUE) return 0;
    int count = 0;
    SP_DEVINFO_DATA d{sizeof(d)};
    for (DWORD i = 0; SetupDiEnumDeviceInfo(set, i, &d); ++i) {
        wchar_t ids[1024] = {};
        if (!SetupDiGetDeviceRegistryPropertyW(set, &d, SPDRP_HARDWAREID, nullptr, reinterpret_cast<PBYTE>(ids), sizeof(ids) - 4, nullptr))
            continue;
        for (const wchar_t* p = ids; *p; p += wcslen(p) + 1) {
            if (_wcsicmp(p, hwid) == 0) {
                ++count;
                fn(set, d);
                break;
            }
        }
    }
    SetupDiDestroyDeviceInfoList(set);
    return count;
}

template <class F>
int forEachVadDevice(F fn) {
    return forEachDevice(kHwid, fn);
}

bool asioRegistered() {
    HKEY k;
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\ASIO\\ToneSphere ASIO", 0, KEY_READ | KEY_WOW64_64KEY, &k) != ERROR_SUCCESS) return false;
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
    if (RegCreateKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\ToneSphere", 0, nullptr, 0, KEY_WRITE | KEY_WOW64_64KEY, nullptr, &k, nullptr) != ERROR_SUCCESS)
        return;
    RegSetValueExW(k, name, 0, REG_SZ, reinterpret_cast<const BYTE*>(value.c_str()), static_cast<DWORD>((value.size() + 1) * 2));
    RegCloseKey(k);
}

int installAsio(const std::wstring& dir, const std::wstring& engine, const std::wstring& ui) {
    const bool a = registerDll(dir + L"\\ToneSphereASIO64.dll", false);
    const bool b = registerDll(dir + L"\\ToneSphereASIO32.dll", false);
    if (!engine.empty()) setPath(L"EnginePath", engine);
    if (!ui.empty()) setPath(L"UiPath", ui);
    return result(a, a ? (b ? "ASIO driver registered (64- and 32-bit)" : "ASIO driver registered (64-bit)") : "Could not register the ASIO driver");
}

int uninstallAsio(const std::wstring& dir) {
    registerDll(dir + L"\\ToneSphereASIO64.dll", true);
    registerDll(dir + L"\\ToneSphereASIO32.dll", true);
    RegDeleteKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\ToneSphere", KEY_WOW64_64KEY, 0);
    return result(true, "ASIO driver removed");
}

int installVad(const std::wstring& dir) {
    const std::wstring inf = dir + L"\\tonespherevad.inf";
    if (GetFileAttributesW(inf.c_str()) == INVALID_FILE_ATTRIBUTES) return result(false, "tonespherevad.inf not found");
    const std::wstring cer = dir + L"\\ToneSphereTest.cer";
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
        const wchar_t hw[] = L"ROOT\\ToneSphereVad\0";
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
            if (a.find("tonespherevad.sys") != std::string::npos || w.find(L"tonespherevad.sys") != std::wstring::npos)
                SetupUninstallOEMInfW(fd.cFileName, SUOI_FORCEDELETE, nullptr);
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }
    return result(true, n ? "Virtual audio driver removed" : "Virtual audio driver was not installed", reboot ? "\"reboot\":true" : "\"reboot\":false");
}

// ---- VB-CABLE (donationware by VB-Audio Software, https://vb-audio.com/Cable/)
// Its licence lets anyone pass the package on unmodified, but not build it into another
// installation procedure. So ToneSphere never bundles or silently installs it: the app downloads
// the official package, this checks it really is VB-Audio's signed setup, and then VB-Audio's own
// installer opens for the user.
std::wstring signerName(const std::wstring& file) {
    HCERTSTORE store = nullptr;
    HCRYPTMSG msg = nullptr;
    std::wstring name;
    if (!CryptQueryObject(CERT_QUERY_OBJECT_FILE, file.c_str(), CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED, CERT_QUERY_FORMAT_FLAG_BINARY, 0, nullptr,
                          nullptr, nullptr, &store, &msg, nullptr))
        return name;
    DWORD size = 0;
    CryptMsgGetParam(msg, CMSG_SIGNER_INFO_PARAM, 0, nullptr, &size);
    std::vector<BYTE> buf(size);
    if (size && CryptMsgGetParam(msg, CMSG_SIGNER_INFO_PARAM, 0, buf.data(), &size)) {
        auto* si = reinterpret_cast<CMSG_SIGNER_INFO*>(buf.data());
        CERT_INFO ci{};
        ci.Issuer = si->Issuer;
        ci.SerialNumber = si->SerialNumber;
        if (PCCERT_CONTEXT cert = CertFindCertificateInStore(store, X509_ASN_ENCODING | PKCS_7_ASN_ENCODING, 0, CERT_FIND_SUBJECT_CERT, &ci, nullptr)) {
            wchar_t n[512] = {};
            CertGetNameStringW(cert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, n, 512);
            name = n;
            CertFreeCertificateContext(cert);
        }
    }
    CryptMsgClose(msg);
    CertCloseStore(store, 0);
    return name;
}

// True when Windows trusts the file's Authenticode signature (unmodified file, valid chain).
bool signatureTrusted(const std::wstring& file) {
    WINTRUST_FILE_INFO fi{sizeof(fi)};
    fi.pcwszFilePath = file.c_str();
    WINTRUST_DATA wd{sizeof(wd)};
    wd.dwUIChoice = WTD_UI_NONE;
    wd.fdwRevocationChecks = WTD_REVOKE_NONE;  // works offline; the chain and the file hash are still checked
    wd.dwUnionChoice = WTD_CHOICE_FILE;
    wd.pFile = &fi;
    wd.dwStateAction = WTD_STATEACTION_VERIFY;
    GUID action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    const LONG r = WinVerifyTrust(static_cast<HWND>(INVALID_HANDLE_VALUE), &action, &wd);
    wd.dwStateAction = WTD_STATEACTION_CLOSE;
    WinVerifyTrust(static_cast<HWND>(INVALID_HANDLE_VALUE), &action, &wd);
    return r == ERROR_SUCCESS;
}

// VB-Audio signs its setups as its author, Vincent Burel ("BUREL VINCENT Entrepreneur individuel").
bool isVbAudioSigner(std::wstring name) {
    for (wchar_t& c : name) c = static_cast<wchar_t>(std::towupper(c));
    return name.find(L"BUREL VINCENT") != std::wstring::npos || name.find(L"VINCENT BUREL") != std::wstring::npos ||
           name.find(L"VB-AUDIO") != std::wstring::npos;
}

int verifySignature(const std::wstring& file) {
    if (GetFileAttributesW(file.c_str()) == INVALID_FILE_ATTRIBUTES) return result(false, "The VB-CABLE setup was not found.");
    const std::wstring signer = signerName(file);
    const std::string extra = "\"signer\":\"" + esc(utf8(signer)) + "\"";
    if (!signatureTrusted(file)) return result(false, "The VB-CABLE setup isn't validly signed, so it was not run.", extra);
    if (!isVbAudioSigner(signer)) return result(false, "The VB-CABLE setup isn't signed by VB-Audio, so it was not run.", extra);
    return result(true, "Signed by VB-Audio", extra);
}

bool vbCablePresent() {
    return forEachDevice(kVbCableHwid, [](HDEVINFO, SP_DEVINFO_DATA&) {}) > 0;
}

int installVbCable(const std::wstring& dir) {
    if (vbCablePresent()) return result(true, "VB-CABLE is already installed");
    const std::wstring setup = dir + L"\\VBCABLE_Setup_x64.exe";
    if (verifySignature(setup) != 0) return 1;
    // VB-Audio's installer, with its own window: the user confirms "Install Driver" there.
    std::wstring cmd = L"\"" + setup + L"\"";
    STARTUPINFOW si{sizeof(si)};
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, dir.c_str(), &si, &pi))
        return result(false, lastError("Starting the VB-CABLE setup"));
    WaitForSingleObject(pi.hProcess, 30 * 60 * 1000);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    if (!vbCablePresent()) return result(false, "VB-CABLE was not installed (its setup was closed before it finished).");
    return result(true, "VB-CABLE installed. Windows may need a restart before every app sees it.", "\"reboot\":true");
}

int status() {
    const int devices = forEachVadDevice([](HDEVINFO, SP_DEVINFO_DATA&) {});
    char extra[256];
    std::snprintf(extra, sizeof(extra), "\"admin\":%s,\"testSigning\":%s,\"vadDevice\":%s,\"asio\":%s,\"vbcable\":%s", isAdmin() ? "true" : "false",
                  testSigningEnabled() ? "true" : "false", devices ? "true" : "false", asioRegistered() ? "true" : "false",
                  vbCablePresent() ? "true" : "false");
    return result(true, "status", extra);
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    const std::wstring cmd = argc > 1 ? argv[1] : L"status";
    std::wstring dir = argc > 2 ? argv[2] : L"";
    std::wstring engine, ui, vbcable;
    for (int i = 3; i + 1 < argc; ++i) {
        if (!wcscmp(argv[i], L"--engine")) engine = argv[++i];
        else if (!wcscmp(argv[i], L"--ui")) ui = argv[++i];
        else if (!wcscmp(argv[i], L"--vbcable")) vbcable = argv[++i];
        else if (!wcscmp(argv[i], L"--result")) resultLog = _wfopen(argv[++i], L"a");
    }
    if (cmd == L"status") return status();
    if (cmd == L"verify-signature") return verifySignature(dir);
    if (!isAdmin()) return result(false, "Administrator rights are required");
    if (cmd == L"install-asio") return installAsio(dir, engine, ui);
    if (cmd == L"uninstall-asio") return uninstallAsio(dir);
    if (cmd == L"install-vad") return installVad(dir);
    if (cmd == L"uninstall-vad") return uninstallVad();
    if (cmd == L"install-vbcable") return installVbCable(dir);
    if (cmd == L"install-all") {  // one elevation for the app's "Install drivers" button
        int failed = installAsio(dir, engine, ui);
        // Public builds leave the test-signed virtual audio driver out (it needs Microsoft's signature).
        if (GetFileAttributesW((dir + L"\\vad\\tonespherevad.inf").c_str()) != INVALID_FILE_ATTRIBUTES) failed |= installVad(dir + L"\\vad");
        if (!vbcable.empty()) failed |= installVbCable(vbcable);
        return failed;
    }
    return result(false, "unknown command");
}
