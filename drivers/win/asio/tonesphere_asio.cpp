// ToneSphere ASIO driver: lets any ASIO host (DAW) exchange audio with the ToneSphere engine.
// Several hosts can use it at once; each appears in ToneSphere as an application with
// outputs ("what the DAW plays") and inputs ("what the DAW records").
//
// The driver runs its own high-resolution clock thread at the host's buffer size and
// exchanges audio through lock-free rings in shared memory (see tonesphere_asio_shared.h).
// The ToneSphere engine compensates any clock difference, so no hardware is required.
#include <windows.h>
#include "version.h"
#include <avrt.h>
#include <objbase.h>
#include <olectl.h>
#include <shellapi.h>
#include <timeapi.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "asiosys.h"
#include "asio.h"
#include "iasiodrv.h"
#include "tonesphere_asio_shared.h"

// {05358F36-792A-4BC6-8C9F-944AD781D1DC}
static const CLSID CLSID_ToneSphereAsio = {0x05358f36, 0x792a, 0x4bc6, {0x8c, 0x9f, 0x94, 0x4a, 0xd7, 0x81, 0xd1, 0xdc}};
static const char* kDriverName = "ToneSphere ASIO";
static HMODULE g_module = nullptr;
static std::atomic<long> g_objects{0};
static std::atomic<long> g_locks{0};

namespace {

int64_t qpc() {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return c.QuadPart;
}
double qpcFreq() {
    static const double f = [] {
        LARGE_INTEGER x;
        QueryPerformanceFrequency(&x);
        return static_cast<double>(x.QuadPart);
    }();
    return f;
}

std::wstring processFileName() {
    wchar_t path[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring p(path, n);
    const auto slash = p.find_last_of(L"\\/");
    return slash == std::wstring::npos ? p : p.substr(slash + 1);
}

// Tries to start the ToneSphere engine: the path written by the desktop installer, else the
// Store app's execution alias (%LOCALAPPDATA%\Microsoft\WindowsApps\tonesphere-engine.exe).
void tryLaunchEngine() {
    wchar_t path[MAX_PATH] = {};
    DWORD size = sizeof(path);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\ToneSphere", L"EnginePath", RRF_RT_REG_SZ | RRF_SUBKEY_WOW6464KEY, nullptr, path, &size) !=
        ERROR_SUCCESS) {
        const DWORD n = ExpandEnvironmentStringsW(L"%LOCALAPPDATA%\\Microsoft\\WindowsApps\\tonesphere-engine.exe", path, MAX_PATH);
        if (n == 0 || n > MAX_PATH || GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES) return;
    }
    STARTUPINFOW si{sizeof(si)};
    PROCESS_INFORMATION pi{};
    std::wstring cmd = L"\"" + std::wstring(path) + L"\" --background";
    if (CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, DETACHED_PROCESS | CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
}

}  // namespace

class ToneSphereAsio final : public IASIO {
public:
    ToneSphereAsio() { ++g_objects; }
    ~ToneSphereAsio() {
        stop();
        disposeBuffers();
        releaseSlot();
        --g_objects;
    }

    // ---- IUnknown (ASIO hosts query with the CLSID as IID)
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (riid == IID_IUnknown || riid == CLSID_ToneSphereAsio) {
            *ppv = static_cast<IASIO*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++ref_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG r = --ref_;
        if (r == 0) delete this;
        return r;
    }

    // ---- IASIO
    ASIOBool init(void* sysHandle) override {
        (void)sysHandle;
        if (server_) return ASIOTrue;
        if (!openServer()) {
            tryLaunchEngine();
            for (int i = 0; i < 40 && !openServer(); ++i) Sleep(100);
        }
        if (!server_) {
            setError("The ToneSphere audio engine is not running. Start ToneSphere and try again.");
            return ASIOFalse;
        }
        inputs_ = std::clamp(server_->inputChannels, 1, TSASIO_MAX_CHANNELS);
        outputs_ = std::clamp(server_->outputChannels, 1, TSASIO_MAX_CHANNELS);
        sampleRate_ = server_->engineSampleRate > 0 ? server_->engineSampleRate : 48000.0;
        if (!claimSlot()) {
            setError("Too many programs are using ToneSphere ASIO at the same time.");
            return ASIOFalse;
        }
        return ASIOTrue;
    }
    void getDriverName(char* name) override { std::snprintf(name, 32, "%s", kDriverName); }
    long getDriverVersion() override { return TONESPHERE_VERSION_MAJOR * 10000 + TONESPHERE_VERSION_MINOR * 100 + TONESPHERE_VERSION_PATCH; }
    void getErrorMessage(char* s) override { std::snprintf(s, 124, "%s", error_.c_str()); }

    ASIOError start() override {
        if (!callbacks_ || bufferSize_ <= 0) return ASE_NotPresent;
        if (running_) return ASE_OK;
        samplePos_ = 0;
        if (client_) {
            client_->fromEngine.readPos = client_->fromEngine.writePos;
            client_->running = 1;
        }
        running_ = true;
        thread_ = std::thread([this] { clockThread(); });
        return ASE_OK;
    }
    ASIOError stop() override {
        running_ = false;
        if (thread_.joinable()) thread_.join();
        if (client_) client_->running = 0;
        return ASE_OK;
    }
    ASIOError getChannels(long* in, long* out) override {
        if (!server_) return ASE_NotPresent;
        *in = inputs_;
        *out = outputs_;
        return ASE_OK;
    }
    ASIOError getLatencies(long* in, long* out) override {
        const long b = bufferSize_ > 0 ? bufferSize_ : preferredBuffer();
        // One buffer here plus the engine's cushion (about two engine blocks).
        *in = b + 256;
        *out = b + 256;
        return ASE_OK;
    }
    ASIOError getBufferSize(long* minSize, long* maxSize, long* preferred, long* granularity) override {
        *minSize = 32;
        *maxSize = 4096;
        *preferred = preferredBuffer();
        *granularity = -1;  // powers of two
        return ASE_OK;
    }
    ASIOError canSampleRate(ASIOSampleRate rate) override {
        // Any common rate works: the engine resamples clients that differ from its own rate.
        static const double ok[] = {44100, 48000, 88200, 96000, 176400, 192000};
        for (double r : ok)
            if (std::fabs(r - rate) < 0.5) return ASE_OK;
        return ASE_NoClock;
    }
    ASIOError getSampleRate(ASIOSampleRate* rate) override {
        *rate = sampleRate_;
        return ASE_OK;
    }
    ASIOError setSampleRate(ASIOSampleRate rate) override {
        if (rate == 0) return ASE_OK;  // "external clock" request: we are always internal
        if (canSampleRate(rate) != ASE_OK) return ASE_NoClock;
        if (std::fabs(rate - sampleRate_) < 0.5) return ASE_OK;
        sampleRate_ = rate;
        if (client_) {
            client_->sampleRate = rate;
            InterlockedIncrement64(&client_->generation);
        }
        if (callbacks_ && callbacks_->sampleRateDidChange) callbacks_->sampleRateDidChange(rate);
        return ASE_OK;
    }
    ASIOError getClockSources(ASIOClockSource* clocks, long* num) override {
        if (*num < 1) return ASE_InvalidParameter;
        clocks[0].index = 0;
        clocks[0].associatedChannel = -1;
        clocks[0].associatedGroup = -1;
        clocks[0].isCurrentSource = ASIOTrue;
        std::snprintf(clocks[0].name, 32, "ToneSphere engine");
        *num = 1;
        return ASE_OK;
    }
    ASIOError setClockSource(long ref) override { return ref == 0 ? ASE_OK : ASE_InvalidParameter; }
    ASIOError getSamplePosition(ASIOSamples* pos, ASIOTimeStamp* ts) override {
        const int64_t p = samplePos_.load();
        pos->hi = static_cast<unsigned long>(static_cast<uint64_t>(p) >> 32);
        pos->lo = static_cast<unsigned long>(p & 0xffffffff);
        const double ns = static_cast<double>(lastSwitchQpc_.load()) / qpcFreq() * 1e9;
        const uint64_t t = static_cast<uint64_t>(ns);
        ts->hi = static_cast<unsigned long>(t >> 32);
        ts->lo = static_cast<unsigned long>(t & 0xffffffff);
        return ASE_OK;
    }
    ASIOError getChannelInfo(ASIOChannelInfo* info) override {
        const bool in = info->isInput == ASIOTrue;
        if (info->channel < 0 || info->channel >= (in ? inputs_ : outputs_)) return ASE_InvalidParameter;
        info->isActive = ASIOFalse;
        for (auto& b : buffers_)
            if (b.isInput == info->isInput && b.channelNum == info->channel) info->isActive = ASIOTrue;
        info->channelGroup = 0;
        info->type = ASIOSTFloat32LSB;
        std::snprintf(info->name, 32, "ToneSphere %s %ld", in ? "In" : "Out", info->channel + 1);
        return ASE_OK;
    }
    ASIOError createBuffers(ASIOBufferInfo* infos, long num, long bufferSize, ASIOCallbacks* cb) override {
        if (!server_ || !cb || bufferSize < 16 || bufferSize > 8192) return ASE_InvalidParameter;
        disposeBuffers();
        bufferSize_ = bufferSize;
        callbacks_ = cb;
        buffers_.assign(infos, infos + num);
        memory_.assign(static_cast<size_t>(num) * 2 * bufferSize, 0.0f);
        for (long i = 0; i < num; ++i) {
            ASIOBufferInfo& bi = infos[i];
            const long limit = bi.isInput ? inputs_ : outputs_;
            if (bi.channelNum < 0 || bi.channelNum >= limit) {
                disposeBuffers();
                return ASE_InvalidParameter;
            }
            bi.buffers[0] = &memory_[static_cast<size_t>(i) * 2 * bufferSize];
            bi.buffers[1] = &memory_[static_cast<size_t>(i) * 2 * bufferSize + bufferSize];
            buffers_[i] = bi;
        }
        timeInfo_ = cb->asioMessage && cb->asioMessage(kAsioSelectorSupported, kAsioSupportsTimeInfo, nullptr, nullptr) == 1 &&
                    cb->asioMessage(kAsioSupportsTimeInfo, 0, nullptr, nullptr) == 1;
        if (!openClientRegion()) {
            setError("Cannot share audio with the ToneSphere engine.");
            return ASE_HWMalfunction;
        }
        client_->bufferSize = bufferSize;
        client_->sampleRate = sampleRate_;
        InterlockedIncrement64(&client_->generation);
        return ASE_OK;
    }
    ASIOError disposeBuffers() override {
        stop();
        buffers_.clear();
        memory_.clear();
        callbacks_ = nullptr;
        bufferSize_ = 0;
        return ASE_OK;
    }
    ASIOError controlPanel() override {
        wchar_t path[MAX_PATH] = {};
        DWORD size = sizeof(path);
        if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\ToneSphere", L"UiPath", RRF_RT_REG_SZ | RRF_SUBKEY_WOW6464KEY, nullptr, path, &size) ==
            ERROR_SUCCESS) {
            ShellExecuteW(nullptr, L"open", path, nullptr, nullptr, SW_SHOWNORMAL);
            return ASE_OK;
        }
        // Store build: the package registers the tonesphere: protocol.
        if (reinterpret_cast<INT_PTR>(ShellExecuteW(nullptr, L"open", L"tonesphere:", nullptr, nullptr, SW_SHOWNORMAL)) > 32) return ASE_OK;
        MessageBoxW(nullptr, L"Open the ToneSphere app to route this program's audio.", L"ToneSphere ASIO", MB_OK | MB_ICONINFORMATION);
        return ASE_OK;
    }
    ASIOError future(long selector, void*) override {
        switch (selector) {
            case kAsioEnableTimeCodeRead:
            case kAsioDisableTimeCodeRead:
                return ASE_NotPresent;
            case kAsioCanTimeInfo:
                return ASE_SUCCESS;
            default:
                return ASE_NotPresent;
        }
    }
    ASIOError outputReady() override { return ASE_NotPresent; }  // we copy outputs after bufferSwitch returns

private:
    long preferredBuffer() const {
        long p = server_ && server_->preferredBufferSize > 0 ? server_->preferredBufferSize : 256;
        long pow2 = 32;
        while (pow2 < p) pow2 <<= 1;
        return pow2;
    }
    void setError(const char* s) { error_ = s; }

    bool openServer() {
        HANDLE h = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, TSASIO_SERVER_NAME);
        if (!h) return false;
        auto* s = static_cast<TSASIO_SERVER*>(MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(TSASIO_SERVER)));
        if (!s || s->magic != TSASIO_MAGIC || s->version != TSASIO_VERSION) {
            if (s) UnmapViewOfFile(s);
            CloseHandle(h);
            return false;
        }
        serverMap_ = h;
        server_ = s;
        return true;
    }

    bool claimSlot() {
        for (int i = 0; i < TSASIO_MAX_CLIENTS; ++i) {
            if (InterlockedCompareExchange(reinterpret_cast<volatile LONG*>(&server_->slots[i].state), TSASIO_CLAIMED, TSASIO_FREE) ==
                TSASIO_FREE) {
                slot_ = i;
                TSASIO_SLOT& s = server_->slots[i];
                s.pid = GetCurrentProcessId();
                const std::wstring exe = processFileName();
                wcsncpy_s(s.exe, exe.c_str(), _TRUNCATE);
                std::wstring disp = exe.substr(0, exe.rfind(L'.'));
                if (!disp.empty()) disp[0] = static_cast<wchar_t>(towupper(disp[0]));
                wcsncpy_s(s.display, disp.c_str(), _TRUNCATE);
                s.heartbeat = qpc();
                return true;
            }
        }
        return false;
    }

    bool openClientRegion() {
        if (client_) return true;
        wchar_t name[96];
        swprintf_s(name, TSASIO_CLIENT_NAME_FMT, static_cast<unsigned>(GetCurrentProcessId()), static_cast<unsigned>(slot_));
        const size_t bytes = ntasio_client_bytes(inputs_, outputs_);
        clientMap_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, static_cast<DWORD>(static_cast<uint64_t>(bytes) >> 32),
                                        static_cast<DWORD>(bytes & 0xffffffff), name);
        if (!clientMap_) return false;
        client_ = static_cast<TSASIO_CLIENT*>(MapViewOfFile(clientMap_, FILE_MAP_ALL_ACCESS, 0, 0, bytes));
        if (!client_) return false;
        std::memset(client_, 0, sizeof(TSASIO_CLIENT));
        client_->magic = TSASIO_MAGIC;
        client_->version = TSASIO_VERSION;
        client_->inputs = inputs_;
        client_->outputs = outputs_;
        client_->sampleRate = sampleRate_;
        MemoryBarrier();
        InterlockedExchange(reinterpret_cast<volatile LONG*>(&server_->slots[slot_].state), TSASIO_ACTIVE);
        return true;
    }

    void releaseSlot() {
        if (server_ && slot_ >= 0) InterlockedExchange(reinterpret_cast<volatile LONG*>(&server_->slots[slot_].state), TSASIO_FREE);
        slot_ = -1;
        if (client_) UnmapViewOfFile(client_);
        if (clientMap_) CloseHandle(clientMap_);
        if (server_) UnmapViewOfFile(server_);
        if (serverMap_) CloseHandle(serverMap_);
        client_ = nullptr;
        clientMap_ = nullptr;
        server_ = nullptr;
        serverMap_ = nullptr;
    }

    // Moves one buffer of audio and calls the host.
    void process(long index) {
        const long n = bufferSize_;
        TSASIO_CLIENT* c = client_;
        // ToneSphere -> host inputs
        if (c) {
            TSASIO_RING& r = c->fromEngine;
            const float* data = ntasio_from_engine(c);
            int64_t rd = r.readPos;
            const int64_t avail = r.writePos - rd;
            const bool ok = avail >= n && avail <= static_cast<int64_t>(TSASIO_RING_FRAMES);
            if (!ok && avail > static_cast<int64_t>(TSASIO_RING_FRAMES)) rd = r.writePos - n;  // resync
            for (auto& b : buffers_) {
                if (!b.isInput) continue;
                float* dst = static_cast<float*>(b.buffers[index]);
                if (!ok) {
                    std::memset(dst, 0, sizeof(float) * n);
                    continue;
                }
                for (long i = 0; i < n; ++i)
                    dst[i] = data[((rd + i) & (TSASIO_RING_FRAMES - 1)) * c->inputs + b.channelNum];
            }
            if (ok) r.readPos = rd + n;
            else InterlockedIncrement64(&r.underflows);
        } else {
            for (auto& b : buffers_)
                if (b.isInput) std::memset(b.buffers[index], 0, sizeof(float) * n);
        }

        const int64_t now = qpc();
        lastSwitchQpc_ = now;
        if (timeInfo_ && callbacks_->bufferSwitchTimeInfo) {
            ASIOTime t{};
            t.timeInfo.speed = 1.0;
            const uint64_t ns = static_cast<uint64_t>(static_cast<double>(now) / qpcFreq() * 1e9);
            t.timeInfo.systemTime.hi = static_cast<unsigned long>(ns >> 32);
            t.timeInfo.systemTime.lo = static_cast<unsigned long>(ns & 0xffffffff);
            const uint64_t pos = static_cast<uint64_t>(samplePos_.load());
            t.timeInfo.samplePosition.hi = static_cast<unsigned long>(pos >> 32);
            t.timeInfo.samplePosition.lo = static_cast<unsigned long>(pos & 0xffffffff);
            t.timeInfo.sampleRate = sampleRate_;
            t.timeInfo.flags = kSystemTimeValid | kSamplePositionValid | kSampleRateValid;
            callbacks_->bufferSwitchTimeInfo(&t, index, ASIOTrue);
        } else {
            callbacks_->bufferSwitch(index, ASIOTrue);
        }
        samplePos_ += n;

        // host outputs -> ToneSphere
        if (c) {
            TSASIO_RING& r = c->toEngine;
            float* data = ntasio_to_engine(c);
            const int64_t w = r.writePos;
            const int64_t used = w - r.readPos;
            if (used < 0 || used + n > static_cast<int64_t>(TSASIO_RING_FRAMES)) {
                InterlockedIncrement64(&r.overflows);  // engine not consuming
            } else {
                for (int ch = 0; ch < c->outputs; ++ch) {
                    const float* src = nullptr;
                    for (auto& b : buffers_)
                        if (!b.isInput && b.channelNum == ch) src = static_cast<const float*>(b.buffers[index]);
                    for (long i = 0; i < n; ++i) data[((w + i) & (TSASIO_RING_FRAMES - 1)) * c->outputs + ch] = src ? src[i] : 0.0f;
                }
                MemoryBarrier();
                r.writePos = w + n;
            }
            if (server_ && slot_ >= 0) server_->slots[slot_].heartbeat = now;
        }
    }

    void clockThread() {
        DWORD task = 0;
        HANDLE av = AvSetMmThreadCharacteristicsW(L"Pro Audio", &task);
        HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        if (!timer) timer = CreateWaitableTimerW(nullptr, FALSE, nullptr);
        timeBeginPeriod(1);
        const double period = static_cast<double>(bufferSize_) / sampleRate_;
        const int64_t t0 = qpc();
        int64_t done = 0;
        long index = 0;
        while (running_) {
            const double elapsed = static_cast<double>(qpc() - t0) / qpcFreq();
            const int64_t due = static_cast<int64_t>(elapsed / period);
            if (due - done > 8) done = due - 1;  // resynchronise after a stall
            while (done < due && running_) {
                process(index);
                index ^= 1;
                ++done;
            }
            const double wait = (static_cast<double>(done + 1) * period) - static_cast<double>(qpc() - t0) / qpcFreq();
            if (wait > 0) {
                LARGE_INTEGER li;
                li.QuadPart = -static_cast<LONGLONG>(wait * 1e7);
                SetWaitableTimer(timer, &li, 0, nullptr, nullptr, FALSE);
                WaitForSingleObject(timer, 100);
            }
        }
        timeEndPeriod(1);
        CloseHandle(timer);
        if (av) AvRevertMmThreadCharacteristics(av);
    }

    std::atomic<ULONG> ref_{1};
    std::string error_;
    HANDLE serverMap_ = nullptr;
    TSASIO_SERVER* server_ = nullptr;
    HANDLE clientMap_ = nullptr;
    TSASIO_CLIENT* client_ = nullptr;
    int slot_ = -1;
    long inputs_ = 8, outputs_ = 8;
    double sampleRate_ = 48000.0;
    long bufferSize_ = 0;
    ASIOCallbacks* callbacks_ = nullptr;
    std::vector<ASIOBufferInfo> buffers_;
    std::vector<float> memory_;
    bool timeInfo_ = false;
    std::atomic<bool> running_{false};
    std::thread thread_;
    std::atomic<int64_t> samplePos_{0};
    std::atomic<int64_t> lastSwitchQpc_{0};
};

// ---------------------------------------------------------------------------- COM plumbing
class Factory final : public IClassFactory {
public:
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (riid == IID_IUnknown || riid == IID_IClassFactory) {
            *ppv = static_cast<IClassFactory*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return 2; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
    HRESULT STDMETHODCALLTYPE CreateInstance(IUnknown* outer, REFIID riid, void** ppv) override {
        if (outer) return CLASS_E_NOAGGREGATION;
        auto* d = new (std::nothrow) ToneSphereAsio();
        if (!d) return E_OUTOFMEMORY;
        const HRESULT hr = d->QueryInterface(riid, ppv);
        d->Release();
        return hr;
    }
    HRESULT STDMETHODCALLTYPE LockServer(BOOL lock) override {
        lock ? ++g_locks : --g_locks;
        return S_OK;
    }
};
static Factory g_factory;

static std::wstring clsidString() {
    wchar_t buf[64];
    StringFromGUID2(CLSID_ToneSphereAsio, buf, 64);
    return buf;
}

extern "C" HRESULT __stdcall DllGetClassObject(REFCLSID clsid, REFIID riid, void** ppv) {
    if (clsid != CLSID_ToneSphereAsio) return CLASS_E_CLASSNOTAVAILABLE;
    return g_factory.QueryInterface(riid, ppv);
}

extern "C" HRESULT __stdcall DllCanUnloadNow() { return g_objects == 0 && g_locks == 0 ? S_OK : S_FALSE; }

extern "C" HRESULT __stdcall DllRegisterServer() {
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(g_module, path, MAX_PATH);
    const std::wstring clsid = clsidString();
    const std::wstring key = L"CLSID\\" + clsid + L"\\InprocServer32";
    HKEY h;
    if (RegCreateKeyExW(HKEY_CLASSES_ROOT, key.c_str(), 0, nullptr, 0, KEY_WRITE, nullptr, &h, nullptr) != ERROR_SUCCESS) return SELFREG_E_CLASS;
    RegSetValueExW(h, nullptr, 0, REG_SZ, reinterpret_cast<const BYTE*>(path), static_cast<DWORD>((wcslen(path) + 1) * 2));
    const wchar_t* tm = L"Apartment";
    RegSetValueExW(h, L"ThreadingModel", 0, REG_SZ, reinterpret_cast<const BYTE*>(tm), static_cast<DWORD>((wcslen(tm) + 1) * 2));
    RegCloseKey(h);
    if (RegCreateKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\ASIO\\ToneSphere ASIO", 0, nullptr, 0, KEY_WRITE, nullptr, &h, nullptr) != ERROR_SUCCESS)
        return SELFREG_E_CLASS;
    RegSetValueExW(h, L"CLSID", 0, REG_SZ, reinterpret_cast<const BYTE*>(clsid.c_str()), static_cast<DWORD>((clsid.size() + 1) * 2));
    const wchar_t* desc = L"ToneSphere ASIO";
    RegSetValueExW(h, L"Description", 0, REG_SZ, reinterpret_cast<const BYTE*>(desc), static_cast<DWORD>((wcslen(desc) + 1) * 2));
    RegCloseKey(h);
    return S_OK;
}

extern "C" HRESULT __stdcall DllUnregisterServer() {
    const std::wstring clsid = clsidString();
    RegDeleteTreeW(HKEY_CLASSES_ROOT, (L"CLSID\\" + clsid).c_str());
    RegDeleteTreeW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\ASIO\\ToneSphere ASIO");
    return S_OK;
}

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_module = module;
        DisableThreadLibraryCalls(module);
    }
    return TRUE;
}
