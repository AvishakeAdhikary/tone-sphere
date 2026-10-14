// Per-application audio (Windows): session discovery, process-loopback capture and
// per-app endpoint redirection.
#include "backends/win/app_audio.h"

#include <windows.h>

#include <audioclient.h>
#include <audioclientactivationparams.h>
#include <audiopolicy.h>
#include <mmdeviceapi.h>
#include <roapi.h>
#include <shellapi.h>
#include <tlhelp32.h>
#include <wincodec.h>
#include <winstring.h>
#include <wrl/implements.h>

#include <algorithm>
#include <cwctype>

#include "backends/win/wasapi.h"
#include "core/engine.h"
#include "util/encoding.h"
#include "util/log.h"
#include "util/platform.h"

#pragma comment(lib, "mmdevapi.lib")
#pragma comment(lib, "runtimeobject.lib")
#pragma comment(lib, "version.lib")
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "shell32.lib")

using Microsoft::WRL::ClassicCom;
using Microsoft::WRL::ComPtr;
using Microsoft::WRL::FtmBase;
using Microsoft::WRL::RuntimeClass;
using Microsoft::WRL::RuntimeClassFlags;

namespace ts {

// Undocumented but stable since Windows 10 1803 (used by Windows Settings and EarTrumpet).
// Only the method we need is declared after the placeholder slots.
MIDL_INTERFACE("ab3d4648-e242-459f-b02f-541c70306324")
IAudioPolicyConfigFactory21H2 : public IInspectable {
public:
    virtual HRESULT STDMETHODCALLTYPE __add_CtxVolumeChange() = 0;
    virtual HRESULT STDMETHODCALLTYPE __remove_CtxVolumeChanged() = 0;
    virtual HRESULT STDMETHODCALLTYPE __add_RingerVibrateStateChanged() = 0;
    virtual HRESULT STDMETHODCALLTYPE __remove_RingerVibrateStateChange() = 0;
    virtual HRESULT STDMETHODCALLTYPE __SetVolumeGroupGainForId() = 0;
    virtual HRESULT STDMETHODCALLTYPE __GetVolumeGroupGainForId() = 0;
    virtual HRESULT STDMETHODCALLTYPE __GetActiveVolumeGroupForEndpointId() = 0;
    virtual HRESULT STDMETHODCALLTYPE __GetVolumeGroupsForEndpoint() = 0;
    virtual HRESULT STDMETHODCALLTYPE __GetCurrentVolumeContext() = 0;
    virtual HRESULT STDMETHODCALLTYPE __SetVolumeGroupMuteForId() = 0;
    virtual HRESULT STDMETHODCALLTYPE __GetVolumeGroupMuteForId() = 0;
    virtual HRESULT STDMETHODCALLTYPE __SetRingerVibrateState() = 0;
    virtual HRESULT STDMETHODCALLTYPE __GetRingerVibrateState() = 0;
    virtual HRESULT STDMETHODCALLTYPE __SetPreferredChatApplication() = 0;
    virtual HRESULT STDMETHODCALLTYPE __ResetPreferredChatApplication() = 0;
    virtual HRESULT STDMETHODCALLTYPE __GetPreferredChatApplication() = 0;
    virtual HRESULT STDMETHODCALLTYPE __GetCurrentChatApplications() = 0;
    virtual HRESULT STDMETHODCALLTYPE __add_ChatContextChanged() = 0;
    virtual HRESULT STDMETHODCALLTYPE __remove_ChatContextChanged() = 0;
    virtual HRESULT STDMETHODCALLTYPE SetPersistedDefaultAudioEndpoint(UINT processId, EDataFlow flow, ERole role, HSTRING deviceId) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetPersistedDefaultAudioEndpoint(UINT processId, EDataFlow flow, int role, HSTRING* deviceId) = 0;
    virtual HRESULT STDMETHODCALLTYPE ClearAllPersistedApplicationDefaultEndpoints() = 0;
};
static const IID kIidPolicyDownlevel = {0x2a59116d, 0x6c4f, 0x45e0, {0xa7, 0x4f, 0x70, 0x7e, 0x3f, 0xef, 0x92, 0x58}};

namespace {

std::string lowerExe(const std::wstring& path) {
    std::wstring f = path.substr(path.find_last_of(L"\\/") + 1);
    std::transform(f.begin(), f.end(), f.begin(), [](wchar_t c) { return static_cast<wchar_t>(std::towlower(c)); });
    return toUtf8(f);
}

std::wstring processPath(DWORD pid) {
    std::wstring out;
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!h) return out;
    wchar_t buf[MAX_PATH * 2];
    DWORD len = static_cast<DWORD>(std::size(buf));
    if (QueryFullProcessImageNameW(h, 0, buf, &len)) out.assign(buf, len);
    CloseHandle(h);
    return out;
}

std::string fileDescription(const std::wstring& path) {
    DWORD dummy = 0;
    const DWORD size = GetFileVersionInfoSizeW(path.c_str(), &dummy);
    if (!size) return {};
    std::vector<BYTE> data(size);
    if (!GetFileVersionInfoW(path.c_str(), 0, size, data.data())) return {};
    struct Lang {
        WORD lang, cp;
    }* langs = nullptr;
    UINT len = 0;
    if (!VerQueryValueW(data.data(), L"\\VarFileInfo\\Translation", reinterpret_cast<void**>(&langs), &len) || len < sizeof(Lang)) return {};
    wchar_t key[128];
    swprintf_s(key, L"\\StringFileInfo\\%04x%04x\\FileDescription", langs[0].lang, langs[0].cp);
    wchar_t* desc = nullptr;
    if (VerQueryValueW(data.data(), key, reinterpret_cast<void**>(&desc), &len) && desc && len > 1) return toUtf8(desc);
    return {};
}

std::string prettyExe(const std::string& exe) {
    std::string n = exe.substr(0, exe.rfind('.'));
    if (!n.empty()) n[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(n[0])));
    return n;
}

class ActivationHandler : public RuntimeClass<RuntimeClassFlags<ClassicCom>, FtmBase, IActivateAudioInterfaceCompletionHandler> {
public:
    ActivationHandler() { done_ = CreateEventW(nullptr, TRUE, FALSE, nullptr); }
    ~ActivationHandler() override { CloseHandle(done_); }
    STDMETHOD(ActivateCompleted)(IActivateAudioInterfaceAsyncOperation* op) override {
        HRESULT hr = S_OK;
        ComPtr<IUnknown> unk;
        if (SUCCEEDED(op->GetActivateResult(&hr, &unk)) && SUCCEEDED(hr)) unk.As(&client_);
        result_ = hr;
        SetEvent(done_);
        return S_OK;
    }
    bool wait(DWORD ms) { return WaitForSingleObject(done_, ms) == WAIT_OBJECT_0; }
    ComPtr<IAudioClient> client_;
    HRESULT result_ = E_FAIL;

private:
    HANDLE done_;
};

}  // namespace

AppAudio::AppAudio() {
    HSTRING cls = nullptr;
    const wchar_t* name = L"Windows.Media.Internal.AudioPolicyConfig";
    if (SUCCEEDED(WindowsCreateString(name, static_cast<UINT32>(wcslen(name)), &cls))) {
        void* f = nullptr;
        if (SUCCEEDED(RoGetActivationFactory(cls, __uuidof(IAudioPolicyConfigFactory21H2), &f))) {
            policy_ = f;
        } else if (SUCCEEDED(RoGetActivationFactory(cls, kIidPolicyDownlevel, &f))) {
            policy_ = f;
            policyNew_ = false;
        }
        WindowsDeleteString(cls);
    }
    if (!policy_) TS_WARN("Per-app device redirection is not available on this Windows version");
}

AppAudio::~AppAudio() { shutdown(); }

void AppAudio::shutdown() {
    // Undo every redirection we made so apps go back to the Windows default device.
    for (auto& [exe, ep] : redirectedOut_) {
        auto it = apps_.find(exe);
        if (it != apps_.end() && !ep.empty()) setAppEndpoint(it->second.pids, false, "");
    }
    for (auto& [exe, ep] : redirectedIn_) {
        auto it = apps_.find(exe);
        if (it != apps_.end() && !ep.empty()) setAppEndpoint(it->second.pids, true, "");
    }
    redirectedOut_.clear();
    redirectedIn_.clear();
    if (policy_) {
        static_cast<IUnknown*>(policy_)->Release();
        policy_ = nullptr;
    }
}

bool AppAudio::poll() {
    std::map<std::string, AppSession> found;
    ComPtr<IMMDeviceEnumerator> en;
    if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&en)))) return false;
    const DWORD self = GetCurrentProcessId();
    for (EDataFlow flow : {eRender, eCapture}) {
        ComPtr<IMMDeviceCollection> coll;
        if (FAILED(en->EnumAudioEndpoints(flow, DEVICE_STATE_ACTIVE, &coll))) continue;
        UINT n = 0;
        coll->GetCount(&n);
        for (UINT i = 0; i < n; ++i) {
            ComPtr<IMMDevice> dev;
            ComPtr<IAudioSessionManager2> mgr;
            ComPtr<IAudioSessionEnumerator> sessions;
            if (FAILED(coll->Item(i, &dev)) || FAILED(dev->Activate(__uuidof(IAudioSessionManager2), CLSCTX_ALL, nullptr, &mgr)) ||
                FAILED(mgr->GetSessionEnumerator(&sessions)))
                continue;
            int count = 0;
            sessions->GetCount(&count);
            for (int k = 0; k < count; ++k) {
                ComPtr<IAudioSessionControl> ctl;
                ComPtr<IAudioSessionControl2> ctl2;
                if (FAILED(sessions->GetSession(k, &ctl)) || FAILED(ctl.As(&ctl2))) continue;
                if (ctl2->IsSystemSoundsSession() == S_OK) continue;
                DWORD pid = 0;
                ctl2->GetProcessId(&pid);
                if (pid == 0 || pid == self) continue;
                AudioSessionState st = AudioSessionStateInactive;
                ctl->GetState(&st);
                if (st == AudioSessionStateExpired) continue;
                const std::wstring path = processPath(pid);
                if (path.empty()) continue;
                const std::string exe = lowerExe(path);
                if (exe == "tonesphere.exe" || exe == "tonesphere-engine.exe" || exe == "audiodg.exe") continue;
                AppSession& a = found[exe];
                a.exe = exe;
                a.path = path;
                if (std::find(a.pids.begin(), a.pids.end(), pid) == a.pids.end()) a.pids.push_back(pid);
                if (flow == eRender) {
                    a.render = true;
                    a.renderActive |= st == AudioSessionStateActive;
                } else {
                    a.capture = true;
                    a.captureActive |= st == AudioSessionStateActive;
                }
            }
        }
    }
    std::string sig;
    for (auto& [exe, a] : found) {
        std::sort(a.pids.begin(), a.pids.end());
        sig += exe + (a.render ? "R" : "") + (a.renderActive ? "a" : "") + (a.capture ? "C" : "") + (a.captureActive ? "a" : "");
        for (auto p : a.pids) sig += ":" + std::to_string(p);
        sig += ";";
        if (a.displayName.empty()) {
            auto old = apps_.find(exe);
            a.displayName = old != apps_.end() && !old->second.displayName.empty() ? old->second.displayName : fileDescription(a.path);
            if (a.displayName.empty()) a.displayName = prettyExe(exe);
        }
    }
    apps_ = std::move(found);
    if (sig == signature_) return false;
    signature_ = sig;
    return true;
}

std::string AppAudio::iconFor(const std::wstring& path, const std::string& exe) {
    auto it = icons_.find(exe);
    if (it != icons_.end()) return it->second;
    std::string url;
    HICON icon = nullptr;
    if (ExtractIconExW(path.c_str(), 0, &icon, nullptr, 1) > 0 && icon) {
        ComPtr<IWICImagingFactory> wic;
        ComPtr<IWICBitmap> bmp;
        ComPtr<IStream> stream;
        ComPtr<IWICBitmapEncoder> enc;
        ComPtr<IWICBitmapFrameEncode> frame;
        if (SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic))) &&
            SUCCEEDED(wic->CreateBitmapFromHICON(icon, &bmp)) && SUCCEEDED(CreateStreamOnHGlobal(nullptr, TRUE, &stream)) &&
            SUCCEEDED(wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc)) &&
            SUCCEEDED(enc->Initialize(stream.Get(), WICBitmapEncoderNoCache)) && SUCCEEDED(enc->CreateNewFrame(&frame, nullptr)) &&
            SUCCEEDED(frame->Initialize(nullptr)) && SUCCEEDED(frame->WriteSource(bmp.Get(), nullptr)) && SUCCEEDED(frame->Commit()) &&
            SUCCEEDED(enc->Commit())) {
            HGLOBAL mem = nullptr;
            if (SUCCEEDED(GetHGlobalFromStream(stream.Get(), &mem))) {
                STATSTG stat{};
                stream->Stat(&stat, STATFLAG_NONAME);
                const void* data = GlobalLock(mem);
                if (data) {
                    url = "data:image/png;base64," + base64Encode(data, static_cast<size_t>(stat.cbSize.QuadPart));
                    GlobalUnlock(mem);
                }
            }
        }
        DestroyIcon(icon);
    }
    icons_[exe] = url;
    return url;
}

void AppAudio::refresh(Engine& e, std::set<std::string>& seen) {
    for (auto& [exe, a] : apps_) {
        auto make = [&](bool out) {
            const std::string id = std::string(out ? "app:out:" : "app:in:") + exe;
            NodeModel& n = e.ensureNode(id);
            n.kind = DeviceKind::App;
            n.bin = out ? Bin::AppOut : Bin::AppIn;
            n.name = a.displayName;
            n.subtitle = exe;
            n.channels = 2;
            n.channelNames = {"L", "R"};
            n.present = true;
            n.device.input = out;
            n.device.kind = DeviceKind::App;
            n.device.bin = n.bin;
            n.device.nodeId = id;
            n.device.name = a.displayName;
            const bool pidsChanged = n.app.pids != a.pids;
            n.app.exe = exe;
            n.app.pids = a.pids;
            n.app.playing = out ? a.renderActive : a.captureActive;
            n.app.iconPng = iconFor(a.path, exe);
            if (pidsChanged && n.stream && out && n.app.mode == "copy") e.closeStreamLocked(n);  // re-target loopback
            seen.insert(id);
        };
        if (a.render) make(true);
        if (a.capture) make(false);
    }
}

uint32_t AppAudio::rootPid(uint32_t pid, const std::string& exe) {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return pid;
    std::map<DWORD, std::pair<DWORD, std::string>> procs;
    PROCESSENTRY32W pe{sizeof(pe)};
    for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe)) {
        std::wstring f = pe.szExeFile;
        std::transform(f.begin(), f.end(), f.begin(), [](wchar_t c) { return static_cast<wchar_t>(std::towlower(c)); });
        procs[pe.th32ProcessID] = {pe.th32ParentProcessID, toUtf8(f)};
    }
    CloseHandle(snap);
    uint32_t cur = pid;
    for (int depth = 0; depth < 16; ++depth) {
        auto it = procs.find(cur);
        if (it == procs.end()) break;
        auto parent = procs.find(it->second.first);
        if (parent == procs.end() || parent->second.second != exe) break;
        cur = parent->first;
    }
    return cur;
}

bool AppAudio::setAppEndpoint(const std::vector<uint32_t>& pids, bool capture, const std::string& endpointId) {
    if (!policy_) return false;
    auto* f = static_cast<IAudioPolicyConfigFactory21H2*>(policy_);
    HSTRING dev = nullptr;
    std::wstring full;
    if (!endpointId.empty()) {
        full = L"\\\\?\\SWD#MMDEVAPI#" + toWide(endpointId) +
               (capture ? L"#{2eef81be-33fa-4800-9670-1cd474972c3f}" : L"#{e6327cad-dcec-4949-ae8a-991e976a79d2}");
        WindowsCreateString(full.c_str(), static_cast<UINT32>(full.size()), &dev);
    }
    bool ok = true;
    for (uint32_t pid : pids) {
        const EDataFlow flow = capture ? eCapture : eRender;
        ok &= SUCCEEDED(f->SetPersistedDefaultAudioEndpoint(pid, flow, eMultimedia, dev));
        ok &= SUCCEEDED(f->SetPersistedDefaultAudioEndpoint(pid, flow, eConsole, dev));
    }
    if (dev) WindowsDeleteString(dev);
    return ok;
}

void AppAudio::applyRouting(Engine& e) {
    // Virtual devices already claimed by other apps.
    std::set<std::string> busy;
    for (auto& [exe, id] : cableOut_) busy.insert(id);
    for (auto& [exe, id] : cableIn_) busy.insert(id);
    auto usedByRoutes = [&](const std::string& id) {
        return std::any_of(e.routes_.begin(), e.routes_.end(), [&](const RouteModel& r) { return r.from == id || r.to == id; });
    };
    // Prefer ToneSphere's own virtual devices; fall back to compatibility cables.
    auto freeDevice = [&](Bin bin) -> NodeModel* {
        for (DeviceKind k : {DeviceKind::Vad, DeviceKind::Compat})
            for (auto& [id, n] : e.nodes_)
                if (n.kind == k && n.bin == bin && n.present && !n.device.appEndpointId.empty() && !busy.count(id) && !usedByRoutes(id))
                    return &n;
        return nullptr;
    };

    for (auto& [id, n] : e.nodes_) {
        if (n.kind != DeviceKind::App) continue;
        const std::string& exe = n.app.exe;
        const bool needed = e.nodeNeededLocked(n);
        if (n.bin == Bin::AppOut) {
            const bool want = needed && n.app.mode == "isolate";
            if (want && !cableOut_.count(exe)) {
                NodeModel* dev = freeDevice(Bin::VirtualOut);
                if (!dev) {
                    n.streamError = "Isolating an app needs a free virtual output (install VB-CABLE in Settings › Devices)";
                    continue;
                }
                if (setAppEndpoint(n.app.pids, false, dev->device.appEndpointId)) {
                    cableOut_[exe] = dev->id;
                    cableOutDev_[exe] = dev->device;
                    redirectedOut_[exe] = dev->device.appEndpointId;
                    busy.insert(dev->id);
                    if (n.stream) e.closeStreamLocked(n);  // switch from loopback to the virtual device
                    TS_INFO("Isolated %s on %s", exe.c_str(), dev->name.c_str());
                } else {
                    n.streamError = "Windows refused to move this app to a virtual output";
                }
            } else if (!want && cableOut_.count(exe)) {
                setAppEndpoint(n.app.pids, false, "");
                cableOut_.erase(exe);
                cableOutDev_.erase(exe);
                redirectedOut_.erase(exe);
                if (n.stream) e.closeStreamLocked(n);
                TS_INFO("Released %s from isolation", exe.c_str());
            }
        } else if (n.bin == Bin::AppIn) {
            if (needed && !cableIn_.count(exe)) {
                NodeModel* dev = freeDevice(Bin::VirtualIn);
                if (!dev) {
                    n.streamError = "Sending sound into an app needs a free virtual microphone (install VB-CABLE in Settings › Devices)";
                    continue;
                }
                if (setAppEndpoint(n.app.pids, true, dev->device.appEndpointId)) {
                    cableIn_[exe] = dev->id;
                    cableInDev_[exe] = dev->device;
                    redirectedIn_[exe] = dev->device.appEndpointId;
                    busy.insert(dev->id);
                    TS_INFO("%s now records from %s", exe.c_str(), dev->name.c_str());
                } else {
                    n.streamError = "Windows refused to change this app's microphone";
                }
            } else if (!needed && cableIn_.count(exe)) {
                setAppEndpoint(n.app.pids, true, "");
                cableIn_.erase(exe);
                cableInDev_.erase(exe);
                redirectedIn_.erase(exe);
                if (n.stream) e.closeStreamLocked(n);
            }
        }
    }
}

const DeviceInfo* AppAudio::assigned(const NodeModel& n) const {
    const auto& m = n.bin == Bin::AppIn ? cableInDev_ : cableOutDev_;
    if (n.bin == Bin::AppOut && n.app.mode != "isolate") return nullptr;
    auto it = m.find(n.app.exe);
    return it == m.end() ? nullptr : &it->second;
}

std::unique_ptr<DeviceStream> AppAudio::open(NodeModel& n, double engineRate, std::string& err) {
    const std::string& exe = n.app.exe;
    if (n.bin == Bin::AppIn) {
        err = "No virtual microphone assigned";
        return nullptr;
    }
    if (n.app.mode == "off") {
        err = "Capture is off for this app";
        return nullptr;
    }
    // Copy mode: process loopback of the app's whole process tree.
    if (n.app.pids.empty()) {
        err = "App is not running";
        return nullptr;
    }
    AUDIOCLIENT_ACTIVATION_PARAMS params{};
    params.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK;
    params.ProcessLoopbackParams.TargetProcessId = rootPid(n.app.pids.front(), exe);
    params.ProcessLoopbackParams.ProcessLoopbackMode = PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE;
    PROPVARIANT pv{};
    pv.vt = VT_BLOB;
    pv.blob.cbSize = sizeof(params);
    pv.blob.pBlobData = reinterpret_cast<BYTE*>(&params);
    auto handler = Microsoft::WRL::Make<ActivationHandler>();
    ComPtr<IActivateAudioInterfaceAsyncOperation> op;
    HRESULT hr = ActivateAudioInterfaceAsync(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, __uuidof(IAudioClient), &pv, handler.Get(), &op);
    if (FAILED(hr) || !handler->wait(3000) || FAILED(handler->result_) || !handler->client_) {
        err = "Capturing this app needs Windows 10 version 2004 or newer";
        return nullptr;
    }
    WAVEFORMATEX fmt{};
    fmt.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
    fmt.nChannels = 2;
    fmt.nSamplesPerSec = static_cast<DWORD>(engineRate);
    fmt.wBitsPerSample = 32;
    fmt.nBlockAlign = 8;
    fmt.nAvgBytesPerSec = fmt.nSamplesPerSec * fmt.nBlockAlign;
    IAudioClient* client = handler->client_.Detach();
    auto s = std::make_unique<WasapiStream>(client, true, n.name);
    if (!s->initShared(20.0, true, &fmt, err)) return nullptr;
    TS_INFO("Capturing %s (pid %u, process tree)", exe.c_str(), params.ProcessLoopbackParams.TargetProcessId);
    return s;
}

void AppAudio::handleRpc(Engine& e, const std::string& method, const nlohmann::json& p) {
    const std::string exe = p.value("exe", "");
    for (auto& [id, n] : e.nodes_) {
        if (n.kind != DeviceKind::App || n.app.exe != exe) continue;
        if (method == "app.setMode" && n.bin == Bin::AppOut) {
            const std::string mode = p.value("mode", "copy");
            if (mode != "off" && mode != "copy" && mode != "isolate") throw std::runtime_error("Unknown capture mode");
            if (n.app.mode != mode) {
                n.app.mode = mode;
                n.streamError.clear();
                if (n.stream) e.closeStreamLocked(n);
            }
        }
        if (method == "app.setRule" && p.contains("pinned")) {
            n.app.pinned = p["pinned"].get<bool>();
            n.persistent = n.app.pinned;
        }
    }
}

}  // namespace ts
