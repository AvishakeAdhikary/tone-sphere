#include "plugins/vst3_host.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "pluginterfaces/gui/iplugview.h"
#include "pluginterfaces/vst/ivstaudioprocessor.h"
#include "pluginterfaces/vst/ivsteditcontroller.h"
#include "pluginterfaces/vst/ivstprocesscontext.h"
#include "pluginterfaces/vst/vstspeaker.h"
#include "public.sdk/source/common/memorystream.h"
#include "public.sdk/source/vst/hosting/hostclasses.h"
#include "public.sdk/source/vst/hosting/module.h"
#include "public.sdk/source/vst/hosting/parameterchanges.h"
#include "public.sdk/source/vst/hosting/plugprovider.h"
#include "public.sdk/source/vst/hosting/processdata.h"
#include "util/encoding.h"
#include "util/log.h"
#include "util/platform.h"
#include "util/ui_thread.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <cstdio>
#endif

namespace fs = std::filesystem;
using nlohmann::json;
using namespace Steinberg;
using namespace Steinberg::Vst;

namespace ts {

namespace {

constexpr const char* kEffectCategory = "Audio Module Class";

// ---------------------------------------------------------------- host context + modules
FUnknown* hostContext() {
    static IPtr<HostApplication> app = [] {
        IPtr<HostApplication> a = owned(new HostApplication());
        PluginContextFactory::instance().setPluginContext(a);
        return a;
    }();
    return app;
}

// Loaded modules stay loaded for the life of the engine (unloading plugin DLLs with
// lingering threads is a classic crash source). Accessed on the main thread only.
VST3::Hosting::Module::Ptr loadModule(const std::string& path) {
    static std::map<std::string, VST3::Hosting::Module::Ptr> modules;
    auto it = modules.find(path);
    if (it != modules.end()) return it->second;
    std::string err;
    auto m = VST3::Hosting::Module::create(path, err);
    if (!m) throw std::runtime_error("Cannot load the plugin: " + (err.empty() ? path : err));
    modules[path] = m;
    return m;
}

std::string blobPack(const std::string& a, const std::string& b) {
    std::string out = "NT3S";
    auto put = [&](const std::string& s) {
        const uint32_t n = static_cast<uint32_t>(s.size());
        out.append(reinterpret_cast<const char*>(&n), 4);
        out += s;
    };
    put(a);
    put(b);
    return base64Encode(out);
}

bool blobUnpack(const std::string& b64, std::string& a, std::string& b) {
    const std::string raw = base64Decode(b64);
    if (raw.size() < 12 || raw.compare(0, 4, "NT3S") != 0) return false;
    size_t pos = 4;
    auto get = [&](std::string& s) {
        if (pos + 4 > raw.size()) return false;
        uint32_t n;
        std::memcpy(&n, raw.data() + pos, 4);
        pos += 4;
        if (pos + n > raw.size()) return false;
        s.assign(raw.data() + pos, n);
        pos += n;
        return true;
    };
    return get(a) && get(b);
}

std::string streamBytes(MemoryStream& s) {
    return std::string(s.getData(), static_cast<size_t>(s.getSize()));
}

#ifdef _MSC_VER
bool safeProcess(IAudioProcessor* p, ProcessData* d) {
    __try {
        p->process(*d);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}
#else
bool safeProcess(IAudioProcessor* p, ProcessData* d) {
    p->process(*d);
    return true;
}
#endif

class Vst3Fx;

// ---------------------------------------------------------------- component handler
class Handler : public IComponentHandler {
public:
    explicit Handler(Vst3Fx& fx) : fx_(fx) {}
    tresult PLUGIN_API queryInterface(const TUID iid, void** obj) override {
        if (FUnknownPrivate::iidEqual(iid, FUnknown::iid) || FUnknownPrivate::iidEqual(iid, IComponentHandler::iid)) {
            *obj = static_cast<IComponentHandler*>(this);
            return kResultOk;
        }
        *obj = nullptr;
        return kNoInterface;
    }
    uint32 PLUGIN_API addRef() override { return 1; }   // lifetime owned by Vst3Fx
    uint32 PLUGIN_API release() override { return 1; }
    tresult PLUGIN_API beginEdit(ParamID) override { return kResultOk; }
    tresult PLUGIN_API performEdit(ParamID id, ParamValue v) override;
    tresult PLUGIN_API endEdit(ParamID) override { return kResultOk; }
    tresult PLUGIN_API restartComponent(int32 flags) override;

private:
    Vst3Fx& fx_;
};

#ifdef _WIN32
// ---------------------------------------------------------------- editor window
class EditorWindow : public IPlugFrame {
public:
    EditorWindow(Vst3Fx& fx, IPtr<IPlugView> view) : fx_(fx), view_(std::move(view)) {}
    bool open(const std::wstring& title);
    void close();
    void raise();
    HWND hwnd() const { return hwnd_; }

    tresult PLUGIN_API queryInterface(const TUID iid, void** obj) override {
        if (FUnknownPrivate::iidEqual(iid, FUnknown::iid) || FUnknownPrivate::iidEqual(iid, IPlugFrame::iid)) {
            *obj = static_cast<IPlugFrame*>(this);
            return kResultOk;
        }
        *obj = nullptr;
        return kNoInterface;
    }
    uint32 PLUGIN_API addRef() override { return 1; }
    uint32 PLUGIN_API release() override { return 1; }
    tresult PLUGIN_API resizeView(IPlugView* view, ViewRect* r) override {
        if (!hwnd_ || !r) return kInvalidArgument;
        resizeClient(r->getWidth(), r->getHeight());
        view->onSize(r);
        return kResultTrue;
    }

private:
    static LRESULT CALLBACK proc(HWND h, UINT msg, WPARAM wp, LPARAM lp);
    void resizeClient(int w, int h);
    Vst3Fx& fx_;
    IPtr<IPlugView> view_;
    HWND hwnd_ = nullptr;
    bool resizable_ = false;
    bool inResize_ = false;
};
#endif

// ---------------------------------------------------------------- the effect
class Vst3Fx final : public Fx {
public:
    Vst3Fx(const std::string& name, IPtr<PlugProvider> provider)
        : Fx("vst3", name, {}), provider_(std::move(provider)), handler_(*this) {
        component_ = provider_->getComponentPtr();
        controller_ = provider_->getControllerPtr();
        processor_ = FUnknownPtr<IAudioProcessor>(component_);
        if (!processor_) throw std::runtime_error("This plugin is not an audio effect");
        if (processor_->canProcessSampleSize(kSample32) != kResultTrue) throw std::runtime_error("This plugin does not support 32-bit audio");
        if (controller_) controller_->setComponentHandler(&handler_);
        const int32 nParams = controller_ ? std::max<int32>(controller_->getParameterCount(), 16) : 16;
        inChanges_.setMaxParameters(nParams);
        outChanges_.setMaxParameters(nParams);
        toAudio_.setMaxParameters(nParams * 4);
        toUi_.setMaxParameters(nParams * 4);
    }

    ~Vst3Fx() override {
        ui::run([this] {
#ifdef _WIN32
            if (editor_) {
                editor_->close();
                editor_.reset();
            }
#endif
            deactivate();
            if (controller_) controller_->setComponentHandler(nullptr);
            processor_ = nullptr;
            controller_ = nullptr;
            component_ = nullptr;
            provider_ = nullptr;  // terminates the plugin
        });
    }

    void prepare(double sampleRate, int maxBlock, int channels) override {
        Fx::prepare(sampleRate, maxBlock, channels);
        ui::run([&] {
            deactivate();
            negotiateBuses(channels);
            ProcessSetup setup{kRealtime, kSample32, maxBlock, sampleRate};
            if (processor_->setupProcessing(setup) != kResultOk) TS_WARN("Plugin %s rejected the processing setup", name().c_str());
            data_.unprepare();
            data_.prepare(*component_, maxBlock, kSample32);
            data_.processMode = kRealtime;
            if (component_->setActive(true) != kResultOk) {
                TS_WARN("Plugin %s could not be activated", name().c_str());
                return;
            }
            processor_->setProcessing(true);
            active_ = true;
        });
    }

    void process(float* const* buf, int channels, int frames) override {
        if (!active_ || failed_.load(std::memory_order_relaxed) || frames > maxBlock_) return;
        // node -> plugin main input
        if (pin_ > 0 && data_.numInputs > 0) {
            float** in = data_.inputs[0].channelBuffers32;
            for (int c = 0; c < pin_; ++c) {
                const int src = channels == 1 ? 0 : c;
                if (src < channels) std::memcpy(in[c], buf[src], sizeof(float) * frames);
                else std::memset(in[c], 0, sizeof(float) * frames);
            }
        }
        data_.numSamples = frames;
        toAudio_.transferChangesTo(inChanges_);
        data_.inputParameterChanges = &inChanges_;
        data_.outputParameterChanges = &outChanges_;
        ctx_.state = ProcessContext::kPlaying | ProcessContext::kTempoValid | ProcessContext::kTimeSigValid |
                     ProcessContext::kProjectTimeMusicValid | ProcessContext::kContTimeValid;
        ctx_.sampleRate = sr_;
        ctx_.tempo = 120.0;
        ctx_.timeSigNumerator = 4;
        ctx_.timeSigDenominator = 4;
        ctx_.projectTimeSamples = static_cast<TSamples>(samplePos_);
        ctx_.continousTimeSamples = static_cast<TSamples>(samplePos_);
        ctx_.projectTimeMusic = static_cast<double>(samplePos_) / sr_ * (ctx_.tempo / 60.0);
        data_.processContext = &ctx_;
        if (!safeProcess(processor_.get(), &data_)) {
            failed_ = true;  // keep the engine alive; the effect is switched off
            return;
        }
        samplePos_ += frames;
        toUi_.transferChangesFrom(outChanges_);
        inChanges_.clearQueue();
        outChanges_.clearQueue();
        // plugin main output -> node (extra node channels pass through)
        if (pout_ > 0 && data_.numOutputs > 0) {
            float** out = data_.outputs[0].channelBuffers32;
            if (channels == 1 && pout_ >= 2) {
                for (int i = 0; i < frames; ++i) buf[0][i] = 0.5f * (out[0][i] + out[1][i]);
            } else {
                for (int c = 0; c < std::min(channels, pout_); ++c) std::memcpy(buf[c], out[c], sizeof(float) * frames);
            }
        }
    }

    bool hasEditor() const override { return controller_ != nullptr; }

    void openEditor() override {
#ifdef _WIN32
        ui::run([this] {
            if (editor_) {
                editor_->raise();
                return;
            }
            if (!controller_) throw std::runtime_error("This plugin has no window");
            IPtr<IPlugView> view = owned(controller_->createView(ViewType::kEditor));
            if (!view || view->isPlatformTypeSupported(kPlatformTypeHWND) != kResultTrue)
                throw std::runtime_error("This plugin has no window");
            auto w = std::make_unique<EditorWindow>(*this, view);
            if (!w->open(toWide(name()))) throw std::runtime_error("The plugin window could not be opened");
            editor_ = std::move(w);
        });
#else
        throw std::runtime_error("Plugin windows are not supported on this platform yet");
#endif
    }

    std::string saveState() const override {
        std::string out;
        ui::run([&] {
            MemoryStream cs, es;
            if (component_->getState(&cs) != kResultOk) return;
            if (controller_) controller_->getState(&es);
            out = blobPack(streamBytes(cs), streamBytes(es));
        });
        return out;
    }

    void loadState(const std::string& b64) override {
        std::string a, b;
        if (b64.empty() || !blobUnpack(b64, a, b)) return;
        ui::run([&] {
            MemoryStream cs(a.data(), static_cast<TSize>(a.size()));
            component_->setState(&cs);
            if (controller_) {
                int64 pos = 0;
                cs.seek(0, IBStream::kIBSeekSet, &pos);
                controller_->setComponentState(&cs);
                if (!b.empty()) {
                    MemoryStream es(b.data(), static_cast<TSize>(b.size()));
                    controller_->setState(&es);
                }
            }
        });
    }

    std::string status() const override {
        return failed_ ? "This plugin stopped after an internal error. Remove it and add it again." : std::string();
    }

    // UI thread: forwards editor edits to the audio thread.
    void editFromUi(ParamID id, ParamValue v) { toAudio_.addChange(id, v, 0); }
    // UI thread (editor timer): output parameter values (meters) -> controller.
    void drainToUi() {
        ParamID id;
        ParamValue v;
        int32 off;
        while (toUi_.getNextChange(id, v, off))
            if (controller_) controller_->setParamNormalized(id, v);
    }
#ifdef _WIN32
    void editorClosed() { editor_.reset(); }
#endif

private:
    void deactivate() {
        if (!active_) return;
        processor_->setProcessing(false);
        component_->setActive(false);
        active_ = false;
    }

    void negotiateBuses(int channels) {
        const int32 nIn = component_->getBusCount(kAudio, kInput);
        const int32 nOut = component_->getBusCount(kAudio, kOutput);
        std::vector<SpeakerArrangement> ins(nIn), outs(nOut);
        for (int32 i = 0; i < nIn; ++i) processor_->getBusArrangement(kInput, i, ins[i]);
        for (int32 i = 0; i < nOut; ++i) processor_->getBusArrangement(kOutput, i, outs[i]);
        auto tryArr = [&](SpeakerArrangement a) {
            auto i2 = ins;
            auto o2 = outs;
            if (nIn > 0) i2[0] = a;
            if (nOut > 0) o2[0] = a;
            return processor_->setBusArrangements(i2.data(), nIn, o2.data(), nOut) == kResultTrue;
        };
        if (!(channels == 1 && tryArr(SpeakerArr::kMono))) tryArr(SpeakerArr::kStereo);
        if (nIn > 0) component_->activateBus(kAudio, kInput, 0, true);
        if (nOut > 0) component_->activateBus(kAudio, kOutput, 0, true);
        SpeakerArrangement a = 0;
        pin_ = (nIn > 0 && processor_->getBusArrangement(kInput, 0, a) == kResultOk) ? SpeakerArr::getChannelCount(a) : 0;
        a = 0;
        pout_ = (nOut > 0 && processor_->getBusArrangement(kOutput, 0, a) == kResultOk) ? SpeakerArr::getChannelCount(a) : 0;
    }

    IPtr<PlugProvider> provider_;
    IPtr<IComponent> component_;
    IPtr<IEditController> controller_;
    IPtr<IAudioProcessor> processor_;
    Handler handler_;
    HostProcessData data_;
    ParameterChanges inChanges_, outChanges_;
    ParameterChangeTransfer toAudio_, toUi_;
    ProcessContext ctx_{};
    int64_t samplePos_ = 0;
    int pin_ = 0, pout_ = 0;
    bool active_ = false;
    std::atomic<bool> failed_{false};
#ifdef _WIN32
    std::unique_ptr<EditorWindow> editor_;
#endif
};

tresult PLUGIN_API Handler::performEdit(ParamID id, ParamValue v) {
    fx_.editFromUi(id, v);
    return kResultOk;
}
tresult PLUGIN_API Handler::restartComponent(int32 flags) {
    if (flags & (kIoChanged | kLatencyChanged)) TS_INFO("Plugin %s requested a restart (flags 0x%x)", fx_.name().c_str(), flags);
    return kResultOk;
}

#ifdef _WIN32
bool EditorWindow::open(const std::wstring& title) {
    static const wchar_t* kClass = L"ToneSpherePluginWindow";
    static bool registered = [] {
        WNDCLASSEXW wc{sizeof(wc)};
        wc.lpfnWndProc = proc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        wc.hIcon = LoadIconW(GetModuleHandleW(nullptr), MAKEINTRESOURCEW(1));
        wc.hbrBackground = reinterpret_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
        wc.lpszClassName = kClass;
        return RegisterClassExW(&wc) != 0;
    }();
    (void)registered;
    resizable_ = view_->canResize() == kResultTrue;
    const DWORD style = resizable_ ? WS_OVERLAPPEDWINDOW : (WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX);
    ViewRect r{0, 0, 600, 400};
    view_->getSize(&r);
    RECT wr{0, 0, std::max<int32>(r.getWidth(), 50), std::max<int32>(r.getHeight(), 30)};
    AdjustWindowRectEx(&wr, style, FALSE, 0);
    hwnd_ = CreateWindowExW(WS_EX_APPWINDOW, kClass, title.c_str(), style, CW_USEDEFAULT, CW_USEDEFAULT, wr.right - wr.left,
                            wr.bottom - wr.top, nullptr, nullptr, GetModuleHandleW(nullptr), this);
    if (!hwnd_) return false;
    view_->setFrame(this);
    if (view_->attached(hwnd_, kPlatformTypeHWND) != kResultOk) {
        view_->setFrame(nullptr);
        DestroyWindow(hwnd_);
        hwnd_ = nullptr;
        return false;
    }
    // Some plugins only know their final size once attached.
    ViewRect r2 = r;
    if (view_->getSize(&r2) == kResultOk && (r2.getWidth() != r.getWidth() || r2.getHeight() != r.getHeight()))
        resizeClient(r2.getWidth(), r2.getHeight());
    SetTimer(hwnd_, 1, 33, nullptr);
    ShowWindow(hwnd_, SW_SHOWNORMAL);
    raise();
    return true;
}

void EditorWindow::raise() {
    if (!hwnd_) return;
    if (IsIconic(hwnd_)) ShowWindow(hwnd_, SW_RESTORE);
    // The engine is a background process, so plain SetForegroundWindow is usually refused:
    // briefly make the window topmost so it appears above the ToneSphere window.
    SetWindowPos(hwnd_, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
    SetWindowPos(hwnd_, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
    SetForegroundWindow(hwnd_);
}

void EditorWindow::close() {
    if (!hwnd_) return;
    KillTimer(hwnd_, 1);
    view_->removed();
    view_->setFrame(nullptr);
    const HWND h = hwnd_;
    hwnd_ = nullptr;
    SetWindowLongPtrW(h, GWLP_USERDATA, 0);
    DestroyWindow(h);
    view_ = nullptr;
}

void EditorWindow::resizeClient(int w, int h) {
    const DWORD style = static_cast<DWORD>(GetWindowLongPtrW(hwnd_, GWL_STYLE));
    RECT wr{0, 0, w, h};
    AdjustWindowRectEx(&wr, style, FALSE, 0);
    inResize_ = true;
    SetWindowPos(hwnd_, nullptr, 0, 0, wr.right - wr.left, wr.bottom - wr.top, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
    inResize_ = false;
}

LRESULT CALLBACK EditorWindow::proc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_NCCREATE) SetWindowLongPtrW(h, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(reinterpret_cast<CREATESTRUCTW*>(lp)->lpCreateParams));
    auto* self = reinterpret_cast<EditorWindow*>(GetWindowLongPtrW(h, GWLP_USERDATA));
    if (self && self->hwnd_ == h) {
        switch (msg) {
        case WM_TIMER:
            self->fx_.drainToUi();
            return 0;
        case WM_SIZE:
            if (self->resizable_ && !self->inResize_ && self->view_ && wp != SIZE_MINIMIZED) {
                ViewRect r{0, 0, LOWORD(lp), HIWORD(lp)};
                if (self->view_->checkSizeConstraint(&r) == kResultTrue &&
                    (r.getWidth() != LOWORD(lp) || r.getHeight() != HIWORD(lp)))
                    self->resizeClient(r.getWidth(), r.getHeight());
                self->view_->onSize(&r);
            }
            return 0;
        case WM_CLOSE:
            self->close();
            self->fx_.editorClosed();  // deletes self
            return 0;
        default:
            break;
        }
    }
    return DefWindowProcW(h, msg, wp, lp);
}
#endif

// ---------------------------------------------------------------- missing plugin placeholder
class MissingPluginFx final : public Fx {
public:
    MissingPluginFx(const std::string& name, std::string state) : Fx("vst3", name, {}), state_(std::move(state)) {}
    void process(float* const*, int, int) override {}
    std::string saveState() const override { return state_; }
    std::string status() const override { return "This plugin is not installed on this PC, so it is skipped."; }

private:
    std::string state_;
};

// ---------------------------------------------------------------- scanning helpers
std::string u8(const fs::path& p) {
    const auto s = p.u8string();
    return std::string(reinterpret_cast<const char*>(s.data()), s.size());
}

int64_t modTime(const std::string& path) {
    std::error_code ec;
    int64_t t = 0;
    auto consider = [&](const fs::path& p) {
        const auto w = fs::last_write_time(p, ec);
        if (!ec) t = std::max<int64_t>(t, static_cast<int64_t>(w.time_since_epoch().count()));
    };
    const fs::path p = fs::u8path(path);
    consider(p);
    if (fs::is_directory(p, ec)) {
        for (auto& e : fs::recursive_directory_iterator(p / "Contents", fs::directory_options::skip_permission_denied, ec)) {
            if (e.is_regular_file(ec)) consider(e.path());
        }
    }
    return t;
}

void findModulesIn(const fs::path& folder, std::vector<std::string>& out) {
    std::error_code ec;
    if (!fs::is_directory(folder, ec)) return;
    for (auto it = fs::recursive_directory_iterator(folder, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        const fs::path& p = it->path();
        std::string ext = u8(p.extension());
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".vst3") {
            out.push_back(u8(p));
            if (it->is_directory(ec)) it.disable_recursion_pending();
        }
    }
}

// Runs the scanner on one module and returns its JSON report (or an error report).
json runScanner(const std::string& scanner, const std::string& module) {
#ifdef _WIN32
    const std::string outFile = dataDir() + "\\plugscan-" + randomHex(4) + ".json";
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE hOut = CreateFileW(toWide(outFile).c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr);
    if (hOut == INVALID_HANDLE_VALUE) return {{"ok", false}, {"error", "Cannot start the plugin scanner"}};
    STARTUPINFOW si{sizeof(si)};
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = hOut;
    si.hStdError = hOut;
    si.hStdInput = nullptr;
    PROCESS_INFORMATION pi{};
    std::wstring cmd = L"\"" + toWide(scanner) + L"\" \"" + toWide(module) + L"\"";
    const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(hOut);
    json r;
    if (!ok) {
        r = {{"ok", false}, {"error", "Cannot start the plugin scanner"}};
    } else {
        CloseHandle(pi.hThread);
        const DWORD w = WaitForSingleObject(pi.hProcess, 60000);
        DWORD code = 0;
        if (w != WAIT_OBJECT_0) {
            TerminateProcess(pi.hProcess, 1);
            r = {{"ok", false}, {"error", "The plugin did not respond while loading"}};
        } else {
            GetExitCodeProcess(pi.hProcess, &code);
        }
        CloseHandle(pi.hProcess);
        if (r.is_null()) {
            std::string text;
            readFile(outFile, text);
            const auto brace = text.rfind("\n{");
            try {
                r = json::parse(brace == std::string::npos ? text : text.substr(brace + 1));
            } catch (...) {
                r = {{"ok", false}, {"error", code ? "The plugin crashed while loading" : "The plugin could not be read"}};
            }
        }
    }
    DeleteFileW(toWide(outFile).c_str());
    return r;
#else
    std::string cmd = "\"" + scanner + "\" \"" + module + "\" 2>/dev/null";
    FILE* f = popen(cmd.c_str(), "r");
    if (!f) return {{"ok", false}, {"error", "Cannot start the plugin scanner"}};
    std::string text;
    char b[4096];
    size_t n;
    while ((n = fread(b, 1, sizeof(b), f)) > 0) text.append(b, n);
    const int rc = pclose(f);
    try {
        const auto brace = text.rfind("\n{");
        return json::parse(brace == std::string::npos ? text : text.substr(brace + 1));
    } catch (...) {
        return {{"ok", false}, {"error", rc ? "The plugin crashed while loading" : "The plugin could not be read"}};
    }
#endif
}

}  // namespace

// ================================================================ public helpers
json describeVst3Module(const std::string& path) {
    std::string err;
    auto m = VST3::Hosting::Module::create(path, err);
    if (!m) return {{"ok", false}, {"error", err.empty() ? "Not a VST3 plugin" : err}};
    json classes = json::array();
    for (const auto& ci : m->getFactory().classInfos()) {
        if (ci.category() != kEffectCategory) continue;
        classes.push_back({{"id", ci.ID().toString()},
                           {"name", ci.name()},
                           {"vendor", ci.vendor()},
                           {"version", ci.version()},
                           {"sdk", ci.sdkVersion()},
                           {"subCategories", ci.subCategoriesString()}});
    }
    return {{"ok", true}, {"classes", classes}};
}

std::vector<std::string> standardVst3Modules(const std::vector<std::string>& extraFolders) {
    std::vector<std::string> folders;
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    if (GetEnvironmentVariableW(L"CommonProgramFiles", buf, MAX_PATH)) folders.push_back(toUtf8(buf) + "\\VST3");
    if (GetEnvironmentVariableW(L"LOCALAPPDATA", buf, MAX_PATH)) folders.push_back(toUtf8(buf) + "\\Programs\\Common\\VST3");
#elif defined(__APPLE__)
    if (const char* h = std::getenv("HOME")) folders.push_back(std::string(h) + "/Library/Audio/Plug-Ins/VST3");
    folders.push_back("/Library/Audio/Plug-Ins/VST3");
#else
    if (const char* h = std::getenv("HOME")) folders.push_back(std::string(h) + "/.vst3");
    folders.push_back("/usr/lib/vst3");
    folders.push_back("/usr/local/lib/vst3");
#endif
    folders.insert(folders.end(), extraFolders.begin(), extraFolders.end());
    std::vector<std::string> out;
    for (auto& f : folders) findModulesIn(fs::u8path(f), out);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::shared_ptr<Fx> createVst3Fx(const std::string& modulePath, const std::string& classId) {
    std::shared_ptr<Fx> fx;
    ui::run([&] {
        hostContext();
        auto mod = loadModule(modulePath);
        const auto& factory = mod->getFactory();
        for (const auto& ci : factory.classInfos()) {
            if (ci.ID().toString() != classId) continue;
            IPtr<PlugProvider> prov = owned(new PlugProvider(factory, ci, true));
            if (!prov->initialize()) throw std::runtime_error("The plugin failed to start");
            auto v = std::make_shared<Vst3Fx>(ci.name(), prov);
            v->pluginId = classId;
            fx = v;
            return;
        }
        throw std::runtime_error("The plugin no longer contains this effect");
    });
    return fx;
}

std::shared_ptr<Fx> createMissingPluginFx(const std::string& pluginId, const std::string& name, const std::string& state) {
    auto fx = std::make_shared<MissingPluginFx>(name.empty() ? "Missing plugin" : name, state);
    fx->pluginId = pluginId;
    return fx;
}

// ================================================================ PluginManager
PluginManager::PluginManager(std::string cacheFile, std::string scannerExe)
    : cacheFile_(std::move(cacheFile)), scannerExe_(std::move(scannerExe)) {
    cache_ = {{"modules", json::object()}};
}

PluginManager::~PluginManager() {
    stop_ = true;
    if (scan_.joinable()) scan_.join();
}

void PluginManager::init() {
    std::string text;
    bool have = false;
    if (readFile(cacheFile_, text)) {
        try {
            json j = json::parse(text);
            if (j.value("version", 0) == 1 && j.contains("modules")) {
                std::lock_guard lk(mu_);
                cache_ = j;
                rebuildListLocked();
                have = true;
            }
        } catch (...) {
        }
    }
    if (!have) scanAsync();
}

void PluginManager::setExtraFolders(std::vector<std::string> folders) {
    std::lock_guard lk(mu_);
    extra_ = std::move(folders);
}

std::vector<std::string> PluginManager::searchFolders() const {
    std::lock_guard lk(mu_);
    return extra_;
}

void PluginManager::scanAsync() {
    if (scanning_.exchange(true)) return;
    if (scan_.joinable()) scan_.join();
    scan_ = std::thread([this] { scanThread(); });
}

void PluginManager::scanThread() {
    setThreadName("ts-plugscan");
    std::vector<std::string> extra;
    json modules;
    {
        std::lock_guard lk(mu_);
        extra = extra_;
        modules = cache_["modules"];
    }
    const auto paths = standardVst3Modules(extra);
    json next = json::object();
    int scanned = 0;
    for (const auto& p : paths) {
        if (stop_) break;
        const int64_t mt = modTime(p);
        if (modules.contains(p) && modules[p].value("mtime", int64_t(-1)) == mt) {
            next[p] = modules[p];
            continue;
        }
        TS_INFO("Scanning plugin %s", p.c_str());
        json r = runScanner(scannerExe_, p);
        r["mtime"] = mt;
        if (!r.value("ok", false)) TS_WARN("Plugin %s: %s", p.c_str(), r.value("error", "error").c_str());
        next[p] = r;
        ++scanned;
    }
    {
        std::lock_guard lk(mu_);
        cache_ = {{"version", 1}, {"modules", next}};
        rebuildListLocked();
        writeFileAtomic(cacheFile_, cache_.dump(1));
    }
    TS_INFO("Plugin scan finished: %zu modules (%d scanned)", paths.size(), scanned);
    scanning_ = false;
    if (onChanged) onChanged();
}

void PluginManager::rebuildListLocked() {
    list_.clear();
    std::map<std::string, bool> seen;
    for (auto& [path, m] : cache_["modules"].items()) {
        if (!m.value("ok", false)) continue;
        for (auto& c : m.value("classes", json::array())) {
            const std::string sub = c.value("subCategories", "");
            // ToneSphere inserts process audio: skip instruments/generators without an Fx role.
            if (sub.find("Instrument") != std::string::npos && sub.find("Fx") == std::string::npos) continue;
            const std::string id = c.value("id", "");
            if (id.empty() || seen[id]) continue;
            seen[id] = true;
            std::string cat = sub;
            if (cat.rfind("Fx|", 0) == 0) cat = cat.substr(3);
            if (cat == "Fx") cat = "Effect";
            list_.push_back({id, c.value("name", "Plugin"), c.value("vendor", ""), cat, c.value("version", ""), path});
        }
    }
    std::sort(list_.begin(), list_.end(), [](auto& a, auto& b) { return a.name < b.name; });
}

std::vector<PluginEntry> PluginManager::list() const {
    std::lock_guard lk(mu_);
    return list_;
}

json PluginManager::listJson() const {
    json a = json::array();
    for (auto& p : list())
        a.push_back({{"id", p.id}, {"name", p.name}, {"vendor", p.vendor}, {"category", p.category}, {"path", p.path}});
    return a;
}

std::shared_ptr<Fx> PluginManager::create(const std::string& pluginId) {
    std::string path;
    for (auto& p : list())
        if (p.id == pluginId) path = p.path;
    if (path.empty()) throw std::runtime_error("This plugin is not installed");
    return createVst3Fx(path, pluginId);
}

}  // namespace ts
