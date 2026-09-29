// VST3 hosting on the Steinberg VST3 SDK (MIT since 3.8), using its own hosting helpers:
// VST3::Hosting::Module to load a module, PlugProvider to create, connect and initialise
// component and controller exactly as the SDK's validator does.
//
// Threads. Every non-real-time call into a plugin runs on one plugin thread, which is
// OLE-initialised and pumps messages, because VST3 controllers expect a single UI thread
// and editors need a message loop. process() runs on the engine's audio thread, through
// Vst3Processor below. Parameter changes cross from the plugin thread to the audio thread
// in a wait-free queue and reach the plugin as fixed-capacity IParameterChanges, so the
// audio thread never allocates on the plugin's behalf.
//
// Faults. Each call into plugin code goes through seh_invoke. A structured exception
// (access violation, a C++ exception escaping the plugin) marks the instance crashed; it
// is never called again, and on the audio thread it is bypassed from then on. Its module
// is deliberately leaked rather than unloaded: code that has just faulted cannot be
// trusted to shut down.
#include "vst3_host.h"

#include <windows.h>
#include <objbase.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "dsp.h"
#include "spsc.h"
#include "tonesphere_native.h"

#include "pluginterfaces/base/funknownimpl.h"
#include "pluginterfaces/gui/iplugview.h"
#include "pluginterfaces/vst/ivstaudioprocessor.h"
#include "pluginterfaces/vst/ivstcomponent.h"
#include "pluginterfaces/vst/ivsteditcontroller.h"
#include "pluginterfaces/vst/ivstparameterchanges.h"
#include "pluginterfaces/vst/ivstprocesscontext.h"
#include "pluginterfaces/vst/vstspeaker.h"
#include "public.sdk/source/common/memorystream.h"
#include "public.sdk/source/vst/hosting/hostclasses.h"
#include "public.sdk/source/vst/hosting/module.h"
#include "public.sdk/source/vst/hosting/plugprovider.h"
#include "public.sdk/source/vst/utility/stringconvert.h"

using namespace Steinberg;
using namespace Steinberg::Vst;

namespace {

thread_local std::string t_error;

// Structured exception boundary. Kept free of C++ objects with destructors, as MSVC
// requires of a function containing __try.
bool seh_invoke(void (*fn)(void*), void* context, DWORD* code) {
    __try {
        fn(context);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *code = GetExceptionCode();
        return false;
    }
}

template <typename F>
bool guarded(F&& f, DWORD& code) {
    return seh_invoke([](void* p) { (*static_cast<F*>(p))(); }, &f, &code);
}

std::string fault_text(const char* what, DWORD code) {
    char buffer[160];
    const char* kind = code == EXCEPTION_ACCESS_VIOLATION ? "access violation"
                     : code == 0xE06D7363u               ? "C++ exception"
                     : code == EXCEPTION_STACK_OVERFLOW  ? "stack overflow"
                                                         : "structured exception";
    std::snprintf(buffer, sizeof(buffer), "plugin crashed during %s (%s, 0x%08lX)", what, kind,
                  static_cast<unsigned long>(code));
    return buffer;
}

void copy_str(char* dst, size_t capacity, const std::string& s) { std::snprintf(dst, capacity, "%s", s.c_str()); }

// ---- The plugin thread -----------------------------------------------------------------

class PluginThread {
public:
    PluginThread() {
        wake_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        HANDLE ready = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        thread_ = std::thread([this, ready] {
            OleInitialize(nullptr);
            id_ = GetCurrentThreadId();
            SetEvent(ready);
            for (;;) {
                MsgWaitForMultipleObjects(1, &wake_, FALSE, INFINITE, QS_ALLINPUT);
                MSG msg;
                while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                    TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
                std::deque<std::function<void()>> work;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    work.swap(queue_);
                }
                for (auto& fn : work) fn();
                if (quit_) break;
            }
            OleUninitialize();
        });
        WaitForSingleObject(ready, INFINITE);
        CloseHandle(ready);
    }

    // Run on the plugin thread and wait. Re-entrant: already on it, just run.
    void call(const std::function<void()>& fn) {
        if (GetCurrentThreadId() == id_) {
            fn();
            return;
        }
        HANDLE done = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back([&fn, done] {
                fn();
                SetEvent(done);
            });
        }
        SetEvent(wake_);
        WaitForSingleObject(done, INFINITE);
        CloseHandle(done);
    }

private:
    std::thread thread_;
    DWORD id_ = 0;
    HANDLE wake_ = nullptr;
    std::mutex mutex_;
    std::deque<std::function<void()>> queue_;
    bool quit_ = false;
};

// Deliberately never destroyed: plugin modules may still hold references to the host
// application object and to this thread when the process exits.
PluginThread& plugin_thread() {
    static PluginThread* thread = new PluginThread();
    return *thread;
}

HostApplication& host_application() {
    static HostApplication* host = [] {
        auto* h = new HostApplication();
        PluginContextFactory::instance().setPluginContext(h);
        return h;
    }();
    return *host;
}

// ---- Fixed-capacity parameter changes for the audio thread ------------------------------

class FixedQueue final : public IParamValueQueue {
public:
    static constexpr int32 kPoints = 16;
    void reset(ParamID id) { id_ = id; count_ = 0; }
    ParamID PLUGIN_API getParameterId() override { return id_; }
    int32 PLUGIN_API getPointCount() override { return count_; }
    tresult PLUGIN_API getPoint(int32 index, int32& offset, ParamValue& value) override {
        if (index < 0 || index >= count_) return kResultFalse;
        offset = offsets_[index];
        value = values_[index];
        return kResultTrue;
    }
    tresult PLUGIN_API addPoint(int32 offset, ParamValue value, int32& index) override {
        if (count_ == kPoints) return kResultFalse;
        offsets_[count_] = offset;
        values_[count_] = value;
        index = count_++;
        return kResultTrue;
    }
    tresult PLUGIN_API queryInterface(const TUID _iid, void** obj) override {
        if (FUnknownPrivate::iidEqual(_iid, IParamValueQueue::iid) || FUnknownPrivate::iidEqual(_iid, FUnknown::iid)) {
            *obj = this;
            return kResultOk;
        }
        *obj = nullptr;
        return kNoInterface;
    }
    uint32 PLUGIN_API addRef() override { return 1; }
    uint32 PLUGIN_API release() override { return 1; }

private:
    ParamID id_ = 0;
    int32 count_ = 0;
    int32 offsets_[kPoints] = {};
    ParamValue values_[kPoints] = {};
};

class FixedChanges final : public IParameterChanges {
public:
    static constexpr int32 kQueues = 64;
    void clear() { count_ = 0; }
    int32 PLUGIN_API getParameterCount() override { return count_; }
    IParamValueQueue* PLUGIN_API getParameterData(int32 index) override {
        return index >= 0 && index < count_ ? &queues_[index] : nullptr;
    }
    IParamValueQueue* PLUGIN_API addParameterData(const ParamID& id, int32& index) override {
        for (int32 i = 0; i < count_; ++i)
            if (queues_[i].getParameterId() == id) {
                index = i;
                return &queues_[i];
            }
        if (count_ == kQueues) return nullptr;
        queues_[count_].reset(id);
        index = count_;
        return &queues_[count_++];
    }
    tresult PLUGIN_API queryInterface(const TUID _iid, void** obj) override {
        if (FUnknownPrivate::iidEqual(_iid, IParameterChanges::iid) || FUnknownPrivate::iidEqual(_iid, FUnknown::iid)) {
            *obj = this;
            return kResultOk;
        }
        *obj = nullptr;
        return kNoInterface;
    }
    uint32 PLUGIN_API addRef() override { return 1; }
    uint32 PLUGIN_API release() override { return 1; }

private:
    FixedQueue queues_[kQueues];
    int32 count_ = 0;
};

struct ParamChange {
    ParamID id;
    ParamValue value;
};

// ---- An open plugin ------------------------------------------------------------------------

class Instance;

class ComponentHandler final : public IComponentHandler {
public:
    explicit ComponentHandler(Instance& owner) : owner_(owner) {}
    tresult PLUGIN_API beginEdit(ParamID) override { return kResultOk; }
    tresult PLUGIN_API performEdit(ParamID id, ParamValue value) override;
    tresult PLUGIN_API endEdit(ParamID) override { return kResultOk; }
    tresult PLUGIN_API restartComponent(int32 flags) override;
    tresult PLUGIN_API queryInterface(const TUID _iid, void** obj) override {
        if (FUnknownPrivate::iidEqual(_iid, IComponentHandler::iid) || FUnknownPrivate::iidEqual(_iid, FUnknown::iid)) {
            *obj = this;
            return kResultOk;
        }
        *obj = nullptr;
        return kNoInterface;
    }
    uint32 PLUGIN_API addRef() override { return 1; }
    uint32 PLUGIN_API release() override { return 1; }

private:
    Instance& owner_;
};

class EditorFrame;

class Instance {
public:
    uint32_t handle = 0;
    uint32_t channels = 0;
    uint32_t max_block = 0;
    double sample_rate = 0;
    VST3::Hosting::Module::Ptr module;
    IPtr<PlugProvider> provider;
    IPtr<IComponent> component;
    IPtr<IAudioProcessor> processor;
    IPtr<IEditController> controller;
    ComponentHandler handler{*this};
    bool has_input = false;
    std::atomic<bool> crashed{false};
    std::atomic<uint32_t> restart_flags{0};
    std::atomic<uint32_t> latency{0};
    std::atomic<uint64_t> blocks{0};
    std::atomic<uint32_t> fault_code{0};  // set by the audio thread when process() faults
    std::string fault;  // control side: written before `crashed` is set, or from fault_code
    ts::ItemQueue<ParamChange, 1024> changes;  // producer: plugin thread; consumer: audio thread
    std::unique_ptr<EditorFrame> editor;

    void mark_crashed(const std::string& what) {
        fault = what;
        crashed.store(true, std::memory_order_release);
    }
    ~Instance();
};

tresult ComponentHandler::performEdit(ParamID id, ParamValue value) {
    // The plugin's own editor moved a control: it already updated its controller, so only
    // the audio thread still needs to hear about it.
    owner_.changes.push(ParamChange{id, value});
    return kResultOk;
}

tresult ComponentHandler::restartComponent(int32 flags) {
    owner_.restart_flags.fetch_or(static_cast<uint32_t>(flags));
    if ((flags & kLatencyChanged) && owner_.processor) owner_.latency.store(owner_.processor->getLatencySamples());
    return kResultOk;
}

// ---- Editor window ----------------------------------------------------------------------

LRESULT CALLBACK editor_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);

class EditorFrame final : public IPlugFrame {
public:
    EditorFrame(IPtr<IPlugView> view) : view_(std::move(view)) {}
    ~EditorFrame() { close(); }

    bool open(const std::string& title) {
        static const bool registered = [] {
            WNDCLASSW wc{};
            wc.lpfnWndProc = editor_proc;
            wc.hInstance = GetModuleHandleW(nullptr);
            wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
            wc.lpszClassName = L"ToneSpherePluginEditor";
            return RegisterClassW(&wc) != 0;
        }();
        (void)registered;
        ViewRect size{0, 0, 400, 300};
        view_->getSize(&size);
        RECT r{0, 0, size.getWidth(), size.getHeight()};
        const DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
        AdjustWindowRect(&r, style, FALSE);
        const std::u16string wide = Steinberg::Vst::StringConvert::convert(title);
        window_ = CreateWindowExW(0, L"ToneSpherePluginEditor", reinterpret_cast<const wchar_t*>(wide.c_str()), style,
                                  CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, nullptr, nullptr,
                                  GetModuleHandleW(nullptr), this);
        if (!window_) return false;
        view_->setFrame(this);
        if (view_->attached(window_, kPlatformTypeHWND) != kResultOk) {
            DestroyWindow(window_);
            window_ = nullptr;
            return false;
        }
        ShowWindow(window_, SW_SHOW);
        return true;
    }

    void close() {
        if (!window_) return;
        view_->removed();
        view_->setFrame(nullptr);
        HWND w = window_;
        window_ = nullptr;
        DestroyWindow(w);
    }

    tresult PLUGIN_API resizeView(IPlugView* view, ViewRect* size) override {
        if (!window_ || !size) return kInvalidArgument;
        RECT r{0, 0, size->getWidth(), size->getHeight()};
        AdjustWindowRect(&r, static_cast<DWORD>(GetWindowLongW(window_, GWL_STYLE)), FALSE);
        SetWindowPos(window_, nullptr, 0, 0, r.right - r.left, r.bottom - r.top, SWP_NOMOVE | SWP_NOZORDER);
        return view->onSize(size);
    }
    tresult PLUGIN_API queryInterface(const TUID _iid, void** obj) override {
        if (FUnknownPrivate::iidEqual(_iid, IPlugFrame::iid) || FUnknownPrivate::iidEqual(_iid, FUnknown::iid)) {
            *obj = this;
            return kResultOk;
        }
        *obj = nullptr;
        return kNoInterface;
    }
    uint32 PLUGIN_API addRef() override { return 1; }
    uint32 PLUGIN_API release() override { return 1; }

    bool is_open() const { return window_ != nullptr; }
    void on_closed_by_user() { close(); }

private:
    IPtr<IPlugView> view_;
    HWND window_ = nullptr;
};

LRESULT CALLBACK editor_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_NCCREATE) {
        auto* create = reinterpret_cast<CREATESTRUCTW*>(lp);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(create->lpCreateParams));
    }
    auto* frame = reinterpret_cast<EditorFrame*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (msg == WM_CLOSE && frame) {
        frame->on_closed_by_user();
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

Instance::~Instance() {
    // Runs on the plugin thread (see release_instance). A crashed plugin is abandoned, not
    // shut down: calling back into it is what the crash made unsafe.
    if (crashed.load()) {
        // take() drops each reference without calling Release() on the plugin.
        (void)editor.release();
        (void)provider.take();
        (void)processor.take();
        (void)component.take();
        (void)controller.take();
        new VST3::Hosting::Module::Ptr(std::move(module));  // leaked: never unloaded
        return;
    }
    DWORD code = 0;
    guarded([&] {
        editor.reset();
        if (controller) controller->setComponentHandler(nullptr);
        if (processor) processor->setProcessing(false);
        if (component) component->setActive(false);
        processor = nullptr;
        component = nullptr;
        controller = nullptr;
        provider = nullptr;
    }, code);
    if (code) new VST3::Hosting::Module::Ptr(std::move(module));
}

// ---- Registry ----------------------------------------------------------------------------

std::mutex g_registry_mutex;
std::unordered_map<uint32_t, std::shared_ptr<Instance>> g_instances;
uint32_t g_next_handle = 1;

void release_instance(Instance* instance) {
    plugin_thread().call([instance] { delete instance; });
}

std::shared_ptr<Instance> find(uint32_t handle) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto it = g_instances.find(handle);
    if (it == g_instances.end()) {
        t_error = "no open plugin with handle " + std::to_string(handle);
        return nullptr;
    }
    return it->second;
}

SpeakerArrangement arrangement_for(uint32_t channels) {
    if (channels == 1) return SpeakerArr::kMono;
    if (channels == 2) return SpeakerArr::kStereo;
    SpeakerArrangement a = 0;
    for (uint32_t c = 0; c < channels && c < 64; ++c) a |= 1ull << c;
    return a;
}

}  // namespace

// ---- The audio-thread side ---------------------------------------------------------------

namespace ts {
namespace {

class Vst3Processor final : public Processor {
public:
    Vst3Processor(std::shared_ptr<Instance> instance, uint32_t max_block)
        : Processor(TS_INSERT_VST3, instance->channels, static_cast<uint32_t>(instance->sample_rate)),
          instance_(std::move(instance)),
          scratch_(new float[static_cast<size_t>(channels_) * max_block]()) {
        for (uint32_t c = 0; c < channels_; ++c) out_ptrs_[c] = scratch_.get() + static_cast<size_t>(c) * max_block;
        in_bus_.numChannels = static_cast<int32>(channels_);
        out_bus_.numChannels = static_cast<int32>(channels_);
        out_bus_.channelBuffers32 = out_ptrs_;
        context_.sampleRate = instance_->sample_rate;
        context_.tempo = 120.0;
        context_.timeSigNumerator = 4;
        context_.timeSigDenominator = 4;
        context_.state = ProcessContext::kTempoValid | ProcessContext::kTimeSigValid;
    }
    uint32_t param_count() const override { return 0; }
    uint32_t plugin() const override { return instance_->handle; }

protected:
    void update() noexcept override {}

    void process(float* const* ch, uint32_t frames) noexcept override {
        Instance& p = *instance_;
        if (p.crashed.load(std::memory_order_acquire)) return;  // bypass: the input passes through

        changes_.clear();
        ParamChange change;
        while (p.changes.pop(change)) {
            int32 index = 0;
            if (IParamValueQueue* q = changes_.addParameterData(change.id, index)) q->addPoint(0, change.value, index);
        }
        out_changes_.clear();

        in_bus_.channelBuffers32 = const_cast<float**>(ch);
        ProcessData data;
        data.processMode = kRealtime;
        data.symbolicSampleSize = kSample32;
        data.numSamples = static_cast<int32>(frames);
        data.numInputs = p.has_input ? 1 : 0;
        data.inputs = p.has_input ? &in_bus_ : nullptr;
        data.numOutputs = 1;
        data.outputs = &out_bus_;
        data.inputParameterChanges = &changes_;
        data.outputParameterChanges = &out_changes_;
        data.processContext = &context_;

        DWORD code = 0;
        Context call{p.processor.get(), &data};
        if (!seh_invoke([](void* c) {
                auto* call = static_cast<Context*>(c);
                call->processor->process(*call->data);
            }, &call, &code)) {
            // Bypass from here on. Formatting the reason allocates, so it is deferred to
            // whoever next reads the status on the control side.
            p.fault_code.store(code, std::memory_order_relaxed);
            p.crashed.store(true, std::memory_order_release);
            return;
        }
        for (uint32_t c = 0; c < channels_; ++c) std::memcpy(ch[c], out_ptrs_[c], sizeof(float) * frames);
        context_.projectTimeSamples += frames;
        p.blocks.fetch_add(1, std::memory_order_relaxed);
    }

private:
    struct Context {
        IAudioProcessor* processor;
        ProcessData* data;
    };

    std::shared_ptr<Instance> instance_;
    std::unique_ptr<float[]> scratch_;
    float* out_ptrs_[TS_MAX_CHANNELS] = {};
    AudioBusBuffers in_bus_{};
    AudioBusBuffers out_bus_{};
    FixedChanges changes_;
    FixedChanges out_changes_;
    ProcessContext context_{};
};

}  // namespace

std::unique_ptr<Processor> make_vst3_processor(uint32_t handle, uint32_t channels, uint32_t max_block,
                                               std::string& error) {
    auto instance = find(handle);
    if (!instance) {
        error = t_error;
        return nullptr;
    }
    if (instance->crashed.load()) {
        error = "plugin " + std::to_string(handle) + " has crashed and cannot be inserted";
        return nullptr;
    }
    if (instance->channels != channels) {
        error = "plugin " + std::to_string(handle) + " was opened for " + std::to_string(instance->channels) +
                " channels, the node has " + std::to_string(channels);
        return nullptr;
    }
    if (instance->max_block < max_block) {
        error = "plugin " + std::to_string(handle) + " was set up for blocks of at most " +
                std::to_string(instance->max_block) + " frames";
        return nullptr;
    }
    return std::make_unique<Vst3Processor>(std::move(instance), max_block);
}

}  // namespace ts

// ---- C ABI -------------------------------------------------------------------------------

extern "C" {

TS_API int32_t ts_vst3_last_error(char* buffer, int32_t capacity) {
    if (!buffer || capacity <= 0) return 0;
    const int32_t n = static_cast<int32_t>(t_error.size()) < capacity - 1 ? static_cast<int32_t>(t_error.size()) : capacity - 1;
    std::memcpy(buffer, t_error.data(), static_cast<size_t>(n));
    buffer[n] = '\0';
    return n;
}

TS_API int32_t ts_vst3_scan(const char* path, ts_vst3_class* out, int32_t capacity) {
    if (!path || capacity < 0 || (capacity && !out)) return TS_ERR_INVALID;
    int32_t result = 0;
    std::string error;
    plugin_thread().call([&] {
        host_application();
        VST3::Hosting::Module::Ptr module;
        DWORD code = 0;
        std::vector<VST3::Hosting::ClassInfo> classes;
        const bool ok = guarded([&] {
            module = VST3::Hosting::Module::create(path, error);
            if (module) classes = module->getFactory().classInfos();
        }, code);
        if (!ok) {
            error = fault_text("module load or factory query", code);
            new VST3::Hosting::Module::Ptr(std::move(module));
            result = TS_ERR_BACKEND;
            return;
        }
        if (!module) {
            if (error.empty()) error = "not a loadable VST3 module";
            result = TS_ERR_BACKEND;
            return;
        }
        for (const auto& c : classes) {
            if (result < capacity) {
                ts_vst3_class& d = out[result];
                std::memset(&d, 0, sizeof(d));
                copy_str(d.uid, sizeof(d.uid), c.ID().toString());
                copy_str(d.name, sizeof(d.name), c.name());
                copy_str(d.vendor, sizeof(d.vendor), c.vendor());
                copy_str(d.version, sizeof(d.version), c.version());
                copy_str(d.category, sizeof(d.category), c.category());
                copy_str(d.subcategories, sizeof(d.subcategories), c.subCategoriesString());
                copy_str(d.sdk_version, sizeof(d.sdk_version), c.sdkVersion());
                d.class_flags = c.classFlags();
                d.is_audio_effect = c.category() == kVstAudioEffectClass ? 1u : 0u;
            }
            ++result;
        }
        guarded([&] { module.reset(); }, code);
    });
    if (result < 0) t_error = error;
    return result;
}

TS_API ts_result ts_vst3_open(const char* path, const char* class_uid, uint32_t sample_rate, uint32_t max_block,
                              uint32_t channels, uint32_t* handle) {
    if (!path || !class_uid || !handle || channels < 1 || channels > TS_MAX_CHANNELS || max_block < 1) {
        t_error = "invalid arguments";
        return TS_ERR_INVALID;
    }
    ts_result result = TS_OK;
    std::string error;
    Instance* raw = new Instance();
    raw->channels = channels;
    raw->max_block = max_block;
    raw->sample_rate = sample_rate;

    plugin_thread().call([&] {
        host_application();
        DWORD code = 0;
        std::string step = "module load";
        const bool ok = guarded([&] {
            raw->module = VST3::Hosting::Module::create(path, error);
            if (!raw->module) return;
            const auto uid = VST3::UID::fromString(std::string(class_uid));
            if (!uid) { error = "malformed class ID"; return; }
            VST3::Hosting::ClassInfo info;
            bool found = false;
            for (const auto& c : raw->module->getFactory().classInfos())
                if (c.ID() == *uid) { info = c; found = true; }
            if (!found) { error = "the module has no class " + std::string(class_uid); return; }
            if (info.category() != kVstAudioEffectClass) { error = "class is not an audio processor"; return; }

            step = "initialize";
            raw->provider = owned(new PlugProvider(raw->module->getFactory(), info, true));
            if (!raw->provider->initialize()) { error = "the plugin failed to initialise"; return; }
            raw->component = raw->provider->getComponentPtr();
            raw->controller = raw->provider->getControllerPtr();
            raw->processor = FUnknownPtr<IAudioProcessor>(raw->component);
            if (!raw->processor) { error = "the component is not an IAudioProcessor"; return; }
            if (raw->controller) raw->controller->setComponentHandler(&raw->handler);

            step = "bus setup";
            const int32 ins = raw->component->getBusCount(kAudio, kInput);
            const int32 outs = raw->component->getBusCount(kAudio, kOutput);
            if (outs < 1) { error = "the plugin has no audio output bus"; return; }
            raw->has_input = ins > 0;
            std::vector<SpeakerArrangement> in_arr(ins), out_arr(outs);
            for (int32 i = 0; i < ins; ++i) raw->processor->getBusArrangement(kInput, i, in_arr[i]);
            for (int32 i = 0; i < outs; ++i) raw->processor->getBusArrangement(kOutput, i, out_arr[i]);
            if (ins) in_arr[0] = arrangement_for(channels);
            out_arr[0] = arrangement_for(channels);
            raw->processor->setBusArrangements(ins ? in_arr.data() : nullptr, ins, out_arr.data(), outs);
            // Whatever the plugin agreed to is what it will process; it must be our width.
            SpeakerArrangement got_out = 0, got_in = 0;
            raw->processor->getBusArrangement(kOutput, 0, got_out);
            if (static_cast<uint32_t>(SpeakerArr::getChannelCount(got_out)) != channels) {
                error = "the plugin will not process " + std::to_string(channels) + " channels (its main output is " +
                        std::to_string(SpeakerArr::getChannelCount(got_out)) + ")";
                return;
            }
            if (ins) {
                raw->processor->getBusArrangement(kInput, 0, got_in);
                if (static_cast<uint32_t>(SpeakerArr::getChannelCount(got_in)) != channels) {
                    error = "the plugin will not take " + std::to_string(channels) + " input channels";
                    return;
                }
                raw->component->activateBus(kAudio, kInput, 0, true);
            }
            raw->component->activateBus(kAudio, kOutput, 0, true);

            step = "setup processing";
            if (raw->processor->canProcessSampleSize(kSample32) != kResultTrue) { error = "the plugin cannot process 32-bit float"; return; }
            ProcessSetup setup{kRealtime, kSample32, static_cast<int32>(max_block), static_cast<SampleRate>(sample_rate)};
            if (raw->processor->setupProcessing(setup) != kResultOk) { error = "the plugin refused the processing setup"; return; }
            step = "activation";
            if (raw->component->setActive(true) != kResultOk) { error = "the plugin refused to activate"; return; }
            raw->processor->setProcessing(true);
            raw->latency.store(raw->processor->getLatencySamples());
        }, code);

        if (!ok) {
            error = fault_text(step.c_str(), code);
            raw->mark_crashed(error);
            result = TS_ERR_BACKEND;
        } else if (!error.empty() || !raw->module) {
            if (error.empty()) error = "not a loadable VST3 module";
            result = TS_ERR_BACKEND;
        }
    });

    if (result != TS_OK) {
        t_error = error;
        release_instance(raw);
        return result;
    }
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    raw->handle = g_next_handle++;
    g_instances.emplace(raw->handle, std::shared_ptr<Instance>(raw, release_instance));
    *handle = raw->handle;
    return TS_OK;
}

TS_API ts_result ts_vst3_close(uint32_t handle) {
    std::shared_ptr<Instance> held;
    {
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        auto it = g_instances.find(handle);
        if (it == g_instances.end()) {
            t_error = "no open plugin with handle " + std::to_string(handle);
            return TS_ERR_NOT_FOUND;
        }
        held = std::move(it->second);
        g_instances.erase(it);
    }
    return TS_OK;  // released now, or when the last plan holding it is freed
}

TS_API int32_t ts_vst3_param_count(uint32_t handle) {
    auto p = find(handle);
    if (!p) return TS_ERR_NOT_FOUND;
    if (!p->controller) return 0;
    int32_t n = 0;
    DWORD code = 0;
    plugin_thread().call([&] { guarded([&] { n = p->controller->getParameterCount(); }, code); });
    if (code) {
        p->mark_crashed(fault_text("getParameterCount", code));
        t_error = p->fault;
        return TS_ERR_BACKEND;
    }
    return n;
}

TS_API ts_result ts_vst3_param_info(uint32_t handle, int32_t index, ts_vst3_param* out) {
    auto p = find(handle);
    if (!p) return TS_ERR_NOT_FOUND;
    if (!out || !p->controller) return TS_ERR_INVALID;
    ts_result result = TS_OK;
    DWORD code = 0;
    plugin_thread().call([&] {
        guarded([&] {
            ParameterInfo info{};
            if (p->controller->getParameterInfo(index, info) != kResultOk) { result = TS_ERR_NOT_FOUND; return; }
            std::memset(out, 0, sizeof(*out));
            out->id = info.id;
            out->step_count = info.stepCount;
            out->default_normalized = info.defaultNormalizedValue;
            out->flags = info.flags;
            out->unit_id = info.unitId;
            out->normalized = p->controller->getParamNormalized(info.id);
            out->plain = p->controller->normalizedParamToPlain(info.id, out->normalized);
            copy_str(out->title, sizeof(out->title), Steinberg::Vst::StringConvert::convert(info.title));
            copy_str(out->short_title, sizeof(out->short_title), Steinberg::Vst::StringConvert::convert(info.shortTitle));
            copy_str(out->units, sizeof(out->units), Steinberg::Vst::StringConvert::convert(info.units));
            String128 text{};
            if (p->controller->getParamStringByValue(info.id, out->normalized, text) == kResultOk)
                copy_str(out->display, sizeof(out->display), Steinberg::Vst::StringConvert::convert(text));
        }, code);
    });
    if (code) {
        p->mark_crashed(fault_text("getParameterInfo", code));
        t_error = p->fault;
        return TS_ERR_BACKEND;
    }
    if (result != TS_OK) t_error = "no parameter at index " + std::to_string(index);
    return result;
}

TS_API ts_result ts_vst3_set_param(uint32_t handle, uint32_t id, double normalized) {
    auto p = find(handle);
    if (!p) return TS_ERR_NOT_FOUND;
    if (normalized < 0.0 || normalized > 1.0 || normalized != normalized) {
        t_error = "a normalised value is between 0 and 1";
        return TS_ERR_INVALID;
    }
    DWORD code = 0;
    bool queued = true;
    plugin_thread().call([&] {
        if (p->controller) guarded([&] { p->controller->setParamNormalized(id, normalized); }, code);
        queued = p->changes.push(ParamChange{id, normalized});
    });
    if (code) {
        p->mark_crashed(fault_text("setParamNormalized", code));
        t_error = p->fault;
        return TS_ERR_BACKEND;
    }
    if (!queued) {
        t_error = "the audio thread has not consumed the previous 1024 parameter changes";
        return TS_ERR_STATE;
    }
    return TS_OK;
}

TS_API int32_t ts_vst3_get_state(uint32_t handle, int32_t which, uint8_t* buffer, int32_t capacity) {
    auto p = find(handle);
    if (!p) return TS_ERR_NOT_FOUND;
    if (which == 1 && !p->controller) return 0;
    int32_t size = 0;
    DWORD code = 0;
    tresult r = kResultOk;
    plugin_thread().call([&] {
        guarded([&] {
            MemoryStream stream;
            r = which == 0 ? p->component->getState(&stream) : p->controller->getState(&stream);
            size = static_cast<int32_t>(stream.getSize());
            if (buffer && capacity > 0) std::memcpy(buffer, stream.getData(), static_cast<size_t>(std::min(size, capacity)));
        }, code);
    });
    if (code) {
        p->mark_crashed(fault_text("getState", code));
        t_error = p->fault;
        return TS_ERR_BACKEND;
    }
    if (r != kResultOk && r != kNotImplemented) {
        t_error = "the plugin refused to report its state";
        return TS_ERR_BACKEND;
    }
    return size;
}

TS_API ts_result ts_vst3_set_state(uint32_t handle, const uint8_t* component, int32_t component_size,
                                   const uint8_t* controller, int32_t controller_size) {
    auto p = find(handle);
    if (!p) return TS_ERR_NOT_FOUND;
    DWORD code = 0;
    tresult r = kResultOk;
    plugin_thread().call([&] {
        guarded([&] {
            if (component && component_size > 0) {
                MemoryStream a(const_cast<uint8_t*>(component), component_size);
                r = p->component->setState(&a);
                if (r == kResultOk && p->controller) {
                    MemoryStream b(const_cast<uint8_t*>(component), component_size);
                    p->controller->setComponentState(&b);
                }
            }
            if (r == kResultOk && controller && controller_size > 0 && p->controller) {
                MemoryStream c(const_cast<uint8_t*>(controller), controller_size);
                r = p->controller->setState(&c);
            }
        }, code);
    });
    if (code) {
        p->mark_crashed(fault_text("setState", code));
        t_error = p->fault;
        return TS_ERR_BACKEND;
    }
    if (r != kResultOk) {
        t_error = "the plugin rejected the state (it may be from another plugin or version)";
        return TS_ERR_INVALID;
    }
    return TS_OK;
}

TS_API ts_result ts_vst3_get_status(uint32_t handle, ts_vst3_status* out) {
    auto p = find(handle);
    if (!p) return TS_ERR_NOT_FOUND;
    if (!out) return TS_ERR_INVALID;
    std::memset(out, 0, sizeof(*out));
    out->crashed = p->crashed.load(std::memory_order_acquire) ? 1u : 0u;
    if (out->crashed && p->fault.empty()) p->fault = fault_text("process", p->fault_code.load());
    out->restart_flags = p->restart_flags.exchange(0);
    out->latency = p->latency.load();
    out->channels = p->channels;
    out->blocks = p->blocks.load();
    copy_str(out->fault, sizeof(out->fault), p->fault);
    return TS_OK;
}

TS_API int32_t ts_vst3_has_editor(uint32_t handle) {
    auto p = find(handle);
    if (!p || !p->controller) return 0;
    int32_t has = 0;
    DWORD code = 0;
    plugin_thread().call([&] {
        guarded([&] {
            IPtr<IPlugView> view = owned(p->controller->createView(ViewType::kEditor));
            has = view && view->isPlatformTypeSupported(kPlatformTypeHWND) == kResultTrue ? 1 : 0;
        }, code);
    });
    return code ? 0 : has;
}

TS_API ts_result ts_vst3_open_editor(uint32_t handle) {
    auto p = find(handle);
    if (!p) return TS_ERR_NOT_FOUND;
    if (!p->controller) {
        t_error = "the plugin has no controller, so no editor";
        return TS_ERR_UNSUPPORTED;
    }
    ts_result result = TS_OK;
    DWORD code = 0;
    plugin_thread().call([&] {
        if (p->editor && p->editor->is_open()) return;
        guarded([&] {
            IPtr<IPlugView> view = owned(p->controller->createView(ViewType::kEditor));
            if (!view || view->isPlatformTypeSupported(kPlatformTypeHWND) != kResultTrue) {
                result = TS_ERR_UNSUPPORTED;
                return;
            }
            p->editor = std::make_unique<EditorFrame>(view);
            if (!p->editor->open("ToneSphere plugin editor")) {
                p->editor.reset();
                result = TS_ERR_BACKEND;
            }
        }, code);
    });
    if (code) {
        p->mark_crashed(fault_text("opening the editor", code));
        t_error = p->fault;
        return TS_ERR_BACKEND;
    }
    if (result == TS_ERR_UNSUPPORTED) t_error = "the plugin offers no Windows editor";
    else if (result != TS_OK) t_error = "the editor could not be attached";
    return result;
}

TS_API ts_result ts_vst3_close_editor(uint32_t handle) {
    auto p = find(handle);
    if (!p) return TS_ERR_NOT_FOUND;
    DWORD code = 0;
    plugin_thread().call([&] { guarded([&] { p->editor.reset(); }, code); });
    return code ? TS_ERR_BACKEND : TS_OK;
}

}  // extern "C"
