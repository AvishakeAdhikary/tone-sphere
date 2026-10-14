#include "backends/linux/pipewire.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <spa/pod/builder.h>

#include "core/bridge.h"
#include "util/log.h"
#include "util/platform.h"

namespace nt {

namespace {

constexpr int kQuantum = 256;  // requested PipeWire period (frames at the engine rate)

struct NodeRec {
    std::string name, desc, mediaClass, appName, appBinary, api, serial;
    int channels = 2;
};

std::string prop(const spa_dict* d, const char* key) {
    const char* v = d ? spa_dict_lookup(d, key) : nullptr;
    return v ? v : "";
}

std::vector<std::string> channelNames(int ch) {
    if (ch == 1) return {"Mono"};
    if (ch == 2) return {"L", "R"};
    std::vector<std::string> v;
    for (int i = 0; i < ch; ++i) v.push_back(std::to_string(i + 1));
    return v;
}

}  // namespace

class Port;

struct PipeWireBackend::Impl {
    pw_thread_loop* loop = nullptr;
    pw_context* context = nullptr;
    pw_core* core = nullptr;
    pw_registry* registry = nullptr;
    spa_hook coreListener{}, registryListener{};
    int syncSeq = -1;
    bool synced = false;

    std::mutex mu;
    std::map<uint32_t, NodeRec> nodes;
    std::atomic<bool> changed{true};

    // NexTone's own virtual devices (persistent while the engine runs).
    struct Virtual {
        std::string nodeName, label;
        bool apps_play_into = true;  // "NexTone Output N" (true) or "NexTone Mic N" (false)
        std::shared_ptr<Port> port;
    };
    std::vector<Virtual> virtuals;
    double rate = 48000.0;

    ~Impl();
};

// A PipeWire stream that is bound to at most one engine bridge at a time.
class Port {
public:
    Port(PipeWireBackend::Impl& pw, bool captureIntoNexTone, int channels, double rate, pw_properties* props, bool autoconnect)
        : pw_(pw), capture_(captureIntoNexTone), channels_(channels) {
        static const pw_stream_events events = [] {
            pw_stream_events e{};
            e.version = PW_VERSION_STREAM_EVENTS;
            e.process = &Port::onProcess;
            e.state_changed = &Port::onState;
            return e;
        }();
        pw_thread_loop_lock(pw_.loop);
        stream_ = pw_stream_new(pw_.core, "NexTone", props);
        if (stream_) {
            pw_stream_add_listener(stream_, &listener_, &events, this);
            uint8_t buf[1024];
            spa_pod_builder b = SPA_POD_BUILDER_INIT(buf, sizeof(buf));
            spa_audio_info_raw info{};
            info.format = SPA_AUDIO_FORMAT_F32;
            info.channels = static_cast<uint32_t>(channels);
            info.rate = static_cast<uint32_t>(rate);
            if (channels == 1) info.position[0] = SPA_AUDIO_CHANNEL_MONO;
            if (channels >= 2) {
                info.position[0] = SPA_AUDIO_CHANNEL_FL;
                info.position[1] = SPA_AUDIO_CHANNEL_FR;
                for (int c = 2; c < channels && c < SPA_AUDIO_MAX_CHANNELS; ++c) info.position[c] = SPA_AUDIO_CHANNEL_AUX0 + (c - 2);
            }
            const spa_pod* params[1] = {spa_format_audio_raw_build(&b, SPA_PARAM_EnumFormat, &info)};
            const auto flags = static_cast<pw_stream_flags>(PW_STREAM_FLAG_MAP_BUFFERS | PW_STREAM_FLAG_RT_PROCESS |
                                                            (autoconnect ? PW_STREAM_FLAG_AUTOCONNECT : 0));
            if (pw_stream_connect(stream_, capture_ ? PW_DIRECTION_INPUT : PW_DIRECTION_OUTPUT, PW_ID_ANY, flags, params, 1) < 0)
                failed_ = true;
        } else {
            failed_ = true;
        }
        pw_thread_loop_unlock(pw_.loop);
    }

    ~Port() {
        detach();
        pw_thread_loop_lock(pw_.loop);
        if (stream_) {
            spa_hook_remove(&listener_);
            pw_stream_destroy(stream_);
        }
        pw_thread_loop_unlock(pw_.loop);
    }

    void attach(std::shared_ptr<CaptureBridge> c, std::shared_ptr<RenderBridge> r) {
        detach();
        capOwn_ = std::move(c);
        renOwn_ = std::move(r);
        cap_.store(capOwn_.get());
        ren_.store(renOwn_.get());
    }

    void detach() {
        cap_.store(nullptr);
        ren_.store(nullptr);
        while (inCallback_.load() != 0) std::this_thread::yield();  // the RT thread is done with the bridge
        capOwn_.reset();
        renOwn_.reset();
    }

    bool failed() const { return failed_.load(); }
    int quantum() const { return quantum_.load(); }

private:
    static void onState(void* data, pw_stream_state, pw_stream_state state, const char* err) {
        auto* self = static_cast<Port*>(data);
        if (state == PW_STREAM_STATE_STREAMING || state == PW_STREAM_STATE_PAUSED) self->connected_ = true;
        if (state == PW_STREAM_STATE_ERROR || (state == PW_STREAM_STATE_UNCONNECTED && self->connected_)) {
            if (err) NT_WARN("PipeWire stream: %s", err);
            self->failed_ = true;
        }
    }

    static void onProcess(void* data) {
        auto* self = static_cast<Port*>(data);
        pw_buffer* b = pw_stream_dequeue_buffer(self->stream_);
        if (!b) return;
        spa_data& d = b->buffer->datas[0];
        const uint32_t stride = static_cast<uint32_t>(sizeof(float) * self->channels_);
        const double now = nowSeconds();
        self->inCallback_.fetch_add(1);
        if (d.data) {
            if (self->capture_) {
                const uint32_t offset = std::min(d.chunk->offset, d.maxsize);
                const uint32_t size = std::min(d.chunk->size, d.maxsize - offset);
                const uint32_t frames = size / stride;
                self->quantum_.store(static_cast<int>(frames), std::memory_order_relaxed);
                if (CaptureBridge* c = self->cap_.load(); c && frames)
                    c->deviceWrite(reinterpret_cast<const float*>(static_cast<uint8_t*>(d.data) + offset), frames, now);
            } else {
                uint32_t frames = d.maxsize / stride;
                if (b->requested) frames = std::min<uint32_t>(frames, static_cast<uint32_t>(b->requested));
                self->quantum_.store(static_cast<int>(frames), std::memory_order_relaxed);
                auto* out = static_cast<float*>(d.data);
                if (RenderBridge* r = self->ren_.load())
                    r->deviceRead(out, frames, now);
                else
                    std::memset(out, 0, static_cast<size_t>(frames) * stride);
                d.chunk->offset = 0;
                d.chunk->stride = static_cast<int32_t>(stride);
                d.chunk->size = frames * stride;
            }
        }
        self->inCallback_.fetch_sub(1);
        pw_stream_queue_buffer(self->stream_, b);
    }

    PipeWireBackend::Impl& pw_;
    bool capture_;
    int channels_;
    pw_stream* stream_ = nullptr;
    spa_hook listener_{};
    std::atomic<CaptureBridge*> cap_{nullptr};
    std::atomic<RenderBridge*> ren_{nullptr};
    std::shared_ptr<CaptureBridge> capOwn_;
    std::shared_ptr<RenderBridge> renOwn_;
    std::atomic<int> inCallback_{0};
    std::atomic<bool> failed_{false};
    std::atomic<bool> connected_{false};
    std::atomic<int> quantum_{0};
};

PipeWireBackend::Impl::~Impl() {
    for (auto& v : virtuals) v.port.reset();
    if (loop) pw_thread_loop_stop(loop);
    if (registry) pw_proxy_destroy(reinterpret_cast<pw_proxy*>(registry));
    if (core) pw_core_disconnect(core);
    if (context) pw_context_destroy(context);
    if (loop) pw_thread_loop_destroy(loop);
}

namespace {

class PwStream final : public DeviceStream {
public:
    PwStream(std::shared_ptr<PipeWireBackend::Impl> pw, DeviceInfo dev, int channels, double rate, std::shared_ptr<Port> persistent)
        : pw_(std::move(pw)), dev_(std::move(dev)), channels_(channels), rate_(rate), port_(std::move(persistent)), persistent_(port_ != nullptr) {}
    ~PwStream() override { stop(); }

    double sampleRate() const override { return rate_; }
    int channels() const override { return channels_; }
    int periodFrames() const override {
        const int q = port_ ? port_->quantum() : 0;
        return q > 0 ? q : kQuantum;
    }
    bool start(std::shared_ptr<CaptureBridge> capture, std::shared_ptr<RenderBridge> render) override {
        if (!port_) {
            pw_properties* props = pw_properties_new(PW_KEY_MEDIA_TYPE, "Audio", PW_KEY_MEDIA_CATEGORY, dev_.input ? "Capture" : "Playback",
                                                     PW_KEY_MEDIA_ROLE, "Production", PW_KEY_NODE_NAME,
                                                     dev_.input ? "nextone.capture" : "nextone.playback", PW_KEY_NODE_DESCRIPTION,
                                                     ("NexTone: " + dev_.name).c_str(), PW_KEY_TARGET_OBJECT, dev_.endpointId.c_str(),
                                                     "node.dont-reconnect", "true", PW_KEY_APP_NAME, "NexTone", nullptr);
            pw_properties_setf(props, PW_KEY_NODE_LATENCY, "%d/%u", kQuantum, static_cast<unsigned>(rate_));
            port_ = std::make_shared<Port>(*pw_, dev_.input, channels_, rate_, props, true);
        }
        if (capture) capture->markStarted();
        if (render) render->markStarted();
        port_->attach(capture, render);
        return !port_->failed();
    }
    void stop() override {
        if (!port_) return;
        port_->detach();
        if (!persistent_) port_.reset();
    }
    std::string error() const override { return port_ && port_->failed() ? "Device was removed" : std::string(); }

private:
    std::shared_ptr<PipeWireBackend::Impl> pw_;
    DeviceInfo dev_;
    int channels_;
    double rate_;
    std::shared_ptr<Port> port_;
    bool persistent_;
};

void onRegistryGlobal(void* data, uint32_t id, uint32_t, const char* type, uint32_t, const spa_dict* props) {
    auto* pw = static_cast<PipeWireBackend::Impl*>(data);
    if (std::strcmp(type, PW_TYPE_INTERFACE_Node) != 0) return;
    NodeRec r;
    r.mediaClass = prop(props, PW_KEY_MEDIA_CLASS);
    if (r.mediaClass.rfind("Audio/", 0) != 0 && r.mediaClass.rfind("Stream/Output/Audio", 0) != 0) return;
    r.name = prop(props, PW_KEY_NODE_NAME);
    r.desc = prop(props, PW_KEY_NODE_DESCRIPTION);
    r.appName = prop(props, PW_KEY_APP_NAME);
    r.appBinary = prop(props, PW_KEY_APP_PROCESS_BINARY);
    r.api = prop(props, "device.api");
    r.serial = prop(props, PW_KEY_OBJECT_SERIAL);
    const std::string ch = prop(props, PW_KEY_AUDIO_CHANNELS);
    if (!ch.empty()) r.channels = std::clamp(std::atoi(ch.c_str()), 1, 32);
    if (r.name.rfind("nextone.", 0) == 0 || r.appName == "NexTone") return;  // our own nodes
    std::lock_guard lk(pw->mu);
    pw->nodes[id] = r;
    pw->changed = true;
}

void onRegistryRemove(void* data, uint32_t id) {
    auto* pw = static_cast<PipeWireBackend::Impl*>(data);
    std::lock_guard lk(pw->mu);
    if (pw->nodes.erase(id)) pw->changed = true;
}

void onCoreDone(void* data, uint32_t id, int seq) {
    auto* pw = static_cast<PipeWireBackend::Impl*>(data);
    if (id == PW_ID_CORE && seq == pw->syncSeq) {
        pw->synced = true;
        pw_thread_loop_signal(pw->loop, false);
    }
}

}  // namespace

PipeWireBackend::PipeWireBackend(int virtualDevices) : impl_(std::make_shared<Impl>()) {
    static std::once_flag once;
    std::call_once(once, [] { pw_init(nullptr, nullptr); });
    Impl& pw = *impl_;
    pw.loop = pw_thread_loop_new("nt-pipewire", nullptr);
    pw.context = pw_context_new(pw_thread_loop_get_loop(pw.loop), pw_properties_new(PW_KEY_APP_NAME, "NexTone", nullptr), 0);
    if (!pw.loop || !pw.context || pw_thread_loop_start(pw.loop) < 0) {
        NT_WARN("PipeWire: cannot start the client loop");
        return;
    }
    pw_thread_loop_lock(pw.loop);
    pw.core = pw_context_connect(pw.context, nullptr, 0);
    if (!pw.core) {
        pw_thread_loop_unlock(pw.loop);
        NT_WARN("PipeWire: no server running");
        return;
    }
    static const pw_core_events coreEvents = [] {
        pw_core_events e{};
        e.version = PW_VERSION_CORE_EVENTS;
        e.done = onCoreDone;
        return e;
    }();
    static const pw_registry_events regEvents = [] {
        pw_registry_events e{};
        e.version = PW_VERSION_REGISTRY_EVENTS;
        e.global = onRegistryGlobal;
        e.global_remove = onRegistryRemove;
        return e;
    }();
    pw_core_add_listener(pw.core, &pw.coreListener, &coreEvents, &pw);
    pw.registry = pw_core_get_registry(pw.core, PW_VERSION_REGISTRY, 0);
    pw_registry_add_listener(pw.registry, &pw.registryListener, &regEvents, &pw);
    pw.syncSeq = pw_core_sync(pw.core, PW_ID_CORE, 0);
    while (!pw.synced) {
        if (pw_thread_loop_timed_wait(pw.loop, 3) != 0) break;  // initial node list (3 s at most)
    }
    pw_thread_loop_unlock(pw.loop);

    // Virtual devices other apps can use right away (also while nothing is routed).
    for (int i = 1; i <= virtualDevices; ++i) {
        for (bool appsPlay : {true, false}) {
            Impl::Virtual v;
            v.apps_play_into = appsPlay;
            v.nodeName = (appsPlay ? "nextone.output." : "nextone.mic.") + std::to_string(i);
            v.label = (appsPlay ? "NexTone Output " : "NexTone Mic ") + std::to_string(i);
            pw_properties* props = pw_properties_new(PW_KEY_MEDIA_TYPE, "Audio", PW_KEY_MEDIA_CLASS,
                                                     appsPlay ? "Audio/Sink" : "Audio/Source", PW_KEY_NODE_NAME,
                                                     v.nodeName.c_str(), PW_KEY_NODE_DESCRIPTION, v.label.c_str(),
                                                     "node.always-process", "true", PW_KEY_APP_NAME, "NexTone", nullptr);
            pw_properties_setf(props, PW_KEY_NODE_LATENCY, "%d/%u", kQuantum, static_cast<unsigned>(pw.rate));
            // A sink receives audio from apps (NexTone captures it); a source sends audio to apps.
            v.port = std::make_shared<Port>(pw, appsPlay, 2, pw.rate, props, false);
            pw.virtuals.push_back(std::move(v));
        }
    }
    NT_INFO("PipeWire connected: %zu audio nodes, %zu virtual devices", pw.nodes.size(), pw.virtuals.size());
}

PipeWireBackend::~PipeWireBackend() = default;

bool PipeWireBackend::connected() const { return impl_->core != nullptr; }

bool PipeWireBackend::poll() { return impl_->changed.exchange(false); }

std::vector<DeviceInfo> PipeWireBackend::enumerate() {
    std::vector<DeviceInfo> out;
    if (!connected()) return out;
    for (size_t i = 0; i < impl_->virtuals.size(); ++i) {
        const auto& v = impl_->virtuals[i];
        DeviceInfo d;
        d.kind = DeviceKind::PipeWire;
        d.nodeId = std::string(v.apps_play_into ? "pw:vout:" : "pw:vmic:") + v.nodeName.substr(v.nodeName.rfind('.') + 1);
        d.endpointId = "virtual:" + std::to_string(i);
        d.appEndpointId = v.nodeName;
        d.name = v.label;
        d.subtitle = v.apps_play_into ? "Apps play into this; NexTone records it" : "NexTone plays into this; apps record it";
        d.bin = v.apps_play_into ? Bin::VirtualOut : Bin::VirtualIn;
        d.input = v.apps_play_into;
        d.channels = 2;
        d.channelNames = channelNames(2);
        d.sampleRate = impl_->rate;
        out.push_back(d);
    }
    std::lock_guard lk(impl_->mu);
    std::map<std::string, bool> seenApps;
    for (const auto& [id, n] : impl_->nodes) {
        DeviceInfo d;
        d.kind = DeviceKind::PipeWire;
        d.channels = n.channels;
        d.channelNames = channelNames(n.channels);
        d.sampleRate = impl_->rate;
        if (n.mediaClass == "Audio/Sink") {
            d.nodeId = "pw:out:" + n.name;
            d.endpointId = n.name;
            d.bin = Bin::PhysicalOut;
            d.input = false;
        } else if (n.mediaClass.rfind("Audio/Source", 0) == 0) {
            d.nodeId = "pw:in:" + n.name;
            d.endpointId = n.name;
            d.bin = Bin::PhysicalIn;
            d.input = true;
        } else if (n.mediaClass.rfind("Stream/Output/Audio", 0) == 0) {
            const std::string app = !n.appBinary.empty() ? n.appBinary : !n.appName.empty() ? n.appName : n.name;
            if (app.empty() || seenApps[app]) continue;
            seenApps[app] = true;
            d.nodeId = "pw:app:" + app;
            d.endpointId = n.serial.empty() ? n.name : n.serial;
            d.bin = Bin::AppOut;
            d.input = true;
            d.name = !n.appName.empty() ? n.appName : app;
            d.subtitle = "Application (copy of its sound)";
            out.push_back(d);
            continue;
        } else {
            continue;
        }
        d.name = !n.desc.empty() ? n.desc : n.name;
        d.subtitle = n.api.empty() ? "PipeWire" : n.api;
        out.push_back(d);
    }
    return out;
}

std::unique_ptr<DeviceStream> PipeWireBackend::open(const DeviceInfo& dev, const StreamRequest& req, std::string& error) {
    if (!connected()) {
        error = "PipeWire is not running";
        return nullptr;
    }
    double rate = req.preferredRate > 0 ? req.preferredRate : impl_->rate;
    std::shared_ptr<Port> persistent;
    if (dev.endpointId.rfind("virtual:", 0) == 0) {
        const size_t i = static_cast<size_t>(std::atoi(dev.endpointId.c_str() + 8));
        if (i >= impl_->virtuals.size()) {
            error = "Unknown virtual device";
            return nullptr;
        }
        persistent = impl_->virtuals[i].port;
        rate = impl_->rate;  // virtual devices run at the rate they were created with
        if (persistent->failed()) {
            error = "The virtual device could not be created";
            return nullptr;
        }
    }
    return std::make_unique<PwStream>(impl_, dev, std::max(1, dev.channels), rate, persistent);
}

}  // namespace nt
