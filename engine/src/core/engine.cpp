#include "core/engine.h"
#include "version.h"

#include <algorithm>
#include <cmath>
#include <set>

#include "net/vban.h"
#include "plugins/vst3_host.h"
#include "util/log.h"
#include "util/platform.h"

#ifdef _WIN32
#include <windows.h>
#include <objbase.h>
#include <timeapi.h>

#include "backends/win/app_audio.h"
#include "backends/win/asio_clients.h"
#include "backends/win/asio_host.h"
#include "backends/win/peak_meters.h"
#include "backends/win/vad.h"
#include "backends/win/wasapi.h"
#endif
#ifdef TONESPHERE_WITH_PIPEWIRE
#include "backends/linux/pipewire.h"
#endif

#ifndef _WIN32
namespace ts {
// Per-app audio and OS peak meters are Windows features; other platforms get empty stand-ins.
class AppAudio {
public:
    void shutdown() {}
};
class PeakMeters {};
}  // namespace ts
#endif

namespace ts {

namespace {
constexpr const char* kVersion = TONESPHERE_VERSION;

double clampRate(double r) { return r >= 8000 && r <= 384000 ? r : 48000.0; }
}  // namespace

std::string Engine::version() const { return kVersion; }

Engine::Engine() {
    tone_ = std::make_shared<ToneGen>();
    recorder_ = std::make_unique<Recorder>();
    settings_.recordFolder = defaultRecordFolder();
}

Engine::~Engine() { shutdown(); }

// ============================================================ lifecycle
bool Engine::start() {
    std::lock_guard lock(mu_);
#ifdef _WIN32
    const char* scanner = "tonesphere-plugscan.exe";
#else
    const char* scanner = "tonesphere-plugscan";
#endif
    plugins_ = std::make_unique<PluginManager>(dataDir() + "/plugins.json", exeDir() + "/" + scanner);
    plugins_->onChanged = [this] {
        if (onStateChanged) onStateChanged();
    };
    loadConfig();
    applyPluginFoldersLocked();
    plugins_->init();
    runningRate_ = clampRate(settings_.sampleRate);
#ifdef _WIN32
    auto wasapi = std::make_unique<WasapiBackend>();
    wasapi->setCompatMode(settings_.compatMode);
    wasapi->onDevicesChanged = [this] {
        devicesChanged_ = true;
        maintCv_.notify_all();
    };
    backends_.push_back(std::move(wasapi));
    backends_.push_back(std::make_unique<VadBackend>());
    auto asioClients = std::make_unique<AsioClientBackend>();
    asioClients->setEngineInfo(runningRate_, settings_.blockSize, 8, 8);
    backends_.push_back(std::move(asioClients));
    auto asioHost = std::make_unique<AsioHostBackend>();
    asioHost->setPreferred(runningRate_, settings_.blockSize);
    backends_.push_back(std::move(asioHost));
    apps_ = std::make_unique<AppAudio>();
    peaks_ = std::make_unique<PeakMeters>();
#endif
#ifdef TONESPHERE_WITH_PIPEWIRE
    backends_.push_back(std::make_unique<PipeWireBackend>());
#endif
    backends_.push_back(std::make_unique<vban::VbanBackend>());

    // Built-in test tone source (hidden in the simple view).
    NodeModel& t = ensureNode("tone");
    t.bin = Bin::PhysicalIn;
    t.kind = DeviceKind::Wasapi;
    t.name = "Test tone";
    t.subtitle = "Built-in generator";
    t.channels = 2;
    t.channelNames = {"L", "R"};
    t.present = true;
    t.persistent = true;

    recorder_->setReplaySeconds(settings_.replayBufferSeconds);
    refreshDevicesLocked();
    rebuildLocked();
    startClock();
    maint_ = std::thread([this] { maintenanceThread(); });
    TS_INFO("Engine started at %.0f Hz, block %d", runningRate_, settings_.blockSize);
    return true;
}

void Engine::shutdown() {
    if (shutdown_.exchange(true)) return;
    maintCv_.notify_all();
    if (maint_.joinable()) maint_.join();
    stopClock();
    std::lock_guard lock(mu_);
    saveConfigLocked();
    for (auto& [id, n] : nodes_) closeStreamLocked(n);
    if (apps_) apps_->shutdown();
    program_.store(nullptr);
    programOwner_.reset();
    retired_.clear();
    // Unload plugins now, while COM and the main thread are still up.
    for (auto& [id, n] : nodes_) n.fx.clear();
    TS_INFO("Engine stopped");
}

// ============================================================ helpers
void Engine::applyPluginFoldersLocked() {
    // User folders from settings, plus TONESPHERE_VST3_PATH (tests, portable setups).
    std::vector<std::string> folders = settings_.pluginFolders;
#ifdef _WIN32
    const char sep = ';';
#else
    const char sep = ':';
#endif
    if (const char* extra = std::getenv("TONESPHERE_VST3_PATH")) {
        const std::string s = extra;
        for (size_t a = 0, b; a <= s.size(); a = b + 1) {
            b = s.find(sep, a);
            if (b == std::string::npos) b = s.size();
            if (b > a) folders.push_back(s.substr(a, b - a));
        }
    }
    plugins_->setExtraFolders(folders);
}
NodeModel* Engine::findNode(const std::string& id) {
    auto it = nodes_.find(id);
    return it == nodes_.end() ? nullptr : &it->second;
}

RouteModel* Engine::findRoute(const std::string& id) {
    for (auto& r : routes_)
        if (r.id == id) return &r;
    return nullptr;
}

NodeModel& Engine::ensureNode(const std::string& id) {
    auto it = nodes_.find(id);
    if (it != nodes_.end()) return it->second;
    NodeModel& n = nodes_[id];
    n.id = id;
    nodeOrder_.push_back(id);
    return n;
}

void Engine::markDirty() {
    configDirty_ = true;
    if (onStateChanged) onStateChanged();
}

void Engine::retireLocked(std::shared_ptr<void> obj) {
    if (obj) retired_.push_back({std::move(obj), cyclesStarted_.load(std::memory_order_acquire)});
}

void Engine::reclaimLocked() {
    const uint64_t done = cyclesDone_.load(std::memory_order_acquire);
    const bool idle = !clockRun_.load();
    retired_.erase(std::remove_if(retired_.begin(), retired_.end(), [&](const Retired& r) { return idle || done >= r.epoch; }),
                   retired_.end());
}

void Engine::applyNodeParamsLocked(NodeModel& n) {
    if (!n.rt) return;
    n.rt->gainDb.store(n.gainDb);
    n.rt->muted.store(n.muted);
    n.rt->solo.store(n.solo);
    n.rt->invert.store(n.settings.invert);
    n.rt->pan.store(static_cast<float>(n.settings.pan));
    n.rt->delayMs.store(static_cast<float>(n.settings.delayMs));
}

void Engine::instantiatePendingFxLocked(NodeModel& n) {
    for (const FxConfig& fc : n.pendingFx) {
        std::shared_ptr<Fx> fx;
        if (fc.type == "vst3") {
            try {
                fx = plugins_->create(fc.pluginId);
                fx->loadState(fc.state);
            } catch (const std::exception& e) {
                TS_WARN("Plugin %s on %s: %s", fc.name.c_str(), n.id.c_str(), e.what());
                fx = createMissingPluginFx(fc.pluginId, fc.name, fc.state);
            }
        } else {
            fx = createBuiltinFx(fc.type);
        }
        if (!fx) continue;
        fx->id = fc.id.empty() ? newId("fx") : fc.id;
        fx->bypass.store(fc.bypass);
        for (auto& [k, v] : fc.params) fx->setParam(k, v);
        fx->sidechainId = fc.sidechain;
        fx->prepare(runningRate_, kMaxBlock, n.channels);
        n.fx.push_back(fx);
    }
    n.pendingFx.clear();
}

// ============================================================ devices
void Engine::refreshDevicesLocked() {
    std::set<std::string> seen;
    for (auto& be : backends_) {
#ifdef _WIN32
        if (auto* w = dynamic_cast<WasapiBackend*>(be.get())) w->setCompatMode(settings_.compatMode);
#endif
        for (DeviceInfo& d : be->enumerate()) {
            seen.insert(d.nodeId);
            NodeModel& n = ensureNode(d.nodeId);
            const bool channelsChanged = n.rt && n.channels != d.channels;
            n.device = d;
            n.bin = d.bin;
            n.kind = d.kind;
            n.name = d.name;
            n.subtitle = d.subtitle;
            n.channels = d.channels;
            n.channelNames = d.channelNames;
            n.present = true;
            if (channelsChanged) {
                closeStreamLocked(n);
                retireLocked(n.rt);
                n.rt.reset();
            }
        }
    }
#ifdef _WIN32
    if (apps_) apps_->refresh(*this, seen);
#endif
    vadPresent_ = std::any_of(nodes_.begin(), nodes_.end(), [](auto& kv) { return kv.second.kind == DeviceKind::Vad && kv.second.present; });
    seen.insert("tone");
    // Devices that (re)appear take over the routing of a preset placeholder or of their old id.
    rebindDevicesLocked(seen);
    // Mark vanished devices; drop ones nobody refers to.
    for (auto it = nodes_.begin(); it != nodes_.end();) {
        NodeModel& n = it->second;
        const bool device = n.kind != DeviceKind::Vban && n.bin != Bin::Bus && n.id != "tone";
        if (device && !seen.count(n.id)) {
            if (n.present) TS_INFO("Device gone: %s", n.name.c_str());
            n.present = false;
            closeStreamLocked(n);
            const bool referenced = std::any_of(routes_.begin(), routes_.end(), [&](const RouteModel& r) { return r.from == n.id || r.to == n.id; });
            if (!referenced && !n.persistent && !n.app.pinned) {
                if (n.rt) retireLocked(n.rt);
                nodeOrder_.erase(std::remove(nodeOrder_.begin(), nodeOrder_.end(), n.id), nodeOrder_.end());
                it = nodes_.erase(it);
                continue;
            }
        }
        ++it;
    }
}

bool Engine::nodeNeededLocked(const NodeModel& n) const {
    if (!n.present || !n.userEnabled) return false;
    if (n.bin == Bin::Bus || n.id == "tone") return false;
    if (forcedOpen_.count(n.id)) return true;
    // (VBAN nodes are opened like devices when connected.)
    for (const RouteModel& r : routes_) {
        if (r.from == n.id || r.to == n.id) return true;
    }
    if (n.bin == Bin::PhysicalOut) {
        for (auto& [id, b] : nodes_)
            if (b.bin == Bin::Bus && b.solo) return true;
    }
    return false;
}

Backend* Engine::backendFor(DeviceKind k) const {
    const char* want = k == DeviceKind::Vad          ? "vad"
                       : k == DeviceKind::AsioClient ? "asioclient"
                       : k == DeviceKind::Asio       ? "asio"
                       : k == DeviceKind::Vban       ? "vban"
                       : k == DeviceKind::PipeWire   ? "pipewire"
                       : (k == DeviceKind::Wasapi || k == DeviceKind::Compat) ? "wasapi"
                                                                              : nullptr;
    if (!want) return nullptr;
    for (auto& b : backends_)
        if (std::string(b->name()) == want) return b.get();
    return nullptr;
}

bool Engine::openStreamLocked(NodeModel& n) {
    if (n.stream) return true;
    DeviceInfo dev = n.device;
    bool viaBackend = true;
#ifdef _WIN32
    if (n.kind == DeviceKind::App) {
        const DeviceInfo* assigned = apps_ ? apps_->assigned(n) : nullptr;
        if (assigned) {
            // Isolated app / app microphone: stream through the assigned virtual device.
            dev = *assigned;
            dev.input = n.bin == Bin::AppOut;
        } else {
            viaBackend = false;
            std::string err;
            n.stream = apps_ ? apps_->open(n, runningRate_, err) : nullptr;
            if (!n.stream) {
                n.streamError = err;
                return false;
            }
            dev.input = true;
        }
    }
#endif
    if (viaBackend) {
        StreamRequest req;
        req.exclusive = n.settings.exclusive;
        req.bufferMs = n.settings.bufferMs;
        req.preferredRate = runningRate_;
        req.safetyMs = n.settings.safetyMs;
        if (n.kind == DeviceKind::Vban) {
            dev.kind = DeviceKind::Vban;
            dev.input = n.bin == Bin::VirtualOut;  // network input = a source
            dev.channels = n.channels;
            dev.name = n.name;
            req.host = n.settings.host;
            req.port = n.settings.port;
            req.streamName = n.settings.streamName;
        }
        Backend* be = backendFor(dev.kind);
        if (!be) return false;
        std::string err;
        n.stream = be->open(dev, req, err);
        if (!n.stream) {
            n.streamError = err.empty() ? "Could not open device" : err;
            TS_WARN("%s: %s", n.name.c_str(), n.streamError.c_str());
            return false;
        }
    }
    n.streamInput = dev.input;

    // Device may report a different channel count than its property store.
    if (n.stream->channels() != n.channels) {
        n.channels = n.stream->channels();
        if (n.channelNames.size() != static_cast<size_t>(n.channels)) {
            n.channelNames.clear();
            for (int c = 0; c < n.channels; ++c) n.channelNames.push_back(std::to_string(c + 1));
        }
        if (n.rt) retireLocked(n.rt);
        n.rt.reset();
    }
    if (!n.rt) {
        n.rt = std::make_shared<NodeRuntime>();
        n.rt->id = n.id;
        n.rt->bin = n.bin;
        n.rt->allocate(n.channels, runningRate_);
        applyNodeParamsLocked(n);
    }
    BridgeConfig cfg;
    cfg.channels = static_cast<size_t>(n.channels);
    cfg.deviceRate = n.stream->sampleRate();
    cfg.engineRate = runningRate_;
    cfg.devicePeriod = static_cast<size_t>(std::max(1, n.stream->periodFrames()));
    cfg.engineMaxBlock = static_cast<size_t>(settings_.blockSize);
    cfg.extraSafetyMs = n.settings.safetyMs;
    cfg.jittery = n.kind == DeviceKind::App || n.kind == DeviceKind::Vban;
    n.streamRate = n.stream->sampleRate();
    if (n.streamInput) {
        n.cap = std::make_shared<CaptureBridge>();
        n.cap->init(cfg);
        n.rt->capture.store(n.cap.get(), std::memory_order_release);
    } else {
        n.ren = std::make_shared<RenderBridge>();
        n.ren->init(cfg);
        n.rt->render.store(n.ren.get(), std::memory_order_release);
    }
    if (!n.stream->start(n.cap, n.ren)) {
        n.streamError = "Device failed to start";
        closeStreamLocked(n);
        return false;
    }
    n.streamError.clear();
    TS_INFO("Opened %s (%.0f Hz, %d ch, period %d)", n.name.c_str(), n.streamRate, n.channels, n.stream->periodFrames());
    return true;
}

void Engine::closeStreamLocked(NodeModel& n) {
    if (n.stream) {
        n.stream->stop();
        n.stream.reset();
        TS_INFO("Closed %s", n.name.c_str());
    }
    if (n.rt) {
        n.rt->capture.store(nullptr, std::memory_order_release);
        n.rt->render.store(nullptr, std::memory_order_release);
    }
    if (n.cap) retireLocked(std::move(n.cap));
    if (n.ren) retireLocked(std::move(n.ren));
    n.cap.reset();
    n.ren.reset();
}

void Engine::updateStreamsLocked() {
#ifdef _WIN32
    // Per-app redirections first: app streams depend on which virtual cable they were given.
    if (apps_) apps_->applyRouting(*this);
#endif
    for (const std::string& id : nodeOrder_) {
        NodeModel& n = nodes_[id];
        if (n.bin == Bin::Bus || n.id == "tone") continue;
        if (nodeNeededLocked(n)) {
            if (!n.stream && n.streamError.empty()) openStreamLocked(n);
        } else if (n.stream) {
            closeStreamLocked(n);
        }
    }
}

void Engine::rebuildLocked() {
    updateStreamsLocked();
    compileAndPublishLocked();
    updateExternalMetersLocked();
    markDirty();
}

std::string Engine::chooseClockMasterLocked() const { return "internal"; }

// ============================================================ compile & publish
void Engine::compileAndPublishLocked() {
    // Every node gets a runtime (so meters and buses work even without devices).
    for (const std::string& id : nodeOrder_) {
        NodeModel& n = nodes_[id];
        if (!n.rt) {
            n.rt = std::make_shared<NodeRuntime>();
            n.rt->id = n.id;
            n.rt->bin = n.bin;
            n.rt->allocate(n.channels, runningRate_);
            applyNodeParamsLocked(n);
        }
        if (n.id == "tone") n.rt->tone = tone_;
        if (!n.pendingFx.empty()) instantiatePendingFxLocked(n);
    }

    // Meter layout: stable while the node set (and channel counts) stays the same.
    std::string sig;
    int off = 0;
    for (const std::string& id : nodeOrder_) {
        NodeModel& n = nodes_[id];
        n.meterOffset = off;
        off += n.channels * 2;
        sig += id + ":" + std::to_string(n.channels) + ";";
    }
    if (!meterBank_ || sig != meterSig_) {
        meterSig_ = sig;
        meterBank_ = std::make_shared<MeterBank>(static_cast<size_t>(off), ++layoutVersion_);
    }

    auto p = std::make_shared<Program>();
    p->sampleRate = runningRate_;
    p->meters = meterBank_;

    auto included = [&](const NodeModel& n) { return n.rt && (n.present || n.bin == Bin::Bus || n.kind == DeviceKind::Vban); };
    std::map<std::string, size_t> index;
    auto add = [&](NodeModel& n) {
        ProgramNode pn;
        pn.rt = n.rt.get();
        pn.isBus = n.bin == Bin::Bus;
        pn.isSource = binIsSource(n.bin);
        pn.isDest = binIsDest(n.bin) && !pn.isBus;
        pn.physicalOut = n.bin == Bin::PhysicalOut;
        pn.fx = n.fx;
        pn.meterOffset = n.meterOffset;
        index[n.id] = p->nodes.size();
        p->nodes.push_back(std::move(pn));
        p->keepNodes.push_back(n.rt);
    };
    // 1) pure sources
    for (const std::string& id : nodeOrder_) {
        NodeModel& n = nodes_[id];
        if (included(n) && n.bin != Bin::Bus && binIsSource(n.bin)) add(n);
    }
    // 2) buses in topological order (cycles fall back to one-block feedback)
    std::vector<std::string> buses;
    for (const std::string& id : nodeOrder_)
        if (nodes_[id].bin == Bin::Bus && included(nodes_[id])) buses.push_back(id);
    std::map<std::string, int> indeg;
    for (auto& b : buses) indeg[b] = 0;
    for (auto& r : routes_)
        if (indeg.count(r.from) && indeg.count(r.to)) indeg[r.to]++;
    std::vector<std::string> order;
    std::set<std::string> placed;
    while (order.size() < buses.size()) {
        bool progressed = false;
        for (auto& b : buses) {
            if (placed.count(b) || indeg[b] > 0) continue;
            order.push_back(b);
            placed.insert(b);
            progressed = true;
            for (auto& r : routes_)
                if (r.from == b && indeg.count(r.to)) indeg[r.to]--;
        }
        if (!progressed) {
            for (auto& b : buses)
                if (!placed.count(b)) {
                    order.push_back(b);
                    placed.insert(b);
                    for (auto& r : routes_)
                        if (r.from == b && indeg.count(r.to)) indeg[r.to]--;
                    break;
                }
        }
    }
    for (auto& b : order) add(nodes_[b]);
    // 3) destinations
    for (const std::string& id : nodeOrder_) {
        NodeModel& n = nodes_[id];
        if (included(n) && n.bin != Bin::Bus && binIsDest(n.bin)) add(n);
    }

    // Routes -> incoming lists.
    for (RouteModel& r : routes_) {
        if (!r.rt) {
            r.rt = std::make_shared<RouteRuntime>();
            r.rt->id = r.id;
        }
        r.rt->gainDb.store(r.gainDb);
        r.rt->muted.store(r.muted);
        r.rt->enabled.store(r.enabled);
        auto fi = index.find(r.from), ti = index.find(r.to);
        if (fi == index.end() || ti == index.end()) continue;
        Incoming inc;
        inc.from = p->nodes[fi->second].rt;
        inc.route = r.rt.get();
        for (auto& [f, t, g] : r.xps) inc.xps.push_back({f, t, g <= -119.f ? 0.0f : std::pow(10.0f, g / 20.0f)});
        p->nodes[ti->second].in.push_back(std::move(inc));
        p->keepRoutes.push_back(r.rt);
    }
    for (auto& [id, n] : nodes_) {
        if (n.bin == Bin::Bus && n.solo && n.rt && index.count(id)) p->soloBuses.push_back(n.rt.get());
        // Ducker sidechains.
        for (auto& fx : n.fx) {
            if (fx->sidechainId.empty()) {
                fx->sidechain.store(nullptr);
                continue;
            }
            NodeModel* sc = findNode(fx->sidechainId);
            if (sc && sc->rt) {
                fx->sidechainChannels.store(sc->rt->channels);
                fx->sidechain.store(sc->rt->buf);
            } else {
                fx->sidechain.store(nullptr);
            }
        }
    }

    // Publish (RCU): the previous program stays alive until the audio thread has moved on.
    program_.store(p.get(), std::memory_order_release);
    if (programOwner_) retireLocked(programOwner_);
    programOwner_ = std::move(p);
    reclaimLocked();
    updateTapsLocked();
}

// Recording taps: nodes being recorded, plus every mix (or the outputs when there are no mixes)
// for the instant-replay buffer.
void Engine::updateTapsLocked() {
    std::vector<Recorder::Tap> taps;
    auto add = [&](const std::string& id) {
        auto it = nodes_.find(id);
        if (it == nodes_.end() || !it->second.rt) return;
        for (auto& t : taps)
            if (t.nodeId == id) return;
        const NodeModel& n = it->second;
        taps.push_back({id, n.customName.empty() ? n.name : n.customName, n.rt, n.channels});
    };
    for (auto& id : recordingNodes_) add(id);
    if (recorder_->replaySeconds() > 0) {
        bool anyBus = false;
        for (auto& id : nodeOrder_)
            if (nodes_[id].bin == Bin::Bus) {
                add(id);
                anyBus = true;
            }
        if (!anyBus)
            for (auto& id : nodeOrder_)
                if (nodes_[id].bin == Bin::PhysicalOut && nodes_[id].present) add(id);
    }
    recorder_->setTaps(taps, runningRate_);
}

// ============================================================ clock
void Engine::startClock() {
    clockRun_ = true;
    clock_ = std::thread([this] { clockThread(); });
}

void Engine::stopClock() {
    clockRun_ = false;
    if (clock_.joinable()) clock_.join();
}

void Engine::cycle(int frames, double now) {
    const uint64_t k = cyclesStarted_.fetch_add(1, std::memory_order_acq_rel) + 1;
    Program* p = program_.load(std::memory_order_acquire);
    if (p) processProgram(*p, frames, now);
    cyclesDone_.store(k, std::memory_order_release);
}

void Engine::clockThread() {
    setAudioThreadPriority("ts-engine");
#ifdef _WIN32
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (!timer) timer = CreateWaitableTimerW(nullptr, FALSE, nullptr);
    timeBeginPeriod(1);
#endif
    const double rate = runningRate_;
    const int block = std::clamp(settings_.blockSize, 16, kMaxBlock);
    const double t0 = nowSeconds();
    uint64_t processed = 0;
    double busy = 0, window = 0, winStart = t0;
    while (clockRun_.load(std::memory_order_relaxed)) {
        const double now = nowSeconds();
        const uint64_t due = static_cast<uint64_t>((now - t0) * rate);
        if (due > processed + static_cast<uint64_t>(rate * 0.25)) {
            // We fell far behind (system sleep / debugger): resynchronise instead of bursting.
            processed = due - static_cast<uint64_t>(block);
            lateCycles_.fetch_add(1);
        }
        while (processed + static_cast<uint64_t>(block) <= due) {
            const double s = nowSeconds();
            cycle(block, s);
            busy += nowSeconds() - s;
            processed += static_cast<uint64_t>(block);
        }
        window = now - winStart;
        if (window >= 0.5) {
            cpuPercent_.store(static_cast<float>(busy / window * 100.0));
            busy = 0;
            winStart = now;
        }
        const double next = t0 + static_cast<double>(processed + static_cast<uint64_t>(block)) / rate;
        const double wait = next - nowSeconds();
        if (wait > 0) {
#ifdef _WIN32
            LARGE_INTEGER li;
            li.QuadPart = -static_cast<LONGLONG>(wait * 1e7);
            SetWaitableTimer(timer, &li, 0, nullptr, nullptr, FALSE);
            WaitForSingleObject(timer, 100);
#else
            timespec ts{static_cast<time_t>(wait), static_cast<long>((wait - static_cast<long>(wait)) * 1e9)};
            nanosleep(&ts, nullptr);
#endif
        }
    }
#ifdef _WIN32
    timeEndPeriod(1);
    CloseHandle(timer);
#endif
}

// ============================================================ maintenance
void Engine::maintenanceThread() {
    setThreadName("ts-maint");
#ifdef _WIN32
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
#endif
    double lastSave = nowSeconds(), lastApps = 0, lastMeters = 0;
    while (!shutdown_) {
        {
            std::unique_lock lk(maintMu_);
            maintCv_.wait_for(lk, std::chrono::milliseconds(250));
        }
        if (shutdown_) break;
        std::lock_guard lock(mu_);
        bool changed = false;
        for (auto& be : backends_)
            if (be->poll()) devicesChanged_ = true;
        if (devicesChanged_.exchange(false)) {
            // Let Windows settle (several notifications arrive per change).
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            for (auto& [id, n] : nodes_) n.streamError.clear();
            refreshDevicesLocked();
            changed = true;
        }
        for (auto& [id, n] : nodes_) {
            if (n.stream && !n.stream->error().empty()) {
                n.streamError = n.stream->error();
                if (n.streamError.rfind("restart:", 0) == 0) {
                    closeStreamLocked(n);  // e.g. format change: reopen right away
                    n.streamError.clear();
                    changed = true;
                    continue;
                }
                TS_WARN("%s: %s", n.name.c_str(), n.streamError.c_str());
                closeStreamLocked(n);
                changed = true;
            }
        }
        const double now = nowSeconds();
#ifdef _WIN32
        if (apps_ && now - lastApps > 1.0) {
            lastApps = now;
            if (apps_->poll()) {
                refreshDevicesLocked();
                changed = true;
            }
        }
#endif
        if (changed) rebuildLocked();
        reclaimLocked();
        if (changed || now - lastMeters > 1.0) {
            lastMeters = now;
            updateExternalMetersLocked();
        }
        if (configDirty_ && now - lastSave > 1.0) {
            saveConfigLocked();
            lastSave = now;
        }
        if (onStateChanged) onStateChanged();  // live stats (cpu, xruns, gain reduction)
    }
#ifdef _WIN32
    CoUninitialize();
#endif
}

void Engine::updateExternalMetersLocked() {
#ifdef _WIN32
    if (!peaks_) return;
    std::vector<PeakMeters::Target> targets;
    for (auto& [id, n] : nodes_) {
        if (!n.rt) continue;
        // Only real Windows devices/apps have an OS peak meter (mixes keep the default kind!).
        const bool osDevice = n.bin != Bin::Bus && id != "tone" && !n.device.endpointId.empty() &&
                              (n.kind == DeviceKind::Wasapi || n.kind == DeviceKind::Compat);
        const bool osApp = n.kind == DeviceKind::App && !n.app.pids.empty();
        const bool candidate = n.present && !n.stream && n.meterOffset >= 0 && (osDevice || osApp);
        if (candidate) {
            PeakMeters::Target t;
            t.offset = n.meterOffset;
            t.channels = n.channels;
            if (n.kind == DeviceKind::App) {
                t.pids = n.app.pids;
                t.capture = n.bin == Bin::AppIn;
            } else {
                t.endpointId = n.device.endpointId;
            }
            targets.push_back(std::move(t));
        }
        n.rt->externalMeter.store(candidate, std::memory_order_relaxed);
    }
    peaks_->setTargets(std::move(targets));
#endif
}

void Engine::pollExternalMeters() {
#ifdef _WIN32
    std::shared_ptr<MeterBank> bank;
    {
        std::lock_guard lock(mu_);
        bank = meterBank_;
    }
    if (bank && peaks_) peaks_->poll(*bank, bank->layoutVersion);
#endif
}

uint32_t Engine::meters(std::vector<float>& out) {
    std::shared_ptr<MeterBank> bank;
    {
        std::lock_guard lock(mu_);
        bank = meterBank_;
    }
    if (!bank) return 0;
    out.resize(bank->size);
    for (size_t i = 0; i < bank->size; ++i) out[i] = bank->values[i].load(std::memory_order_relaxed);
    return bank->layoutVersion;
}

std::vector<std::tuple<int, int, float>> Engine::defaultCrosspoints(const NodeModel& a, const NodeModel& b, const std::string& mode) const {
    std::vector<std::tuple<int, int, float>> x;
    const int nf = a.channels, nt = b.channels;
    if (mode == "mono-sum") {
        const float g = nf > 1 ? static_cast<float>(-10.0 * std::log10(static_cast<double>(nf))) : 0.0f;
        for (int f = 0; f < nf; ++f)
            for (int t = 0; t < nt; ++t) x.emplace_back(f, t, std::round(g * 10) / 10);
    } else if (mode == "spread" || (mode == "auto" && nf == 1)) {
        for (int f = 0; f < nf; ++f)
            for (int t = 0; t < nt; ++t)
                if (nf == 1 || f % nt == t) x.emplace_back(f, t, 0.0f);
    } else if (mode == "auto" && nf > nt && nt == 2) {
        // Surround -> stereo downmix: L,R full; centre/LFE/surrounds at -3 dB.
        for (int f = 0; f < nf; ++f) {
            if (f == 0) x.emplace_back(0, 0, 0.0f);
            else if (f == 1) x.emplace_back(1, 1, 0.0f);
            else if (f == 2) { x.emplace_back(2, 0, -3.0f); x.emplace_back(2, 1, -3.0f); }
            else if (f == 3) continue;  // LFE dropped
            else x.emplace_back(f, f % 2, -3.0f);
        }
    } else {
        for (int i = 0; i < std::min(nf, nt); ++i) x.emplace_back(i, i, 0.0f);
    }
    return x;
}

}  // namespace ts
