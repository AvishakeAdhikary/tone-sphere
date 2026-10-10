// RPC method dispatch (see shared/protocol/PROTOCOL.md). Errors are thrown as
// std::runtime_error with messages meant for end users.
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/engine.h"
#include "util/log.h"
#include "util/platform.h"

#ifdef _WIN32
#include "backends/win/app_audio.h"
#include "backends/win/asio_host.h"
#endif

namespace nt {

Bin binFromString(const std::string& s);

namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error(msg); }

std::string str(const json& p, const char* k) {
    if (!p.contains(k) || !p[k].is_string()) fail(std::string("Missing parameter: ") + k);
    return p[k].get<std::string>();
}

}  // namespace

json Engine::call(const std::string& method, const json& params) {
    std::lock_guard lock(mu_);
    const json& p = params.is_object() ? params : json::object();

    // ------------------------------------------------------------ engine
    if (method == "engine.getState") return state();
    if (method == "engine.shutdown") {
        if (onShutdownRequested) onShutdownRequested();
        return nullptr;
    }
    if (method == "engine.restart" || method == "engine.setSettings") {
        bool restartAudio = method == "engine.restart";
        bool refresh = false;
        if (p.contains("sampleRate") && p["sampleRate"].get<double>() != settings_.sampleRate) {
            settings_.sampleRate = std::clamp(p["sampleRate"].get<double>(), 8000.0, 384000.0);
            restartAudio = true;
        }
        if (p.contains("blockSize") && p["blockSize"].get<int>() != settings_.blockSize) {
            settings_.blockSize = std::clamp(p["blockSize"].get<int>(), 16, kMaxBlock);
            restartAudio = true;
        }
        if (p.contains("clockMaster")) settings_.clockMaster = p["clockMaster"].get<std::string>();
        if (p.contains("compatMode") && p["compatMode"].get<bool>() != settings_.compatMode) {
            settings_.compatMode = p["compatMode"].get<bool>();
            refresh = true;
        }
        if (p.contains("startWithSystem")) settings_.startWithSystem = p["startWithSystem"].get<bool>();
        if (p.contains("startMinimized")) settings_.startMinimized = p["startMinimized"].get<bool>();
        if (p.contains("hotkeys") && p["hotkeys"].is_object()) settings_.hotkeys = p["hotkeys"].get<std::map<std::string, std::string>>();
        if (p.contains("replayBufferSeconds")) {
            settings_.replayBufferSeconds = std::clamp(p["replayBufferSeconds"].get<int>(), 0, 3600);
            recorder_->setReplaySeconds(settings_.replayBufferSeconds);
            updateTapsLocked();
        }
        if (p.contains("recordFolder")) settings_.recordFolder = p["recordFolder"].get<std::string>();
        if (p.contains("theme")) settings_.theme = p["theme"].get<std::string>();
        if (p.contains("advancedMode")) settings_.advancedMode = p["advancedMode"].get<bool>();
        if (restartAudio) {
            NT_INFO("Restarting audio engine");
            stopClock();
            for (auto& [id, n] : nodes_) {
                closeStreamLocked(n);
                n.streamError.clear();
                if (n.rt) retireLocked(n.rt);
                n.rt.reset();
                for (auto& f : n.fx) f->prepare(settings_.sampleRate, kMaxBlock, n.channels);
            }
            runningRate_ = settings_.sampleRate;
            for (auto& [id, n] : nodes_)
                for (auto& f : n.fx) f->prepare(runningRate_, kMaxBlock, n.channels);
            rebuildLocked();
            startClock();
        } else if (refresh) {
            refreshDevicesLocked();
            rebuildLocked();
        } else {
            markDirty();
        }
        return state()["settings"];
    }
    if (method == "devices.refresh") {
        for (auto& [id, n] : nodes_) n.streamError.clear();
        refreshDevicesLocked();
        rebuildLocked();
        return nullptr;
    }

    // ------------------------------------------------------------ nodes
    if (method == "node.update") {
        NodeModel* n = findNode(str(p, "id"));
        if (!n) fail("That device or app no longer exists");
        bool reopen = false, rebuild = false;
        if (p.contains("gainDb")) n->gainDb = std::clamp(p["gainDb"].get<float>(), -120.0f, 24.0f);
        if (p.contains("muted")) n->muted = p["muted"].get<bool>();
        if (p.contains("solo")) {
            n->solo = p["solo"].get<bool>();
            rebuild = true;
        }
        if (p.contains("enabled")) {
            n->userEnabled = p["enabled"].get<bool>();
            rebuild = true;
        }
        if (p.contains("name")) {
            std::string nm = p["name"].get<std::string>();
            if (n->bin == Bin::Bus || n->kind == DeviceKind::Vban) n->name = nm, n->customName.clear();
            else n->customName = nm == n->name ? std::string() : nm;
        }
        if (p.contains("settings") && p["settings"].is_object()) {
            const json& s = p["settings"];
            NodeSettingsData& d = n->settings;
            if (s.contains("exclusive") && s["exclusive"].get<bool>() != d.exclusive) d.exclusive = s["exclusive"].get<bool>(), reopen = true;
            if (s.contains("bufferMs") && s["bufferMs"].get<double>() != d.bufferMs) d.bufferMs = std::clamp(s["bufferMs"].get<double>(), 1.0, 500.0), reopen = true;
            if (s.contains("safetyMs") && s["safetyMs"].get<double>() != d.safetyMs) d.safetyMs = std::clamp(s["safetyMs"].get<double>(), 0.0, 1000.0), reopen = true;
            if (s.contains("delayMs")) d.delayMs = std::clamp(s["delayMs"].get<double>(), 0.0, 2000.0);
            if (s.contains("pan")) d.pan = std::clamp(s["pan"].get<double>(), -1.0, 1.0);
            if (s.contains("invert")) d.invert = s["invert"].get<bool>();
            if (s.contains("host")) d.host = s["host"].get<std::string>(), reopen = true;
            if (s.contains("port")) d.port = s["port"].get<int>(), reopen = true;
            if (s.contains("streamName")) d.streamName = s["streamName"].get<std::string>().substr(0, 16), reopen = true;
        }
        if (n->kind == DeviceKind::Vban)
            n->subtitle = n->settings.streamName + " · " + n->settings.host + ":" + std::to_string(n->settings.port);
        applyNodeParamsLocked(*n);
        if (reopen) {
            closeStreamLocked(*n);
            n->streamError.clear();
            rebuild = true;
        }
        if (rebuild) rebuildLocked();
        else markDirty();
        return nullptr;
    }
    if (method == "node.openControlPanel") {
        NodeModel* n = findNode(str(p, "id"));
        if (!n || n->kind != DeviceKind::Asio) fail("This device has no control panel");
#ifdef _WIN32
        if (auto* be = dynamic_cast<AsioHostBackend*>(backendFor(DeviceKind::Asio))) {
            std::string err;
            if (!be->controlPanel(n->device.endpointId, err)) fail(err);
            return nullptr;
        }
#endif
        fail("ASIO is not available");
    }

    // ------------------------------------------------------------ routes
    if (method == "route.connect") {
        const std::string from = str(p, "from"), to = str(p, "to");
        NodeModel* a = findNode(from);
        NodeModel* b = findNode(to);
        if (!a || !b) fail("That device or app no longer exists");
        if (from == to) fail("A device can't be connected to itself");
        if (!binIsSource(a->bin)) fail("“" + a->name + "” can't send audio");
        if (!binIsDest(b->bin)) fail("“" + b->name + "” can't receive audio");
        for (auto& r : routes_)
            if (r.from == from && r.to == to) return json{{"id", r.id}, {"from", r.from}, {"to", r.to}};
        // A compatibility cable carries one direction at a time: refuse feedback loops.
        if (a->kind == DeviceKind::Compat || b->kind == DeviceKind::Compat) {
            const NodeModel* cable = a->kind == DeviceKind::Compat ? a : b;
            for (auto& r : routes_) {
                const NodeModel* x = findNode(r.from);
                const NodeModel* y = findNode(r.to);
                for (const NodeModel* m : {x, y}) {
                    if (m && m != cable && m->kind == DeviceKind::Compat && m->subtitle == cable->subtitle && m->bin != cable->bin)
                        fail("“" + cable->subtitle + "” is already in use in the other direction. A virtual cable can carry sound only one way at a time.");
                }
            }
        }
        RouteModel r;
        r.id = newId("r");
        r.from = from;
        r.to = to;
        r.xps = defaultCrosspoints(*a, *b, p.value("mode", "auto"));
        routes_.push_back(r);
        rebuildLocked();
        NT_INFO("Connected %s -> %s", a->name.c_str(), b->name.c_str());
        return json{{"id", r.id}, {"from", from}, {"to", to}, {"gainDb", 0}, {"muted", false}, {"enabled", true}};
    }
    if (method == "route.disconnect") {
        const std::string id = str(p, "id");
        auto it = std::find_if(routes_.begin(), routes_.end(), [&](const RouteModel& r) { return r.id == id; });
        if (it == routes_.end()) fail("Connection not found");
        if (it->rt) retireLocked(it->rt);
        routes_.erase(it);
        rebuildLocked();
        return nullptr;
    }
    if (method == "route.update") {
        RouteModel* r = findRoute(str(p, "id"));
        if (!r) fail("Connection not found");
        if (p.contains("gainDb")) r->gainDb = std::clamp(p["gainDb"].get<float>(), -120.0f, 24.0f);
        if (p.contains("muted")) r->muted = p["muted"].get<bool>();
        if (p.contains("enabled")) r->enabled = p["enabled"].get<bool>();
        if (r->rt) {
            r->rt->gainDb.store(r->gainDb);
            r->rt->muted.store(r->muted);
            r->rt->enabled.store(r->enabled);
        }
        markDirty();
        return nullptr;
    }
    if (method == "route.setCrosspoint" || method == "route.applyPreset") {
        RouteModel* r = findRoute(str(p, "id"));
        if (!r) fail("Connection not found");
        if (method == "route.setCrosspoint") {
            const int f = p.value("fromCh", 0), t = p.value("toCh", 0);
            r->xps.erase(std::remove_if(r->xps.begin(), r->xps.end(), [&](auto& x) { return std::get<0>(x) == f && std::get<1>(x) == t; }), r->xps.end());
            if (p.contains("gainDb") && !p["gainDb"].is_null()) r->xps.emplace_back(f, t, std::clamp(p["gainDb"].get<float>(), -120.0f, 24.0f));
        } else {
            const std::string mode = p.value("mode", "one-to-one");
            NodeModel* a = findNode(r->from);
            NodeModel* b = findNode(r->to);
            r->xps = mode == "clear" || !a || !b ? decltype(r->xps){} : defaultCrosspoints(*a, *b, mode);
        }
        compileAndPublishLocked();
        markDirty();
        return nullptr;
    }

    // ------------------------------------------------------------ buses
    if (method == "bus.create") {
        const std::string id = "bus:" + randomHex(4);
        NodeModel& n = ensureNode(id);
        n.bin = Bin::Bus;
        n.name = p.value("name", "Mix");
        if (n.name.empty()) n.name = "Mix";
        n.channels = std::clamp(p.value("channels", 2), 1, 16);
        n.channelNames = n.channels == 2 ? std::vector<std::string>{"L", "R"} : n.channels == 1 ? std::vector<std::string>{"Mono"} : std::vector<std::string>{};
        if (n.channelNames.empty())
            for (int c = 0; c < n.channels; ++c) n.channelNames.push_back(std::to_string(c + 1));
        n.persistent = true;
        n.present = true;
        rebuildLocked();
        return nodeJsonLocked(n);
    }
    if (method == "bus.delete" || method == "vban.remove") {
        const std::string id = str(p, "id");
        NodeModel* n = findNode(id);
        if (!n) fail("Not found");
        closeStreamLocked(*n);
        for (auto& r : routes_)
            if ((r.from == id || r.to == id) && r.rt) retireLocked(r.rt);
        routes_.erase(std::remove_if(routes_.begin(), routes_.end(), [&](const RouteModel& r) { return r.from == id || r.to == id; }), routes_.end());
        if (n->rt) retireLocked(n->rt);
        nodes_.erase(id);
        nodeOrder_.erase(std::remove(nodeOrder_.begin(), nodeOrder_.end(), id), nodeOrder_.end());
        rebuildLocked();
        return nullptr;
    }

    // ------------------------------------------------------------ effects
    if (method == "fx.add") {
        NodeModel* n = findNode(str(p, "nodeId"));
        if (!n) fail("Not found");
        const std::string type = str(p, "type");
        std::shared_ptr<Fx> fx = createBuiltinFx(type);
        if (!fx) fail(type == "vst3" ? "VST3 plugin hosting is not available in this build yet" : "Unknown effect");
        fx->id = newId("fx");
        fx->prepare(runningRate_, kMaxBlock, n->channels);
        const size_t idx = std::min<size_t>(p.value("index", n->fx.size()), n->fx.size());
        n->fx.insert(n->fx.begin() + static_cast<long>(idx), fx);
        compileAndPublishLocked();
        markDirty();
        return json{{"id", fx->id}, {"type", fx->type()}, {"name", fx->name()}};
    }
    if (method == "fx.remove" || method == "fx.update" || method == "fx.move" || method == "fx.openEditor") {
        NodeModel* n = findNode(str(p, "nodeId"));
        if (!n) fail("Not found");
        const std::string fxId = str(p, "fxId");
        auto it = std::find_if(n->fx.begin(), n->fx.end(), [&](auto& f) { return f->id == fxId; });
        if (it == n->fx.end()) fail("Effect not found");
        if (method == "fx.remove") {
            n->fx.erase(it);
            compileAndPublishLocked();
        } else if (method == "fx.move") {
            auto fx = *it;
            n->fx.erase(it);
            const size_t idx = std::min<size_t>(static_cast<size_t>(std::max(0, p.value("index", 0))), n->fx.size());
            n->fx.insert(n->fx.begin() + static_cast<long>(idx), fx);
            compileAndPublishLocked();
        } else if (method == "fx.openEditor") {
            (*it)->openEditor();
        } else {
            if (p.contains("bypass")) (*it)->bypass.store(p["bypass"].get<bool>());
            if (p.contains("params") && p["params"].is_object())
                for (auto& [k, v] : p["params"].items())
                    if (v.is_number()) (*it)->setParam(k, v.get<float>());
            if (p.contains("sidechain")) {
                (*it)->sidechainId = p["sidechain"].is_string() ? p["sidechain"].get<std::string>() : "";
                compileAndPublishLocked();
            }
        }
        markDirty();
        return nullptr;
    }
    if (method == "plugins.scan") return nullptr;

    // ------------------------------------------------------------ apps
#ifdef _WIN32
    if (method == "app.setMode" || method == "app.setRule") {
        if (!apps_) fail("Per-app audio is not available");
        apps_->handleRpc(*this, method, p);
        rebuildLocked();
        return nullptr;
    }
#endif

    // ------------------------------------------------------------ scenes
    if (method == "scene.save") {
        const std::string name = p.value("name", "Scene");
        std::string id = p.value("id", "");
        SceneModel* s = nullptr;
        for (auto& sc : scenes_)
            if (sc.id == id) s = &sc;
        if (!s) {
            scenes_.push_back({newId("sc"), name, "", json::object()});
            s = &scenes_.back();
        }
        s->name = name;
        if (p.contains("hotkey")) s->hotkey = p["hotkey"].is_string() ? p["hotkey"].get<std::string>() : std::string();
        s->snapshot = snapshotLocked();
        activeScene_ = s->id;
        markDirty();
        return json{{"id", s->id}, {"name", s->name}};
    }
    if (method == "scene.load") {
        const std::string id = str(p, "id");
        for (auto& sc : scenes_) {
            if (sc.id != id) continue;
            restoreSnapshotLocked(sc.snapshot);
            activeScene_ = id;
            rebuildLocked();
            return nullptr;
        }
        fail("Scene not found");
    }
    if (method == "scene.delete") {
        const std::string id = str(p, "id");
        scenes_.erase(std::remove_if(scenes_.begin(), scenes_.end(), [&](auto& s) { return s.id == id; }), scenes_.end());
        if (activeScene_ == id) activeScene_.clear();
        markDirty();
        return nullptr;
    }

    // ------------------------------------------------------------ tone / misc
    if (method == "tone.set") {
        tone_->enabled.store(p.value("enabled", false));
        tone_->freq.store(std::clamp(p.value("freq", 440.0f), 10.0f, 20000.0f));
        tone_->levelDb.store(std::clamp(p.value("levelDb", -18.0f), -90.0f, 0.0f));
        if (p.contains("target") && p["target"].is_string()) {
            const std::string to = p["target"].get<std::string>();
            const bool exists = std::any_of(routes_.begin(), routes_.end(), [&](auto& r) { return r.from == "tone" && r.to == to; });
            if (!exists && findNode(to)) {
                RouteModel r;
                r.id = newId("r");
                r.from = "tone";
                r.to = to;
                r.xps = defaultCrosspoints(*findNode("tone"), *findNode(to), "auto");
                routes_.push_back(r);
                rebuildLocked();
            }
        }
        markDirty();
        return nullptr;
    }
    if (method == "vban.add") {
        const bool in = p.value("direction", "in") == "in";
        const std::string id = std::string(in ? "vban:in:" : "vban:out:") + randomHex(4);
        NodeModel& n = ensureNode(id);
        n.kind = DeviceKind::Vban;
        n.bin = in ? Bin::VirtualOut : Bin::VirtualIn;  // network input is a source, output a destination
        n.name = p.value("name", in ? std::string("Network in") : std::string("Network out"));
        n.channels = std::clamp(p.value("channels", 2), 1, 16);
        n.channelNames.clear();
        if (n.channels == 2) n.channelNames = {"L", "R"};
        else for (int c = 0; c < n.channels; ++c) n.channelNames.push_back(std::to_string(c + 1));
        n.settings.host = p.value("host", in ? std::string("0.0.0.0") : std::string("127.0.0.1"));
        n.settings.port = std::clamp(p.value("port", 6980), 1, 65535);
        n.settings.streamName = p.value("streamName", std::string("Stream1")).substr(0, 16);
        n.settings.safetyMs = 20.0;  // network jitter cushion
        n.subtitle = n.settings.streamName + " · " + n.settings.host + ":" + std::to_string(n.settings.port);
        n.persistent = true;
        n.present = true;
        rebuildLocked();
        return nodeJsonLocked(n);
    }
    if (method == "recorder.start") {
        std::vector<std::string> ids;
        if (p.contains("nodeIds") && p["nodeIds"].is_array())
            for (auto& v : p["nodeIds"])
                if (v.is_string() && findNode(v.get<std::string>())) ids.push_back(v.get<std::string>());
        if (ids.empty()) fail("Choose something to record");
        if (p.value("format", "wav") != "wav") fail("Only WAV recording is available in this version");
        recordingNodes_ = ids;
        updateTapsLocked();
        std::string err;
        const std::string folder = p.value("folder", settings_.recordFolder);
        if (!recorder_->start(ids, folder.empty() ? defaultRecordFolder() : folder, err)) {
            recordingNodes_.clear();
            updateTapsLocked();
            fail(err);
        }
        markDirty();
        return nullptr;
    }
    if (method == "recorder.stop") {
        auto files = recorder_->stop();
        recordingNodes_.clear();
        updateTapsLocked();
        lastRecordings_.insert(lastRecordings_.end(), files.begin(), files.end());
        if (lastRecordings_.size() > 20) lastRecordings_.erase(lastRecordings_.begin(), lastRecordings_.end() - 20);
        markDirty();
        return json{{"files", files}};
    }
    if (method == "recorder.saveReplay") {
        if (recorder_->replaySeconds() <= 0) fail("Turn on the instant replay buffer in Settings first");
        std::string err;
        auto files = recorder_->saveReplay(p.value("seconds", recorder_->replaySeconds()), settings_.recordFolder, err);
        if (files.empty()) fail(err);
        lastRecordings_.insert(lastRecordings_.end(), files.begin(), files.end());
        markDirty();
        return json{{"file", files.front()}, {"files", files}};
    }
    if (method == "latency.measure") fail("Latency measurement arrives in a later update of this build");

    fail("Unknown method: " + method);
}

}  // namespace nt
