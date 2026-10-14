// Engine configuration persistence (config.json) and scene snapshots.
#include <algorithm>
#include <cmath>

#include "core/engine.h"
#include "core/serialize.h"
#include "util/log.h"
#include "util/platform.h"

namespace ts {

Bin binFromString(const std::string& s) {
    if (s == "physical-in") return Bin::PhysicalIn;
    if (s == "physical-out") return Bin::PhysicalOut;
    if (s == "virtual-in") return Bin::VirtualIn;
    if (s == "virtual-out") return Bin::VirtualOut;
    if (s == "app-in") return Bin::AppIn;
    if (s == "app-out") return Bin::AppOut;
    return Bin::Bus;
}

DeviceKind kindFromString(const std::string& s) {
    if (s == "asio") return DeviceKind::Asio;
    if (s == "vad") return DeviceKind::Vad;
    if (s == "compat") return DeviceKind::Compat;
    if (s == "app") return DeviceKind::App;
    if (s == "asioclient") return DeviceKind::AsioClient;
    if (s == "vban") return DeviceKind::Vban;
    if (s == "pipewire") return DeviceKind::PipeWire;
    if (s == "coreaudio") return DeviceKind::CoreAudio;
    return DeviceKind::Wasapi;
}

namespace {

std::string configPath() { return dataDir() + "/config.json"; }

}  // namespace

json settingsJson(const NodeSettingsData& s) {
    return {{"exclusive", s.exclusive}, {"bufferMs", s.bufferMs}, {"safetyMs", s.safetyMs}, {"delayMs", s.delayMs},
            {"pan", s.pan},             {"invert", s.invert},     {"host", s.host},         {"port", s.port},
            {"streamName", s.streamName}, {"channelNames", s.channelNames}};
}

void settingsFrom(NodeSettingsData& s, const json& j) {
    s.exclusive = j.value("exclusive", s.exclusive);
    s.bufferMs = j.value("bufferMs", s.bufferMs);
    s.safetyMs = j.value("safetyMs", s.safetyMs);
    s.delayMs = j.value("delayMs", s.delayMs);
    s.pan = j.value("pan", s.pan);
    s.invert = j.value("invert", s.invert);
    s.host = j.value("host", s.host);
    s.port = j.value("port", s.port);
    s.streamName = j.value("streamName", s.streamName);
    if (j.contains("channelNames") && j["channelNames"].is_array()) s.channelNames = j["channelNames"].get<std::vector<std::string>>();
}

json fxJson(const Fx& fx) {
    json params = json::object();
    for (size_t i = 0; i < fx.paramDefs().size(); ++i) params[fx.paramDefs()[i].id] = fx.param(i);
    return {{"id", fx.id},           {"type", fx.type()},         {"pluginId", fx.pluginId},
            {"bypass", fx.bypass.load()}, {"params", params},     {"sidechain", fx.sidechainId},
            {"state", fx.saveState()}, {"name", fx.name()}};
}

FxConfig fxConfigFrom(const json& j) {
    FxConfig c;
    c.id = j.value("id", "");
    c.type = j.value("type", "");
    c.pluginId = j.value("pluginId", "");
    c.bypass = j.value("bypass", false);
    c.sidechain = j.value("sidechain", "");
    c.state = j.value("state", "");
    c.name = j.value("name", "");
    if (j.contains("params") && j["params"].is_object())
        for (auto& [k, v] : j["params"].items())
            if (v.is_number()) c.params[k] = v.get<float>();
    return c;
}

json routeJson(const RouteModel& r) {
    json x = json::array();
    for (auto& [f, t, g] : r.xps) x.push_back({f, t, g});
    return {{"id", r.id}, {"from", r.from}, {"to", r.to}, {"gainDb", r.gainDb}, {"muted", r.muted}, {"enabled", r.enabled}, {"xps", x}};
}

RouteModel routeFrom(const json& j) {
    RouteModel r;
    r.id = j.value("id", newId("r"));
    r.from = j.value("from", "");
    r.to = j.value("to", "");
    r.gainDb = j.value("gainDb", 0.0f);
    r.muted = j.value("muted", false);
    r.enabled = j.value("enabled", true);
    if (j.contains("xps"))
        for (auto& x : j["xps"])
            if (x.is_array() && x.size() == 3) r.xps.emplace_back(x[0].get<int>(), x[1].get<int>(), x[2].get<float>());
    return r;
}

void Engine::saveConfigLocked() {
    json j;
    j["version"] = 1;
    j["settings"] = {{"sampleRate", settings_.sampleRate},   {"blockSize", settings_.blockSize},
                     {"clockMaster", settings_.clockMaster}, {"compatMode", settings_.compatMode},
                     {"startWithSystem", settings_.startWithSystem}, {"startMinimized", settings_.startMinimized},
                     {"hotkeys", settings_.hotkeys},         {"replayBufferSeconds", settings_.replayBufferSeconds},
                     {"recordFolder", settings_.recordFolder}, {"theme", settings_.theme}, {"pluginFolders", settings_.pluginFolders}, {"oscPort", settings_.oscPort}, {"midiMappings", settings_.midiMappings}, {"skin", settings_.skin}, {"customSkins", settings_.customSkins},
                     {"advancedMode", settings_.advancedMode}, {"dspThreads", settings_.dspThreads}, {"midiInstrument", settings_.midiInstrument}};
    json nodes = json::array();
    for (const std::string& id : nodeOrder_) {
        const NodeModel& n = nodes_.at(id);
        if (id == "tone") continue;
        json fx = json::array();
        for (auto& f : n.fx) fx.push_back(fxJson(*f));
        for (auto& pf : n.pendingFx) {
            json params = json::object();
            for (auto& [k, v] : pf.params) params[k] = v;
            fx.push_back({{"id", pf.id}, {"type", pf.type}, {"pluginId", pf.pluginId}, {"bypass", pf.bypass}, {"params", params},
                          {"sidechain", pf.sidechain}, {"state", pf.state}, {"name", pf.name}});
        }
        json nj = {{"id", n.id},
                   {"bin", binName(n.bin)},
                   {"kind", deviceKindName(n.kind)},
                   {"name", n.name},
                   {"customName", n.customName},
                   {"subtitle", n.subtitle},
                   {"channels", n.channels},
                   {"channelNames", n.channelNames},
                   {"persistent", n.persistent},
                   {"userEnabled", n.userEnabled},
                   {"gainDb", n.gainDb},
                   {"muted", n.muted},
                   {"solo", n.solo},
                   {"settings", settingsJson(n.settings)},
                   {"fx", fx}};
        if (n.kind == DeviceKind::App) nj["app"] = {{"exe", n.app.exe}, {"mode", n.app.mode}, {"pinned", n.app.pinned}};
        if (!n.match.is_null()) nj["match"] = n.match;
        nodes.push_back(nj);
    }
    j["nodes"] = nodes;
    json routes = json::array();
    for (auto& r : routes_) routes.push_back(routeJson(r));
    j["routes"] = routes;
    json scenes = json::array();
    for (auto& s : scenes_) scenes.push_back({{"id", s.id}, {"name", s.name}, {"hotkey", s.hotkey}, {"snapshot", s.snapshot}});
    j["scenes"] = scenes;
    j["activeScene"] = activeScene_;
    if (writeFileAtomic(configPath(), j.dump(2))) configDirty_ = false;
}

void Engine::loadConfig() {
    std::string text;
    if (!readFile(configPath(), text)) {
        TS_INFO("No configuration yet; starting fresh");
        return;
    }
    json j;
    try {
        j = json::parse(text);
    } catch (const std::exception& e) {
        TS_ERROR("Configuration is damaged (%s); keeping a backup and starting fresh", e.what());
        writeFileAtomic(configPath() + ".damaged", text);
        return;
    }
    if (j.contains("settings")) {
        const json& s = j["settings"];
        settings_.sampleRate = s.value("sampleRate", settings_.sampleRate);
        settings_.blockSize = s.value("blockSize", settings_.blockSize);
        settings_.clockMaster = s.value("clockMaster", settings_.clockMaster);
        settings_.compatMode = s.value("compatMode", settings_.compatMode);
        settings_.startWithSystem = s.value("startWithSystem", settings_.startWithSystem);
        settings_.startMinimized = s.value("startMinimized", settings_.startMinimized);
        if (s.contains("hotkeys") && s["hotkeys"].is_object()) settings_.hotkeys = s["hotkeys"].get<std::map<std::string, std::string>>();
        settings_.replayBufferSeconds = s.value("replayBufferSeconds", settings_.replayBufferSeconds);
        settings_.recordFolder = s.value("recordFolder", settings_.recordFolder);
    if (s.contains("pluginFolders") && s["pluginFolders"].is_array()) settings_.pluginFolders = s["pluginFolders"].get<std::vector<std::string>>();
    settings_.oscPort = s.value("oscPort", 0);
    if (s.contains("midiMappings") && s["midiMappings"].is_array()) settings_.midiMappings = s["midiMappings"];
    settings_.skin = s.value("skin", settings_.skin);
    if (s.contains("customSkins") && s["customSkins"].is_array()) settings_.customSkins = s["customSkins"];
        settings_.theme = s.value("theme", settings_.theme);
        settings_.advancedMode = s.value("advancedMode", settings_.advancedMode);
        settings_.dspThreads = std::clamp(s.value("dspThreads", settings_.dspThreads), 0, 64);
        if (s.contains("midiInstrument") && s["midiInstrument"].is_object()) settings_.midiInstrument = s["midiInstrument"];
    }
    for (const json& nj : j.value("nodes", json::array())) {
        const std::string id = nj.value("id", "");
        if (id.empty()) continue;
        NodeModel& n = ensureNode(id);
        n.bin = binFromString(nj.value("bin", "bus"));
        n.kind = kindFromString(nj.value("kind", "wasapi"));
        n.name = nj.value("name", id);
        n.customName = nj.value("customName", "");
        n.subtitle = nj.value("subtitle", "");
        n.channels = std::clamp(nj.value("channels", 2), 1, kMaxChannels);
        if (nj.contains("channelNames")) n.channelNames = nj["channelNames"].get<std::vector<std::string>>();
        n.persistent = nj.value("persistent", false);
        n.userEnabled = nj.value("userEnabled", true);
        n.gainDb = nj.value("gainDb", 0.0f);
        n.muted = nj.value("muted", false);
        n.solo = nj.value("solo", false);
        if (nj.contains("settings")) settingsFrom(n.settings, nj["settings"]);
        for (const json& f : nj.value("fx", json::array())) n.pendingFx.push_back(fxConfigFrom(f));
        if (nj.contains("app")) {
            n.app.exe = nj["app"].value("exe", "");
            n.app.mode = nj["app"].value("mode", "copy");
            n.app.pinned = nj["app"].value("pinned", false);
        }
        if (nj.contains("match")) n.match = nj["match"];
        n.present = n.bin == Bin::Bus || n.kind == DeviceKind::Vban;
    }
    for (const json& rj : j.value("routes", json::array())) routes_.push_back(routeFrom(rj));
    for (const json& sj : j.value("scenes", json::array()))
        scenes_.push_back({sj.value("id", newId("sc")), sj.value("name", "Scene"), sj.value("hotkey", ""), sj.value("snapshot", json::object())});
    activeScene_ = j.value("activeScene", "");
    TS_INFO("Loaded configuration: %zu nodes, %zu routes, %zu scenes", nodes_.size(), routes_.size(), scenes_.size());
}

json Engine::snapshotLocked() const {
    json nodes = json::object();
    json buses = json::array();
    for (const std::string& id : nodeOrder_) {
        const NodeModel& n = nodes_.at(id);
        json fx = json::array();
        for (auto& f : n.fx) fx.push_back(fxJson(*f));
        nodes[id] = {{"gainDb", n.gainDb}, {"muted", n.muted}, {"solo", n.solo}, {"userEnabled", n.userEnabled},
                     {"settings", settingsJson(n.settings)}, {"fx", fx}};
        if (n.bin == Bin::Bus) buses.push_back({{"id", id}, {"name", n.customName.empty() ? n.name : n.customName}, {"channels", n.channels}});
    }
    json routes = json::array();
    for (auto& r : routes_) routes.push_back(routeJson(r));
    return {{"nodes", nodes}, {"routes", routes}, {"buses", buses}};
}

void Engine::restoreSnapshotLocked(const json& s) {
    for (const json& b : s.value("buses", json::array())) {
        const std::string id = b.value("id", "");
        if (id.empty() || findNode(id)) continue;
        NodeModel& n = ensureNode(id);
        n.bin = Bin::Bus;
        n.kind = DeviceKind::Wasapi;
        n.name = b.value("name", "Mix");
        n.channels = b.value("channels", 2);
        n.channelNames = n.channels == 2 ? std::vector<std::string>{"L", "R"} : std::vector<std::string>{};
        n.persistent = true;
        n.present = true;
    }
    if (s.contains("nodes")) {
        for (auto& [id, v] : s["nodes"].items()) {
            NodeModel* n = findNode(id);
            if (!n) continue;
            n->gainDb = v.value("gainDb", n->gainDb);
            n->muted = v.value("muted", n->muted);
            n->solo = v.value("solo", n->solo);
            n->userEnabled = v.value("userEnabled", n->userEnabled);
            if (v.contains("settings")) settingsFrom(n->settings, v["settings"]);
            if (v.contains("fx")) {
                n->fx.clear();
                n->pendingFx.clear();
                for (const json& f : v["fx"]) n->pendingFx.push_back(fxConfigFrom(f));
            }
            applyNodeParamsLocked(*n);
        }
    }
    for (auto& r : routes_)
        if (r.rt) retireLocked(r.rt);
    routes_.clear();
    for (const json& rj : s.value("routes", json::array())) routes_.push_back(routeFrom(rj));
}

}  // namespace ts
