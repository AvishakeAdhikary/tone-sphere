// Serialises the engine model into the protocol's EngineState (see ui/src/shared/protocol.ts).
#include <algorithm>

#include "core/engine.h"
#include "plugins/vst3_host.h"
#include "util/platform.h"

#ifdef _WIN32
#include <windows.h>

#include "backends/win/app_audio.h"
#endif

namespace nt {

namespace {

// The NexTone ASIO driver is registered by the desktop installer (nextone-setup).
bool asioDriverInstalled() {
#ifdef _WIN32
    HKEY k;
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\ASIO\\NexTone ASIO", 0, KEY_READ | KEY_WOW64_64KEY, &k) != ERROR_SUCCESS) return false;
    RegCloseKey(k);
    return true;
#else
    return false;
#endif
}

const char* protocolKind(const NodeModel& n) {
    if (n.id == "tone") return "tone";
    if (n.bin == Bin::Bus) return "bus";
    return deviceKindName(n.kind);
}

json fxInfo(const Fx& fx) {
    json params = json::array();
    const auto& defs = fx.paramDefs();
    for (size_t i = 0; i < defs.size(); ++i) {
        json p = {{"id", defs[i].id}, {"name", defs[i].name}, {"min", defs[i].min}, {"max", defs[i].max}, {"value", fx.param(i)}};
        if (!defs[i].unit.empty()) p["unit"] = defs[i].unit;
        if (defs[i].log) p["scale"] = "log";
        if (!defs[i].choices.empty()) p["choices"] = defs[i].choices;
        params.push_back(p);
    }
    json j = {{"id", fx.id}, {"type", fx.type()}, {"name", fx.name()}, {"bypass", fx.bypass.load()}, {"params", params},
              {"hasEditor", fx.hasEditor()}, {"reductionDb", fx.reductionDb.load()}};
    if (fx.type() == "ducker") j["sidechain"] = fx.sidechainId.empty() ? json(nullptr) : json(fx.sidechainId);
    if (!fx.pluginId.empty()) j["pluginId"] = fx.pluginId;
    if (const std::string s = fx.status(); !s.empty()) j["error"] = s;
    return j;
}

}  // namespace

json Engine::nodeJsonLocked(const NodeModel& n) const {
    std::string state = "active", status;
    const bool isApp = n.kind == DeviceKind::App;
    if (n.bin == Bin::Bus || n.id == "tone") {
        state = "active";
    } else if (!n.present) {
        state = "unavailable";
        status = isApp ? "Not running" : n.kind == DeviceKind::Vban ? "Not connected" : "Disconnected";
    } else if (!n.userEnabled) {
        state = "idle";
        status = "Off";
    } else if (!n.streamError.empty()) {
        state = "error";
        status = n.streamError;
    } else if (isApp && !n.app.playing && n.bin == Bin::AppOut) {
        state = "idle";
        status = "Silent";
    }

    json caps = {{"canDisable", n.bin != Bin::Bus && n.id != "tone"},
                 {"canRename", n.id != "tone"},
                 {"canDelete", n.bin == Bin::Bus || n.kind == DeviceKind::Vban},
                 {"hasControlPanel", n.device.hasControlPanel}};
    if (n.bin == Bin::AppOut) caps["canIsolate"] = true;
    if (n.kind == DeviceKind::Wasapi && (n.bin == Bin::PhysicalIn || n.bin == Bin::PhysicalOut) && n.id != "tone") caps["exclusiveMode"] = true;

    const NodeSettingsData& s = n.settings;
    json settings = {{"exclusive", s.exclusive}, {"bufferMs", s.bufferMs}, {"safetyMs", s.safetyMs},
                     {"delayMs", s.delayMs},     {"pan", s.pan},           {"invert", s.invert}};
    if (n.kind == DeviceKind::Vban) {
        settings["host"] = s.host;
        settings["port"] = s.port;
        settings["streamName"] = s.streamName;
    }

    json fx = json::array();
    for (auto& f : n.fx) fx.push_back(fxInfo(*f));

    json j = {{"id", n.id},
              {"bin", binName(n.bin)},
              {"kind", protocolKind(n)},
              {"name", n.customName.empty() ? n.name : n.customName},
              {"channels", n.channels},
              {"channelNames", n.channelNames},
              {"state", state},
              {"enabled", n.userEnabled},
              {"gainDb", n.gainDb},
              {"muted", n.muted},
              {"solo", n.solo},
              {"meter", {{"offset", std::max(0, n.meterOffset)}, {"channels", n.channels}}},
              {"fx", fx},
              {"caps", caps},
              {"settings", settings}};
    if (!n.subtitle.empty()) j["subtitle"] = n.subtitle;
    if (!status.empty()) j["statusText"] = status;
    if (n.device.isDefault) j["isDefault"] = true;
    if (n.device.isDefaultComms) j["isDefaultComms"] = true;
    if (n.rt) j["cpuUs"] = n.rt->cpuUs.load();
    const BridgeStats* bs = n.cap ? &n.cap->stats() : n.ren ? &n.ren->stats() : nullptr;
    if (bs && n.stream) {
        j["sampleRate"] = n.streamRate;
        j["latencyMs"] = bs->latencyMs.load() + n.stream->deviceLatencyMs();
        j["driftPpm"] = bs->ppm.load();
        const uint64_t glitches = n.stream->deviceGlitches();
        j["xruns"] = bs->underruns.load() + bs->overruns.load() + glitches;
        j["diag"] = {{"underruns", bs->underruns.load()}, {"overruns", bs->overruns.load()}, {"fillMs", bs->fillMs.load()},
                     {"targetMs", bs->targetMs.load()}, {"devicePeriod", n.stream->periodFrames()}, {"deviceGlitches", glitches},
                     {"settlingUnderruns", bs->settling.load()}};
    }
    if (isApp) {
        json pids = json::array();
        for (auto p : n.app.pids) pids.push_back(p);
        j["app"] = {{"exe", n.app.exe}, {"pids", pids}, {"mode", n.app.mode}, {"pinned", n.app.pinned}, {"playing", n.app.playing}};
        if (!n.app.iconPng.empty()) j["icon"] = n.app.iconPng;
    }
    return j;
}

json Engine::state() {
    std::lock_guard lock(mu_);
    json nodes = json::array();
    uint64_t xruns = 0;
    double capLat = 0, renLat = 0;
    bool anyOpen = false;
    for (const std::string& id : nodeOrder_) {
        const NodeModel& n = nodes_.at(id);
        nodes.push_back(nodeJsonLocked(n));
        if (n.cap) {
            xruns += n.cap->stats().underruns + n.cap->stats().overruns;
            capLat = std::max(capLat, static_cast<double>(n.cap->stats().latencyMs.load()) + (n.stream ? n.stream->deviceLatencyMs() : 0));
            anyOpen = true;
        }
        if (n.stream) xruns += n.stream->deviceGlitches();
        if (n.ren) {
            xruns += n.ren->stats().underruns + n.ren->stats().overruns;
            renLat = std::max(renLat, static_cast<double>(n.ren->stats().latencyMs.load()) + (n.stream ? n.stream->deviceLatencyMs() : 0));
            anyOpen = true;
        }
    }
    json routes = json::array();
    for (const RouteModel& r : routes_) {
        json xps = json::array();
        for (auto& [f, t, g] : r.xps) xps.push_back({{"f", f}, {"t", t}, {"g", g}});
        routes.push_back({{"id", r.id}, {"from", r.from}, {"to", r.to}, {"gainDb", r.gainDb}, {"muted", r.muted}, {"enabled", r.enabled}, {"crosspoints", xps}});
    }
    json scenes = json::array();
    for (auto& s : scenes_) {
        json sj = {{"id", s.id}, {"name", s.name}};
        if (!s.hotkey.empty()) sj["hotkey"] = s.hotkey;
        scenes.push_back(sj);
    }
    const double blockMs = settings_.blockSize / runningRate_ * 1000.0;
    json st = {{"running", clockRun_.load()},
               {"sampleRate", runningRate_},
               {"blockSize", settings_.blockSize},
               {"cpuPercent", cpuPercent_.load()},
               {"xruns", xruns + lateCycles_.load()},
               {"latencyMs", anyOpen ? capLat + renLat + blockMs : blockMs},
               {"clockMaster", chooseClockMasterLocked()},
#ifdef _WIN32
               {"platform", "windows"},
#elif defined(__APPLE__)
               {"platform", "macos"},
#else
               {"platform", "linux"},
#endif
               {"driverPack", {{"vad", vadPresent_}, {"asio", asioDriverInstalled()}}}};

    json settings = {{"sampleRate", settings_.sampleRate},     {"blockSize", settings_.blockSize},
                     {"clockMaster", settings_.clockMaster},   {"compatMode", settings_.compatMode},
                     {"startWithSystem", settings_.startWithSystem}, {"startMinimized", settings_.startMinimized},
                     {"hotkeys", settings_.hotkeys},           {"replayBufferSeconds", settings_.replayBufferSeconds},
                     {"recordFolder", settings_.recordFolder}, {"theme", settings_.theme}, {"pluginFolders", settings_.pluginFolders},
                     {"advancedMode", settings_.advancedMode}};
    json j = {{"version", version()},
              {"protocol", 1},
              {"layoutVersion", layoutVersion_},
              {"settings", settings},
              {"status", st},
              {"nodes", nodes},
              {"routes", routes},
              {"scenes", scenes},
              {"plugins", plugins_ ? plugins_->listJson() : json::array()},
              {"pluginsScanning", plugins_ && plugins_->scanning()},
              {"recorder", {{"recording", recorder_->recording()}, {"files", lastRecordings_}, {"folder", settings_.recordFolder},
                            {"replaySeconds", settings_.replayBufferSeconds}}}};
    if (!activeScene_.empty()) j["activeScene"] = activeScene_;
    return j;
}

}  // namespace nt
