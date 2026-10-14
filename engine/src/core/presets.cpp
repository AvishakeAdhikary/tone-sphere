// Portable routing presets: import/export, and re-binding devices that come back (or appear for
// the first time) so routing survives unplugging and moving to another PC. See core/presets.h.
#include "core/presets.h"

#include <algorithm>
#include <map>
#include <stdexcept>

#include "core/engine.h"
#include "core/serialize.h"
#include "util/log.h"
#include "util/platform.h"

namespace ts {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::vector<std::string> patternsOf(const json& rule) {
    std::vector<std::string> out;
    if (!rule.contains("match")) return out;
    const json& m = rule["match"];
    if (m.is_string()) out.push_back(m.get<std::string>());
    if (m.is_array())
        for (const json& p : m)
            if (p.is_string() && !p.get<std::string>().empty()) out.push_back(p.get<std::string>());
    return out;
}

std::string slug(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c)) out += static_cast<char>(std::tolower(c));
        else if (!out.empty() && out.back() != '-') out += '-';
    }
    while (!out.empty() && out.back() == '-') out.pop_back();
    return out.empty() ? "device" : out;
}

bool isDeviceNode(const NodeModel& n) { return n.bin != Bin::Bus && n.id != "tone" && n.kind != DeviceKind::Vban; }

// Exe-addressed nodes (ToneSphere ASIO clients, per-app audio) have deterministic ids.
std::string exeNodeId(const json& ref) {
    const std::string kind = ref.value("kind", ""), exe = lower(ref.value("exe", "")), bin = ref.value("bin", "");
    if (exe.empty() || (kind != "asioclient" && kind != "app")) return {};
    const bool out = bin == "app-out";
    return kind + (out ? ":out:" : ":in:") + exe;
}

}  // namespace

std::string deviceLabel(const std::string& name, const std::string& subtitle) {
    return subtitle.empty() ? name : name + " (" + subtitle + ")";
}

int presetMatchScore(const json& rule, const std::string& bin, const std::string& kind, const std::string& name, const std::string& subtitle) {
    if (!rule.is_object() || rule.value("bin", "") != bin) return 0;
    const std::string k = rule.value("kind", "");
    if (!k.empty() && k != kind) return 0;
    const std::string label = lower(deviceLabel(name, subtitle)), lname = lower(name);
    const auto pats = patternsOf(rule);
    int best = 0;
    for (size_t i = 0; i < pats.size(); ++i) {
        const std::string p = lower(pats[i]);
        const int s = label == p ? 3 : lname == p ? 2 : label.find(p) != std::string::npos ? 1 : 0;
        if (s) best = std::max(best, (100 - static_cast<int>(std::min<size_t>(i, 90))) * 10 + s);
    }
    return best;
}

// ------------------------------------------------------------------ re-binding
void Engine::renameNodeRefsLocked(const std::string& from, const std::string& to) {
    for (auto& r : routes_) {
        if (r.from == from) r.from = to;
        if (r.to == from) r.to = to;
    }
    // The same connection may now exist twice: keep the first.
    for (size_t i = 0; i < routes_.size(); ++i)
        for (size_t j = routes_.size(); j-- > i + 1;)
            if (routes_[j].from == routes_[i].from && routes_[j].to == routes_[i].to) {
                if (routes_[j].rt) retireLocked(routes_[j].rt);
                routes_.erase(routes_.begin() + static_cast<long>(j));
            }
    for (auto& sc : scenes_) {
        json& snap = sc.snapshot;
        if (snap.contains("routes"))
            for (json& r : snap["routes"]) {
                if (r.value("from", "") == from) r["from"] = to;
                if (r.value("to", "") == from) r["to"] = to;
            }
        if (snap.contains("nodes") && snap["nodes"].contains(from) && !snap["nodes"].contains(to)) {
            snap["nodes"][to] = snap["nodes"][from];
            snap["nodes"].erase(from);
        }
    }
    for (auto& [id, n] : nodes_) {
        for (auto& f : n.fx)
            if (f->sidechainId == from) f->sidechainId = to;
        for (auto& f : n.pendingFx)
            if (f.sidechain == from) f.sidechain = to;
    }
    if (settings_.clockMaster == from) settings_.clockMaster = to;
}

void Engine::rebindDevicesLocked(const std::set<std::string>& seen) {
    auto routed = [&](const std::string& id) {
        return std::any_of(routes_.begin(), routes_.end(), [&](const RouteModel& r) { return r.from == id || r.to == id; });
    };
    std::set<std::string> taken;
    for (const std::string& nid : seen) {
        auto nit = nodes_.find(nid);
        if (nit == nodes_.end() || !isDeviceNode(nit->second) || !nit->second.present || routed(nid)) continue;
        NodeModel& n = nit->second;
        if (n.kind == DeviceKind::AsioClient || n.kind == DeviceKind::App) continue;  // ids are already stable
        const std::string bin = binName(n.bin), kind = deviceKindName(n.kind), label = deviceLabel(n.name, n.subtitle);
        std::string bestId;
        int bestScore = 0;
        for (auto& [mid, m] : nodes_) {
            if (seen.count(mid) || taken.count(mid) || !isDeviceNode(m) || m.present) continue;
            int score = 0;
            if (!m.match.is_null()) score = presetMatchScore(m.match, bin, kind, n.name, n.subtitle);
            // The same device back under a new id (another USB port, re-installed driver, …).
            if (m.bin == n.bin && m.kind == n.kind && deviceLabel(m.name, m.subtitle) == label) score = std::max(score, 5000);
            if (score > 0 && (routed(mid) || mid.rfind("ph:", 0) == 0) && score > bestScore) {
                bestScore = score;
                bestId = mid;
            }
        }
        if (bestId.empty()) continue;
        NodeModel& m = nodes_.at(bestId);
        n.gainDb = m.gainDb;
        n.muted = m.muted;
        n.solo = m.solo;
        n.userEnabled = m.userEnabled;
        n.settings = m.settings;
        if (!m.customName.empty()) n.customName = m.customName;
        if (!m.match.is_null()) n.match = m.match;
        if (!m.fx.empty() || !m.pendingFx.empty()) {
            n.fx = std::move(m.fx);
            n.pendingFx = std::move(m.pendingFx);
            for (auto& f : n.fx) f->prepare(runningRate_, kMaxBlock, n.channels);
        }
        applyNodeParamsLocked(n);
        TS_INFO("Device %s connected for %s", label.c_str(), deviceLabel(m.name, m.subtitle).c_str());
        renameNodeRefsLocked(bestId, nid);
        if (m.rt) retireLocked(m.rt);
        closeStreamLocked(m);
        nodeOrder_.erase(std::remove(nodeOrder_.begin(), nodeOrder_.end(), bestId), nodeOrder_.end());
        nodes_.erase(bestId);
        taken.insert(bestId);
    }
}

// ------------------------------------------------------------------ import
std::string Engine::resolvePresetDeviceLocked(const json& ref, bool& waiting) {
    waiting = false;
    const std::string bin = ref.value("bin", ""), kind = ref.value("kind", "");
    const Bin b = binFromString(bin);
    if (b == Bin::Bus) throw std::runtime_error("A preset device has no valid bin");
    const std::string label = ref.value("label", ref.value("exe", ref.value("key", "Device")));

    if (const std::string id = exeNodeId(ref); !id.empty()) {
        NodeModel& n = ensureNode(id);
        if (!n.present) {
            n.bin = b;
            n.kind = kindFromString(kind);
            if (n.name.empty() || n.name == id) n.name = label;
            n.subtitle = kind == "asioclient" ? "Waiting for the app to open ToneSphere ASIO" : "Waiting for the app to play";
            n.channels = std::clamp(ref.value("channels", 2), 1, kMaxChannels);
            n.channelNames.clear();
            for (int c = 0; c < n.channels; ++c) n.channelNames.push_back(std::to_string(c + 1));
            if (n.kind == DeviceKind::App) {
                n.app.exe = lower(ref.value("exe", ""));
                n.app.pinned = true;
            }
            waiting = true;
        }
        n.persistent = true;
        return id;
    }

    // Best matching device we know (connected ones first).
    std::string best;
    int bestScore = 0;
    for (auto& [id, n] : nodes_) {
        if (!isDeviceNode(n)) continue;
        int s = presetMatchScore(ref, bin, deviceKindName(n.kind), n.name, n.subtitle);
        if (s > 0 && n.present) s += 10000;
        if (s > bestScore) {
            bestScore = s;
            best = id;
        }
    }
    if (best.empty() && ref.value("fallback", "") == "default") {
        for (auto& [id, n] : nodes_)
            if (n.bin == b && n.present && n.device.isDefault && isDeviceNode(n)) best = id;
    }
    json rule = {{"bin", bin}, {"match", ref.contains("match") ? ref["match"] : json::array()}};
    if (!kind.empty()) rule["kind"] = kind;
    if (!best.empty()) {
        NodeModel& n = nodes_.at(best);
        n.match = rule;
        waiting = !n.present;
        return best;
    }
    // Not seen yet: a placeholder that binds to the device as soon as it is plugged in.
    std::string id = "ph:" + bin + ":" + slug(ref.value("key", label));
    for (int i = 2; findNode(id); ++i) id = "ph:" + bin + ":" + slug(ref.value("key", label)) + "-" + std::to_string(i);
    NodeModel& n = ensureNode(id);
    n.bin = b;
    n.kind = kind.empty() ? DeviceKind::Wasapi : kindFromString(kind);
    n.name = label;
    n.subtitle = "Not connected yet: plug it in and ToneSphere connects it";
    n.channels = std::clamp(ref.value("channels", 2), 1, kMaxChannels);
    n.channelNames = n.channels == 2 ? std::vector<std::string>{"L", "R"} : std::vector<std::string>{};
    for (int c = static_cast<int>(n.channelNames.size()); c < n.channels; ++c) n.channelNames.push_back(std::to_string(c + 1));
    n.match = rule;
    n.present = false;
    n.persistent = true;  // kept while scenes refer to it; replaced by the real device when it appears
    waiting = true;
    return id;
}

json Engine::importPresetLocked(const json& pr) {
    // "nextone-preset" files were exported before the rename and have the same layout.
    const std::string format = pr.is_object() ? pr.value("format", "") : "";
    if (format != "tonesphere-preset" && format != "nextone-preset") throw std::runtime_error("This file isn't a ToneSphere preset");
    if (pr.value("version", 1) > 1) throw std::runtime_error("This preset was made by a newer ToneSphere. Please update ToneSphere first.");
    const std::string name = pr.value("name", "Imported preset");
    std::map<std::string, std::string> ids;
    json waiting = json::array();

    auto applyProps = [&](NodeModel& n, const json& j) {
        if (j.contains("gainDb")) n.gainDb = std::clamp(j["gainDb"].get<float>(), -90.0f, 12.0f);
        if (j.contains("muted")) n.muted = j["muted"].get<bool>();
        if (j.contains("settings") && j["settings"].is_object()) settingsFrom(n.settings, j["settings"]);
        if (j.contains("fx") && j["fx"].is_array()) {
            n.fx.clear();
            n.pendingFx.clear();
            for (const json& f : j["fx"]) {
                FxConfig c = fxConfigFrom(f);
                c.id.clear();
                n.pendingFx.push_back(c);
            }
        }
        applyNodeParamsLocked(n);
    };

    for (const json& m : pr.value("mixes", json::array())) {
        const std::string key = m.value("key", ""), mname = m.value("name", "Mix");
        std::string id;
        for (auto& [nid, n] : nodes_)
            if (n.bin == Bin::Bus && (n.customName.empty() ? n.name : n.customName) == mname) id = nid;
        if (id.empty()) {
            id = "bus:" + randomHex(4);
            NodeModel& n = ensureNode(id);
            n.bin = Bin::Bus;
            n.name = mname;
            n.channels = std::clamp(m.value("channels", 2), 1, 16);
            n.channelNames.clear();
            for (int c = 0; c < n.channels; ++c) n.channelNames.push_back(n.channels == 2 ? (c ? "R" : "L") : std::to_string(c + 1));
            n.persistent = true;
            n.present = true;
        }
        applyProps(nodes_.at(id), m);
        ids[key] = id;
    }
    for (const json& d : pr.value("devices", json::array())) {
        bool wait = false;
        const std::string id = resolvePresetDeviceLocked(d, wait);
        applyProps(nodes_.at(id), d);
        ids[d.value("key", id)] = id;
        if (wait) waiting.push_back(d.value("label", nodes_.at(id).name));
    }

    // The preset is the whole routing map: it replaces the current connections.
    for (auto& r : routes_)
        if (r.rt) retireLocked(r.rt);
    routes_.clear();
    int skipped = 0;
    for (const json& rj : pr.value("routes", json::array())) {
        const auto f = ids.find(rj.value("from", "")), t = ids.find(rj.value("to", ""));
        if (f == ids.end() || t == ids.end()) {
            ++skipped;
            continue;
        }
        NodeModel& a = nodes_.at(f->second);
        NodeModel& b = nodes_.at(t->second);
        if (!binIsSource(a.bin) || !binIsDest(b.bin) || a.id == b.id) {
            ++skipped;
            continue;
        }
        RouteModel r;
        r.id = newId("r");
        r.from = a.id;
        r.to = b.id;
        r.gainDb = std::clamp(rj.value("gainDb", 0.0f), -120.0f, 24.0f);
        r.muted = rj.value("muted", false);
        if (rj.contains("channels") && rj["channels"].is_array()) {
            for (const json& x : rj["channels"])
                if (x.is_array() && x.size() >= 2)
                    r.xps.emplace_back(x[0].get<int>(), x[1].get<int>(), x.size() > 2 ? x[2].get<float>() : 0.0f);
        } else {
            r.xps = defaultCrosspoints(a, b, rj.value("mode", "auto"));
        }
        routes_.push_back(std::move(r));
    }
    rebuildLocked();

    // Keep it as a scene so it can be switched to later (replaces a scene with the same name).
    SceneModel* sc = nullptr;
    for (auto& s : scenes_)
        if (s.name == name) sc = &s;
    if (!sc) {
        scenes_.push_back({newId("sc"), name, "", json::object()});
        sc = &scenes_.back();
    }
    sc->snapshot = snapshotLocked();
    activeScene_ = sc->id;
    markDirty();
    TS_INFO("Imported preset \"%s\": %zu connections, %zu waiting for devices", name.c_str(), routes_.size(), waiting.size());
    return {{"sceneId", sc->id}, {"name", name}, {"connections", routes_.size()}, {"waiting", waiting}, {"skipped", skipped}};
}

// ------------------------------------------------------------------ export
json Engine::exportPresetLocked(const std::string& name, const std::string& description) const {
    std::map<std::string, std::string> keyOf;
    std::set<std::string> usedKeys;
    auto uniqueKey = [&](const std::string& base) {
        std::string k = slug(base);
        for (int i = 2; usedKeys.count(k); ++i) k = slug(base) + "-" + std::to_string(i);
        usedKeys.insert(k);
        return k;
    };
    auto props = [&](const NodeModel& n, json& j) {
        j["gainDb"] = n.gainDb;
        j["muted"] = n.muted;
        j["settings"] = {{"exclusive", n.settings.exclusive}, {"bufferMs", n.settings.bufferMs}, {"delayMs", n.settings.delayMs},
                         {"pan", n.settings.pan},             {"invert", n.settings.invert}};
        json fx = json::array();
        for (auto& f : n.fx) {
            json fj = fxJson(*f);
            fj.erase("id");
            fx.push_back(fj);
        }
        j["fx"] = fx;
    };
    std::set<std::string> used;
    for (auto& r : routes_) {
        used.insert(r.from);
        used.insert(r.to);
    }
    json mixes = json::array(), devices = json::array(), routes = json::array();
    for (const std::string& id : nodeOrder_) {
        const NodeModel& n = nodes_.at(id);
        if (n.bin == Bin::Bus) {
            const std::string mname = n.customName.empty() ? n.name : n.customName;
            json m = {{"key", keyOf[id] = uniqueKey(mname)}, {"name", mname}, {"channels", n.channels}};
            props(n, m);
            mixes.push_back(m);
        } else if (used.count(id) && id != "tone") {
            const std::string label = n.match.is_null() || n.present ? deviceLabel(n.name, n.subtitle) : n.name;
            json d = {{"key", keyOf[id] = uniqueKey(n.customName.empty() ? label : n.customName)},
                      {"bin", binName(n.bin)},
                      {"kind", deviceKindName(n.kind)},
                      {"label", n.customName.empty() ? label : n.customName},
                      {"channels", n.channels}};
            if (n.kind == DeviceKind::AsioClient || n.kind == DeviceKind::App) {
                std::string exe = id.substr(id.find(':', id.find(':') + 1) + 1);
                if (const auto hash = exe.find('#'); hash != std::string::npos) exe.resize(hash);
                d["exe"] = exe;
            } else {
                json pats = json::array();
                if (!n.match.is_null())
                    for (const auto& p : patternsOf(n.match)) pats.push_back(p);
                if (n.present || n.match.is_null()) {
                    pats.push_back(deviceLabel(n.name, n.subtitle));
                    if (!n.subtitle.empty()) pats.push_back(n.name);
                }
                json uniq = json::array();
                for (const json& p : pats)
                    if (std::find(uniq.begin(), uniq.end(), p) == uniq.end()) uniq.push_back(p);
                d["match"] = uniq;
            }
            props(n, d);
            devices.push_back(d);
        }
    }
    for (auto& r : routes_) {
        if (!keyOf.count(r.from) || !keyOf.count(r.to)) continue;  // e.g. the test tone
        json x = json::array();
        for (auto& [f, t, g] : r.xps) x.push_back({f, t, g});
        routes.push_back({{"from", keyOf[r.from]}, {"to", keyOf[r.to]}, {"gainDb", r.gainDb}, {"muted", r.muted}, {"channels", x}});
    }
    return {{"format", "tonesphere-preset"}, {"version", 1}, {"name", name}, {"description", description}, {"app", "ToneSphere " + version()},
            {"devices", devices}, {"mixes", mixes}, {"routes", routes}};
}

}  // namespace ts
