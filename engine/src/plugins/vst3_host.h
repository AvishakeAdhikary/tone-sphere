#pragma once
// VST3 plugin hosting.
//  - Scanning runs out of process (tonesphere-plugscan), so a plugin that crashes while
//    loading cannot take the engine down; results are cached per file modification time.
//  - Plugins run in-process as effects (Fx). Creation, state, editors and destruction happen
//    on the main (UI) thread; process() runs on the audio thread. A plugin that faults while
//    processing is switched off instead of crashing the engine.
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "dsp/fx.h"

namespace ts {

struct PluginEntry {
    std::string id;  // VST3 class id (32 hex digits)
    std::string name, vendor, category, version, path;
    bool instrument = false;  // has a MIDI input and no audio input role (synths, samplers)
};

class PluginManager {
public:
    /// `cacheFile`: JSON scan cache; `scannerExe`: path of tonesphere-plugscan.
    PluginManager(std::string cacheFile, std::string scannerExe);
    ~PluginManager();

    /// Loads the cache; starts a background scan when there is none.
    void init();
    /// Extra folders searched in addition to the standard VST3 folders.
    void setExtraFolders(std::vector<std::string> folders);
    /// Starts a background scan (no-op while one is running).
    void scanAsync();
    bool scanning() const { return scanning_.load(); }
    std::vector<PluginEntry> list() const;
    nlohmann::json listJson() const;

    /// Creates an effect instance (inactive until prepare()). Throws std::runtime_error.
    std::shared_ptr<Fx> create(const std::string& pluginId);

    /// Called (from the scan thread) when the plugin list changed.
    std::function<void()> onChanged;

    /// Folders that are searched (standard VST3 folders + extra folders).
    std::vector<std::string> searchFolders() const;

private:
    void scanThread();
    std::string cacheFile_, scannerExe_;
    mutable std::mutex mu_;
    std::vector<std::string> extra_;
    nlohmann::json cache_;  // {"modules": {path: {mtime, ok, error, classes:[...]}}}
    std::vector<PluginEntry> list_;
    std::thread scan_;
    std::atomic<bool> scanning_{false};
    std::atomic<bool> stop_{false};
    void rebuildListLocked();
};

/// Placeholder for a saved plugin that is not installed (keeps its settings and state).
std::shared_ptr<Fx> createMissingPluginFx(const std::string& pluginId, const std::string& name, const std::string& state);

/// Loads a plugin module and lists its effect classes (used by tonesphere-plugscan).
nlohmann::json describeVst3Module(const std::string& path);
/// Standard VST3 module locations on this platform.
std::vector<std::string> standardVst3Modules(const std::vector<std::string>& extraFolders);
/// Loads and instantiates one class directly (tests / tonesphere-plugscan --probe).
std::shared_ptr<Fx> createVst3Fx(const std::string& modulePath, const std::string& classId);

}  // namespace ts
