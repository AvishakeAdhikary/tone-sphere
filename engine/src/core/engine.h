#pragma once
// The ToneSphere engine: owns the routing model, devices, clock and the published audio Program.
// All public methods are thread-safe (serialised by an internal mutex) and are called from
// the IPC thread; the audio path never takes that mutex.
#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "backends/device.h"
#include "core/dsp_pool.h"
#include "core/graph.h"
#include "core/recorder.h"

namespace ts {

using json = nlohmann::json;

struct EngineSettings {
    double sampleRate = 48000.0;
    int blockSize = 128;
    std::string clockMaster = "auto";
    bool compatMode = true;
    bool startWithSystem = true;
    bool startMinimized = true;
    std::map<std::string, std::string> hotkeys;
    int replayBufferSeconds = 0;
    std::string recordFolder;
    std::string theme = "system";
    bool advancedMode = false;
    std::vector<std::string> pluginFolders;  // extra VST3 folders
    int oscPort = 0;                         // OSC control over UDP (0 = off)
    json midiMappings = json::array();       // MIDI controller mappings (see control/remote.h)
    std::string skin = "fluent";           // UI skin id
    json customSkins = json::array();        // imported UI skins (validated by the UI)
    int dspThreads = 0;                      // 0 = auto (multi-core when it helps), 1 = one core, N = N cores
    json midiInstrument = nullptr;
    std::string language = "system";         // UI language: system | en | hi           // {nodeId, fxId}: the instrument MIDI devices play (or null)
};

struct NodeSettingsData {
    bool exclusive = false;
    double bufferMs = 10.0;
    double safetyMs = 0.0;
    double delayMs = 0.0;
    double pan = 0.0;
    bool invert = false;
    std::string host;
    int port = 6980;
    std::string streamName;
    std::vector<std::string> channelNames;
};

struct AppData {
    std::string exe;
    std::vector<uint32_t> pids;
    std::string mode = "copy";  // off | copy | isolate
    bool pinned = false;
    bool playing = false;
    std::string iconPng;  // data URL
};

struct FxConfig {
    std::string id, type, pluginId, sidechain, state, name;
    bool bypass = false;
    std::map<std::string, float> params;
};

/// Editable model of one node (control thread).
struct NodeModel {
    std::string id;
    Bin bin = Bin::PhysicalIn;
    DeviceKind kind = DeviceKind::Wasapi;
    std::string name, customName, subtitle;
    int channels = 2;
    std::vector<std::string> channelNames;
    DeviceInfo device;     // device-backed nodes
    bool present = false;  // device/app currently available
    bool persistent = false;  // bus / vban / pinned app: kept even when absent
    bool userEnabled = true;
    float gainDb = 0;
    bool muted = false, solo = false;
    NodeSettingsData settings;
    AppData app;
    std::vector<std::shared_ptr<Fx>> fx;
    std::vector<FxConfig> pendingFx;  // fx from config not yet instantiated
    json match;  // preset device rule (bin/kind/name patterns): re-binds this node to a matching device

    std::shared_ptr<NodeRuntime> rt;
    std::unique_ptr<DeviceStream> stream;
    bool streamInput = true;  // the open stream captures into ToneSphere
    std::shared_ptr<CaptureBridge> cap;
    std::shared_ptr<RenderBridge> ren;
    std::string streamError;
    double streamRate = 0;
    int meterOffset = -1;
    double lastNeededAt = 0;  // for delayed close
};

struct RouteModel {
    std::string id, from, to;
    float gainDb = 0;
    bool muted = false, enabled = true;
    std::vector<std::tuple<int, int, float>> xps;  // from ch, to ch, gain dB
    std::shared_ptr<RouteRuntime> rt;
};

struct SceneModel {
    std::string id, name, hotkey;
    json snapshot;
};

class AppAudio;    // per-application audio (Windows)
class PeakMeters;  // OS peak meters for idle devices (Windows)
class PluginManager;  // VST3 plugins

class Engine {
public:
    Engine();
    ~Engine();

    bool start();
    void shutdown();
    bool isShuttingDown() const { return shutdown_.load(); }

    /// Dispatches one RPC request; throws std::runtime_error with a user-facing message on failure.
    json call(const std::string& method, const json& params);
    /// Full state snapshot (protocol EngineState).
    json state();
    /// Fills `out` with the current meter values; returns the layout version.
    uint32_t meters(std::vector<float>& out);
    /// Refreshes levels of idle devices/apps from the OS peak meters (meter thread).
    void pollExternalMeters();

    /// Called whenever the observable state changed (coalesced by the IPC layer).
    std::function<void()> onStateChanged;
    /// MIDI learn / device list (set by the host process; see control/remote.h).
    std::function<json(double)> onMidiLearn;
    std::function<json()> onMidiDevices;
    /// Plays a MIDI short message on the instrument chosen in settings (MIDI devices). Lock-free
    /// for the caller: safe from MIDI driver callbacks. Returns false when no instrument is set.
    bool playMidi(uint8_t status, uint8_t data1, uint8_t data2);
    /// Called to request process exit.
    std::function<void()> onShutdownRequested;

    std::string version() const;

private:
    friend class AppAudio;
    // ---- model operations (mutex held) ----
    void refreshDevicesLocked();
    void rebuildLocked();
    void compileAndPublishLocked();
    void updateStreamsLocked();
    bool openStreamLocked(NodeModel& n);
    Backend* backendFor(DeviceKind k) const;
    void closeStreamLocked(NodeModel& n);
    bool nodeNeededLocked(const NodeModel& n) const;
    NodeModel* findNode(const std::string& id);
    RouteModel* findRoute(const std::string& id);
    NodeModel& ensureNode(const std::string& id);
    void applyNodeParamsLocked(NodeModel& n);
    void instantiatePendingFxLocked(NodeModel& n);
    void applyPluginFoldersLocked();
    // ---- presets (core/presets.cpp) ----
    json importPresetLocked(const json& preset);
    json exportPresetLocked(const std::string& name, const std::string& description) const;
    std::string resolvePresetDeviceLocked(const json& ref, bool& waiting);
    void rebindDevicesLocked(const std::set<std::string>& seen);
    void renameNodeRefsLocked(const std::string& from, const std::string& to);
    /// latency.measure: runs without holding the engine lock while it waits for audio.
    json measureLatency(const json& params);
    std::set<std::string> forcedOpen_;  // nodes kept open during a measurement
    std::vector<std::tuple<int, int, float>> defaultCrosspoints(const NodeModel& a, const NodeModel& b, const std::string& mode) const;
    void markDirty();
    void saveConfigLocked();
    void loadConfig();
    json snapshotLocked() const;
    void restoreSnapshotLocked(const json& s);
    json nodeJsonLocked(const NodeModel& n) const;
    std::string chooseClockMasterLocked() const;

    // ---- clock ----
    void startClock();
    void stopClock();
    void clockThread();
    void cycle(int frames, double now);

    // ---- background maintenance (device changes, stream errors, autosave) ----
    void maintenanceThread();

    mutable std::recursive_mutex mu_;
    EngineSettings settings_;
    std::map<std::string, NodeModel> nodes_;
    std::vector<std::string> nodeOrder_;
    std::vector<RouteModel> routes_;
    std::vector<SceneModel> scenes_;
    std::string activeScene_;
    std::vector<std::unique_ptr<Backend>> backends_;
    std::unique_ptr<AppAudio> apps_;
    std::unique_ptr<PeakMeters> peaks_;
    std::unique_ptr<PluginManager> plugins_;
    void updateExternalMetersLocked();
    std::shared_ptr<ToneGen> tone_;

    // published program (RCU)
    std::atomic<Program*> program_{nullptr};
    struct Retired {
        std::shared_ptr<void> obj;
        uint64_t epoch;
    };
    std::vector<Retired> retired_;
    void retireLocked(std::shared_ptr<void> obj);
    void reclaimLocked();
    std::atomic<uint64_t> cyclesStarted_{0};
    std::atomic<uint64_t> cyclesDone_{0};
    std::shared_ptr<Program> programOwner_;
    std::shared_ptr<MeterBank> meterBank_;
    std::string meterSig_;
    uint32_t layoutVersion_ = 0;

    // clock
    std::thread clock_;
    std::atomic<bool> clockRun_{false};
    // Multi-core DSP: created with the clock, destroyed after it stops (core/dsp_pool.h).
    std::unique_ptr<DspPool> dspPool_;
    DspScheduler dsp_;
    // The instrument MIDI devices play (settings.midiInstrument), resolved on every publish.
    std::atomic<std::shared_ptr<Fx>> midiTarget_;
    void resolveMidiTargetLocked();
    double runningRate_ = 48000.0;
    std::atomic<float> cpuPercent_{0};
    std::atomic<uint64_t> lateCycles_{0};

    // maintenance
    std::thread maint_;
    std::mutex maintMu_;
    std::condition_variable maintCv_;
    std::atomic<bool> devicesChanged_{false};
    std::atomic<bool> configDirty_{false};
    std::atomic<bool> shutdown_{false};
    std::unique_ptr<Recorder> recorder_;
    std::vector<std::string> recordingNodes_;
    std::vector<std::string> lastRecordings_;
    void updateTapsLocked();
    bool vadPresent_ = false;
};

}  // namespace ts
