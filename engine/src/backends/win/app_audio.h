#pragma once
// Per-application audio on Windows:
//  * discovers apps with audio sessions (playing → "Application outputs", recording → "Application inputs"),
//  * captures one app's sound via process loopback ("copy" mode, Windows 10 2004+),
//  * redirects an app to a private virtual device via the per-app default endpoint
//    ("isolate" mode and app microphones), the same mechanism Windows' "App volume and
//    device preferences" page uses.
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "backends/device.h"

namespace nt {

class Engine;
struct NodeModel;

class AppAudio {
public:
    AppAudio();
    ~AppAudio();

    /// Polls audio sessions; returns true when the set of apps / their state changed.
    bool poll();
    /// Creates/updates app nodes in the engine (engine mutex held).
    void refresh(Engine& e, std::set<std::string>& seen);
    /// Opens a process-loopback stream for an app node in copy mode (engine mutex held).
    std::unique_ptr<DeviceStream> open(NodeModel& n, double engineRate, std::string& err);
    /// Virtual device assigned to an app node (isolate mode / app microphone), or nullptr.
    const DeviceInfo* assigned(const NodeModel& n) const;
    /// Applies per-app endpoint redirection according to modes and routes (engine mutex held).
    void applyRouting(Engine& e);
    void handleRpc(Engine& e, const std::string& method, const nlohmann::json& p);
    void shutdown();

    struct AppSession {
        std::string exe;  // lower-case file name, e.g. "spotify.exe"
        std::string displayName;
        std::wstring path;
        std::vector<uint32_t> pids;
        bool render = false, capture = false;
        bool renderActive = false, captureActive = false;
        int channels = 2;
    };

private:
    std::string iconFor(const std::wstring& path, const std::string& exe);
    bool setAppEndpoint(const std::vector<uint32_t>& pids, bool capture, const std::string& endpointId);
    uint32_t rootPid(uint32_t pid, const std::string& exe);

    std::map<std::string, AppSession> apps_;  // by exe
    std::string signature_;
    std::map<std::string, std::string> icons_;
    // exe -> endpoint redirected to (empty when not redirected)
    std::map<std::string, std::string> redirectedOut_, redirectedIn_;
    // exe -> virtual device (node id + device) assigned for isolate / microphone feed
    std::map<std::string, std::string> cableOut_, cableIn_;
    std::map<std::string, DeviceInfo> cableOutDev_, cableInDev_;
    void* policy_ = nullptr;  // IAudioPolicyConfigFactory*
    bool policyNew_ = true;
};

}  // namespace nt
