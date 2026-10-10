#pragma once
// Levels for devices/apps NexTone isn't streaming from, read from Windows' own peak meters
// (IAudioMeterInformation on endpoints and audio sessions). Lets users see their microphone
// or an app move before anything is routed.
#include <memory>
#include <string>
#include <vector>

namespace nt {

struct MeterBank;

class PeakMeters {
public:
    struct Target {
        int offset = 0;
        int channels = 2;
        std::string endpointId;      // device endpoint (empty for apps)
        std::vector<uint32_t> pids;  // app sessions
        bool capture = false;        // app: capture sessions instead of render sessions
    };
    PeakMeters();
    ~PeakMeters();
    /// Control side (about once per second): which nodes to meter.
    void setTargets(std::vector<Target> targets);
    /// Meter thread (~30 Hz): writes peak values into the bank.
    void poll(MeterBank& bank, uint32_t layoutVersion);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nt
