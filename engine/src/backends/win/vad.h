#pragma once
// NexTone virtual audio driver backend: opens the driver's shared-memory rings
// (Global\NexToneVadOut<N> / Global\NexToneVadMic<N>) and exposes them as devices:
//   "NexTone Output N" -> Virtual outputs (apps play into it, NexTone records it)
//   "NexTone Mic N"    -> Virtual inputs  (NexTone plays into it, apps record it)
#include <memory>
#include <string>
#include <vector>

#include "backends/device.h"

namespace nt {

class VadBackend final : public Backend {
public:
    const char* name() const override { return "vad"; }
    std::vector<DeviceInfo> enumerate() override;
    std::unique_ptr<DeviceStream> open(const DeviceInfo& dev, const StreamRequest& req, std::string& error) override;
    /// True when the NexTone driver is installed and running (at least one ring exists).
    static bool driverPresent();
    bool poll() override;

private:
    int lastPresent_ = -1;  // bit set of rings seen at the last poll
};

}  // namespace nt
