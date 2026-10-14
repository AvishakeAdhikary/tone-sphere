#pragma once
// ToneSphere virtual audio driver backend: opens the driver's shared-memory rings
// (Global\ToneSphereVadOut<N> / Global\ToneSphereVadMic<N>) and exposes them as devices:
//   "ToneSphere Output N" -> Virtual outputs (apps play into it, ToneSphere records it)
//   "ToneSphere Mic N"    -> Virtual inputs  (ToneSphere plays into it, apps record it)
#include <memory>
#include <string>
#include <vector>

#include "backends/device.h"

namespace ts {

class VadBackend final : public Backend {
public:
    const char* name() const override { return "vad"; }
    std::vector<DeviceInfo> enumerate() override;
    std::unique_ptr<DeviceStream> open(const DeviceInfo& dev, const StreamRequest& req, std::string& error) override;
    /// True when the ToneSphere driver is installed and running (at least one ring exists).
    static bool driverPresent();
    bool poll() override;

private:
    int lastPresent_ = -1;  // bit set of rings seen at the last poll
};

}  // namespace ts
