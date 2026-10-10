#pragma once
// PipeWire backend (Linux).
//  - Physical inputs/outputs: every Audio/Source and Audio/Sink node in the graph.
//  - Virtual outputs/inputs: NexTone creates its own nodes that stay visible to other apps:
//      "NexTone Output N" (Audio/Sink: apps play into it, NexTone records it)
//      "NexTone Mic N"    (Audio/Source/Virtual: NexTone plays into it, apps record it)
//  - Application outputs: playback streams of other apps (captured as a copy).
// Streams exchange float32 audio with the engine's bridges from PipeWire's real-time thread.
#include <memory>
#include <string>
#include <vector>

#include "backends/device.h"

namespace nt {

class PipeWireBackend final : public Backend {
public:
    explicit PipeWireBackend(int virtualDevices = 2);
    ~PipeWireBackend() override;
    const char* name() const override { return "pipewire"; }
    std::vector<DeviceInfo> enumerate() override;
    std::unique_ptr<DeviceStream> open(const DeviceInfo& dev, const StreamRequest& req, std::string& error) override;
    bool poll() override;
    /// False when no PipeWire server could be reached.
    bool connected() const;

    struct Impl;

private:
    std::shared_ptr<Impl> impl_;
};

}  // namespace nt
