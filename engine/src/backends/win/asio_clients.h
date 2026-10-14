#pragma once
// DAWs using the "ToneSphere ASIO" driver. The engine owns the shared registry region; each
// connected DAW appears as an application with outputs (DAW playback) and inputs (DAW recording).
#include <string>
#include <vector>

#include "backends/device.h"

struct TSASIO_SERVER;  // drivers/win/asio/tonesphere_asio_shared.h (global scope)

namespace ts {

class AsioClientBackend final : public Backend {
public:
    AsioClientBackend();
    ~AsioClientBackend() override;
    const char* name() const override { return "asioclient"; }
    std::vector<DeviceInfo> enumerate() override;
    std::unique_ptr<DeviceStream> open(const DeviceInfo& dev, const StreamRequest& req, std::string& error) override;
    bool poll() override;
    /// Published to drivers so DAWs default to the engine's rate/buffer.
    void setEngineInfo(double sampleRate, int blockSize, int inputs, int outputs);
    bool available() const { return server_ != nullptr; }

private:
    void* map_ = nullptr;
    ::TSASIO_SERVER* server_ = nullptr;
    std::string signature_;
};

}  // namespace ts
