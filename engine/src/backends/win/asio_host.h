#pragma once
// Hosts one real ASIO driver (audio interface) inside the engine and exposes its inputs
// and outputs as physical devices. Driver calls run on a dedicated STA thread (ASIO drivers
// are apartment-threaded COM objects and some need a message loop).
#include <string>
#include <vector>

#include "backends/device.h"

namespace nt {

class AsioHostBackend final : public Backend {
public:
    AsioHostBackend();
    ~AsioHostBackend() override;
    const char* name() const override { return "asio"; }
    std::vector<DeviceInfo> enumerate() override;
    std::unique_ptr<DeviceStream> open(const DeviceInfo& dev, const StreamRequest& req, std::string& error) override;
    bool poll() override;
    bool controlPanel(const std::string& driverName, std::string& error);
    /// Sample rate / buffer size the engine would like the interface to run at.
    void setPreferred(double sampleRate, int bufferSize);
};

}  // namespace nt
