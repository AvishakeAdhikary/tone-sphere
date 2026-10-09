#pragma once
// Platform-neutral audio device abstraction.
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/bridge.h"
#include "core/graph.h"

namespace nt {

enum class DeviceKind { Wasapi, Asio, Vad, Compat, App, AsioClient, Vban, PipeWire, CoreAudio };
const char* deviceKindName(DeviceKind k);

struct DeviceInfo {
    std::string nodeId;      // stable node id, e.g. "wasapi:out:{0.0.0.00000000}.{guid}"
    std::string endpointId;  // platform endpoint id
    std::string name;
    std::string subtitle;
    DeviceKind kind = DeviceKind::Wasapi;
    Bin bin = Bin::PhysicalIn;
    bool input = true;  // true: captures into NexTone (source); false: renders (destination)
    int channels = 2;
    double sampleRate = 48000.0;
    std::vector<std::string> channelNames;
    bool isDefault = false;
    bool isDefaultComms = false;
    bool supportsExclusive = false;
    bool hasControlPanel = false;
    bool present = true;
};

struct StreamRequest {
    bool exclusive = false;
    double bufferMs = 10.0;
    double preferredRate = 0.0;  // 0 = device default
    int preferredBlock = 0;      // ASIO buffer size / exclusive period hint (frames)
};

/// An open device stream. Created (initialised) by a backend; started with a bridge.
class DeviceStream {
public:
    virtual ~DeviceStream() = default;
    virtual double sampleRate() const = 0;
    virtual int channels() const = 0;
    virtual int periodFrames() const = 0;
    virtual double deviceLatencyMs() const { return 0.0; }
    /// Starts I/O. Exactly one of capture/render is used depending on direction.
    virtual bool start(std::shared_ptr<CaptureBridge> capture, std::shared_ptr<RenderBridge> render) = 0;
    virtual void stop() = 0;
    /// Non-empty when the stream failed after starting (device removed, etc.).
    virtual std::string error() const { return {}; }
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char* name() const = 0;
    virtual std::vector<DeviceInfo> enumerate() = 0;
    virtual std::unique_ptr<DeviceStream> open(const DeviceInfo& dev, const StreamRequest& req, std::string& error) = 0;
    /// Invoked (from any thread) when devices appear/disappear or defaults change.
    std::function<void()> onDevicesChanged;
};

}  // namespace nt
