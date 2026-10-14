#pragma once
// WASAPI backend: physical endpoints, compatibility-mode virtual cables, and shared helpers
// used by the application-capture backend.
#include <windows.h>
#include <mmreg.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "backends/device.h"

struct IAudioClient;
struct IMMDeviceEnumerator;

namespace ts {

class WasapiNotifier;

class WasapiBackend final : public Backend {
public:
    WasapiBackend();
    ~WasapiBackend() override;
    const char* name() const override { return "wasapi"; }
    std::vector<DeviceInfo> enumerate() override;
    std::unique_ptr<DeviceStream> open(const DeviceInfo& dev, const StreamRequest& req, std::string& error) override;

    /// When enabled, known virtual-cable endpoints (VB-CABLE, VoiceMeeter, VAC) appear as
    /// virtual inputs/outputs instead of physical devices.
    void setCompatMode(bool on) { compat_ = on; }
    /// Interface names that belong to the ToneSphere driver (handled by the VAD backend, hidden here).
    static bool isToneSphereEndpoint(const std::wstring& interfaceName);
    static bool isVirtualCable(const std::wstring& interfaceName, const std::wstring& name);

private:
    IMMDeviceEnumerator* enumerator_ = nullptr;
    WasapiNotifier* notifier_ = nullptr;
    std::atomic<bool> compat_{true};
};

/// Sample format of a WASAPI buffer.
enum class SampleFormat { Float32, Int16, Int24, Int32 };

/// Generic event-driven WASAPI stream on an already-activated IAudioClient.
class WasapiStream final : public DeviceStream {
public:
    /// Takes ownership of `client` (one reference).
    WasapiStream(IAudioClient* client, bool capture, std::string label);
    ~WasapiStream() override;

    /// Shared-mode init (format = device mix format as float32, or `forced` if given).
    bool initShared(double bufferMs, bool loopback, const WAVEFORMATEX* forced, std::string& err);
    bool initExclusive(double bufferMs, double preferredRate, int preferredChannels, std::string& err);

    double sampleRate() const override { return rate_; }
    int channels() const override { return channels_; }
    int periodFrames() const override { return period_; }
    double deviceLatencyMs() const override { return latencyMs_; }
    bool start(std::shared_ptr<CaptureBridge> capture, std::shared_ptr<RenderBridge> render) override;
    void stop() override;
    std::string error() const override;
    uint64_t deviceGlitches() const override { return glitches_.load(); }
    bool deviceClocked() const override { return exclusive_; }

private:
    void run();
    void captureLoop();
    void renderLoop();

    IAudioClient* client_ = nullptr;
    bool capture_ = true;
    bool exclusive_ = false;
    std::string label_;
    HANDLE event_ = nullptr;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::shared_ptr<CaptureBridge> cap_;
    std::shared_ptr<RenderBridge> ren_;
    double rate_ = 48000;
    int channels_ = 2;
    int period_ = 480;
    unsigned bufferFrames_ = 0;
    double latencyMs_ = 0;
    SampleFormat fmt_ = SampleFormat::Float32;
    int bytesPerSample_ = 4;
    mutable std::atomic<bool> hasError_{false};
    std::string error_;
    std::atomic<uint64_t> glitches_{0};
};

}  // namespace ts
