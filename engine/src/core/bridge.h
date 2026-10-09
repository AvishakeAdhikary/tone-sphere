#pragma once
// Bridges between a device's own clock/thread and the engine clock/thread.
//
//   CaptureBridge:  device thread --(ring)--> resampler --> engine (planar)
//   RenderBridge:   engine (planar) --> resampler --(ring)--> device thread
//
// The engine side runs the resampler with a ratio corrected by a DriftController,
// so devices with independent crystals stay in sync without clicks. When a bridge
// belongs to the clock-master device, drift correction is disabled.
#include <atomic>
#include <cstddef>
#include <vector>

#include "core/drift.h"
#include "core/resampler.h"
#include "core/ring_buffer.h"

namespace nt {

struct BridgeConfig {
    size_t channels = 2;
    double deviceRate = 48000.0;
    double engineRate = 48000.0;
    size_t devicePeriod = 480;   // typical frames per device callback
    size_t engineMaxBlock = 1024;
    double extraSafetyMs = 0.0;  // user "safety buffer"
    bool master = false;         // device drives the engine clock: no drift correction
    bool jittery = false;        // source arrives in bursts (network/app loopback): bigger margin
};

/// Timestamped device position, published by the device thread (seqlock).
/// Lets the engine estimate a smooth, continuous fill level instead of sampling
/// the sawtooth caused by device periods (which aliases into slow beats).
class DevicePosition {
public:
    void publish(double timeSec, uint64_t totalFrames) {
        const uint32_t s = seq_.load(std::memory_order_relaxed);
        seq_.store(s + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        time_.store(timeSec, std::memory_order_relaxed);
        total_.store(totalFrames, std::memory_order_relaxed);
        seq_.store(s + 2, std::memory_order_release);
    }
    bool read(double& timeSec, uint64_t& totalFrames) const {
        for (int i = 0; i < 8; ++i) {
            const uint32_t s0 = seq_.load(std::memory_order_acquire);
            if (s0 & 1u) continue;
            timeSec = time_.load(std::memory_order_relaxed);
            totalFrames = total_.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq_.load(std::memory_order_relaxed) == s0) return s0 != 0;
        }
        return false;
    }

private:
    std::atomic<uint32_t> seq_{0};
    std::atomic<double> time_{0.0};
    std::atomic<uint64_t> total_{0};
};

struct BridgeStats {
    std::atomic<uint64_t> underruns{0};
    std::atomic<uint64_t> overruns{0};
    std::atomic<float> ppm{0.0f};
    std::atomic<float> fillMs{0.0f};
    std::atomic<float> latencyMs{0.0f};
    std::atomic<bool> active{false};
};

class CaptureBridge {
public:
    void init(const BridgeConfig& cfg);
    const BridgeConfig& config() const { return cfg_; }
    BridgeStats& stats() { return stats_; }

    // ---- device thread ----
    void deviceWrite(const float* interleaved, size_t frames, double nowSec);
    void deviceWriteSilence(size_t frames, double nowSec);

    // ---- engine thread ----
    /// Produces exactly `frames` planar frames (silence when not enough data).
    void engineRead(float* const* out, size_t frames, double nowSec);
    /// Signals that the device stopped (engine outputs silence, state resets on next start).
    void markStopped() { stopped_.store(true, std::memory_order_release); }
    void markStarted() { stopped_.store(false, std::memory_order_release); restart_.store(true, std::memory_order_release); }

private:
    BridgeConfig cfg_;
    BridgeStats stats_;
    FrameRing ring_;
    VarispeedResampler rs_;
    DriftController ctl_;
    std::vector<float> scratch_;
    double nominal_ = 1.0;
    double target_ = 0.0;
    bool primed_ = false;
    DevicePosition devPos_;
    uint64_t devTotal_ = 0;  // device thread
    uint64_t engTotal_ = 0;  // engine thread
    std::atomic<bool> stopped_{true};
    std::atomic<bool> restart_{false};
};

class RenderBridge {
public:
    void init(const BridgeConfig& cfg);
    const BridgeConfig& config() const { return cfg_; }
    BridgeStats& stats() { return stats_; }

    // ---- engine thread ----
    void engineWrite(const float* const* in, size_t frames, double nowSec);

    // ---- device thread ----
    /// Fills `frames` interleaved frames (silence on underrun or before priming).
    void deviceRead(float* interleaved, size_t frames, double nowSec);
    size_t deviceAvailable() const { return ring_.available(); }
    void markStopped() { stopped_.store(true, std::memory_order_release); }
    void markStarted() { stopped_.store(false, std::memory_order_release); restart_.store(true, std::memory_order_release); }

private:
    BridgeConfig cfg_;
    BridgeStats stats_;
    FrameRing ring_;
    VarispeedResampler rs_;
    DriftController ctl_;
    std::vector<float> scratch_;
    double nominal_ = 1.0;
    double target_ = 0.0;
    std::atomic<bool> devicePrimed_{false};
    DevicePosition devPos_;
    uint64_t devTotal_ = 0;  // device thread
    uint64_t engTotal_ = 0;  // engine thread
    std::atomic<bool> stopped_{true};
    std::atomic<bool> restart_{false};
};

}  // namespace nt
