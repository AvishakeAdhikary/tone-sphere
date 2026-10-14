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

namespace ts {

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
    std::atomic<float> targetMs{0.0f};
    std::atomic<uint64_t> settling{0};  // adaptations during the first seconds (not dropouts)
    std::atomic<bool> active{false};
};

/// Observes how a device really delivers/consumes audio: largest burst and longest gap.
/// Written by the device thread, harvested by the engine thread once per window.
class BurstMeter {
public:
    void observe(size_t frames, double now) {
        if (frames > maxFrames_.load(std::memory_order_relaxed)) maxFrames_.store(frames, std::memory_order_relaxed);
        if (last_ > 0.0) {
            const double gap = now - last_;
            if (gap < 1.0 && gap > maxGap_.load(std::memory_order_relaxed)) maxGap_.store(gap, std::memory_order_relaxed);
        }
        last_ = now;
    }
    void resetLast() { last_ = 0.0; }
    size_t peekFrames() const { return maxFrames_.load(std::memory_order_relaxed); }
    double peekGap() const { return maxGap_.load(std::memory_order_relaxed); }
    void harvest(size_t& frames, double& gap) {
        frames = maxFrames_.exchange(0, std::memory_order_relaxed);
        gap = maxGap_.exchange(0.0, std::memory_order_relaxed);
    }

private:
    std::atomic<size_t> maxFrames_{0};
    std::atomic<double> maxGap_{0.0};
    double last_ = 0.0;  // device thread only
};

/// Latency cushion that adapts to the device: grows at once when bursts are larger than
/// expected or after a dropout, and shrinks slowly while there is unused headroom.
class AdaptiveTarget {
public:
    /// @param margin frames always needed besides one device period (engine block + jitter + safety)
    /// @param startPeriod initial (pessimistic) period; adapts down towards nominalPeriod.
    void init(double rate, double nominalPeriod, double margin, bool fixed, double startPeriod = 0.0);
    /// Engine thread, every cycle. `fillBefore` = real ring fill just before consuming. Returns the target.
    double update(BurstMeter& meter, double fillBefore, double now);
    void onUnderrun();
    double target() const { return target_; }
    double period() const { return period_; }

private:
    double rate_ = 48000, nominal_ = 480, margin_ = 0, period_ = 480, target_ = 0, max_ = 0;
    bool fixed_ = false;
    double windowStart_ = -1, minFill_ = 1e18;
    size_t prevBurst_ = 0;
    double prevGap_ = 0;
};

/// Measures a device's real sample rate from its (arrival time, total frames) positions, over a
/// window that grows from the last reset: cheap interfaces can run 1 % or more off their nominal
/// rate, far more than the PI loop is meant to absorb.
class RateEstimator {
public:
    void reset() { t0_ = -1; }
    /// Measured frames per second once the window is long enough; 0 before that.
    double update(double t, uint64_t total) {
        if (t0_ < 0) {
            t0_ = t;
            f0_ = total;
            return 0.0;
        }
        const double w = t - t0_;
        return w >= kMinWindowSec ? static_cast<double>(total - f0_) / w : 0.0;
    }
    static constexpr double kMinWindowSec = 3.0;  // packet jitter / window ≈ 0.3 % at 3 s, shrinking after

private:
    double t0_ = -1;
    uint64_t f0_ = 0;
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
    /// Clock master: the device thread also publishes its position here (null = not the master).
    void setClockMirror(DevicePosition* p) { mirror_.store(p, std::memory_order_release); }

private:
    BridgeConfig cfg_;
    BridgeStats stats_;
    FrameRing ring_;
    VarispeedResampler rs_;
    DriftController ctl_;
    AdaptiveTarget target_;
    BurstMeter bursts_;
    std::vector<float> scratch_;
    std::atomic<DevicePosition*> mirror_{nullptr};
    double nominal_ = 1.0;
    bool primed_ = false;
    double startedAt_ = -1;  // engine thread: first read after (re)start
    DevicePosition devPos_;
    RateEstimator rate_;  // engine thread
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
    /// Clock master: the device thread also publishes its position here (null = not the master).
    void setClockMirror(DevicePosition* p) { mirror_.store(p, std::memory_order_release); }

private:
    std::atomic<DevicePosition*> mirror_{nullptr};
    BridgeConfig cfg_;
    BridgeStats stats_;
    FrameRing ring_;
    VarispeedResampler rs_;
    DriftController ctl_;
    AdaptiveTarget target_;
    BurstMeter bursts_;
    std::vector<float> scratch_;
    double nominal_ = 1.0;
    std::atomic<double> primeLevel_{0.0};      // engine -> device: fill required to (re)start
    std::atomic<bool> devicePrimed_{false};
    std::atomic<uint32_t> deviceUnderruns_{0};  // device -> engine: grow the cushion
    uint32_t seenUnderruns_ = 0;
    uint64_t masterSlack_ = 0;  // device thread: frames the device played as silence (clock master)
    double startedAt_ = -1;  // engine thread
    DevicePosition devPos_;
    RateEstimator rate_;  // engine thread
    uint64_t devTotal_ = 0;  // device thread
    uint64_t engTotal_ = 0;  // engine thread
    std::atomic<bool> stopped_{true};
    std::atomic<bool> restart_{false};
};

}  // namespace ts
