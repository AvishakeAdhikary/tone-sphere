// Drift correction between two device clocks, grown from the legacy `DriftResampler`
// (tonesphere/engine/dsp.py) — this one adds a calibration phase, a dead band and an
// integral term, all found necessary on real hardware. The consumer of a ring between two
// devices reads through this. It holds the ring's fill level near a setpoint by consuming
// slightly faster or slower than it produces output — a ratio within ±0.5 %, which only a device
// that far off its own nominal rate ever needs — using linear interpolation with a carried
// fractional position so block boundaries are seamless.
//
// Allocation-free after construction; process() is called on the consumer's thread only.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>

#include "spsc.h"

namespace ts {

class DriftResampler {
public:
    DriftResampler(uint32_t channels, uint32_t max_block, uint32_t target_fill)
        : channels_(channels),
          target_(static_cast<double>(target_fill)),
          staging_(new float[static_cast<size_t>(max_block * 2 + 4) * channels]()),
          history_(new float[static_cast<size_t>(channels) * 2]()) {}

    double ratio() const { return ratio_; }
    bool primed() const { return primed_; }

    // Fill `out` (interleaved, `frames` frames) from `ring`. Returns frames it had to
    // invent as silence because the ring ran dry (0 in steady state).
    uint32_t process(FrameRing& ring, float* out, uint32_t frames) noexcept {
        uint32_t fill = ring.available();
        if (!primed_) {
            // Wait for a cushion before consuming, or the first blocks all underrun.
            if (fill < target_) {
                std::fill(out, out + static_cast<size_t>(frames) * channels_, 0.0f);
                return 0;
            }
            // Start from exactly the target cushion. Audio arrives in packets, so the fill
            // that first reaches the target overshoots it by up to one packet; keeping that
            // excess would add it to the latency of this run, and a different amount to the
            // next.
            const uint32_t excess = fill - static_cast<uint32_t>(target_);
            if (excess) ring.skip(excess);
            fill -= excess;
            primed_ = true;
            calibrating_ = kCalibrationBlocks;
            fill_sum_ = 0.0;
        }

        if (calibrating_ > 0) {
            // Priming leaves exactly the target in the ring, but the fill is read just before
            // each consumption, while the producer delivers in packets: its average over a
            // packet cycle sits below that by up to half a packet (on the AI-04's loopback,
            // 31 % of the target). Steering towards the target would then correct a drift that
            // does not exist, time-warping the first seconds of every stream (and failing
            // round-trip measurements). So the first blocks run at ratio 1 and learn the fill
            // this stream actually settles at; only a departure from that is drift.
            fill_sum_ += fill;
            ratio_ = 1.0;
            if (--calibrating_ == 0) {
                setpoint_ = std::max(1.0, fill_sum_ / kCalibrationBlocks);
                smoothed_fill_ = setpoint_;
            }
        } else {
            // The fill level jitters by a whole device period as the two threads interleave,
            // so steer on a slow average of it, not on each reading.
            smoothed_fill_ += (fill - smoothed_fill_) * 0.01;
            double error = (smoothed_fill_ - setpoint_) / setpoint_;
            // Inside the dead band the ratio stays where it is: what is left of the packet
            // jitter after smoothing is not drift, and steering on it wobbles the ratio by
            // ~10^-4, enough to smear a 12 kHz sweep (the round-trip measurement) across a
            // fraction of a sample. Two streams on one clock therefore run at exactly 1.
            error = error > kDeadBand ? error - kDeadBand : error < -kDeadBand ? error + kDeadBand : 0.0;
            // Proportional-integral: the integral learns a constant drift, so the fill returns
            // to the setpoint instead of sagging by drift / gain, which on a small cushion is
            // the margin against running dry. Two crystals disagree by tens to hundreds of ppm,
            // but a USB device can be far worse: the Audio Array AI-04 captures "44.1 kHz"
            // about 0.26 % slow against its own playback clock (tests/hardware/
            // test_interface.py); the limit covers that with room to spare. The gains are
            // overdamped (zeta about 1.8-2.5) for block/cushion ratios from 1/4 to 1/2, so
            // the ratio settles without overshoot.
            integral_ = std::clamp(integral_ + error * kIntegralGain, -kLimit, kLimit);
            ratio_ = 1.0 + std::clamp(error * kProportionalGain + integral_, -kLimit, kLimit);
        }

        // Output j samples the input at u = position_ + j * ratio_, where u = 0 is the
        // newest frame already consumed (x[-1]) and u = 1 the first new one (x[0]). Read
        // exactly the frames the last output needs — reading by the block's end instead
        // leaves the last sample one frame short whenever the ratio is below 1, which is a
        // click at every such block boundary. position_ then lands in [ratio - 1, ratio),
        // so it can dip just below zero; two frames of history (x[-2], x[-1]) cover it.
        const double last_u = position_ + (frames - 1) * ratio_;
        const int64_t needed_signed = static_cast<int64_t>(std::floor(last_u)) + 1;
        const uint32_t needed = needed_signed > 0 ? static_cast<uint32_t>(needed_signed) : 0;
        const uint32_t got = ring.read(staging_.get(), needed);
        uint32_t missing = 0;
        if (got < needed) {
            missing = needed - got;
            std::fill(staging_.get() + static_cast<size_t>(got) * channels_,
                      staging_.get() + static_cast<size_t>(needed) * channels_, 0.0f);
        }

        auto at = [&](int64_t index, uint32_t c) -> float {
            if (index >= 0) return staging_[static_cast<size_t>(index) * channels_ + c];
            return history_[static_cast<size_t>(index + 2) * channels_ + c];  // -2 -> 0, -1 -> 1
        };
        for (uint32_t j = 0; j < frames; ++j) {
            const double u = position_ + j * ratio_;
            const int64_t k = static_cast<int64_t>(std::floor(u));
            const float a = static_cast<float>(u - static_cast<double>(k));
            for (uint32_t c = 0; c < channels_; ++c) {
                const float x0 = at(k - 1, c);
                const float x1 = at(k, c);
                out[static_cast<size_t>(j) * channels_ + c] = x0 + (x1 - x0) * a;
            }
        }

        for (uint32_t c = 0; c < channels_; ++c) {
            const float older = at(static_cast<int64_t>(needed) - 2, c);
            const float newest = at(static_cast<int64_t>(needed) - 1, c);
            history_[c] = older;
            history_[channels_ + c] = newest;
        }
        position_ = position_ + frames * ratio_ - needed;
        if (missing) primed_ = ring.available() >= target_;
        return missing;
    }

private:
    const uint32_t channels_;
    const double target_;
    std::unique_ptr<float[]> staging_;
    std::unique_ptr<float[]> history_;  // x[-2] then x[-1], interleaved by channel
    double position_ = 0.0;
    static constexpr uint32_t kCalibrationBlocks = 64;
    static constexpr double kDeadBand = 0.05;
    static constexpr double kProportionalGain = 0.01;
    static constexpr double kIntegralGain = 2e-6;
    static constexpr double kLimit = 0.005;
    double ratio_ = 1.0;
    double smoothed_fill_ = 0.0;
    double setpoint_ = 1.0;
    double fill_sum_ = 0.0;
    double integral_ = 0.0;
    uint32_t calibrating_ = 0;
    bool primed_ = false;
};

}  // namespace ts
