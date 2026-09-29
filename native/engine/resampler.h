// Drift correction between two device clocks: the native form of the legacy
// `DriftResampler` (tonesphere/engine/dsp.py). The consumer of a ring between two devices
// reads through this. It holds the ring's fill level near a target by consuming slightly
// faster or slower than it produces output — a ratio within ±0.1 %, far below anything
// audible as pitch — using linear interpolation with a carried fractional position so
// block boundaries are seamless.
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
        const uint32_t fill = ring.available();
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
            primed_ = true;
            smoothed_fill_ = target_;
        }

        // The fill level jitters by a whole device period as the two threads interleave,
        // so steer on a slow average of it, not on each reading.
        smoothed_fill_ += (fill - smoothed_fill_) * 0.01;
        const double error = (smoothed_fill_ - target_) / target_;
        ratio_ = 1.0 + std::clamp(error * 0.002, -0.001, 0.001);

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
    double ratio_ = 1.0;
    double smoothed_fill_ = 0.0;
    bool primed_ = false;
};

}  // namespace ts
