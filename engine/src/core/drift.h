#pragma once
// PI controller that keeps a ring buffer's fill level at a target by nudging the
// resampling ratio. Output is a fractional correction (e.g. 1e-4 = +100 ppm).
// A feed-forward term carries the device's measured rate error (cheap USB interfaces can be
// off by 1 % or more); the PI loop then only trims the remainder within ±maxCorrection.
#include <algorithm>
#include <cmath>

namespace nt {

class DriftController {
public:
    void configure(double rate, double targetFrames, double maxCorrection = 0.005) {
        rate_ = rate;
        target_ = targetFrames;
        maxCorr_ = maxCorrection;
        reset();
    }
    void reset() {
        avg_ = target_;
        integ_ = 0.0;
        corr_ = ff_;
        settled_ = false;
    }
    void setTarget(double t) { target_ = t; }
    /// Measured device rate error (fraction), applied on top of the PI correction.
    void setFeedForward(double ff) { ff_ = std::clamp(ff, -kMaxFeedForward, kMaxFeedForward); }
    double feedForward() const { return ff_; }
    double target() const { return target_; }
    double correction() const { return corr_; }
    double ppm() const { return corr_ * 1e6; }
    double smoothedFill() const { return avg_; }

    /// @param fill  current fill in frames (device-rate frames)
    /// @param dt    seconds since the previous update
    double update(double fill, double dt) {
        // Low-pass the sawtooth caused by device periods.
        const double alpha = 1.0 - std::exp(-dt / kSmoothSeconds);
        avg_ += (fill - avg_) * alpha;
        const double err = (avg_ - target_) / rate_;  // seconds
        integ_ += err * dt;
        const double iMax = maxCorr_ / kKi;
        integ_ = std::clamp(integ_, -iMax, iMax);
        corr_ = ff_ + std::clamp(kKp * err + kKi * integ_, -maxCorr_, maxCorr_);
        return corr_;
    }

    // Loop tuned for ~0.3 rad/s natural frequency, damping ~0.7 (see tests/drift_test.cpp).
    static constexpr double kKp = 0.42;
    static constexpr double kKi = 0.09;
    static constexpr double kSmoothSeconds = 0.4;
    static constexpr double kMaxFeedForward = 0.03;  // ±3 %: anything further off is a misconfigured device

private:
    double rate_ = 48000.0;
    double target_ = 0.0;
    double maxCorr_ = 0.005;
    double avg_ = 0.0;
    double integ_ = 0.0;
    double corr_ = 0.0;
    double ff_ = 0.0;
    bool settled_ = false;
};

}  // namespace nt
