// Built-in processors. Formulas are the ones proven in tonesphere/engine/effects.py and
// dsp.py (RBJ cookbook biquads, soft-knee compressor, delay with feedback), moved from
// per-block to per-sample where that was a compromise forced by Python rather than a
// choice: the compressor's detector and the limiter now follow every sample.
//
// Every processor preallocates in its constructor (control thread). process() is
// allocation-free and noexcept. Parameters are atomics written by the control thread;
// the audio thread notices a change through `version` and recomputes coefficients itself.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>

#include "tonesphere_native.h"

namespace ts {

constexpr uint32_t kMaxParams = 64;

class Processor {
public:
    Processor(uint32_t type, uint32_t channels, uint32_t sample_rate)
        : type_(type), channels_(channels), sample_rate_(sample_rate) {}
    virtual ~Processor() = default;

    uint32_t type() const { return type_; }
    uint32_t channels() const { return channels_; }

    // Control thread.
    bool set_param(uint32_t index, float value) {
        if (index >= param_count() || !std::isfinite(value)) return false;
        params_[index].store(value, std::memory_order_relaxed);
        version_.fetch_add(1, std::memory_order_release);
        return true;
    }
    float param(uint32_t index) const { return params_[index].load(std::memory_order_relaxed); }
    virtual uint32_t param_count() const = 0;
    // The VST3 handle this processor runs, or 0 for a built-in one.
    virtual uint32_t plugin() const { return 0; }
    // A value the audio thread reports back (gain reduction), or NaN if none.
    virtual float readout() const { return NAN; }

    // Audio thread.
    void run(float* const* ch, uint32_t frames) noexcept {
        const uint32_t v = version_.load(std::memory_order_acquire);
        if (v != seen_) {
            seen_ = v;
            update();
        }
        process(ch, frames);
    }

protected:
    virtual void update() noexcept = 0;
    virtual void process(float* const* ch, uint32_t frames) noexcept = 0;
    float p(uint32_t index) const { return params_[index].load(std::memory_order_relaxed); }

    const uint32_t type_;
    const uint32_t channels_;
    const uint32_t sample_rate_;
    std::atomic<float> params_[kMaxParams] = {};
    std::atomic<uint32_t> version_{1};
    uint32_t seen_ = 0;
};

// ---- Parametric EQ: up to 8 biquad bands in series --------------------------------------

class ParametricEq final : public Processor {
public:
    static constexpr uint32_t kBands = 8;
    // Per band: [type, frequency Hz, Q (or shelf slope), gain dB]. Type 0 is off.
    ParametricEq(uint32_t channels, uint32_t sample_rate)
        : Processor(TS_INSERT_EQ, channels, sample_rate), state_(new double[kBands * channels * 4]()) {
        for (uint32_t b = 0; b < kBands; ++b) {
            params_[b * 4 + 0].store(0.0f);
            params_[b * 4 + 1].store(1000.0f);
            params_[b * 4 + 2].store(0.707f);
            params_[b * 4 + 3].store(0.0f);
        }
    }
    uint32_t param_count() const override { return kBands * 4; }

protected:
    struct Coeffs { double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0; bool active = false; };

    void update() noexcept override {
        for (uint32_t b = 0; b < kBands; ++b) {
            const int type = static_cast<int>(p(b * 4 + 0));
            const double sr = sample_rate_;
            const double f = std::clamp<double>(p(b * 4 + 1), 10.0, sr * 0.49);
            const double q = std::max<double>(p(b * 4 + 2), 0.01);
            const double gain_db = std::clamp<double>(p(b * 4 + 3), -48.0, 48.0);
            const double w = 2.0 * std::numbers::pi * f / sr;
            const double cw = std::cos(w), sw = std::sin(w);
            const double A = std::pow(10.0, gain_db / 40.0);
            double b0 = 1, b1 = 0, b2 = 0, a0 = 1, a1 = 0, a2 = 0;
            Coeffs& c = coeffs_[b];
            c.active = type >= TS_EQ_PEAKING && type <= TS_EQ_LOWPASS;
            switch (type) {
                case TS_EQ_PEAKING: {
                    const double alpha = sw / (2.0 * q);
                    b0 = 1 + alpha * A; b1 = -2 * cw; b2 = 1 - alpha * A;
                    a0 = 1 + alpha / A; a1 = -2 * cw; a2 = 1 - alpha / A;
                    break;
                }
                case TS_EQ_LOW_SHELF:
                case TS_EQ_HIGH_SHELF: {
                    const double slope = q;
                    const double alpha = (sw / 2.0) * std::sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0);
                    const double beta = 2.0 * std::sqrt(A) * alpha;
                    if (type == TS_EQ_LOW_SHELF) {
                        b0 = A * ((A + 1) - (A - 1) * cw + beta);
                        b1 = 2 * A * ((A - 1) - (A + 1) * cw);
                        b2 = A * ((A + 1) - (A - 1) * cw - beta);
                        a0 = (A + 1) + (A - 1) * cw + beta;
                        a1 = -2 * ((A - 1) + (A + 1) * cw);
                        a2 = (A + 1) + (A - 1) * cw - beta;
                    } else {
                        b0 = A * ((A + 1) + (A - 1) * cw + beta);
                        b1 = -2 * A * ((A - 1) + (A + 1) * cw);
                        b2 = A * ((A + 1) + (A - 1) * cw - beta);
                        a0 = (A + 1) - (A - 1) * cw + beta;
                        a1 = 2 * ((A - 1) - (A + 1) * cw);
                        a2 = (A + 1) - (A - 1) * cw - beta;
                    }
                    break;
                }
                case TS_EQ_HIGHPASS: {
                    const double alpha = sw / (2.0 * q);
                    b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2;
                    a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha;
                    break;
                }
                case TS_EQ_LOWPASS: {
                    const double alpha = sw / (2.0 * q);
                    b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = (1 - cw) / 2;
                    a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha;
                    break;
                }
                default:
                    break;
            }
            c.b0 = b0 / a0; c.b1 = b1 / a0; c.b2 = b2 / a0; c.a1 = a1 / a0; c.a2 = a2 / a0;
        }
    }

    void process(float* const* ch, uint32_t frames) noexcept override {
        for (uint32_t b = 0; b < kBands; ++b) {
            const Coeffs& c = coeffs_[b];
            if (!c.active) continue;
            for (uint32_t k = 0; k < channels_; ++k) {
                // Direct form I in double: the recursion needs the precision at low
                // frequencies, where float coefficients leave audible limit cycles.
                double* s = state_.get() + (static_cast<size_t>(b) * channels_ + k) * 4;
                double x1 = s[0], x2 = s[1], y1 = s[2], y2 = s[3];
                float* x = ch[k];
                for (uint32_t i = 0; i < frames; ++i) {
                    const double x0 = x[i];
                    const double y0 = c.b0 * x0 + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
                    x2 = x1; x1 = x0; y2 = y1; y1 = y0;
                    x[i] = static_cast<float>(y0);
                }
                // Flush denormals out of the feedback path; a filter fed silence would
                // otherwise decay into subnormals that cost 100x per operation. And a
                // state that has gone non-finite is reset rather than kept: it would
                // never recover on its own.
                if (std::abs(y1) < 1e-20) y1 = 0.0;
                if (std::abs(y2) < 1e-20) y2 = 0.0;
                if (!std::isfinite(y1) || !std::isfinite(y2)) x1 = x2 = y1 = y2 = 0.0;
                s[0] = x1; s[1] = x2; s[2] = y1; s[3] = y2;
            }
        }
    }

private:
    Coeffs coeffs_[kBands];
    std::unique_ptr<double[]> state_;
};

// ---- Compressor: soft knee, per-sample peak detector, channels linked ------------------

class Compressor final : public Processor {
public:
    // [threshold dB, ratio, attack ms, release ms, knee dB, makeup dB]
    Compressor(uint32_t channels, uint32_t sample_rate) : Processor(TS_INSERT_COMPRESSOR, channels, sample_rate) {
        params_[0].store(-18.0f); params_[1].store(4.0f); params_[2].store(10.0f);
        params_[3].store(100.0f); params_[4].store(6.0f); params_[5].store(0.0f);
    }
    uint32_t param_count() const override { return 6; }
    float readout() const override { return reduction_db_.load(std::memory_order_relaxed); }

protected:
    void update() noexcept override {
        threshold_ = p(0);
        ratio_ = std::max(1.0f, p(1));
        attack_ = coeff(p(2));
        release_ = coeff(p(3));
        knee_ = std::max(0.0f, p(4));
        makeup_ = p(5);
    }

    void process(float* const* ch, uint32_t frames) noexcept override {
        float worst = 0.0f;
        for (uint32_t i = 0; i < frames; ++i) {
            float a = 0.0f;
            for (uint32_t k = 0; k < channels_; ++k) a = std::max(a, std::abs(ch[k][i]));
            // Attack when the level rises, release when it falls: one time constant for
            // both is what makes a naive compressor pump.
            env_ = a > env_ ? a + (env_ - a) * attack_ : a + (env_ - a) * release_;
            const float level = 20.0f * std::log10(std::max(env_, 1e-7f));
            const float reduction = gain_reduction(level);
            worst = std::min(worst, reduction);
            const float g = std::pow(10.0f, (reduction + makeup_) / 20.0f);
            for (uint32_t k = 0; k < channels_; ++k) ch[k][i] *= g;
        }
        reduction_db_.store(worst, std::memory_order_relaxed);
    }

private:
    float coeff(float ms) const {
        return std::exp(-1.0f / (std::max(1e-3f, ms / 1000.0f) * static_cast<float>(sample_rate_)));
    }
    float gain_reduction(float level_db) const {
        const float over = level_db - threshold_;
        const float half = knee_ / 2.0f;
        if (over <= -half) return 0.0f;
        if (over >= half || knee_ == 0.0f) return -(over - over / ratio_);
        const float k = (over + half) * (over + half) / (2.0f * knee_);
        return -(k - k / ratio_);
    }

    float threshold_ = -18.0f, ratio_ = 4.0f, attack_ = 0.0f, release_ = 0.0f, knee_ = 6.0f, makeup_ = 0.0f;
    float env_ = 0.0f;
    std::atomic<float> reduction_db_{0.0f};
};

// ---- Limiter: instant attack, sample-accurate, no lookahead ------------------------------
//
// |output| never exceeds the threshold, sample by sample: the gain is clamped to at most
// threshold/|x| on every sample, and release only ever moves it back up towards that
// bound. There is no lookahead, deliberately — lookahead is latency, and this sits on the
// monitoring path a player hears. The cost is that a transient is flattened within one
// sample rather than eased into, which is what a safety limiter is for.

class Limiter final : public Processor {
public:
    // [threshold (linear), release ms]
    Limiter(uint32_t channels, uint32_t sample_rate) : Processor(TS_INSERT_LIMITER, channels, sample_rate) {
        params_[0].store(0.99f);
        params_[1].store(80.0f);
    }
    uint32_t param_count() const override { return 2; }
    float readout() const override { return reduction_db_.load(std::memory_order_relaxed); }

protected:
    void update() noexcept override {
        threshold_ = std::clamp(p(0), 1e-4f, 1.0f);
        release_ = std::exp(-1.0f / (std::max(1e-3f, p(1) / 1000.0f) * static_cast<float>(sample_rate_)));
    }

    void process(float* const* ch, uint32_t frames) noexcept override {
        float lowest = 1.0f;
        for (uint32_t i = 0; i < frames; ++i) {
            float a = 0.0f;
            for (uint32_t k = 0; k < channels_; ++k) a = std::max(a, std::abs(ch[k][i]));
            const float bound = a > threshold_ ? threshold_ / a : 1.0f;
            gain_ = bound < gain_ ? bound : bound + (gain_ - bound) * release_;
            lowest = std::min(lowest, gain_);
            if (gain_ < 1.0f)
                for (uint32_t k = 0; k < channels_; ++k) ch[k][i] *= gain_;
        }
        reduction_db_.store(20.0f * std::log10(std::max(lowest, 1e-6f)), std::memory_order_relaxed);
    }

private:
    float threshold_ = 0.99f, release_ = 0.0f, gain_ = 1.0f;
    std::atomic<float> reduction_db_{0.0f};
};

// ---- Delay with feedback ----------------------------------------------------------------

class Delay final : public Processor {
public:
    static constexpr float kMaxSeconds = 2.0f;
    // [delay ms, feedback 0..0.95, mix 0..1]
    Delay(uint32_t channels, uint32_t sample_rate)
        : Processor(TS_INSERT_DELAY, channels, sample_rate),
          capacity_(static_cast<uint32_t>(sample_rate * kMaxSeconds) + 1),
          buffer_(new float[static_cast<size_t>(capacity_) * channels]()) {
        params_[0].store(250.0f); params_[1].store(0.35f); params_[2].store(0.25f);
    }
    uint32_t param_count() const override { return 3; }

protected:
    void update() noexcept override {
        const float samples = p(0) / 1000.0f * static_cast<float>(sample_rate_);
        delay_ = std::clamp<uint32_t>(static_cast<uint32_t>(samples), 1, capacity_ - 1);
        feedback_ = std::clamp(p(1), 0.0f, 0.95f);
        mix_ = std::clamp(p(2), 0.0f, 1.0f);
    }

    void process(float* const* ch, uint32_t frames) noexcept override {
        for (uint32_t k = 0; k < channels_; ++k) {
            float* line = buffer_.get() + static_cast<size_t>(k) * capacity_;
            uint32_t w = write_;
            for (uint32_t i = 0; i < frames; ++i) {
                const uint32_t r = w >= delay_ ? w - delay_ : w + capacity_ - delay_;
                const float tap = line[r];
                line[w] = ch[k][i] + tap * feedback_;
                ch[k][i] += tap * mix_;
                if (++w == capacity_) w = 0;
            }
        }
        write_ = (write_ + frames) % capacity_;
    }

private:
    const uint32_t capacity_;
    std::unique_ptr<float[]> buffer_;
    uint32_t write_ = 0, delay_ = 1;
    float feedback_ = 0.35f, mix_ = 0.25f;
};

inline std::unique_ptr<Processor> make_processor(uint32_t type, uint32_t channels, uint32_t sample_rate) {
    switch (type) {
        case TS_INSERT_EQ: return std::make_unique<ParametricEq>(channels, sample_rate);
        case TS_INSERT_COMPRESSOR: return std::make_unique<Compressor>(channels, sample_rate);
        case TS_INSERT_LIMITER: return std::make_unique<Limiter>(channels, sample_rate);
        case TS_INSERT_DELAY: return std::make_unique<Delay>(channels, sample_rate);
        default: return nullptr;
    }
}

}  // namespace ts
