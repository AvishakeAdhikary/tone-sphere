#include "core/resampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ts {

namespace {
constexpr double kPi = 3.14159265358979323846;

double besselI0(double x) {
    double sum = 1.0, term = 1.0;
    for (int k = 1; k < 50; ++k) {
        term *= (x / (2.0 * k)) * (x / (2.0 * k));
        sum += term;
        if (term < 1e-12 * sum) break;
    }
    return sum;
}
}  // namespace

void VarispeedResampler::init(size_t channels, size_t maxInputPerCall, double cutoff, int halfTaps, int phases) {
    channels_ = channels;
    halfTaps_ = halfTaps;
    phases_ = phases;
    const int taps = 2 * halfTaps;
    const double beta = 8.6;
    const double i0b = besselI0(beta);
    table_.assign(static_cast<size_t>(phases + 1) * taps, 0.0f);
    for (int k = 0; k <= phases; ++k) {
        const double frac = static_cast<double>(k) / phases;
        double sum = 0.0;
        std::vector<double> row(taps);
        for (int j = 0; j < taps; ++j) {
            const double t = frac + halfTaps - 1 - j;
            const double x = cutoff * t;
            const double sinc = std::abs(x) < 1e-9 ? 1.0 : std::sin(kPi * x) / (kPi * x);
            const double r = t / halfTaps;
            const double win = std::abs(r) >= 1.0 ? 0.0 : besselI0(beta * std::sqrt(1.0 - r * r)) / i0b;
            row[j] = cutoff * sinc * win;
            sum += row[j];
        }
        // Normalise every phase to unity DC gain (removes interpolation ripple at DC).
        for (int j = 0; j < taps; ++j) table_[static_cast<size_t>(k) * taps + j] = static_cast<float>(row[j] / sum);
    }
    fifoCap_ = maxInputPerCall + static_cast<size_t>(taps) * 4 + 64;
    fifo_.assign(channels, std::vector<float>(fifoCap_, 0.0f));
    reset();
}

void VarispeedResampler::reset() {
    for (auto& f : fifo_) std::fill(f.begin(), f.end(), 0.0f);
    fifoLen_ = static_cast<size_t>(halfTaps_ - 1);
    pos_ = static_cast<double>(halfTaps_ - 1);
}

size_t VarispeedResampler::buffered() const {
    const double b = static_cast<double>(fifoLen_) - pos_;
    return b > 0 ? static_cast<size_t>(b) : 0;
}

size_t VarispeedResampler::push(const float* const* in, size_t frames) {
    const size_t n = std::min(frames, space());
    for (size_t c = 0; c < channels_; ++c) std::memcpy(&fifo_[c][fifoLen_], in[c], n * sizeof(float));
    fifoLen_ += n;
    return n;
}

size_t VarispeedResampler::pushInterleaved(const float* in, size_t frames) {
    const size_t n = std::min(frames, space());
    for (size_t c = 0; c < channels_; ++c) {
        float* dst = &fifo_[c][fifoLen_];
        const float* src = in + c;
        for (size_t i = 0; i < n; ++i) dst[i] = src[i * channels_];
    }
    fifoLen_ += n;
    return n;
}

size_t VarispeedResampler::inputNeeded(size_t outFrames, double step) const {
    if (outFrames == 0) return 0;
    const double last = pos_ + static_cast<double>(outFrames - 1) * step;
    const long long need = static_cast<long long>(std::floor(last)) + halfTaps_ + 1 - static_cast<long long>(fifoLen_);
    return need > 0 ? static_cast<size_t>(need) : 0;
}

inline float VarispeedResampler::tap(size_t ch, double pos) const {
    const long long n0 = static_cast<long long>(pos);
    const double frac = pos - static_cast<double>(n0);
    const double idx = frac * phases_;
    const int k = static_cast<int>(idx);
    const float a = static_cast<float>(idx - k);
    const int taps = 2 * halfTaps_;
    const float* r0 = &table_[static_cast<size_t>(k) * taps];
    const float* r1 = r0 + taps;
    const float* x = &fifo_[ch][static_cast<size_t>(n0 - halfTaps_ + 1)];
    float s0 = 0.0f, s1 = 0.0f;
    for (int j = 0; j < taps; ++j) {
        s0 += x[j] * r0[j];
        s1 += x[j] * r1[j];
    }
    return s0 + (s1 - s0) * a;
}

void VarispeedResampler::compact() {
    const long long drop = static_cast<long long>(pos_) - (halfTaps_ - 1);
    if (drop <= 0) return;
    const size_t d = static_cast<size_t>(std::min<long long>(drop, static_cast<long long>(fifoLen_)));
    const size_t remain = fifoLen_ - d;
    for (auto& f : fifo_) std::memmove(f.data(), f.data() + d, remain * sizeof(float));
    fifoLen_ = remain;
    pos_ -= static_cast<double>(d);
}

size_t VarispeedResampler::pull(float* const* out, size_t outFrames, double step) {
    size_t produced = 0;
    while (produced < outFrames) {
        const long long n0 = static_cast<long long>(pos_);
        if (n0 + halfTaps_ >= static_cast<long long>(fifoLen_)) break;
        for (size_t c = 0; c < channels_; ++c) out[c][produced] = tap(c, pos_);
        pos_ += step;
        ++produced;
    }
    compact();
    return produced;
}

size_t VarispeedResampler::pullInterleaved(float* out, size_t maxOut, double step) {
    size_t produced = 0;
    while (produced < maxOut) {
        const long long n0 = static_cast<long long>(pos_);
        if (n0 + halfTaps_ >= static_cast<long long>(fifoLen_)) break;
        for (size_t c = 0; c < channels_; ++c) out[produced * channels_ + c] = tap(c, pos_);
        pos_ += step;
        ++produced;
    }
    compact();
    return produced;
}

}  // namespace ts
