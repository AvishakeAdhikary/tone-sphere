#pragma once
// Variable-ratio ("varispeed") resampler: polyphase windowed-sinc with linear
// interpolation between phases. Used both for real sample-rate conversion and
// for continuous clock-drift compensation between independent audio devices.
// Real-time safe: all memory is allocated in init().
#include <cstddef>
#include <vector>

namespace ts {

class VarispeedResampler {
public:
    /// @param maxInputPerCall upper bound on frames pushed between pulls (sizes the FIFO)
    /// @param cutoff normalised cutoff (1.0 = input Nyquist). Use min(1, 1/step)*0.95 when downsampling.
    void init(size_t channels, size_t maxInputPerCall, double cutoff = 0.95, int halfTaps = 16, int phases = 256);
    void reset();

    size_t channels() const { return channels_; }
    /// Frames buffered and not yet consumed (beyond the filter history).
    size_t buffered() const;
    /// Space left in the input FIFO.
    size_t space() const { return fifoCap_ - fifoLen_; }
    /// Group delay in input frames.
    int latencyFrames() const { return halfTaps_; }

    /// Appends planar input.
    size_t push(const float* const* in, size_t frames);
    /// Appends interleaved input.
    size_t pushInterleaved(const float* in, size_t frames);

    /// Input frames that must still be pushed to produce `outFrames` at `step` (input frames per output frame).
    size_t inputNeeded(size_t outFrames, double step) const;
    /// Produces up to `outFrames` planar outputs; returns frames produced.
    size_t pull(float* const* out, size_t outFrames, double step);
    /// Produces as many outputs as the buffered input allows (up to maxOut), interleaved.
    size_t pullInterleaved(float* out, size_t maxOut, double step);

private:
    float tap(size_t ch, double pos) const;
    void compact();

    size_t channels_ = 0;
    int halfTaps_ = 16;
    int phases_ = 256;
    std::vector<float> table_;  // (phases+1) * (2*halfTaps)
    std::vector<std::vector<float>> fifo_;
    size_t fifoCap_ = 0;
    size_t fifoLen_ = 0;
    double pos_ = 0.0;  // read position (input frames) relative to fifo start
};

}  // namespace ts
