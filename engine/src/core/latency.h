#pragma once
// Round-trip latency measurement helpers (unit tested). The engine side is Engine::measureLatency.
#include <cstddef>
#include <vector>

namespace nt {

/// Test signal: a Hann-windowed linear chirp (500 Hz .. 8 kHz), 10 ms, peak 0.5.
std::vector<float> makeLatencyChirp(double sampleRate);

struct LatencyEstimate {
    double samples = 0;     // delay of the chirp in the capture, relative to when it was played
    double confidence = 0;  // normalised correlation at the peak (0..1)
};

/// Finds `stimulus` in `captured`, where the stimulus was played starting at index `emitAt`.
LatencyEstimate estimateLatency(const std::vector<float>& captured, const std::vector<float>& stimulus, size_t emitAt);

}  // namespace nt
