#include "core/latency.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>

#include "core/engine.h"
#include "util/log.h"

namespace nt {

std::vector<float> makeLatencyChirp(double sampleRate) {
    const size_t n = static_cast<size_t>(sampleRate * 0.010);
    const double f0 = 500.0, f1 = 8000.0, dur = static_cast<double>(n) / sampleRate;
    std::vector<float> s(n);
    for (size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / sampleRate;
        const double phase = 2.0 * 3.14159265358979 * (f0 * t + 0.5 * (f1 - f0) / dur * t * t);
        const double w = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979 * static_cast<double>(i) / static_cast<double>(n - 1));
        s[i] = static_cast<float>(0.5 * w * std::sin(phase));
    }
    return s;
}

LatencyEstimate estimateLatency(const std::vector<float>& x, const std::vector<float>& s, size_t emitAt) {
    LatencyEstimate r;
    const size_t m = s.size();
    if (m == 0 || x.size() < emitAt + m) return r;
    double sNorm = 0;
    for (float v : s) sNorm += double(v) * v;
    sNorm = std::sqrt(sNorm);
    // Plain cross-correlation over every lag after the chirp was emitted (a few ms of work).
    const size_t lags = x.size() - m - emitAt + 1;
    std::vector<double> c(lags);
    size_t best = 0;
    for (size_t L = 0; L < lags; ++L) {
        double acc = 0;
        const float* xp = x.data() + emitAt + L;
        for (size_t k = 0; k < m; ++k) acc += double(s[k]) * xp[k];
        c[L] = acc;
        if (std::fabs(acc) > std::fabs(c[best])) best = L;
    }
    double xNorm = 0;
    for (size_t k = 0; k < m; ++k) xNorm += double(x[emitAt + best + k]) * x[emitAt + best + k];
    xNorm = std::sqrt(xNorm);
    r.confidence = (xNorm > 0 && sNorm > 0) ? std::fabs(c[best]) / (xNorm * sNorm) : 0.0;
    // Sub-sample peak (parabola through the neighbours of |c|).
    double frac = 0;
    if (best > 0 && best + 1 < lags) {
        const double a = std::fabs(c[best - 1]), b = std::fabs(c[best]), d = std::fabs(c[best + 1]);
        const double den = a - 2 * b + d;
        if (den != 0) frac = std::clamp(0.5 * (a - d) / den, -0.5, 0.5);
    }
    r.samples = static_cast<double>(best) + frac;
    return r;
}

// ------------------------------------------------------------------ engine side
json Engine::measureLatency(const json& p) {
    const std::string outId = p.value("output", ""), inId = p.value("input", "");
    auto probe = std::make_shared<LatencyProbe>();
    std::shared_ptr<NodeRuntime> outRt, inRt;
    double sr = 48000.0;
    {
        std::lock_guard lock(mu_);
        NodeModel* o = findNode(outId);
        NodeModel* i = findNode(inId);
        if (!o || !i) throw std::runtime_error("Choose an output and an input to measure");
        if (!binIsDest(o->bin) || o->bin == Bin::Bus) throw std::runtime_error("The output must be a device or app NexTone plays into");
        if (!binIsSource(i->bin) || i->bin == Bin::Bus || i->id == "tone")
            throw std::runtime_error("The input must be a device or app NexTone records from");
        if (!o->present || !i->present) throw std::runtime_error("Both devices must be connected");
        forcedOpen_.insert(outId);
        forcedOpen_.insert(inId);
        rebuildLocked();
        sr = runningRate_;
        outRt = o->rt;
        inRt = i->rt;
    }
    probe->stimulus = makeLatencyChirp(sr);
    probe->emitAt = static_cast<size_t>(sr * 0.1);
    probe->captured.assign(static_cast<size_t>(sr * 1.1), 0.0f);  // round trips up to ~1 s

    auto cleanup = [&] {
        std::lock_guard lock(mu_);
        if (outRt) outRt->probeOut.store(nullptr, std::memory_order_release);
        if (inRt) inRt->probeIn.store(nullptr, std::memory_order_release);
        retireLocked(probe);  // the audio thread may still hold the pointer for one cycle
        forcedOpen_.erase(outId);
        forcedOpen_.erase(inId);
        rebuildLocked();
    };
    try {
        if (!outRt || !inRt) throw std::runtime_error("The devices are not available");
        // Let freshly opened streams start and their buffers settle.
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        inRt->probeIn.store(probe.get(), std::memory_order_release);
        outRt->probeOut.store(probe.get(), std::memory_order_release);
        probe->armed.store(true, std::memory_order_release);
        for (int t = 0; t < 400 && !probe->done.load(std::memory_order_acquire); ++t) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (!probe->done.load(std::memory_order_acquire)) throw std::runtime_error("The output isn't running, so nothing could be measured");
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
    const LatencyEstimate e = estimateLatency(probe->captured, probe->stimulus, probe->emitAt);
    NT_INFO("Latency %s -> %s: %.2f ms (confidence %.2f)", outId.c_str(), inId.c_str(), e.samples / sr * 1000.0, e.confidence);
    if (e.confidence < 0.3)
        throw std::runtime_error("The test signal didn't come back. Connect the output to the input (cable or loopback) and try again.");
    return {{"latencyMs", e.samples / sr * 1000.0}, {"samples", e.samples}, {"sampleRate", sr}, {"confidence", e.confidence}};
}

}  // namespace nt
