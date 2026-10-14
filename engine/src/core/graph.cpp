#include "core/graph.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "util/platform.h"

namespace ts {

const char* binName(Bin b) {
    switch (b) {
        case Bin::PhysicalIn: return "physical-in";
        case Bin::PhysicalOut: return "physical-out";
        case Bin::VirtualIn: return "virtual-in";
        case Bin::VirtualOut: return "virtual-out";
        case Bin::AppIn: return "app-in";
        case Bin::AppOut: return "app-out";
        case Bin::Bus: return "bus";
    }
    return "?";
}
bool binIsSource(Bin b) { return b == Bin::PhysicalIn || b == Bin::VirtualOut || b == Bin::AppOut || b == Bin::Bus; }
bool binIsDest(Bin b) { return b == Bin::PhysicalOut || b == Bin::VirtualIn || b == Bin::AppIn || b == Bin::Bus; }

void NodeRuntime::allocate(int ch, double sampleRate) {
    channels = std::clamp(ch, 1, kMaxChannels);
    storage.assign(static_cast<size_t>(channels) * kMaxBlock, 0.0f);
    for (int c = 0; c < kMaxChannels; ++c) buf[c] = c < channels ? &storage[static_cast<size_t>(c) * kMaxBlock] : nullptr;
    delayCap = static_cast<int>(sampleRate * 2.0) + kMaxBlock;
    delayLine.assign(static_cast<size_t>(channels) * delayCap, 0.0f);
    tapScratch.assign(static_cast<size_t>(channels) * kMaxBlock, 0.0f);
    delayPos = 0;
}

namespace {

inline float dbToLin(float db) { return db <= -119.0f ? 0.0f : std::pow(10.0f, db * 0.05f); }

void zero(NodeRuntime& n, int frames) {
    for (int c = 0; c < n.channels; ++c) std::memset(n.buf[c], 0, static_cast<size_t>(frames) * sizeof(float));
}

void mixIncoming(NodeRuntime& n, Incoming& inc, int frames) {
    RouteRuntime& r = *inc.route;
    const bool on = r.enabled.load(std::memory_order_relaxed) && !r.muted.load(std::memory_order_relaxed);
    const float target = on ? dbToLin(r.gainDb.load(std::memory_order_relaxed)) : 0.0f;
    const float start = r.curGain;
    r.curGain = target;
    if (target == 0.0f && start == 0.0f) return;
    const float step = (target - start) / static_cast<float>(frames);
    NodeRuntime& from = *inc.from;
    for (const Crosspoint& xp : inc.xps) {
        if (xp.from >= from.channels || xp.to >= n.channels) continue;
        const float* src = from.buf[xp.from];
        float* dst = n.buf[xp.to];
        if (step == 0.0f) {
            const float g = xp.gain * target;
            for (int i = 0; i < frames; ++i) dst[i] += src[i] * g;
        } else {
            float g = start;
            for (int i = 0; i < frames; ++i) {
                g += step;
                dst[i] += src[i] * g * xp.gain;
            }
        }
    }
}

void applyDelay(NodeRuntime& n, int frames, double sr) {
    const float ms = n.delayMs.load(std::memory_order_relaxed);
    if (ms <= 0.0f || n.delayCap <= 0) return;
    const int d = std::clamp(static_cast<int>(ms * 0.001 * sr), 0, n.delayCap - 1);
    for (int c = 0; c < n.channels; ++c) {
        float* line = &n.delayLine[static_cast<size_t>(c) * n.delayCap];
        float* x = n.buf[c];
        int w = n.delayPos;
        for (int i = 0; i < frames; ++i) {
            line[w] = x[i];
            int r = w - d;
            if (r < 0) r += n.delayCap;
            x[i] = line[r];
            if (++w == n.delayCap) w = 0;
        }
    }
    n.delayPos = (n.delayPos + frames) % n.delayCap;
}

void applyGainPanInvert(NodeRuntime& n, int frames) {
    const bool inv = n.invert.load(std::memory_order_relaxed);
    const float target = n.muted.load(std::memory_order_relaxed) ? 0.0f : dbToLin(n.gainDb.load(std::memory_order_relaxed));
    const float start = n.curGain;
    n.curGain = target;
    const float pan = n.channels == 2 ? n.pan.load(std::memory_order_relaxed) : 0.0f;
    const float sign = inv ? -1.0f : 1.0f;
    for (int c = 0; c < n.channels; ++c) {
        float balance = 1.0f;
        if (n.channels == 2) balance = c == 0 ? std::min(1.0f, 1.0f - pan) : std::min(1.0f, 1.0f + pan);
        float* x = n.buf[c];
        if (start == target) {
            const float g = target * balance * sign;
            if (g == 1.0f) continue;
            for (int i = 0; i < frames; ++i) x[i] *= g;
        } else {
            const float step = (target - start) / static_cast<float>(frames);
            float g = start;
            for (int i = 0; i < frames; ++i) {
                g += step;
                x[i] *= g * balance * sign;
            }
        }
    }
}

void meter(NodeRuntime& n, MeterBank& bank, int offset, int frames, double sr) {
    if (offset < 0 || static_cast<size_t>(offset + n.channels * 2) > bank.size) return;
    const float decay = static_cast<float>(std::exp(-static_cast<double>(frames) / (sr * 0.12)));
    for (int c = 0; c < n.channels; ++c) {
        const float* x = n.buf[c];
        float peak = 0.0f;
        double sum = 0.0;
        for (int i = 0; i < frames; ++i) {
            const float a = std::fabs(x[i]);
            peak = a > peak ? a : peak;
            sum += static_cast<double>(x[i]) * x[i];
        }
        const float rms = static_cast<float>(std::sqrt(sum / frames));
        std::atomic<float>& pv = bank.values[static_cast<size_t>(offset + c * 2)];
        std::atomic<float>& rv = bank.values[static_cast<size_t>(offset + c * 2 + 1)];
        pv.store(std::max(peak, pv.load(std::memory_order_relaxed) * decay), std::memory_order_relaxed);
        rv.store(std::max(rms, rv.load(std::memory_order_relaxed) * decay), std::memory_order_relaxed);
    }
}

void tone(NodeRuntime& n, int frames, double sr) {
    ToneGen& t = *n.tone;
    if (!t.enabled.load(std::memory_order_relaxed)) {
        zero(n, frames);
        return;
    }
    const double inc = 2.0 * 3.14159265358979323846 * t.freq.load(std::memory_order_relaxed) / sr;
    const float amp = dbToLin(t.levelDb.load(std::memory_order_relaxed));
    for (int i = 0; i < frames; ++i) {
        const float v = amp * static_cast<float>(std::sin(t.phase));
        t.phase += inc;
        if (t.phase > 6.283185307179586) t.phase -= 6.283185307179586;
        for (int c = 0; c < n.channels; ++c) n.buf[c][i] = v;
    }
}

}  // namespace

void processProgram(Program& p, int frames, double nowSec) {
    const double sr = p.sampleRate;
    MeterBank* bank = p.meters.get();
    for (ProgramNode& pn : p.nodes) {
        const double t0 = nowSeconds();
        NodeRuntime& n = *pn.rt;
        if (pn.isSource && !pn.isBus) {
            if (CaptureBridge* cb = n.capture.load(std::memory_order_acquire))
                cb->engineRead(n.buf, static_cast<size_t>(frames), nowSec);
            else if (n.tone)
                tone(n, frames, sr);
            else
                zero(n, frames);
            // Latency probe: record the raw input (sources run before destinations in a cycle).
            if (LatencyProbe* lp = n.probeIn.load(std::memory_order_acquire);
                lp && lp->armed.load(std::memory_order_acquire) && !lp->done.load(std::memory_order_relaxed)) {
                const size_t room = lp->captured.size() > lp->pos ? lp->captured.size() - lp->pos : 0;
                const size_t k = std::min(room, static_cast<size_t>(frames));
                std::copy(n.buf[0], n.buf[0] + k, lp->captured.begin() + static_cast<std::ptrdiff_t>(lp->pos));
            }
        } else {
            zero(n, frames);
            if (pn.physicalOut && !p.soloBuses.empty()) {
                // "Listen" mode: physical outputs carry only the soloed mixes.
                for (NodeRuntime* b : p.soloBuses) {
                    for (int c = 0; c < n.channels; ++c) {
                        const int sc = b->channels == 1 ? 0 : c % b->channels;
                        const float* src = b->buf[sc];
                        float* dst = n.buf[c];
                        for (int i = 0; i < frames; ++i) dst[i] += src[i];
                    }
                }
            } else {
                for (Incoming& inc : pn.in) mixIncoming(n, inc, frames);
            }
        }

        for (auto& fx : pn.fx) {
            if (!fx->bypass.load(std::memory_order_relaxed)) fx->process(n.buf, n.channels, frames);
        }
        applyDelay(n, frames, sr);
        applyGainPanInvert(n, frames);
        if (bank && !n.externalMeter.load(std::memory_order_relaxed)) meter(n, *bank, pn.meterOffset, frames, sr);
        if (FrameRing* tap = n.tap.load(std::memory_order_acquire)) {
            float* dst = n.tapScratch.data();
            for (int i = 0; i < frames; ++i)
                for (int c = 0; c < n.channels; ++c) *dst++ = n.buf[c][i];
            tap->write(n.tapScratch.data(), static_cast<size_t>(frames));  // drops if the writer falls behind
        }
        if (LatencyProbe* lp = n.probeOut.load(std::memory_order_acquire);
            lp && lp->armed.load(std::memory_order_acquire) && !lp->done.load(std::memory_order_relaxed)) {
            // Latency probe: add the chirp (after this node's gain, so muting can't hide it).
            for (int i = 0; i < frames; ++i) {
                const size_t t = lp->pos + static_cast<size_t>(i);
                if (t < lp->emitAt || t - lp->emitAt >= lp->stimulus.size()) continue;
                for (int c = 0; c < n.channels; ++c) n.buf[c][i] += lp->stimulus[t - lp->emitAt];
            }
            lp->pos += static_cast<size_t>(frames);
            if (lp->pos >= lp->captured.size()) lp->done.store(true, std::memory_order_release);
        }
        if (pn.isDest) {
            if (RenderBridge* rb = n.render.load(std::memory_order_acquire)) rb->engineWrite(n.buf, static_cast<size_t>(frames), nowSec);
        }

        const float us = static_cast<float>((nowSeconds() - t0) * 1e6);
        const float prev = n.cpuUs.load(std::memory_order_relaxed);
        n.cpuUs.store(prev + (us - prev) * 0.05f, std::memory_order_relaxed);
    }
}

}  // namespace ts
