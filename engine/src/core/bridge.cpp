#include "core/bridge.h"

#include <algorithm>
#include <cstring>

namespace nt {

namespace {
constexpr int kHalfTaps = 16;
constexpr double kWindowSec = 8.0;

double jitterMs(const BridgeConfig& c) { return c.jittery ? 12.0 : 2.5; }
}  // namespace

// ============================== AdaptiveTarget ==============================

void AdaptiveTarget::init(double rate, double nominalPeriod, double margin, bool fixed, double startPeriod) {
    rate_ = rate;
    nominal_ = nominalPeriod;
    margin_ = margin;
    period_ = std::max(nominalPeriod, startPeriod);
    target_ = period_ + margin_;
    max_ = rate * 0.3;
    fixed_ = fixed;
    windowStart_ = -1;
    minFill_ = 1e18;
    prevBurst_ = 0;
    prevGap_ = 0;
}

double AdaptiveTarget::update(BurstMeter& meter, double fillBefore, double now) {
    if (fixed_) return target_;
    if (windowStart_ < 0) windowStart_ = now;
    minFill_ = std::min(minFill_, fillBefore);

    // Grow immediately when the device delivers/consumes in bigger chunks than assumed.
    const double observed = std::max({static_cast<double>(meter.peekFrames()), meter.peekGap() * rate_,
                                      static_cast<double>(prevBurst_), prevGap_ * rate_});
    if (observed > period_ * 1.05) {
        period_ = std::min(observed, max_);
        target_ = std::max(target_, period_ + margin_);
    }

    if (now - windowStart_ >= kWindowSec) {
        size_t b = 0;
        double g = 0;
        meter.harvest(b, g);
        prevBurst_ = b;
        prevGap_ = g;
        const double seen = std::max({nominal_, static_cast<double>(b), g * rate_});
        if (seen < period_) period_ = std::max(seen, period_ * 0.9);  // shrink slowly
        const double desired = period_ + margin_;
        // Give back latency only when the real cushion never got close to empty.
        if (target_ > desired && minFill_ > margin_ * 0.75) {
            const double give = std::min(minFill_ - margin_ * 0.5, rate_ * 0.002);  // <= 2 ms per window
            if (give > 0) target_ = std::max(desired, target_ - give);
        }
        windowStart_ = now;
        minFill_ = 1e18;
    }
    target_ = std::min(target_, max_);
    return target_;
}

void AdaptiveTarget::onUnderrun() {
    if (fixed_) return;
    target_ = std::min(max_, target_ + std::max(period_ * 0.5, rate_ * 0.003));
}

// ============================== CaptureBridge ==============================

void CaptureBridge::init(const BridgeConfig& cfg) {
    cfg_ = cfg;
    nominal_ = cfg.deviceRate / cfg.engineRate;
    if (cfg.master) {
        target_.init(cfg.deviceRate, 2.0 * kHalfTaps, 0.0, true);
    } else {
        const double margin = static_cast<double>(cfg.engineMaxBlock) * nominal_ + (jitterMs(cfg) + cfg.extraSafetyMs) * cfg.deviceRate / 1000.0;
        target_.init(cfg.deviceRate, static_cast<double>(cfg.devicePeriod), margin, false);
    }
    // Room for the largest adaptive target plus bursts.
    const size_t cap = static_cast<size_t>(cfg.deviceRate * 0.3) * 3 + cfg.devicePeriod * 4 + cfg.engineMaxBlock * 4 + 4096;
    ring_.init(cfg.channels, cap);
    const size_t maxIn = static_cast<size_t>(static_cast<double>(cfg.engineMaxBlock) * nominal_ * 1.05) + 4 * kHalfTaps + 64;
    rs_.init(cfg.channels, maxIn, 0.95 * std::min(1.0, 1.0 / nominal_), kHalfTaps);
    ctl_.configure(cfg.deviceRate, target_.target());
    scratch_.assign(maxIn * cfg.channels, 0.0f);
    primed_ = false;
    devTotal_ = 0;
    engTotal_ = 0;
    stats_.latencyMs.store(static_cast<float>((target_.target() + kHalfTaps) / cfg.deviceRate * 1000.0));
}

void CaptureBridge::deviceWrite(const float* interleaved, size_t frames, double nowSec) {
    bursts_.observe(frames, nowSec);
    const size_t w = ring_.write(interleaved, frames);
    if (w < frames) stats_.overruns.fetch_add(1, std::memory_order_relaxed);
    devTotal_ += w;
    devPos_.publish(nowSec, devTotal_);
}

void CaptureBridge::deviceWriteSilence(size_t frames, double nowSec) {
    bursts_.observe(frames, nowSec);
    devTotal_ += ring_.writeSilence(frames);
    devPos_.publish(nowSec, devTotal_);
}

void CaptureBridge::engineRead(float* const* out, size_t frames, double nowSec) {
    auto silence = [&] {
        for (size_t c = 0; c < cfg_.channels; ++c) std::memset(out[c], 0, frames * sizeof(float));
    };
    if (restart_.exchange(false, std::memory_order_acq_rel)) {
        rs_.reset();
        ctl_.setFeedForward(0.0);
        ctl_.reset();
        rate_.reset();
        primed_ = false;
        startedAt_ = -1;
    }
    if (stopped_.load(std::memory_order_acquire)) {
        stats_.active.store(false, std::memory_order_relaxed);
        silence();
        return;
    }
    stats_.active.store(true, std::memory_order_relaxed);
    if (startedAt_ < 0) startedAt_ = nowSec;

    size_t avail = ring_.available();
    const double tgt = target_.update(bursts_, static_cast<double>(avail), nowSec);
    ctl_.setTarget(tgt);
    if (!primed_) {
        if (static_cast<double>(avail) < tgt) {
            silence();
            return;
        }
        primed_ = true;
        // Start with exactly the target cushion.
        engTotal_ += ring_.skip(avail - static_cast<size_t>(tgt));
        avail = ring_.available();
        ctl_.reset();
    }
    // Far too much data (device burst / stall recovery): drop back to the target.
    if (static_cast<double>(avail) > tgt * 3.0 + target_.period() * 2.0) {
        engTotal_ += ring_.skip(avail - static_cast<size_t>(tgt));
        avail = ring_.available();
        ctl_.reset();
        rate_.reset();  // a burst or stall would skew the rate measurement
        stats_.overruns.fetch_add(1, std::memory_order_relaxed);
    }

    const double step = nominal_ * (1.0 + (cfg_.master ? 0.0 : ctl_.correction()));
    const size_t need = rs_.inputNeeded(frames, step);
    const size_t take = std::min({need, avail, scratch_.size() / cfg_.channels});
    if (take > 0) {
        ring_.read(scratch_.data(), take);
        engTotal_ += take;
        rs_.pushInterleaved(scratch_.data(), take);
    }
    const size_t got = rs_.pull(out, frames, step);
    if (got < frames) {
        for (size_t c = 0; c < cfg_.channels; ++c) std::memset(out[c] + got, 0, (frames - got) * sizeof(float));
        // The first seconds calibrate the cushion to the device; later dropouts are real.
        (nowSec - startedAt_ < 2.0 ? stats_.settling : stats_.underruns).fetch_add(1, std::memory_order_relaxed);
        target_.onUnderrun();
        rate_.reset();
        primed_ = false;  // rebuild the (now larger) cushion before continuing
        return;
    }

    // Continuous ("virtual") fill: frames the device has produced by now, minus frames consumed.
    double fill = static_cast<double>(ring_.available()) + static_cast<double>(rs_.buffered());
    double tW = 0;
    uint64_t totalW = 0;
    if (devPos_.read(tW, totalW)) {
        const double maxElapsed = 2.0 * target_.period() / cfg_.deviceRate;
        const double elapsed = std::clamp(nowSec - tW, 0.0, maxElapsed);
        fill = static_cast<double>(static_cast<int64_t>(totalW - engTotal_)) + elapsed * cfg_.deviceRate +
               static_cast<double>(rs_.buffered());
        // Real device rate: consume device frames as fast as they actually arrive.
        if (const double r = rate_.update(tW, totalW); r > 0 && !cfg_.master) ctl_.setFeedForward(r / cfg_.deviceRate - 1.0);
    }
    if (!cfg_.master) ctl_.update(fill, static_cast<double>(frames) / cfg_.engineRate);
    stats_.ppm.store(static_cast<float>(ctl_.ppm()), std::memory_order_relaxed);
    stats_.fillMs.store(static_cast<float>(fill / cfg_.deviceRate * 1000.0), std::memory_order_relaxed);
    stats_.targetMs.store(static_cast<float>(tgt / cfg_.deviceRate * 1000.0), std::memory_order_relaxed);
    stats_.latencyMs.store(static_cast<float>((tgt + kHalfTaps) / cfg_.deviceRate * 1000.0), std::memory_order_relaxed);
}

// ============================== RenderBridge ===============================

void RenderBridge::init(const BridgeConfig& cfg) {
    cfg_ = cfg;
    nominal_ = cfg.engineRate / cfg.deviceRate;  // engine frames per device frame
    if (cfg.master) {
        target_.init(cfg.deviceRate, static_cast<double>(cfg.devicePeriod), 0.0, true);
    } else {
        const double margin = static_cast<double>(cfg.engineMaxBlock) / nominal_ + (jitterMs(cfg) + cfg.extraSafetyMs) * cfg.deviceRate / 1000.0;
        // Start with a generous cushion (an extra device period); it shrinks once headroom is proven.
        target_.init(cfg.deviceRate, static_cast<double>(cfg.devicePeriod), margin, false, static_cast<double>(cfg.devicePeriod) * 2.0);
    }
    const size_t cap = static_cast<size_t>(cfg.deviceRate * 0.3) * 3 + cfg.devicePeriod * 4 + cfg.engineMaxBlock * 4 + 4096;
    ring_.init(cfg.channels, cap);
    rs_.init(cfg.channels, cfg.engineMaxBlock + 4 * kHalfTaps + 64, 0.95 * std::min(1.0, 1.0 / nominal_), kHalfTaps);
    ctl_.configure(cfg.deviceRate, target_.target());
    const size_t maxOut = static_cast<size_t>(static_cast<double>(cfg.engineMaxBlock) / nominal_ * 1.05) + 4 * kHalfTaps + 64;
    scratch_.assign(maxOut * cfg.channels, 0.0f);
    primeLevel_.store(target_.target());
    devicePrimed_.store(false);
    deviceUnderruns_.store(0);
    seenUnderruns_ = 0;
    devTotal_ = 0;
    engTotal_ = 0;
    stats_.latencyMs.store(static_cast<float>((target_.target() + kHalfTaps) / cfg.deviceRate * 1000.0));
}

void RenderBridge::engineWrite(const float* const* in, size_t frames, double nowSec) {
    if (restart_.exchange(false, std::memory_order_acq_rel)) {
        rs_.reset();
        ctl_.setFeedForward(0.0);
        ctl_.reset();
        rate_.reset();
        devicePrimed_.store(false, std::memory_order_release);
        startedAt_ = -1;
        if (cfg_.master) engTotal_ += ring_.writeSilence(2 * kHalfTaps);
    }
    if (stopped_.load(std::memory_order_acquire)) {
        stats_.active.store(false, std::memory_order_relaxed);
        return;
    }
    stats_.active.store(true, std::memory_order_relaxed);
    if (startedAt_ < 0) startedAt_ = nowSec;

    // Dropouts reported by the device thread grow the cushion.
    const uint32_t u = deviceUnderruns_.load(std::memory_order_acquire);
    if (u != seenUnderruns_) {
        const uint32_t n = u - seenUnderruns_;
        seenUnderruns_ = u;
        target_.onUnderrun();
        rate_.reset();
        (nowSec - startedAt_ < 2.0 ? stats_.settling : stats_.underruns).fetch_add(n, std::memory_order_relaxed);
    }
    const double fillNow = static_cast<double>(ring_.fillApprox());
    const double tgt = target_.update(bursts_, fillNow, nowSec);
    primeLevel_.store(tgt, std::memory_order_relaxed);
    ctl_.setTarget(tgt);

    const double step = nominal_ * (1.0 + (cfg_.master ? 0.0 : ctl_.correction()));
    rs_.push(in, frames);
    const size_t maxOut = scratch_.size() / cfg_.channels;
    const size_t produced = rs_.pullInterleaved(scratch_.data(), maxOut, step);

    if (!cfg_.master && fillNow > tgt * 3.0 + target_.period() * 2.0) {
        // Device is not consuming (stalled or much slower): drop this block and re-lock.
        stats_.overruns.fetch_add(1, std::memory_order_relaxed);
        ctl_.reset();
        rate_.reset();
    } else if (produced > 0) {
        const size_t w = ring_.write(scratch_.data(), produced);
        engTotal_ += w;
        if (w < produced) stats_.overruns.fetch_add(1, std::memory_order_relaxed);
    }

    double fill = static_cast<double>(ring_.fillApprox());
    double tR = 0;
    uint64_t totalR = 0;
    if (devicePrimed_.load(std::memory_order_acquire) && devPos_.read(tR, totalR)) {
        const double maxElapsed = 2.0 * target_.period() / cfg_.deviceRate;
        const double elapsed = std::clamp(nowSec - tR, 0.0, maxElapsed);
        fill = static_cast<double>(static_cast<int64_t>(engTotal_ - totalR)) - elapsed * cfg_.deviceRate;
        // Real device rate: produce device frames as fast as the device actually consumes them.
        if (const double r = rate_.update(tR, totalR); r > 0 && !cfg_.master) ctl_.setFeedForward(cfg_.deviceRate / r - 1.0);
    } else {
        rate_.reset();  // (re)priming: no frames are consumed while time passes
    }
    fill += static_cast<double>(rs_.buffered()) / nominal_;
    if (!cfg_.master && devicePrimed_.load(std::memory_order_relaxed)) {
        ctl_.update(fill, static_cast<double>(frames) / cfg_.engineRate);
    }
    stats_.ppm.store(static_cast<float>(ctl_.ppm()), std::memory_order_relaxed);
    stats_.fillMs.store(static_cast<float>(fill / cfg_.deviceRate * 1000.0), std::memory_order_relaxed);
    stats_.targetMs.store(static_cast<float>(tgt / cfg_.deviceRate * 1000.0), std::memory_order_relaxed);
    stats_.latencyMs.store(static_cast<float>((tgt + kHalfTaps) / cfg_.deviceRate * 1000.0), std::memory_order_relaxed);
}

void RenderBridge::deviceRead(float* interleaved, size_t frames, double nowSec) {
    bursts_.observe(frames, nowSec);
    if (!devicePrimed_.load(std::memory_order_acquire)) {
        if (static_cast<double>(ring_.available()) >= primeLevel_.load(std::memory_order_relaxed)) {
            devicePrimed_.store(true, std::memory_order_release);
        } else {
            std::memset(interleaved, 0, frames * cfg_.channels * sizeof(float));
            return;
        }
    }
    const size_t got = ring_.read(interleaved, frames);
    devTotal_ += got;
    devPos_.publish(nowSec, devTotal_);
    if (got < frames) {
        std::memset(interleaved + got * cfg_.channels, 0, (frames - got) * cfg_.channels * sizeof(float));
        // Counted by the engine thread (which knows whether we are still settling).
        if (!stopped_.load(std::memory_order_relaxed)) deviceUnderruns_.fetch_add(1, std::memory_order_release);
        devicePrimed_.store(false, std::memory_order_release);
    }
}

}  // namespace nt
