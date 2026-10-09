#include "core/bridge.h"

#include <algorithm>
#include <cstring>

namespace nt {

namespace {
constexpr int kHalfTaps = 16;

double jitterMs(const BridgeConfig& c) { return c.jittery ? 12.0 : 2.5; }
}  // namespace

// ============================== CaptureBridge ==============================

void CaptureBridge::init(const BridgeConfig& cfg) {
    cfg_ = cfg;
    nominal_ = cfg.deviceRate / cfg.engineRate;
    if (cfg.master) {
        target_ = 2.0 * kHalfTaps;
    } else {
        target_ = static_cast<double>(cfg.devicePeriod) + static_cast<double>(cfg.engineMaxBlock) * nominal_ +
                  (jitterMs(cfg) + cfg.extraSafetyMs) * cfg.deviceRate / 1000.0;
    }
    const size_t cap = static_cast<size_t>(target_ * 4) + cfg.devicePeriod * 4 + cfg.engineMaxBlock * 4 + 1024;
    ring_.init(cfg.channels, cap);
    const size_t maxIn = static_cast<size_t>(static_cast<double>(cfg.engineMaxBlock) * nominal_ * 1.02) + 4 * kHalfTaps + 64;
    rs_.init(cfg.channels, maxIn, 0.95 * std::min(1.0, 1.0 / nominal_), kHalfTaps);
    ctl_.configure(cfg.deviceRate, target_);
    scratch_.assign(maxIn * cfg.channels, 0.0f);
    primed_ = false;
    devTotal_ = 0;
    engTotal_ = 0;
    stats_.latencyMs.store(static_cast<float>((target_ + kHalfTaps) / cfg.deviceRate * 1000.0));
}

void CaptureBridge::deviceWrite(const float* interleaved, size_t frames, double nowSec) {
    const size_t w = ring_.write(interleaved, frames);
    if (w < frames) stats_.overruns.fetch_add(1, std::memory_order_relaxed);
    devTotal_ += w;
    devPos_.publish(nowSec, devTotal_);
}

void CaptureBridge::deviceWriteSilence(size_t frames, double nowSec) {
    devTotal_ += ring_.writeSilence(frames);
    devPos_.publish(nowSec, devTotal_);
}

void CaptureBridge::engineRead(float* const* out, size_t frames, double nowSec) {
    auto silence = [&] {
        for (size_t c = 0; c < cfg_.channels; ++c) std::memset(out[c], 0, frames * sizeof(float));
    };
    if (restart_.exchange(false, std::memory_order_acq_rel)) {
        rs_.reset();
        ctl_.reset();
        primed_ = false;
    }
    if (stopped_.load(std::memory_order_acquire)) {
        stats_.active.store(false, std::memory_order_relaxed);
        silence();
        return;
    }
    stats_.active.store(true, std::memory_order_relaxed);

    size_t avail = ring_.available();
    if (!primed_) {
        if (static_cast<double>(avail) < target_) {
            silence();
            return;
        }
        primed_ = true;
        // Start with exactly the target cushion.
        const size_t excess = avail - static_cast<size_t>(target_);
        engTotal_ += ring_.skip(excess);
        avail = ring_.available();
        ctl_.reset();
    }
    // Far too much data (device burst / stall recovery): drop back to the target.
    if (static_cast<double>(avail) > target_ * 3.0 + static_cast<double>(cfg_.devicePeriod) * 2.0) {
        engTotal_ += ring_.skip(avail - static_cast<size_t>(target_));
        avail = ring_.available();
        ctl_.reset();
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
        stats_.underruns.fetch_add(1, std::memory_order_relaxed);
        primed_ = false;  // rebuild the cushion before continuing
        return;
    }

    // Continuous ("virtual") fill: frames the device has produced by now, minus frames consumed.
    double fill = static_cast<double>(ring_.available()) + static_cast<double>(rs_.buffered());
    double tW = 0;
    uint64_t totalW = 0;
    if (devPos_.read(tW, totalW)) {
        const double maxElapsed = 2.0 * static_cast<double>(cfg_.devicePeriod) / cfg_.deviceRate;
        const double elapsed = std::clamp(nowSec - tW, 0.0, maxElapsed);
        fill = static_cast<double>(static_cast<int64_t>(totalW - engTotal_)) + elapsed * cfg_.deviceRate +
               static_cast<double>(rs_.buffered());
    }
    if (!cfg_.master) ctl_.update(fill, static_cast<double>(frames) / cfg_.engineRate);
    stats_.ppm.store(static_cast<float>(ctl_.ppm()), std::memory_order_relaxed);
    stats_.fillMs.store(static_cast<float>(fill / cfg_.deviceRate * 1000.0), std::memory_order_relaxed);
}

// ============================== RenderBridge ===============================

void RenderBridge::init(const BridgeConfig& cfg) {
    cfg_ = cfg;
    nominal_ = cfg.engineRate / cfg.deviceRate;  // engine frames per device frame
    if (cfg.master) {
        target_ = static_cast<double>(cfg.devicePeriod);
    } else {
        target_ = static_cast<double>(cfg.devicePeriod) + static_cast<double>(cfg.engineMaxBlock) / nominal_ +
                  (jitterMs(cfg) + cfg.extraSafetyMs) * cfg.deviceRate / 1000.0;
    }
    const size_t cap = static_cast<size_t>(target_ * 4) + cfg.devicePeriod * 4 + cfg.engineMaxBlock * 4 + 1024;
    ring_.init(cfg.channels, cap);
    rs_.init(cfg.channels, cfg.engineMaxBlock + 4 * kHalfTaps + 64, 0.95 * std::min(1.0, 1.0 / nominal_), kHalfTaps);
    ctl_.configure(cfg.deviceRate, target_);
    const size_t maxOut = static_cast<size_t>(static_cast<double>(cfg.engineMaxBlock) / nominal_ * 1.02) + 4 * kHalfTaps + 64;
    scratch_.assign(maxOut * cfg.channels, 0.0f);
    devicePrimed_.store(false);
    devTotal_ = 0;
    engTotal_ = 0;
    stats_.latencyMs.store(static_cast<float>((target_ + kHalfTaps) / cfg.deviceRate * 1000.0));
}

void RenderBridge::engineWrite(const float* const* in, size_t frames, double nowSec) {
    if (restart_.exchange(false, std::memory_order_acq_rel)) {
        rs_.reset();
        ctl_.reset();
        devicePrimed_.store(false, std::memory_order_release);
        if (cfg_.master) engTotal_ += ring_.writeSilence(2 * kHalfTaps);
    }
    if (stopped_.load(std::memory_order_acquire)) {
        stats_.active.store(false, std::memory_order_relaxed);
        return;
    }
    stats_.active.store(true, std::memory_order_relaxed);

    const double step = nominal_ * (1.0 + (cfg_.master ? 0.0 : ctl_.correction()));
    rs_.push(in, frames);
    const size_t maxOut = scratch_.size() / cfg_.channels;
    const size_t produced = rs_.pullInterleaved(scratch_.data(), maxOut, step);

    const double fillNow = static_cast<double>(ring_.fillApprox());
    if (!cfg_.master && fillNow > target_ * 3.0 + static_cast<double>(cfg_.devicePeriod) * 2.0) {
        // Device is not consuming (stalled or much slower): drop this block and re-lock.
        stats_.overruns.fetch_add(1, std::memory_order_relaxed);
        ctl_.reset();
    } else if (produced > 0) {
        const size_t w = ring_.write(scratch_.data(), produced);
        engTotal_ += w;
        if (w < produced) stats_.overruns.fetch_add(1, std::memory_order_relaxed);
    }

    double fill = static_cast<double>(ring_.fillApprox());
    double tR = 0;
    uint64_t totalR = 0;
    if (devicePrimed_.load(std::memory_order_acquire) && devPos_.read(tR, totalR)) {
        const double maxElapsed = 2.0 * static_cast<double>(cfg_.devicePeriod) / cfg_.deviceRate;
        const double elapsed = std::clamp(nowSec - tR, 0.0, maxElapsed);
        fill = static_cast<double>(static_cast<int64_t>(engTotal_ - totalR)) - elapsed * cfg_.deviceRate;
    }
    fill += static_cast<double>(rs_.buffered()) / nominal_;
    if (!cfg_.master && devicePrimed_.load(std::memory_order_relaxed)) {
        ctl_.update(fill, static_cast<double>(frames) / cfg_.engineRate);
    }
    stats_.ppm.store(static_cast<float>(ctl_.ppm()), std::memory_order_relaxed);
    stats_.fillMs.store(static_cast<float>(fill / cfg_.deviceRate * 1000.0), std::memory_order_relaxed);
}

void RenderBridge::deviceRead(float* interleaved, size_t frames, double nowSec) {
    if (!devicePrimed_.load(std::memory_order_acquire)) {
        if (static_cast<double>(ring_.available()) >= target_) {
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
        if (!stopped_.load(std::memory_order_relaxed)) stats_.underruns.fetch_add(1, std::memory_order_relaxed);
        devicePrimed_.store(false, std::memory_order_release);
    }
}

}  // namespace nt
