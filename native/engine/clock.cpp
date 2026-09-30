// A clock with no device: runs the engine's blocks at real-time pace when nothing routed
// touches hardware — a bus feeding a network stream, a per-process capture feeding a bus.
// Without it such a plan would never run, because only a device callback drives blocks.
//
// A high-resolution waitable timer paced against an absolute schedule (QueryPerformance-
// Counter), so lateness in one period is made up in the next rather than accumulating.
#include "clock.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>

#include <avrt.h>

#include "engine.h"

namespace ts {
namespace {

class ClockBackend final : public DeviceBackend {
public:
    ClockBackend(Engine& engine, uint32_t block_frames)
        : engine_(engine), frames_per_block_(block_frames ? std::min(block_frames, engine.max_block()) : engine.max_block()) {
        timer_ = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        engine_.set_backend_running(true);
        thread_ = std::thread([this] { run(); });
    }
    ~ClockBackend() override {
        stop();
        if (timer_) CloseHandle(timer_);
    }

    void stop() override {
        if (stopping_.exchange(true)) return;
        if (thread_.joinable()) thread_.join();
        engine_.set_backend_running(false);
    }

    int32_t status(ts_stream_status* out, int32_t capacity) override {
        if (capacity < 1) return 0;
        std::memset(out, 0, sizeof(*out));
        out->kind = TS_STREAM_CLOCK;
        out->state = stopping_.load() ? TS_STREAM_STATE_STOPPED : TS_STREAM_STATE_RUNNING;
        out->is_master = 1;
        out->sample_rate = engine_.sample_rate();
        out->buffer_frames = frames_per_block_;
        out->period_frames = frames_per_block_;
        out->frames = frames_.load();
        out->glitches = late_.load();
        out->drift_ratio = 1.0;
        std::snprintf(out->error, sizeof(out->error), "no device routed: blocks paced by a timer");
        return 1;
    }

private:
    void run() {
        DWORD task = 0;
        HANDLE mmcss = AvSetMmThreadCharacteristicsW(L"Pro Audio", &task);
        LARGE_INTEGER frequency, now;
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&now);
        const uint32_t frames = frames_per_block_;
        const double period_ticks = static_cast<double>(frames) * frequency.QuadPart / engine_.sample_rate();
        double deadline = static_cast<double>(now.QuadPart) + period_ticks;
        while (!stopping_.load(std::memory_order_acquire)) {
            QueryPerformanceCounter(&now);
            const double remaining = deadline - static_cast<double>(now.QuadPart);
            if (remaining > 0) {
                LARGE_INTEGER due;
                due.QuadPart = -static_cast<LONGLONG>(remaining * 10'000'000.0 / frequency.QuadPart);
                if (timer_ && SetWaitableTimer(timer_, &due, 0, nullptr, nullptr, FALSE))
                    WaitForSingleObject(timer_, 100);
                else
                    Sleep(1);
            } else if (-remaining > 2 * period_ticks) {
                // More than two periods behind (the machine slept, the thread was starved):
                // count it, and restart the schedule rather than racing to catch up.
                late_.fetch_add(1);
                deadline = static_cast<double>(now.QuadPart);
            }
            engine_.run_block(nullptr, 0, nullptr, 0, frames);
            frames_.fetch_add(frames);
            deadline += period_ticks;
        }
        if (mmcss) AvRevertMmThreadCharacteristics(mmcss);
    }

    Engine& engine_;
    const uint32_t frames_per_block_;
    HANDLE timer_ = nullptr;
    std::thread thread_;
    std::atomic<bool> stopping_{false};
    std::atomic<uint64_t> frames_{0};
    std::atomic<uint64_t> late_{0};
};

}  // namespace

std::unique_ptr<DeviceBackend> start_clock(Engine& engine, uint32_t block_frames) {
    return std::make_unique<ClockBackend>(engine, block_frames);
}

}  // namespace ts
