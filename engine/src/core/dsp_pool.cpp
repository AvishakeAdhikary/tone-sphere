#include "core/dsp_pool.h"

#include <algorithm>
#include <string>

#include "util/platform.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sched.h>
#endif

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define TS_X86 1
#endif

namespace ts {

namespace {
inline void cpuRelax() {
#ifdef TS_X86
    _mm_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
#if defined(_MSC_VER)
    __yield();
#else
    asm volatile("yield");
#endif
#endif
}
// How long a worker keeps spinning for the next level before it parks. Levels of one block
// follow each other within microseconds; blocks are milliseconds apart.
constexpr double kSpinSeconds = 60e-6;
// How long the caller spins for the level's last items before it starts yielding its core.
constexpr double kWaitSpinSeconds = 200e-6;
// Lets any ready thread on this core run, lower-priority ones included (unlike Sleep(0)), so a
// worker that was pushed off can finish its item. No lock and no wait: returns at once when
// nothing else is ready.
inline void yieldCore() {
#ifdef _WIN32
    SwitchToThread();
#else
    sched_yield();
#endif
}
}  // namespace

void enableFlushToZero() {
#ifdef TS_X86
    _mm_setcsr(_mm_getcsr() | 0x8040);  // FTZ (bit 15) | DAZ (bit 6)
#elif defined(__aarch64__) && !defined(_MSC_VER)
    uint64_t fpcr;
    asm volatile("mrs %0, fpcr" : "=r"(fpcr));
    asm volatile("msr fpcr, %0" : : "r"(fpcr | (1ull << 24)));
#endif
}

int DspPool::defaultWorkers() {
    // The cores this process may actually use (an affinity mask, a container or a CI runner can
    // allow fewer than the machine has): more spinning threads than cores only adds stalls.
    unsigned hw = std::thread::hardware_concurrency();
#ifdef _WIN32
    DWORD_PTR proc = 0, sys = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys) && proc) {
        unsigned n = 0;
        for (; proc; proc &= proc - 1) ++n;
        hw = hw ? std::min(hw, n) : n;
    }
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const unsigned n = static_cast<unsigned>(CPU_COUNT(&set));
        if (n) hw = hw ? std::min(hw, n) : n;
    }
#endif
    return hw <= 1 ? 0 : static_cast<int>(std::min(hw - 1, 7u));
}

DspPool::DspPool(int workers) {
    threads_.reserve(static_cast<size_t>(std::max(0, workers)));
    // Workers start from the epoch at construction, not whatever it is when their thread first
    // runs: on a busy core a thread can start after a job, or even the shutdown, was published,
    // and would then wait for a change that never comes.
    const uint32_t start = epoch_.load(std::memory_order_acquire);
    for (int i = 0; i < workers; ++i) threads_.emplace_back([this, i, start] { workerLoop(i, start); });
}

DspPool::~DspPool() {
    stop_.store(true, std::memory_order_seq_cst);
    epoch_.fetch_add(1, std::memory_order_seq_cst);
    epoch_.notify_all();
    for (auto& t : threads_) t.join();
}

void DspPool::work(uint32_t epoch) {
    uint64_t t = ticket_.load(std::memory_order_acquire);
    for (;;) {
        if (static_cast<uint32_t>(t >> 32) != epoch) return;
        const uint32_t count = static_cast<uint32_t>(t >> 16) & 0xFFFF;
        const uint32_t index = static_cast<uint32_t>(t) & 0xFFFF;
        if (index >= count) return;
        if (!ticket_.compare_exchange_weak(t, t + 1, std::memory_order_acq_rel, std::memory_order_acquire)) continue;
        // Claimed: the job can't change until this item is done, so fn_/ctx_ are this job's.
        fn_(ctx_, static_cast<int>(index));
        remaining_.fetch_sub(1, std::memory_order_acq_rel);
        t = ticket_.load(std::memory_order_acquire);
    }
}

void DspPool::workerLoop(int index, uint32_t startEpoch) {
    const std::string name = "ts-dsp-" + std::to_string(index + 1);
    setAudioThreadPriority(name.c_str());
    enableFlushToZero();
    uint32_t seen = startEpoch;
    for (;;) {
        uint32_t e = epoch_.load(std::memory_order_acquire);
        if (e == seen) {
            const double spinUntil = nowSeconds() + kSpinSeconds;
            for (int k = 0; (e = epoch_.load(std::memory_order_acquire)) == seen; ++k) {
                if ((k & 63) == 63 && nowSeconds() > spinUntil) {
                    // Park. The caller publishes the epoch before reading parked_, and we count
                    // ourselves before re-checking the epoch (both seq_cst): no wake-up is lost.
                    parked_.fetch_add(1, std::memory_order_seq_cst);
                    epoch_.wait(seen, std::memory_order_seq_cst);
                    parked_.fetch_sub(1, std::memory_order_seq_cst);
                    k = 0;
                    continue;
                }
                cpuRelax();
            }
        }
        seen = e;
        if (stop_.load(std::memory_order_acquire)) return;
        if (index < helpers_.load(std::memory_order_relaxed)) work(e);
    }
}

void DspPool::parallelFor(int count, Fn fn, void* ctx, int helpers) {
    if (count <= 0) return;
    helpers = std::clamp(helpers, 0, workers());
    if (count == 1 || helpers == 0) {
        for (int i = 0; i < count; ++i) fn(ctx, i);
        return;
    }
    fn_ = fn;
    ctx_ = ctx;
    helpers_.store(helpers, std::memory_order_relaxed);
    remaining_.store(count, std::memory_order_relaxed);
    const uint32_t e = ++nextEpoch_;
    ticket_.store(static_cast<uint64_t>(e) << 32 | static_cast<uint64_t>(count & 0xFFFF) << 16, std::memory_order_release);
    epoch_.store(e, std::memory_order_seq_cst);
    if (parked_.load(std::memory_order_seq_cst) > 0) epoch_.notify_all();
    work(e);
    // Our share is done; the last items may still be running on workers. Spin while they finish
    // (microseconds), but past kWaitSpinSeconds give the core away: with more threads than free
    // cores (a busy or small machine), the worker holding the last item may be waiting for this
    // very core, demoted by MMCSS or preempted, and a pure spin then never ends.
    if (remaining_.load(std::memory_order_acquire) == 0) return;
    const double spinUntil = nowSeconds() + kWaitSpinSeconds;
    for (int k = 0; remaining_.load(std::memory_order_acquire) != 0; ++k) {
        if ((k & 63) == 63 && nowSeconds() > spinUntil) {
            yieldCore();
            continue;
        }
        cpuRelax();
    }
}

}  // namespace ts
