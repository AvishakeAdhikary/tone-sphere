#pragma once
// Real-time worker pool for the audio graph.
//
// The clock thread hands one level of independent graph nodes to the pool and works on it too
// (parallelFor). Workers are real-time threads that never lock, allocate or do I/O while
// running: they claim items with a lock-free ticket, spin briefly between levels and park on an
// atomic between blocks. Nothing here blocks the caller beyond waiting for the level's items.
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace ts {

class DspPool {
public:
    /// Starts `workers` helper threads (0 = everything runs on the caller).
    explicit DspPool(int workers);
    ~DspPool();
    DspPool(const DspPool&) = delete;
    DspPool& operator=(const DspPool&) = delete;

    /// Helper threads available (the caller is one more).
    int workers() const { return static_cast<int>(threads_.size()); }
    /// The helpers this machine should get: one core stays free for everything else; at most 7.
    static int defaultWorkers();

    using Fn = void (*)(void* ctx, int index);
    /// Runs fn(ctx, i) for every i in [0, count) on the caller plus up to `helpers` workers and
    /// returns when all are done. Audio thread only, never re-entrant. count must be < 65536.
    void parallelFor(int count, Fn fn, void* ctx, int helpers);

private:
    void workerLoop(int index, uint32_t startEpoch);
    // Claims and runs items of job `epoch` until none are left.
    void work(uint32_t epoch);

    std::vector<std::thread> threads_;
    std::atomic<bool> stop_{false};
    // Published job: wakes workers (they wait on it).
    alignas(64) std::atomic<uint32_t> epoch_{0};
    // Ticket = epoch << 32 | count << 16 | next index. A claim is a CAS that only succeeds for the
    // current job, so a worker that wakes late can never run an item of the next one.
    alignas(64) std::atomic<uint64_t> ticket_{0};
    alignas(64) std::atomic<int> remaining_{0};
    alignas(64) std::atomic<int> parked_{0};
    std::atomic<int> helpers_{0};
    // Read only after a successful claim, written only once every item of the last job is done.
    Fn fn_ = nullptr;
    void* ctx_ = nullptr;
    uint32_t nextEpoch_ = 0;  // caller only
};

/// Flush denormals to zero on this thread: decaying filter/compressor tails would otherwise
/// cost up to 100x per sample on x86. Call on every audio thread.
void enableFlushToZero();

}  // namespace ts
