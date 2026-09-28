// Single-producer single-consumer rings.
//
// Guarantee, stated exactly: with one producer thread and one consumer thread, both ends
// are wait-free — every call completes in a bounded number of steps and never blocks or
// retries. With more than one thread on either end the structure is simply wrong; callers
// that can have several producers (the Python side) serialise them before calling in.
//
// Indices are 64-bit and only ever increase, so full and empty are distinguished without
// a sentinel slot, and they cannot wrap in any realistic lifetime (2^64 frames at 384 kHz
// is 1.5 million years).
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>

namespace ts {

inline uint32_t next_pow2(uint32_t v) {
    if (v <= 1) return 1;
    --v;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

// Interleaved float32 frames.
class FrameRing {
public:
    FrameRing(uint32_t capacity_frames, uint32_t channels)
        : capacity_(next_pow2(capacity_frames)), mask_(capacity_ - 1), channels_(channels),
          data_(new float[static_cast<size_t>(capacity_) * channels]()) {}

    uint32_t capacity() const { return capacity_; }
    uint32_t channels() const { return channels_; }

    uint32_t available() const {
        return static_cast<uint32_t>(write_.load(std::memory_order_acquire) - read_.load(std::memory_order_acquire));
    }

    // Producer. Writes as many frames as fit; the rest are dropped (the newest audio) and
    // the caller counts them — dropping the oldest instead would need the producer to move
    // the consumer's index, which is exactly the cross-thread write SPSC exists to avoid.
    uint32_t write(const float* src, uint32_t frames) {
        const uint64_t w = write_.load(std::memory_order_relaxed);
        const uint64_t r = read_.load(std::memory_order_acquire);
        const uint32_t free_frames = capacity_ - static_cast<uint32_t>(w - r);
        const uint32_t n = frames < free_frames ? frames : free_frames;
        copy_in(static_cast<uint32_t>(w & mask_), src, n);
        write_.store(w + n, std::memory_order_release);
        return n;
    }

    // Consumer. Reads up to `frames`; returns how many were available.
    uint32_t read(float* dst, uint32_t frames) {
        const uint64_t r = read_.load(std::memory_order_relaxed);
        const uint64_t w = write_.load(std::memory_order_acquire);
        const uint32_t avail = static_cast<uint32_t>(w - r);
        const uint32_t n = frames < avail ? frames : avail;
        copy_out(static_cast<uint32_t>(r & mask_), dst, n);
        read_.store(r + n, std::memory_order_release);
        return n;
    }

private:
    void copy_in(uint32_t at, const float* src, uint32_t n) {
        const uint32_t first = n < capacity_ - at ? n : capacity_ - at;
        std::memcpy(data_.get() + static_cast<size_t>(at) * channels_, src, sizeof(float) * first * channels_);
        if (n > first)
            std::memcpy(data_.get(), src + static_cast<size_t>(first) * channels_, sizeof(float) * (n - first) * channels_);
    }
    void copy_out(uint32_t at, float* dst, uint32_t n) const {
        const uint32_t first = n < capacity_ - at ? n : capacity_ - at;
        std::memcpy(dst, data_.get() + static_cast<size_t>(at) * channels_, sizeof(float) * first * channels_);
        if (n > first)
            std::memcpy(dst + static_cast<size_t>(first) * channels_, data_.get(), sizeof(float) * (n - first) * channels_);
    }

    const uint32_t capacity_;
    const uint32_t mask_;
    const uint32_t channels_;
    std::unique_ptr<float[]> data_;
    // Separate cache lines: the producer and consumer each write one index, and sharing a
    // line would make every write on one side invalidate the other side's cache.
    alignas(64) std::atomic<uint64_t> write_{0};
    alignas(64) std::atomic<uint64_t> read_{0};
};

// Fixed-capacity queue of trivially copyable items, same guarantee as FrameRing.
template <typename T, uint32_t Capacity>
class ItemQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "capacity must be a power of two");

public:
    bool push(const T& item) {
        const uint64_t w = write_.load(std::memory_order_relaxed);
        if (w - read_.load(std::memory_order_acquire) == Capacity) return false;
        items_[w & (Capacity - 1)] = item;
        write_.store(w + 1, std::memory_order_release);
        return true;
    }
    bool pop(T& item) {
        const uint64_t r = read_.load(std::memory_order_relaxed);
        if (r == write_.load(std::memory_order_acquire)) return false;
        item = items_[r & (Capacity - 1)];
        read_.store(r + 1, std::memory_order_release);
        return true;
    }

private:
    T items_[Capacity];
    alignas(64) std::atomic<uint64_t> write_{0};
    alignas(64) std::atomic<uint64_t> read_{0};
};

}  // namespace ts
