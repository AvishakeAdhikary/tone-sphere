#pragma once
// Lock-free single-producer / single-consumer ring of interleaved float frames.
// Safe for one writer thread and one reader thread; never allocates after init().
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ts {

class FrameRing {
public:
    FrameRing() = default;
    FrameRing(const FrameRing&) = delete;
    FrameRing& operator=(const FrameRing&) = delete;

    /// Allocates storage for at least `minFrames` frames (rounded up to a power of two).
    void init(size_t channels, size_t minFrames) {
        size_t cap = 1;
        while (cap < minFrames) cap <<= 1;
        channels_ = channels;
        capacity_ = cap;
        mask_ = cap - 1;
        data_.assign(cap * channels, 0.0f);
        write_.store(0, std::memory_order_relaxed);
        read_.store(0, std::memory_order_relaxed);
    }

    size_t channels() const { return channels_; }
    size_t capacity() const { return capacity_; }

    /// Frames ready to read (reader side).
    size_t available() const {
        return static_cast<size_t>(write_.load(std::memory_order_acquire) - read_.load(std::memory_order_relaxed));
    }
    /// Free frames (writer side).
    size_t space() const {
        return capacity_ - static_cast<size_t>(write_.load(std::memory_order_relaxed) - read_.load(std::memory_order_acquire));
    }
    /// Approximate fill as seen from any thread (for statistics).
    size_t fillApprox() const {
        return static_cast<size_t>(write_.load(std::memory_order_acquire) - read_.load(std::memory_order_acquire));
    }

    /// Writes up to `frames` interleaved frames; returns frames written.
    size_t write(const float* src, size_t frames) {
        const uint64_t w = write_.load(std::memory_order_relaxed);
        const size_t free = space();  // read once: the reader may free more concurrently
        const size_t n = frames < free ? frames : free;
        copyIn(w, src, n);
        write_.store(w + n, std::memory_order_release);
        return n;
    }
    /// Writes `frames` frames of silence; returns frames written.
    size_t writeSilence(size_t frames) {
        const uint64_t w = write_.load(std::memory_order_relaxed);
        const size_t free = space();
        const size_t n = frames < free ? frames : free;
        for (size_t i = 0; i < n; ++i) std::memset(&data_[((w + i) & mask_) * channels_], 0, channels_ * sizeof(float));
        write_.store(w + n, std::memory_order_release);
        return n;
    }
    /// Reads up to `frames` interleaved frames; returns frames read.
    size_t read(float* dst, size_t frames) {
        const uint64_t r = read_.load(std::memory_order_relaxed);
        const size_t avail = available();
        const size_t n = frames < avail ? frames : avail;
        copyOut(r, dst, n);
        read_.store(r + n, std::memory_order_release);
        return n;
    }
    /// Discards up to `frames` frames from the read side.
    size_t skip(size_t frames) {
        const uint64_t r = read_.load(std::memory_order_relaxed);
        const size_t avail = available();
        const size_t n = frames < avail ? frames : avail;
        read_.store(r + n, std::memory_order_release);
        return n;
    }

private:
    void copyIn(uint64_t pos, const float* src, size_t n) {
        const size_t start = static_cast<size_t>(pos & mask_);
        const size_t first = n < capacity_ - start ? n : capacity_ - start;
        std::memcpy(&data_[start * channels_], src, first * channels_ * sizeof(float));
        if (n > first) std::memcpy(&data_[0], src + first * channels_, (n - first) * channels_ * sizeof(float));
    }
    void copyOut(uint64_t pos, float* dst, size_t n) const {
        const size_t start = static_cast<size_t>(pos & mask_);
        const size_t first = n < capacity_ - start ? n : capacity_ - start;
        std::memcpy(dst, &data_[start * channels_], first * channels_ * sizeof(float));
        if (n > first) std::memcpy(dst + first * channels_, &data_[0], (n - first) * channels_ * sizeof(float));
    }

    std::vector<float> data_;
    size_t channels_ = 0;
    size_t capacity_ = 0;
    size_t mask_ = 0;
    alignas(64) std::atomic<uint64_t> write_{0};
    alignas(64) std::atomic<uint64_t> read_{0};
};

}  // namespace ts
