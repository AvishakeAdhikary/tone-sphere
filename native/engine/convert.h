// Sample format conversion at the device boundary. Inside the engine everything is
// float32; a device may hand over int16, packed int24, int32 (24 valid bits or 32) or
// float32/64, interleaved. These convert one interleaved block, allocation-free.
#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace ts {

enum class SampleFormat : uint32_t {
    Float32 = 1,
    Float64 = 2,
    Int16 = 3,
    Int24 = 4,      // packed, 3 bytes
    Int32 = 5,      // 32-bit container; valid bits may be 24 (left-justified) or 32
};

inline uint32_t bytes_per_sample(SampleFormat f) {
    switch (f) {
        case SampleFormat::Float32: return 4;
        case SampleFormat::Float64: return 8;
        case SampleFormat::Int16: return 2;
        case SampleFormat::Int24: return 3;
        case SampleFormat::Int32: return 4;
    }
    return 0;
}

inline void to_float(SampleFormat f, const void* src, float* dst, size_t samples) noexcept {
    const auto* b = static_cast<const uint8_t*>(src);
    switch (f) {
        case SampleFormat::Float32:
            std::memcpy(dst, src, samples * 4);
            break;
        case SampleFormat::Float64:
            for (size_t i = 0; i < samples; ++i) {
                double d;
                std::memcpy(&d, b + i * 8, 8);
                dst[i] = static_cast<float>(d);
            }
            break;
        case SampleFormat::Int16:
            for (size_t i = 0; i < samples; ++i) {
                int16_t v;
                std::memcpy(&v, b + i * 2, 2);
                dst[i] = static_cast<float>(v) * (1.0f / 32768.0f);
            }
            break;
        case SampleFormat::Int24:
            for (size_t i = 0; i < samples; ++i) {
                const uint8_t* p = b + i * 3;
                const int32_t v = static_cast<int32_t>((static_cast<uint32_t>(p[0]) << 8) |
                                                       (static_cast<uint32_t>(p[1]) << 16) |
                                                       (static_cast<uint32_t>(p[2]) << 24)) >> 8;
                dst[i] = static_cast<float>(v) * (1.0f / 8388608.0f);
            }
            break;
        case SampleFormat::Int32:
            for (size_t i = 0; i < samples; ++i) {
                int32_t v;
                std::memcpy(&v, b + i * 4, 4);
                dst[i] = static_cast<float>(static_cast<double>(v) * (1.0 / 2147483648.0));
            }
            break;
    }
}

// Float to integer clips rather than wraps: an over-range sample converted by a plain cast
// flips sign and becomes a full-scale click. Both directions use the same 2^(N-1) scale,
// so a round trip is exact to within half a step and carries no gain error; the one price
// is that +1.0 itself clips to the largest positive code.
inline void from_float(SampleFormat f, const float* src, void* dst, size_t samples) noexcept {
    auto* b = static_cast<uint8_t*>(dst);
    auto clip = [](float x) { return x > 1.0f ? 1.0f : (x < -1.0f ? -1.0f : x); };
    switch (f) {
        case SampleFormat::Float32:
            std::memcpy(dst, src, samples * 4);
            break;
        case SampleFormat::Float64:
            for (size_t i = 0; i < samples; ++i) {
                const double d = src[i];
                std::memcpy(b + i * 8, &d, 8);
            }
            break;
        case SampleFormat::Int16:
            for (size_t i = 0; i < samples; ++i) {
                const auto v = static_cast<int16_t>(std::clamp<long>(std::lrint(clip(src[i]) * 32768.0f), -32768, 32767));
                std::memcpy(b + i * 2, &v, 2);
            }
            break;
        case SampleFormat::Int24:
            for (size_t i = 0; i < samples; ++i) {
                const auto v = static_cast<int32_t>(std::clamp<long>(std::lrint(clip(src[i]) * 8388608.0f), -8388608, 8388607));
                b[i * 3 + 0] = static_cast<uint8_t>(v);
                b[i * 3 + 1] = static_cast<uint8_t>(v >> 8);
                b[i * 3 + 2] = static_cast<uint8_t>(v >> 16);
            }
            break;
        case SampleFormat::Int32:
            for (size_t i = 0; i < samples; ++i) {
                const auto v = static_cast<int32_t>(std::clamp<long long>(
                    std::llrint(static_cast<double>(clip(src[i])) * 2147483648.0), INT32_MIN, INT32_MAX));
                std::memcpy(b + i * 4, &v, 4);
            }
            break;
    }
}

}  // namespace ts
