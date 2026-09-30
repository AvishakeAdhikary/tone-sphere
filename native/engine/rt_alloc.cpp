// Global allocation functions for this DLL, counting any allocation made on the audio
// thread. A real-time rule nobody measures is a rule nobody keeps: this makes "the block
// allocates nothing" a number a test can assert on (ts_stats.rt_allocations).
//
// With the static CRT these replace operator new/delete for this DLL's own code only.
// Allocations inside a third-party VST3 plugin happen in the plugin's module and are not
// seen here — that is a limit of the measurement, stated rather than hidden.
#include <atomic>
#include <cstdlib>
#include <new>

#include "rt_alloc.h"

namespace ts {
thread_local bool t_on_audio_thread = false;
std::atomic<uint64_t> g_rt_allocations{0};
}  // namespace ts

namespace {

void* counted_alloc(std::size_t size) {
    if (ts::t_on_audio_thread) ts::g_rt_allocations.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size ? size : 1);
}

void* counted_aligned_alloc(std::size_t size, std::align_val_t alignment) {
    if (ts::t_on_audio_thread) ts::g_rt_allocations.fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
    return _aligned_malloc(size ? size : 1, static_cast<std::size_t>(alignment));
#else
    void* p = nullptr;
    return posix_memalign(&p, static_cast<std::size_t>(alignment), size ? size : 1) == 0 ? p : nullptr;
#endif
}

void aligned_free(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}

}  // namespace

void* operator new(std::size_t size) {
    if (void* p = counted_alloc(size)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) {
    if (void* p = counted_alloc(size)) return p;
    throw std::bad_alloc();
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept { return counted_alloc(size); }
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept { return counted_alloc(size); }
void* operator new(std::size_t size, std::align_val_t a) {
    if (void* p = counted_aligned_alloc(size, a)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size, std::align_val_t a) {
    if (void* p = counted_aligned_alloc(size, a)) return p;
    throw std::bad_alloc();
}
void* operator new(std::size_t size, std::align_val_t a, const std::nothrow_t&) noexcept {
    return counted_aligned_alloc(size, a);
}
void* operator new[](std::size_t size, std::align_val_t a, const std::nothrow_t&) noexcept {
    return counted_aligned_alloc(size, a);
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { aligned_free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { aligned_free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { aligned_free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { aligned_free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { aligned_free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { aligned_free(p); }
