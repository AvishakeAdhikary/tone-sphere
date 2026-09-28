#pragma once

#include <atomic>
#include <cstdint>

namespace ts {

extern thread_local bool t_on_audio_thread;
extern std::atomic<uint64_t> g_rt_allocations;

// Marks the current thread as the audio thread for the scope's lifetime, so any heap
// allocation this DLL makes inside it is counted.
struct AudioThreadScope {
    AudioThreadScope() { t_on_audio_thread = true; }
    ~AudioThreadScope() { t_on_audio_thread = false; }
    AudioThreadScope(const AudioThreadScope&) = delete;
    AudioThreadScope& operator=(const AudioThreadScope&) = delete;
};

}  // namespace ts
