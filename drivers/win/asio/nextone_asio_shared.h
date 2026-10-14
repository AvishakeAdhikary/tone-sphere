// NexTone ASIO driver <-> NexTone engine shared-memory contract (user mode only).
//
// The engine creates a registry region  Local\NexToneAsioServer  (NTASIO_SERVER) with
// NTASIO_MAX_CLIENTS slots. Every DAW that opens the "NexTone ASIO" driver claims a free slot
// (state FREE -> CLAIMED via interlocked compare-exchange), creates its own audio region
// Local\NexToneAsioClient<pid>_<slot> (NTASIO_CLIENT header + two float rings) and finally
// marks the slot ACTIVE. The engine polls the slots and exposes each active client as
// "Application outputs" (what the DAW plays) and "Application inputs" (what the DAW records).
#pragma once
#include <stdint.h>

#define NTASIO_SERVER_NAME L"Local\\NexToneAsioServer"
#define NTASIO_CLIENT_NAME_FMT L"Local\\NexToneAsioClient%u_%u"
#define NTASIO_MAGIC 0x4F53414Eu  // 'NASO'
#define NTASIO_VERSION 1u
#define NTASIO_MAX_CLIENTS 16
#define NTASIO_MAX_CHANNELS 32
#define NTASIO_RING_FRAMES 16384u  // per direction, power of two

enum NtAsioSlotState : int32_t { NTASIO_FREE = 0, NTASIO_CLAIMED = 1, NTASIO_ACTIVE = 2 };

#pragma pack(push, 8)
typedef struct NTASIO_SLOT {
    volatile int32_t state;       // NtAsioSlotState
    uint32_t pid;
    volatile int64_t heartbeat;   // client: QPC ticks, updated every buffer
    wchar_t exe[64];              // client process file name
    wchar_t display[64];          // friendly app name
} NTASIO_SLOT;

typedef struct NTASIO_SERVER {
    uint32_t magic;
    uint32_t version;
    uint32_t enginePid;
    double engineSampleRate;      // engine's current rate (preferred for DAWs)
    int32_t preferredBufferSize;  // engine block size suggestion
    int32_t inputChannels;        // channels offered to DAWs (NexTone -> DAW)
    int32_t outputChannels;       // channels offered to DAWs (DAW -> NexTone)
    volatile int64_t engineHeartbeat;
    NTASIO_SLOT slots[NTASIO_MAX_CLIENTS];
} NTASIO_SERVER;

// Ring of interleaved float32 frames; positions are monotonic frame counters.
typedef struct NTASIO_RING {
    volatile int64_t writePos;
    volatile int64_t readPos;
    volatile int64_t overflows;
    volatile int64_t underflows;
} NTASIO_RING;

typedef struct NTASIO_CLIENT {
    uint32_t magic;
    uint32_t version;
    double sampleRate;            // client's running rate
    int32_t bufferSize;           // client's ASIO buffer size (frames)
    int32_t inputs;               // DAW input channels (fromEngine ring)
    int32_t outputs;              // DAW output channels (toEngine ring)
    volatile int32_t running;     // 1 between ASIO start() and stop()
    volatile int64_t generation;  // incremented when format/buffers change
    NTASIO_RING toEngine;         // DAW outputs -> NexTone
    NTASIO_RING fromEngine;       // NexTone -> DAW inputs
    // followed by: float toEngineData[NTASIO_RING_FRAMES * outputs]
    //              float fromEngineData[NTASIO_RING_FRAMES * inputs]
} NTASIO_CLIENT;
#pragma pack(pop)

static inline size_t ntasio_client_bytes(int32_t inputs, int32_t outputs) {
    return sizeof(NTASIO_CLIENT) + (size_t)NTASIO_RING_FRAMES * sizeof(float) * (size_t)(inputs + outputs);
}
static inline float* ntasio_to_engine(NTASIO_CLIENT* c) { return (float*)(c + 1); }
static inline float* ntasio_from_engine(NTASIO_CLIENT* c) {
    return (float*)(c + 1) + (size_t)NTASIO_RING_FRAMES * (size_t)c->outputs;
}
