// ToneSphere ASIO driver <-> ToneSphere engine shared-memory contract (user mode only).
//
// The engine creates a registry region  Local\ToneSphereAsioServer  (TSASIO_SERVER) with
// TSASIO_MAX_CLIENTS slots. Every DAW that opens the "ToneSphere ASIO" driver claims a free slot
// (state FREE -> CLAIMED via interlocked compare-exchange), creates its own audio region
// Local\ToneSphereAsioClient<pid>_<slot> (TSASIO_CLIENT header + two float rings) and finally
// marks the slot ACTIVE. The engine polls the slots and exposes each active client as
// "Application outputs" (what the DAW plays) and "Application inputs" (what the DAW records).
#pragma once
#include <stdint.h>

#define TSASIO_SERVER_NAME L"Local\\ToneSphereAsioServer"
#define TSASIO_CLIENT_NAME_FMT L"Local\\ToneSphereAsioClient%u_%u"
#define TSASIO_MAGIC 0x53415354u  // 'TSAS'
#define TSASIO_VERSION 1u
#define TSASIO_MAX_CLIENTS 16
#define TSASIO_MAX_CHANNELS 32
#define TSASIO_RING_FRAMES 16384u  // per direction, power of two

enum TsAsioSlotState : int32_t { TSASIO_FREE = 0, TSASIO_CLAIMED = 1, TSASIO_ACTIVE = 2 };

#pragma pack(push, 8)
typedef struct TSASIO_SLOT {
    volatile int32_t state;       // TsAsioSlotState
    uint32_t pid;
    volatile int64_t heartbeat;   // client: QPC ticks, updated every buffer
    wchar_t exe[64];              // client process file name
    wchar_t display[64];          // friendly app name
} TSASIO_SLOT;

typedef struct TSASIO_SERVER {
    uint32_t magic;
    uint32_t version;
    uint32_t enginePid;
    double engineSampleRate;      // engine's current rate (preferred for DAWs)
    int32_t preferredBufferSize;  // engine block size suggestion
    int32_t inputChannels;        // channels offered to DAWs (ToneSphere -> DAW)
    int32_t outputChannels;       // channels offered to DAWs (DAW -> ToneSphere)
    volatile int64_t engineHeartbeat;
    TSASIO_SLOT slots[TSASIO_MAX_CLIENTS];
} TSASIO_SERVER;

// Ring of interleaved float32 frames; positions are monotonic frame counters.
typedef struct TSASIO_RING {
    volatile int64_t writePos;
    volatile int64_t readPos;
    volatile int64_t overflows;
    volatile int64_t underflows;
} TSASIO_RING;

typedef struct TSASIO_CLIENT {
    uint32_t magic;
    uint32_t version;
    double sampleRate;            // client's running rate
    int32_t bufferSize;           // client's ASIO buffer size (frames)
    int32_t inputs;               // DAW input channels (fromEngine ring)
    int32_t outputs;              // DAW output channels (toEngine ring)
    volatile int32_t running;     // 1 between ASIO start() and stop()
    volatile int64_t generation;  // incremented when format/buffers change
    TSASIO_RING toEngine;         // DAW outputs -> ToneSphere
    TSASIO_RING fromEngine;       // ToneSphere -> DAW inputs
    // followed by: float toEngineData[TSASIO_RING_FRAMES * outputs]
    //              float fromEngineData[TSASIO_RING_FRAMES * inputs]
} TSASIO_CLIENT;
#pragma pack(pop)

static inline size_t ntasio_client_bytes(int32_t inputs, int32_t outputs) {
    return sizeof(TSASIO_CLIENT) + (size_t)TSASIO_RING_FRAMES * sizeof(float) * (size_t)(inputs + outputs);
}
static inline float* ntasio_to_engine(TSASIO_CLIENT* c) { return (float*)(c + 1); }
static inline float* ntasio_from_engine(TSASIO_CLIENT* c) {
    return (float*)(c + 1) + (size_t)TSASIO_RING_FRAMES * (size_t)c->outputs;
}
