// NexTone Virtual Audio Driver <-> NexTone engine shared-memory contract.
// Included by the kernel driver (MS-PL component) and by the engine (GPL component);
// it only describes a data layout and carries no implementation.
//
// Each virtual endpoint owns one named section:
//   kernel: \BaseNamedObjects\NexToneVadOut<N> / \BaseNamedObjects\NexToneVadMic<N>
//   user:   Global\NexToneVadOut<N>            / Global\NexToneVadMic<N>          (N = 1..)
// Layout: NTVAD_RING_HEADER (padded to NTVAD_HEADER_BYTES) followed by a byte ring of
// `dataBytes` bytes (power of two) in the Windows stream's native sample format.
#pragma once

#define NTVAD_MAGIC 0x5256544Eu  // 'NTVR'
#define NTVAD_VERSION 1u
#define NTVAD_MAX_OUTPUTS 8
#define NTVAD_MAX_MICS 8
#define NTVAD_HEADER_BYTES 4096u
#define NTVAD_DATA_BYTES (1024u * 1024u)
#define NTVAD_SECTION_BYTES (NTVAD_HEADER_BYTES + NTVAD_DATA_BYTES)

// Direction of the Windows endpoint.
#define NTVAD_DIR_RENDER 0u   // "NexTone Output N": apps play into it; driver produces, engine consumes
#define NTVAD_DIR_CAPTURE 1u  // "NexTone Mic N": apps record from it; engine produces, driver consumes

#pragma pack(push, 8)
typedef struct _NTVAD_RING_HEADER {
    unsigned int magic;
    unsigned int version;
    unsigned int direction;
    unsigned int index;      // 1-based endpoint number
    unsigned int dataBytes;  // ring size in bytes (power of two)

    // Format of the running Windows stream (written by the driver when a stream opens).
    unsigned int sampleRate;
    unsigned int channels;
    unsigned int bitsPerSample;  // container bits: 16, 24 or 32
    unsigned int validBits;
    unsigned int isFloat;        // 1 = IEEE float32
    unsigned int blockAlign;     // bytes per frame
    volatile long streamActive;  // 1 while Windows has the endpoint's stream in RUN state
    volatile long formatGeneration;  // incremented on every format change
    unsigned int reserved0;

    // Monotonic byte positions. Producer owns writePos, consumer owns readPos.
    volatile __int64 writePos;
    volatile __int64 readPos;

    volatile __int64 engineHeartbeat;  // engine: QPC ticks of its last poll (0 = never attached)
    volatile __int64 driverBytes;      // driver: bytes moved by Windows' stream since it started
    volatile __int64 overflows;        // producer had no room (data dropped)
    volatile __int64 underflows;       // consumer had no data (silence inserted)

    wchar_t name[64];  // display name, e.g. L"NexTone Output 1"
} NTVAD_RING_HEADER;
#pragma pack(pop)
