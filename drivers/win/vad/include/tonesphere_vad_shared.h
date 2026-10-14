// ToneSphere Virtual Audio Driver <-> ToneSphere engine shared-memory contract.
// Included by the kernel driver (MS-PL component) and by the engine (GPL component);
// it only describes a data layout and carries no implementation.
//
// Each virtual endpoint owns one named section:
//   kernel: \BaseNamedObjects\ToneSphereVadOut<N> / \BaseNamedObjects\ToneSphereVadMic<N>
//   user:   Global\ToneSphereVadOut<N>            / Global\ToneSphereVadMic<N>          (N = 1..)
// Layout: TSVAD_RING_HEADER (padded to TSVAD_HEADER_BYTES) followed by a byte ring of
// `dataBytes` bytes (power of two) in the Windows stream's native sample format.
#pragma once

#define TSVAD_MAGIC 0x52565354u  // 'TSVR'
#define TSVAD_VERSION 1u
#define TSVAD_MAX_OUTPUTS 8
#define TSVAD_MAX_MICS 8
#define TSVAD_HEADER_BYTES 4096u
#define TSVAD_DATA_BYTES (1024u * 1024u)
#define TSVAD_SECTION_BYTES (TSVAD_HEADER_BYTES + TSVAD_DATA_BYTES)

// Direction of the Windows endpoint.
#define TSVAD_DIR_RENDER 0u   // "ToneSphere Output N": apps play into it; driver produces, engine consumes
#define TSVAD_DIR_CAPTURE 1u  // "ToneSphere Mic N": apps record from it; engine produces, driver consumes

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

    wchar_t name[64];  // display name, e.g. L"ToneSphere Output 1"
} TSVAD_RING_HEADER;
#pragma pack(pop)
