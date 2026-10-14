/*++
    ToneSphere Virtual Audio Driver - shared-memory rings between WaveRT streams and the
    ToneSphere engine. See include/tonesphere_vad_shared.h for the layout.
--*/
#pragma once

#define TS_OUTPUTS 4  // "ToneSphere Output 1..4"  (render endpoints)
#define TS_MICS 4     // "ToneSphere Mic 1..4"     (capture endpoints)

struct TSVAD_RING;  // opaque

// PASSIVE_LEVEL: create/destroy all rings (device start / stop).
extern "C" NTSTATUS TsRingsCreateAll();
extern "C" void TsRingsDestroyAll();

// Lookup by direction and 0-based index. Returns NULL if unavailable.
extern "C" TSVAD_RING* TsRingGet(BOOLEAN capture, ULONG index);

// Any IRQL <= DISPATCH_LEVEL once created.
extern "C" void TsRingSetFormat(TSVAD_RING* ring, ULONG sampleRate, ULONG channels, ULONG bitsPerSample, ULONG validBits,
                                BOOLEAN isFloat, ULONG blockAlign);
extern "C" void TsRingSetActive(TSVAD_RING* ring, BOOLEAN active);
// Render endpoints: Windows' audio leaves the DMA buffer into the ring.
extern "C" void TsRingProduce(TSVAD_RING* ring, const UCHAR* src, ULONG bytes);
// Capture endpoints: fills `dst` from the ring, silence where no data is available.
extern "C" void TsRingConsume(TSVAD_RING* ring, UCHAR* dst, ULONG bytes, BOOLEAN isFloat, ULONG bitsPerSample);
