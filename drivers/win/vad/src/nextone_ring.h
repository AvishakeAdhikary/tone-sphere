/*++
    NexTone Virtual Audio Driver - shared-memory rings between WaveRT streams and the
    NexTone engine. See include/nextone_vad_shared.h for the layout.
--*/
#pragma once

#define NT_OUTPUTS 4  // "NexTone Output 1..4"  (render endpoints)
#define NT_MICS 4     // "NexTone Mic 1..4"     (capture endpoints)

struct NTVAD_RING;  // opaque

// PASSIVE_LEVEL: create/destroy all rings (device start / stop).
extern "C" NTSTATUS NtRingsCreateAll();
extern "C" void NtRingsDestroyAll();

// Lookup by direction and 0-based index. Returns NULL if unavailable.
extern "C" NTVAD_RING* NtRingGet(BOOLEAN capture, ULONG index);

// Any IRQL <= DISPATCH_LEVEL once created.
extern "C" void NtRingSetFormat(NTVAD_RING* ring, ULONG sampleRate, ULONG channels, ULONG bitsPerSample, ULONG validBits,
                                BOOLEAN isFloat, ULONG blockAlign);
extern "C" void NtRingSetActive(NTVAD_RING* ring, BOOLEAN active);
// Render endpoints: Windows' audio leaves the DMA buffer into the ring.
extern "C" void NtRingProduce(NTVAD_RING* ring, const UCHAR* src, ULONG bytes);
// Capture endpoints: fills `dst` from the ring, silence where no data is available.
extern "C" void NtRingConsume(NTVAD_RING* ring, UCHAR* dst, ULONG bytes, BOOLEAN isFloat, ULONG bitsPerSample);
