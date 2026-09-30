/*++

ToneSphere virtual audio cable.

Copyright (c) 2026 Neural Nexus Studios. Part of a work derived from Microsoft's
SimpleAudioSample and distributed under the same Microsoft Public License (see LICENSE in
driver/windows_virtual_audio).

This is the only component of the driver that is not Microsoft's sample. It joins the two
endpoints of one device: what applications render into the cable's render endpoint is what
applications capturing its capture endpoint receive. Both endpoints run the same format
(48 kHz, 32-bit PCM, stereo), so this copies bytes and never converts.

Each device instance — each cable the user adds — owns one Cable, created with its adapter
object and freed with it. The adapter is reference-counted by the miniports, and the
miniports by their streams, so a cable outlives every stream that writes or reads it, even
across a surprise removal.

It does nothing else, deliberately: no mixing, no resampling, no DSP. All of that belongs in
user mode, in ToneSphere.

Callers are the WaveRT streams' position updates, which run at up to DISPATCH_LEVEL under
the stream's position spin lock. A spin lock here serialises the render and capture sides;
every operation is a bounded memcpy.

--*/

#include <ntddk.h>

#include "cable.h"

namespace {

// One second at 48 kHz, 32-bit, stereo: the storage.
constexpr ULONG kCableBytes = 48000 * 8;

// At most 100 ms may queue. The render and capture streams advance on their own timers,
// so the fill level swings by about a period between them; anything beyond that bound is
// audio no one has asked for yet, and keeping it would only make every later sample late.
constexpr ULONG kMaxQueuedBytes = 4800 * 8;

constexpr ULONG kPoolTag = 'bCsT';

}  // namespace

struct Cable
{
    UCHAR* buffer;
    ULONG read;
    ULONG fill;
    KSPIN_LOCK lock;
};

#pragma code_seg("PAGE")
Cable* CableCreate()
{
    PAGED_CODE();
    Cable* cable = static_cast<Cable*>(ExAllocatePool2(POOL_FLAG_NON_PAGED, sizeof(Cable), kPoolTag));
    if (cable == nullptr)
    {
        return nullptr;
    }
    cable->buffer = static_cast<UCHAR*>(ExAllocatePool2(POOL_FLAG_NON_PAGED, kCableBytes, kPoolTag));
    if (cable->buffer == nullptr)
    {
        ExFreePoolWithTag(cable, kPoolTag);
        return nullptr;
    }
    KeInitializeSpinLock(&cable->lock);
    cable->read = 0;
    cable->fill = 0;
    return cable;
}

VOID CableDestroy(_In_opt_ Cable* cable)
{
    PAGED_CODE();
    if (cable != nullptr)
    {
        ExFreePoolWithTag(cable->buffer, kPoolTag);
        ExFreePoolWithTag(cable, kPoolTag);
    }
}

#pragma code_seg()
VOID CableWrite(_In_opt_ Cable* cable, _In_reads_bytes_(Bytes) const UCHAR* Source, _In_ ULONG Bytes)
{
    if (cable == nullptr || Bytes == 0)
    {
        return;
    }
    if (Bytes > kMaxQueuedBytes)
    {
        Source += Bytes - kMaxQueuedBytes;
        Bytes = kMaxQueuedBytes;
    }

    KIRQL irql;
    KeAcquireSpinLock(&cable->lock, &irql);

    // Over the bound: drop the oldest queued audio, never the newest.
    if (cable->fill + Bytes > kMaxQueuedBytes)
    {
        const ULONG drop = cable->fill + Bytes - kMaxQueuedBytes;
        cable->read = (cable->read + drop) % kCableBytes;
        cable->fill -= drop;
    }

    ULONG write = (cable->read + cable->fill) % kCableBytes;
    ULONG remaining = Bytes;
    while (remaining > 0)
    {
        const ULONG run = min(remaining, kCableBytes - write);
        RtlCopyMemory(cable->buffer + write, Source, run);
        Source += run;
        remaining -= run;
        write = (write + run) % kCableBytes;
    }
    cable->fill += Bytes;

    KeReleaseSpinLock(&cable->lock, irql);
}

VOID CableRead(_In_opt_ Cable* cable, _Out_writes_bytes_(Bytes) UCHAR* Destination, _In_ ULONG Bytes)
{
    if (cable == nullptr)
    {
        RtlZeroMemory(Destination, Bytes);
        return;
    }

    KIRQL irql;
    KeAcquireSpinLock(&cable->lock, &irql);

    const ULONG available = min(Bytes, cable->fill);
    ULONG remaining = available;
    while (remaining > 0)
    {
        const ULONG run = min(remaining, kCableBytes - cable->read);
        RtlCopyMemory(Destination, cable->buffer + cable->read, run);
        Destination += run;
        remaining -= run;
        cable->read = (cable->read + run) % kCableBytes;
    }
    cable->fill -= available;

    KeReleaseSpinLock(&cable->lock, irql);

    // An empty cable is silence: nothing is rendering into the input.
    if (available < Bytes)
    {
        RtlZeroMemory(Destination, Bytes - available);
    }
}

// A capture that starts begins with an empty cable: whatever is queued was rendered before
// anyone was listening, and replaying it would put a previous application's audio into a new
// recording (up to kMaxQueuedBytes of it).
VOID CableFlush(_In_opt_ Cable* cable)
{
    if (cable == nullptr)
    {
        return;
    }
    KIRQL irql;
    KeAcquireSpinLock(&cable->lock, &irql);
    cable->read = 0;
    cable->fill = 0;
    KeReleaseSpinLock(&cable->lock, irql);
}
