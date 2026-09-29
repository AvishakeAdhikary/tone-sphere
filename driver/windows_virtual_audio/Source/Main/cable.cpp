/*++

ToneSphere virtual audio cable.

Copyright (c) 2026 Neural Nexus Studios. Part of a work derived from Microsoft's
SimpleAudioSample and distributed under the same Microsoft Public License (see LICENSE in
driver/windows_virtual_audio).

This is the only component of the driver that is not Microsoft's sample. It joins the two
endpoints: what applications render into the cable's render endpoint is what applications
capturing its capture endpoint receive. Both endpoints run the same format (48 kHz,
32-bit PCM, stereo), so this copies bytes and never converts.

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

UCHAR* g_buffer = nullptr;
ULONG g_read = 0;
ULONG g_fill = 0;
KSPIN_LOCK g_lock;

}  // namespace

#pragma code_seg("PAGE")
NTSTATUS CableInitialize()
{
    PAGED_CODE();
    g_buffer = static_cast<UCHAR*>(ExAllocatePool2(POOL_FLAG_NON_PAGED, kCableBytes, kPoolTag));
    if (g_buffer == nullptr)
    {
        return STATUS_INSUFFICIENT_RESOURCES;
    }
    KeInitializeSpinLock(&g_lock);
    g_read = 0;
    g_fill = 0;
    return STATUS_SUCCESS;
}

VOID CableFree()
{
    PAGED_CODE();
    if (g_buffer != nullptr)
    {
        ExFreePoolWithTag(g_buffer, kPoolTag);
        g_buffer = nullptr;
    }
}

#pragma code_seg()
VOID CableWrite(_In_reads_bytes_(Bytes) const UCHAR* Source, _In_ ULONG Bytes)
{
    if (g_buffer == nullptr || Bytes == 0)
    {
        return;
    }
    if (Bytes > kMaxQueuedBytes)
    {
        Source += Bytes - kMaxQueuedBytes;
        Bytes = kMaxQueuedBytes;
    }

    KIRQL irql;
    KeAcquireSpinLock(&g_lock, &irql);

    // Over the bound: drop the oldest queued audio, never the newest.
    if (g_fill + Bytes > kMaxQueuedBytes)
    {
        const ULONG drop = g_fill + Bytes - kMaxQueuedBytes;
        g_read = (g_read + drop) % kCableBytes;
        g_fill -= drop;
    }

    ULONG write = (g_read + g_fill) % kCableBytes;
    ULONG remaining = Bytes;
    while (remaining > 0)
    {
        const ULONG run = min(remaining, kCableBytes - write);
        RtlCopyMemory(g_buffer + write, Source, run);
        Source += run;
        remaining -= run;
        write = (write + run) % kCableBytes;
    }
    g_fill += Bytes;

    KeReleaseSpinLock(&g_lock, irql);
}

VOID CableRead(_Out_writes_bytes_(Bytes) UCHAR* Destination, _In_ ULONG Bytes)
{
    if (g_buffer == nullptr)
    {
        RtlZeroMemory(Destination, Bytes);
        return;
    }

    KIRQL irql;
    KeAcquireSpinLock(&g_lock, &irql);

    const ULONG available = min(Bytes, g_fill);
    ULONG remaining = available;
    while (remaining > 0)
    {
        const ULONG run = min(remaining, kCableBytes - g_read);
        RtlCopyMemory(Destination, g_buffer + g_read, run);
        Destination += run;
        remaining -= run;
        g_read = (g_read + run) % kCableBytes;
    }
    g_fill -= available;

    KeReleaseSpinLock(&g_lock, irql);

    // An empty cable is silence: nothing is rendering into the input.
    if (available < Bytes)
    {
        RtlZeroMemory(Destination, Bytes - available);
    }
}

// A capture that starts begins with an empty cable: whatever is queued was rendered before
// anyone was listening, and replaying it would put a previous application's audio into a new
// recording (up to kMaxQueuedBytes of it).
VOID CableFlush()
{
    if (g_buffer == nullptr)
    {
        return;
    }
    KIRQL irql;
    KeAcquireSpinLock(&g_lock, &irql);
    g_read = 0;
    g_fill = 0;
    KeReleaseSpinLock(&g_lock, irql);
}
