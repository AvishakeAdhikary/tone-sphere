/*++
    NexTone Virtual Audio Driver - shared-memory rings.

    Each endpoint gets a named, page-file backed section that the engine opens with
    OpenFileMapping("Global\NexToneVad..."). The driver maps the same section into system
    space and locks its pages so the WaveRT position DPC can copy audio at DISPATCH_LEVEL.
--*/
#include <ntifs.h>
#include <ntstrsafe.h>

#include "nextone_ring.h"
#include "../include/nextone_vad_shared.h"

#define NTRING_POOLTAG 'gRtN'

struct NTVAD_RING {
    HANDLE section;
    PVOID view;
    SIZE_T viewSize;
    PMDL mdl;
    BOOLEAN locked;
    NTVAD_RING_HEADER* hdr;
    UCHAR* data;
    ULONG mask;
};

static NTVAD_RING g_Out[NT_OUTPUTS];
static NTVAD_RING g_Mic[NT_MICS];
static BOOLEAN g_Created = FALSE;

static void DestroyRing(NTVAD_RING* r) {
    if (r->mdl) {
        if (r->locked) MmUnlockPages(r->mdl);
        IoFreeMdl(r->mdl);
    }
    if (r->view) MmUnmapViewInSystemSpace(r->view);
    if (r->section) ZwClose(r->section);
    RtlZeroMemory(r, sizeof(*r));
}

// Security descriptor: SYSTEM and Administrators full access, everyone else may map read/write.
static NTSTATUS BuildSecurity(PSECURITY_DESCRIPTOR sd, PACL* outAcl) {
    *outAcl = NULL;
    NTSTATUS st = RtlCreateSecurityDescriptor(sd, SECURITY_DESCRIPTOR_REVISION);
    if (!NT_SUCCESS(st)) return st;
    const ULONG aclSize = sizeof(ACL) + 3 * (sizeof(ACCESS_ALLOWED_ACE) - sizeof(ULONG)) + RtlLengthSid(SeExports->SeWorldSid) +
                          RtlLengthSid(SeExports->SeLocalSystemSid) + RtlLengthSid(SeExports->SeAliasAdminsSid);
    PACL acl = (PACL)ExAllocatePool2(POOL_FLAG_PAGED, aclSize, NTRING_POOLTAG);
    if (!acl) return STATUS_INSUFFICIENT_RESOURCES;
    st = RtlCreateAcl(acl, aclSize, ACL_REVISION);
    if (NT_SUCCESS(st)) st = RtlAddAccessAllowedAce(acl, ACL_REVISION, SECTION_ALL_ACCESS, SeExports->SeLocalSystemSid);
    if (NT_SUCCESS(st)) st = RtlAddAccessAllowedAce(acl, ACL_REVISION, SECTION_ALL_ACCESS, SeExports->SeAliasAdminsSid);
    if (NT_SUCCESS(st))
        st = RtlAddAccessAllowedAce(acl, ACL_REVISION, SECTION_MAP_READ | SECTION_MAP_WRITE | SECTION_QUERY, SeExports->SeWorldSid);
    if (NT_SUCCESS(st)) st = RtlSetDaclSecurityDescriptor(sd, TRUE, acl, FALSE);
    if (!NT_SUCCESS(st)) {
        ExFreePoolWithTag(acl, NTRING_POOLTAG);
        return st;
    }
    *outAcl = acl;
    return STATUS_SUCCESS;
}

static NTSTATUS CreateRing(NTVAD_RING* r, BOOLEAN capture, ULONG index /* 1-based */) {
    RtlZeroMemory(r, sizeof(*r));
    WCHAR nameBuf[64];
    UNICODE_STRING name;
    RtlInitEmptyUnicodeString(&name, nameBuf, sizeof(nameBuf));
    NTSTATUS st = RtlUnicodeStringPrintf(&name, capture ? L"\\BaseNamedObjects\\NexToneVadMic%u" : L"\\BaseNamedObjects\\NexToneVadOut%u", index);
    if (!NT_SUCCESS(st)) return st;

    SECURITY_DESCRIPTOR sd;
    PACL acl = NULL;
    st = BuildSecurity(&sd, &acl);
    if (!NT_SUCCESS(st)) return st;

    OBJECT_ATTRIBUTES oa;
    InitializeObjectAttributes(&oa, &name, OBJ_KERNEL_HANDLE | OBJ_OPENIF, NULL, &sd);
    LARGE_INTEGER size;
    size.QuadPart = NTVAD_SECTION_BYTES;
    st = ZwCreateSection(&r->section, SECTION_ALL_ACCESS, &oa, &size, PAGE_READWRITE, SEC_COMMIT, NULL);
    ExFreePoolWithTag(acl, NTRING_POOLTAG);
    if (!NT_SUCCESS(st)) return st;

    PVOID sectionObject = NULL;
    st = ObReferenceObjectByHandle(r->section, SECTION_MAP_READ | SECTION_MAP_WRITE, NULL, KernelMode, &sectionObject, NULL);
    if (!NT_SUCCESS(st)) {
        DestroyRing(r);
        return st;
    }
    r->viewSize = 0;
    st = MmMapViewInSystemSpace(sectionObject, &r->view, &r->viewSize);
    ObDereferenceObject(sectionObject);
    if (!NT_SUCCESS(st)) {
        r->view = NULL;
        DestroyRing(r);
        return st;
    }

    // Lock the pages: the position DPC touches them at DISPATCH_LEVEL.
    r->mdl = IoAllocateMdl(r->view, (ULONG)NTVAD_SECTION_BYTES, FALSE, FALSE, NULL);
    if (!r->mdl) {
        DestroyRing(r);
        return STATUS_INSUFFICIENT_RESOURCES;
    }
    __try {
        MmProbeAndLockPages(r->mdl, KernelMode, IoWriteAccess);
        r->locked = TRUE;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        DestroyRing(r);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    r->hdr = (NTVAD_RING_HEADER*)r->view;
    r->data = (UCHAR*)r->view + NTVAD_HEADER_BYTES;
    r->mask = NTVAD_DATA_BYTES - 1;
    NTVAD_RING_HEADER* h = r->hdr;
    if (h->magic != NTVAD_MAGIC) {  // first creation (OBJ_OPENIF may reopen an existing section)
        RtlZeroMemory(h, NTVAD_HEADER_BYTES);
        h->magic = NTVAD_MAGIC;
        h->version = NTVAD_VERSION;
        h->sampleRate = 48000;
        h->channels = 2;
        h->bitsPerSample = 16;
        h->validBits = 16;
        h->blockAlign = 4;
    }
    h->direction = capture ? NTVAD_DIR_CAPTURE : NTVAD_DIR_RENDER;
    h->index = index;
    h->dataBytes = NTVAD_DATA_BYTES;
    h->streamActive = 0;
    UNICODE_STRING disp;
    RtlInitEmptyUnicodeString(&disp, h->name, sizeof(h->name) - sizeof(WCHAR));
    RtlUnicodeStringPrintf(&disp, capture ? L"NexTone Mic %u" : L"NexTone Output %u", index);
    return STATUS_SUCCESS;
}

extern "C" NTSTATUS NtRingsCreateAll() {
    PAGED_CODE();
    if (g_Created) return STATUS_SUCCESS;
    NTSTATUS result = STATUS_SUCCESS;
    for (ULONG i = 0; i < NT_OUTPUTS; ++i) {
        NTSTATUS st = CreateRing(&g_Out[i], FALSE, i + 1);
        if (!NT_SUCCESS(st)) result = st;
    }
    for (ULONG i = 0; i < NT_MICS; ++i) {
        NTSTATUS st = CreateRing(&g_Mic[i], TRUE, i + 1);
        if (!NT_SUCCESS(st)) result = st;
    }
    g_Created = TRUE;
    return result;
}

extern "C" void NtRingsDestroyAll() {
    PAGED_CODE();
    if (!g_Created) return;
    for (ULONG i = 0; i < NT_OUTPUTS; ++i) DestroyRing(&g_Out[i]);
    for (ULONG i = 0; i < NT_MICS; ++i) DestroyRing(&g_Mic[i]);
    g_Created = FALSE;
}

extern "C" NTVAD_RING* NtRingGet(BOOLEAN capture, ULONG index) {
    if (capture) return index < NT_MICS && g_Mic[index].hdr ? &g_Mic[index] : NULL;
    return index < NT_OUTPUTS && g_Out[index].hdr ? &g_Out[index] : NULL;
}

extern "C" void NtRingSetFormat(NTVAD_RING* r, ULONG sampleRate, ULONG channels, ULONG bitsPerSample, ULONG validBits, BOOLEAN isFloat,
                                ULONG blockAlign) {
    if (!r || !r->hdr) return;
    NTVAD_RING_HEADER* h = r->hdr;
    h->sampleRate = sampleRate;
    h->channels = channels;
    h->bitsPerSample = bitsPerSample;
    h->validBits = validBits;
    h->isFloat = isFloat ? 1 : 0;
    h->blockAlign = blockAlign;
    KeMemoryBarrier();
    InterlockedIncrement(&h->formatGeneration);
}

extern "C" void NtRingSetActive(NTVAD_RING* r, BOOLEAN active) {
    if (!r || !r->hdr) return;
    if (active) r->hdr->driverBytes = 0;
    InterlockedExchange(&r->hdr->streamActive, active ? 1 : 0);
}

extern "C" void NtRingProduce(NTVAD_RING* r, const UCHAR* src, ULONG bytes) {
    if (!r || !r->hdr || bytes == 0) return;
    NTVAD_RING_HEADER* h = r->hdr;
    h->driverBytes += bytes;
    const LONG64 w = h->writePos;
    const LONG64 rd = InterlockedCompareExchange64(&h->readPos, 0, 0);  // atomic read
    const LONG64 used = w - rd;
    if (used < 0 || used > (LONG64)NTVAD_DATA_BYTES) {
        // Engine never attached / reset: start over at the current position.
        InterlockedExchange64(&h->readPos, w);
    } else if ((LONG64)NTVAD_DATA_BYTES - used < (LONG64)bytes) {
        InterlockedIncrement64(&h->overflows);  // engine not consuming: drop
        return;
    }
    ULONG off = (ULONG)(w & r->mask);
    ULONG first = min(bytes, NTVAD_DATA_BYTES - off);
    RtlCopyMemory(r->data + off, src, first);
    if (bytes > first) RtlCopyMemory(r->data, src + first, bytes - first);
    KeMemoryBarrier();
    InterlockedExchange64(&h->writePos, w + bytes);
}

extern "C" void NtRingConsume(NTVAD_RING* r, UCHAR* dst, ULONG bytes, BOOLEAN isFloat, ULONG bitsPerSample) {
    UNREFERENCED_PARAMETER(isFloat);
    UNREFERENCED_PARAMETER(bitsPerSample);
    if (bytes == 0) return;
    if (!r || !r->hdr) {
        RtlZeroMemory(dst, bytes);  // all supported formats are signed PCM/float: 0 == silence
        return;
    }
    NTVAD_RING_HEADER* h = r->hdr;
    h->driverBytes += bytes;
    const LONG64 rd = h->readPos;
    const LONG64 w = InterlockedCompareExchange64(&h->writePos, 0, 0);
    LONG64 avail = w - rd;
    if (avail < 0 || avail > (LONG64)NTVAD_DATA_BYTES) {
        // Producer restarted: resynchronise to its position.
        InterlockedExchange64(&h->readPos, w);
        avail = 0;
    }
    ULONG take = (ULONG)min((LONG64)bytes, avail);
    // Keep whole frames.
    const ULONG align = h->blockAlign ? h->blockAlign : 1;
    take -= take % align;
    if (take) {
        ULONG off = (ULONG)(rd & r->mask);
        ULONG first = min(take, NTVAD_DATA_BYTES - off);
        RtlCopyMemory(dst, r->data + off, first);
        if (take > first) RtlCopyMemory(dst + first, r->data, take - first);
        KeMemoryBarrier();
        InterlockedExchange64(&h->readPos, rd + take);
    }
    if (take < bytes) {
        RtlZeroMemory(dst + take, bytes - take);
        if (h->engineHeartbeat) InterlockedIncrement64(&h->underflows);
    }
}
