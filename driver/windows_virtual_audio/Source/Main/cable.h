/*++

ToneSphere virtual audio cable — see cable.cpp. Microsoft Public License, like the rest of
driver/windows_virtual_audio.

--*/

#pragma once

// One per device instance: every ToneSphere cable is its own PnP device, so each has its
// own buffer and nothing crosses from one cable to another.
struct Cable;

_IRQL_requires_(PASSIVE_LEVEL) Cable* CableCreate();
_IRQL_requires_(PASSIVE_LEVEL) VOID CableDestroy(_In_opt_ Cable* cable);
VOID CableWrite(_In_opt_ Cable* cable, _In_reads_bytes_(Bytes) const UCHAR* Source, _In_ ULONG Bytes);
VOID CableRead(_In_opt_ Cable* cable, _Out_writes_bytes_(Bytes) UCHAR* Destination, _In_ ULONG Bytes);
VOID CableFlush(_In_opt_ Cable* cable);
