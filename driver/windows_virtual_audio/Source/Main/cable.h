/*++

ToneSphere virtual audio cable — see cable.cpp. Microsoft Public License, like the rest of
driver/windows_virtual_audio.

--*/

#pragma once

NTSTATUS CableInitialize();
VOID CableFree();
VOID CableWrite(_In_reads_bytes_(Bytes) const UCHAR* Source, _In_ ULONG Bytes);
VOID CableRead(_Out_writes_bytes_(Bytes) UCHAR* Destination, _In_ ULONG Bytes);
VOID CableFlush();
