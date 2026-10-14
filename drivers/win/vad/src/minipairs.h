/*++

Copyright (c) Microsoft Corporation All Rights Reserved

Module Name:

    minipairs.h

Abstract:

    Local audio endpoint filter definitions. 
--*/

#ifndef _SIMPLEAUDIOSAMPLE_MINIPAIRS_H_
#define _SIMPLEAUDIOSAMPLE_MINIPAIRS_H_

#include "speakertopo.h"
#include "speakertoptable.h"
#include "speakerwavtable.h"

#include "micarraytopo.h"
#include "micarray1toptable.h"
#include "micarraywavtable.h"
#include "../include/tonesphere_vad_shared.h"


NTSTATUS
CreateMiniportWaveRTSimpleAudioSample
( 
    _Out_       PUNKNOWN *,
    _In_        REFCLSID,
    _In_opt_    PUNKNOWN,
    _In_        POOL_FLAGS,
    _In_        PUNKNOWN,
    _In_opt_    PVOID,
    _In_        PENDPOINT_MINIPAIR
);

NTSTATUS
CreateMiniportTopologySimpleAudioSample
( 
    _Out_       PUNKNOWN *,
    _In_        REFCLSID,
    _In_opt_    PUNKNOWN,
    _In_        POOL_FLAGS,
    _In_        PUNKNOWN,
    _In_opt_    PVOID,
    _In_        PENDPOINT_MINIPAIR
);

//
// Render miniports.
//

/*********************************************************************
* Topology/Wave bridge connection for speaker (internal)             *
*                                                                    *
*              +------+                +------+                      *
*              | Wave |                | Topo |                      *
*              |      |                |      |                      *
* System   --->|0    1|--------------->|0    1|---> Line Out         *
*              |      |                |      |                      *
*              +------+                +------+                      *
*********************************************************************/
static
PHYSICALCONNECTIONTABLE SpeakerTopologyPhysicalConnections[] =
{
    {
        KSPIN_TOPO_WAVEOUT_SOURCE,  // TopologyIn
        KSPIN_WAVE_RENDER3_SOURCE,   // WaveOut
        CONNECTIONTYPE_WAVE_OUTPUT
    }
};

static
ENDPOINT_MINIPAIR SpeakerMiniports =
{
    eSpeakerDevice,
    L"TopologySpeaker",                                     // make sure this or the template name matches with KSNAME_TopologySpeaker in the inf's [Strings] section 
    NULL,                                                   // optional template name
    CreateMiniportTopologySimpleAudioSample,
    &SpeakerTopoMiniportFilterDescriptor,
    0, NULL,                                                // Interface properties
    L"WaveSpeaker",                                         // make sure this or the template name matches with KSNAME_WaveSpeaker in the inf's [Strings] section
    NULL,                                                   // optional template name
    CreateMiniportWaveRTSimpleAudioSample,
    &SpeakerWaveMiniportFilterDescriptor,
    0,                                                      // Interface properties
    NULL,
    SPEAKER_DEVICE_MAX_CHANNELS,
    SpeakerPinDeviceFormatsAndModes,
    SIZEOF_ARRAY(SpeakerPinDeviceFormatsAndModes),
    SpeakerTopologyPhysicalConnections,
    SIZEOF_ARRAY(SpeakerTopologyPhysicalConnections),
    ENDPOINT_NO_FLAGS,
};

//
// Capture miniports.
//

/*********************************************************************
* Topology/Wave bridge connection for mic array  1 (front)           *
*                                                                    *
*              +------+    +------+                                  *
*              | Topo |    | Wave |                                  *
*              |      |    |      |                                  *
*  Mic in  --->|0    1|===>|0    1|---> Capture Host Pin             *
*              |      |    |      |                                  *
*              +------+    +------+                                  *
*********************************************************************/
static
PHYSICALCONNECTIONTABLE MicArray1TopologyPhysicalConnections[] =
{
    {
        KSPIN_TOPO_BRIDGE,          // TopologyOut
        KSPIN_WAVE_BRIDGE,          // WaveIn
        CONNECTIONTYPE_TOPOLOGY_OUTPUT
    }
};

static
ENDPOINT_MINIPAIR MicArray1Miniports =
{
    eMicArrayDevice1,
    L"TopologyMicArray1",                   // make sure this or the template name matches with KSNAME_TopologyMicArray1 in the inf's [Strings] section 
    NULL,                                   // optional template name
    CreateMicArrayMiniportTopology,
    &MicArray1TopoMiniportFilterDescriptor,
    0, NULL,                                // Interface properties
    L"WaveMicArray1",                       // make sure this or the tempalte name matches with KSNAME_WaveMicArray1 in the inf's [Strings] section
    NULL,                                   // optional template name
    CreateMiniportWaveRTSimpleAudioSample,
    &MicArrayWaveMiniportFilterDescriptor,
    0,                                      // Interface properties
    NULL,
    MICARRAY_DEVICE_MAX_CHANNELS,
    MicArrayPinDeviceFormatsAndModes,
    SIZEOF_ARRAY(MicArrayPinDeviceFormatsAndModes),
    MicArray1TopologyPhysicalConnections,
    SIZEOF_ARRAY(MicArray1TopologyPhysicalConnections),
    ENDPOINT_NO_FLAGS,
};


//=============================================================================
//
// ToneSphere endpoints: TS_OUTPUTS render endpoints ("ToneSphere Output N") and TS_MICS
// capture endpoints ("ToneSphere Mic N"), built at DriverEntry from the speaker and mic
// templates above. Each endpoint gets unique KS filter names (matching the INF) and a
// pin Name GUID whose display name is registered under MediaCategories by the INF.
//
// {124670D4-2423-459E-848E-97798DC64E01} .. 08 : outputs
// {124670D4-2423-459E-848E-97798DC64F01} .. 08 : mics
static const GUID g_NtOutNameGuids[TSVAD_MAX_OUTPUTS] = {
    {0x124670d4, 0x2423, 0x459e, {0x84, 0x8e, 0x97, 0x79, 0x8d, 0xc6, 0x4e, 0x01}},
    {0x124670d4, 0x2423, 0x459e, {0x84, 0x8e, 0x97, 0x79, 0x8d, 0xc6, 0x4e, 0x02}},
    {0x124670d4, 0x2423, 0x459e, {0x84, 0x8e, 0x97, 0x79, 0x8d, 0xc6, 0x4e, 0x03}},
    {0x124670d4, 0x2423, 0x459e, {0x84, 0x8e, 0x97, 0x79, 0x8d, 0xc6, 0x4e, 0x04}},
    {0x124670d4, 0x2423, 0x459e, {0x84, 0x8e, 0x97, 0x79, 0x8d, 0xc6, 0x4e, 0x05}},
    {0x124670d4, 0x2423, 0x459e, {0x84, 0x8e, 0x97, 0x79, 0x8d, 0xc6, 0x4e, 0x06}},
    {0x124670d4, 0x2423, 0x459e, {0x84, 0x8e, 0x97, 0x79, 0x8d, 0xc6, 0x4e, 0x07}},
    {0x124670d4, 0x2423, 0x459e, {0x84, 0x8e, 0x97, 0x79, 0x8d, 0xc6, 0x4e, 0x08}},
};
static const GUID g_NtMicNameGuids[TSVAD_MAX_MICS] = {
    {0x124670d4, 0x2423, 0x459e, {0x84, 0x8e, 0x97, 0x79, 0x8d, 0xc6, 0x4f, 0x01}},
    {0x124670d4, 0x2423, 0x459e, {0x84, 0x8e, 0x97, 0x79, 0x8d, 0xc6, 0x4f, 0x02}},
    {0x124670d4, 0x2423, 0x459e, {0x84, 0x8e, 0x97, 0x79, 0x8d, 0xc6, 0x4f, 0x03}},
    {0x124670d4, 0x2423, 0x459e, {0x84, 0x8e, 0x97, 0x79, 0x8d, 0xc6, 0x4f, 0x04}},
    {0x124670d4, 0x2423, 0x459e, {0x84, 0x8e, 0x97, 0x79, 0x8d, 0xc6, 0x4f, 0x05}},
    {0x124670d4, 0x2423, 0x459e, {0x84, 0x8e, 0x97, 0x79, 0x8d, 0xc6, 0x4f, 0x06}},
    {0x124670d4, 0x2423, 0x459e, {0x84, 0x8e, 0x97, 0x79, 0x8d, 0xc6, 0x4f, 0x07}},
    {0x124670d4, 0x2423, 0x459e, {0x84, 0x8e, 0x97, 0x79, 0x8d, 0xc6, 0x4f, 0x08}},
};

static ENDPOINT_MINIPAIR   g_NtOutPairs[TS_OUTPUTS];
static ENDPOINT_MINIPAIR   g_NtMicPairs[TS_MICS];
static PCFILTER_DESCRIPTOR g_NtOutTopoDesc[TS_OUTPUTS];
static PCFILTER_DESCRIPTOR g_NtMicTopoDesc[TS_MICS];
static PCPIN_DESCRIPTOR    g_NtOutTopoPins[TS_OUTPUTS][SIZEOF_ARRAY(SpeakerTopoMiniportPins)];
static PCPIN_DESCRIPTOR    g_NtMicTopoPins[TS_MICS][SIZEOF_ARRAY(MicArray1TopoMiniportPins)];
static WCHAR               g_NtNames[(TS_OUTPUTS + TS_MICS) * 2][32];

static PENDPOINT_MINIPAIR  g_RenderEndpoints[TS_OUTPUTS];
static PENDPOINT_MINIPAIR  g_CaptureEndpoints[TS_MICS];

#define g_cRenderEndpoints  (SIZEOF_ARRAY(g_RenderEndpoints))
#define g_cCaptureEndpoints (SIZEOF_ARRAY(g_CaptureEndpoints))

static void TsSetName(WCHAR* dst, PCWSTR prefix, ULONG n)
{
    UNICODE_STRING s;
    RtlInitEmptyUnicodeString(&s, dst, 32 * sizeof(WCHAR) - sizeof(WCHAR));
    RtlUnicodeStringPrintf(&s, L"%ws%u", prefix, n);
    dst[s.Length / sizeof(WCHAR)] = L'\0';
}

// Called once from DriverEntry (PASSIVE_LEVEL).
static void TsBuildEndpoints()
{
    ULONG nameSlot = 0;
    for (ULONG i = 0; i < TS_OUTPUTS; ++i)
    {
        RtlCopyMemory(g_NtOutTopoPins[i], SpeakerTopoMiniportPins, sizeof(SpeakerTopoMiniportPins));
        g_NtOutTopoPins[i][KSPIN_TOPO_LINEOUT_DEST].KsPinDescriptor.Name = &g_NtOutNameGuids[i];
        // Windows names KSNODETYPE_SPEAKER endpoints "Speakers" regardless of the pin name;
        // a line connector keeps our own name ("ToneSphere Output N").
        g_NtOutTopoPins[i][KSPIN_TOPO_LINEOUT_DEST].KsPinDescriptor.Category = &KSNODETYPE_LINE_CONNECTOR;
        g_NtOutTopoDesc[i] = SpeakerTopoMiniportFilterDescriptor;
        g_NtOutTopoDesc[i].Pins = g_NtOutTopoPins[i];

        g_NtOutPairs[i] = SpeakerMiniports;
        WCHAR* topo = g_NtNames[nameSlot++];
        WCHAR* wave = g_NtNames[nameSlot++];
        TsSetName(topo, L"TopologyToneSphereOut", i + 1);
        TsSetName(wave, L"WaveToneSphereOut", i + 1);
        g_NtOutPairs[i].TopoName = topo;
        g_NtOutPairs[i].WaveName = wave;
        g_NtOutPairs[i].TopoDescriptor = &g_NtOutTopoDesc[i];
        g_NtOutPairs[i].RingIndex = i;
        g_RenderEndpoints[i] = &g_NtOutPairs[i];
    }
    for (ULONG i = 0; i < TS_MICS; ++i)
    {
        RtlCopyMemory(g_NtMicTopoPins[i], MicArray1TopoMiniportPins, sizeof(MicArray1TopoMiniportPins));
        g_NtMicTopoPins[i][KSPIN_TOPO_MIC_ELEMENTS].KsPinDescriptor.Name = &g_NtMicNameGuids[i];
        g_NtMicTopoDesc[i] = MicArray1TopoMiniportFilterDescriptor;
        g_NtMicTopoDesc[i].Pins = g_NtMicTopoPins[i];

        g_NtMicPairs[i] = MicArray1Miniports;
        WCHAR* topo = g_NtNames[nameSlot++];
        WCHAR* wave = g_NtNames[nameSlot++];
        TsSetName(topo, L"TopologyToneSphereMic", i + 1);
        TsSetName(wave, L"WaveToneSphereMic", i + 1);
        g_NtMicPairs[i].TopoName = topo;
        g_NtMicPairs[i].WaveName = wave;
        g_NtMicPairs[i].TopoDescriptor = &g_NtMicTopoDesc[i];
        g_NtMicPairs[i].RingIndex = i;
        g_CaptureEndpoints[i] = &g_NtMicPairs[i];
    }
}

//=============================================================================
//
// Total miniports = # endpoints * 2 (topology + wave).
//
#define g_MaxMiniports  ((g_cRenderEndpoints + g_cCaptureEndpoints) * 2)

#endif // _SIMPLEAUDIOSAMPLE_MINIPAIRS_H_