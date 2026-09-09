/*
    ToneSphereAudio.c — a macOS AudioServerPlugIn (CoreAudio HAL plug-in) that publishes
    one loopback device named "ToneSphere Audio": everything written to its output stream
    comes back out of its input stream, through coreaudiod's own mix graph.

    WHY THIS EXISTS
    ---------------
    `docs/VIRTUAL_AUDIO_DRIVER.md` explains that macOS is the one platform where an
    OS-visible virtual audio device needs neither kernel code nor a paid certificate: a
    user-space bundle in /Library/Audio/Plug-Ins/HAL is enough, and ad-hoc signing is
    enough to load it locally. Once coreaudiod has rescanned, PortAudio enumerates this
    device exactly like a sound card, so ToneSphere needs no new IPC of its own —
    coreaudiod *is* the transport. That is the same architectural insight the Linux
    null-sink backend (`tonesphere/engine/linux_virtual.py`) is built on.

    WHAT IS AND IS NOT VERIFIED ABOUT THIS FILE — READ BEFORE TRUSTING IT
    --------------------------------------------------------------------
    This file was written on a Windows machine. It has never been compiled, installed, or
    listened to by its author. That is not a figure of speech: there was no Mac, no Xcode,
    no clang, and no coreaudiod involved in writing it.

    Verified, and how:
      * The `AudioServerPlugInDriverInterface` member set and the exact
        `kAudioServerPlugInTypeUUID` / `kAudioServerPlugInDriverInterfaceUUID` byte
        sequences were read from a published copy of Apple's real
        `CoreAudio.framework/Headers/AudioServerPlugIn.h` (the MacOSX SDK mirror at
        github.com/phracker/MacOSX-SDKs), not recalled from memory. The plug-in type UUID
        bytes 44 3A BA B8 E7 B3 49 1A B9 85 BE B9 18 70 30 DB are the same value written
        out as a string in Info.plist's CFPlugInTypes key.
      * The IO operation selectors are used *by name* (kAudioServerPlugInIOOperation*),
        never by four-char-code literal, so their numeric values cannot be transcribed
        wrongly here.
      * The vtable below is filled in with C99 designated initializers (`.StartIO = ...`)
        rather than positionally, so a mistake about *slot order* is structurally
        impossible — the compiler binds each function to the member Apple's header names.
        Any mismatch is a compile error on the macOS runner, not a silent misdispatch.
      * Every class id, property selector and error code used below was checked by name
        against published copies of AudioHardwareBase.h and AudioServerPlugIn.h, with one
        exception: kAudioDevicePropertyZeroTimeStampPeriod, which is in neither and is
        believed to be in AudioHardware.h — hence the explicit include of that header
        below. If that belief is wrong the build fails with an undeclared identifier,
        which is the right way for it to be wrong.

    NOT verified, and only a real Mac can settle it:
      * That it compiles at all against the current SDK. Checking that a selector name
        exists is not the same as checking that this file's use of it type-checks.
      * That the property set published here is sufficient for coreaudiod to expose the
        device — a HAL plug-in that misses a property the HAL requires typically loads
        and then simply never appears, with no error anywhere.
      * That the ring-buffer/zero-timestamp timing model below actually produces clean,
        continuous audio rather than clicks, drift, or an unstable device.
      * Anything about System Settings, Audio MIDI Setup, Discord/OBS/DAW device pickers,
        sleep/wake, or Gatekeeper.
    The `build-macos-plugin` job in `.github/workflows/ci.yml` is what turns the first
    three of those from "unverified" into a real, measured pass or a red build: it
    compiles this file, installs the bundle, restarts coreaudiod, and then writes a 1 kHz
    sine into the device and reads it back, asserting the frequency and RMS survive the
    trip. Until that job has actually run green, treat this file as a careful draft.

    DESIGN
    ------
    One plug-in object, one device, two streams (one input, one output), no box and no
    volume/mute controls — the smallest object graph that can carry audio. The absence of
    controls is deliberate: fewer objects is less that can be subtly wrong, and a device
    with no volume control is a real, honest thing (the system volume slider is simply
    inapplicable to it) rather than a control that claims to attenuate and does not.

    Structure follows Apple's own NullAudio sample plug-in, which is also what BlackHole
    (MIT, cited by docs/VIRTUAL_AUDIO_DRIVER.md as the reference implementation) is built
    from. BlackHole was the recommended starting point in the plan for this work; it was
    not forked here because a faithful fork means reproducing ~3800 lines of someone
    else's C exactly, which is not something that can be done reliably from a machine that
    cannot fetch, diff, or compile it. Writing the smaller thing from the documented
    interface, and having CI prove it, was judged the more honest path. Anyone who would
    rather have BlackHole's much more widely exercised implementation should fork it
    directly and keep its MIT notice; the Python side of this feature only depends on the
    device name and UID below, not on which implementation publishes them.
*/

#include <CoreAudio/AudioServerPlugIn.h>
// AudioServerPlugIn.h pulls in AudioHardwareBase.h but NOT AudioHardware.h, and
// kAudioDevicePropertyZeroTimeStampPeriod — the one property a HAL plug-in cannot omit
// without its device having no clock — lives in the latter. If this include turns out to
// be unnecessary on the current SDK it is harmless; if it were missing and the constant is
// where it is believed to be, the build would fail with an undeclared identifier.
#include <CoreAudio/AudioHardware.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <string.h>

#pragma mark - Identity shared with the Python side

/*
    kDevice_Name and kDevice_UID are duplicated in `tonesphere/engine/macos_virtual.py`
    as MACOS_DEVICE_NAME / MACOS_DEVICE_UID: the Python side finds this device by name in
    PortAudio's enumeration and must not match anything else — in particular it must never
    claim a user's separately installed BlackHole as ToneSphere's own. The duplication is
    checked by `tests/test_macos_virtual_device.py`, which reads this very file, so the two
    cannot drift apart silently.
*/
#define kDevice_Name                "ToneSphere Audio"
#define kDevice_UID                 "ToneSphereAudio_UID"
#define kDevice_ModelUID            "ToneSphereAudio_ModelUID"
#define kManufacturer_Name          "ToneSphere"

#define kDevice_Channels            2u
#define kDevice_RingBufferFrames    16384
#define kDevice_DefaultSampleRate   48000.0

enum
{
    kObjectID_PlugIn        = kAudioObjectPlugInObject,
    kObjectID_Device        = 2,
    kObjectID_Stream_Input  = 3,
    kObjectID_Stream_Output = 4
};

static const Float64 kSupportedSampleRates[] = { 44100.0, 48000.0, 88200.0, 96000.0 };
#define kSupportedSampleRateCount (sizeof(kSupportedSampleRates) / sizeof(Float64))

#pragma mark - State

static pthread_mutex_t  gPlugIn_StateMutex      = PTHREAD_MUTEX_INITIALIZER;
static UInt32           gPlugIn_RefCount        = 0;
static AudioServerPlugInHostRef gPlugIn_Host    = NULL;

static Float64          gDevice_SampleRate      = kDevice_DefaultSampleRate;
static UInt64           gDevice_IOClientCount   = 0;
static Float64          gDevice_HostTicksPerFrame = 0.0;
static UInt64           gDevice_NumberTimeStamps  = 0;
static UInt64           gDevice_AnchorHostTime    = 0;
static pthread_mutex_t  gDevice_IOMutex         = PTHREAD_MUTEX_INITIALIZER;

/*
    The loopback itself. Indexed by absolute sample time modulo its length, so the input
    side reads at whatever offset behind the output side the HAL happens to schedule it
    without either needing to know the other's timing.

    Read in DoIOOperation without taking gDevice_IOMutex, deliberately: that call is on
    the realtime IO thread, where blocking on a mutex held by a non-realtime thread is
    exactly how a driver produces dropouts. The only concurrent writer of these samples is
    another IO cycle on the same device, and a torn Float32 is a click, not a crash.
*/
static Float32 gDevice_RingBuffer[kDevice_RingBufferFrames * kDevice_Channels];

#pragma mark - Prototypes

// The one symbol this bundle exports (everything else is compiled -fvisibility=hidden):
// CFPlugIn looks it up by the name Info.plist gives in CFPlugInFactories, so it cannot be
// static and cannot be renamed on one side only.
void*           ToneSphereAudio_Create(CFAllocatorRef inAllocator, CFUUIDRef inRequestedTypeUUID)
                    __attribute__((visibility("default")));

static HRESULT  ToneSphere_QueryInterface(void* inDriver, REFIID inUUID, LPVOID* outInterface);
static ULONG    ToneSphere_AddRef(void* inDriver);
static ULONG    ToneSphere_Release(void* inDriver);
static OSStatus ToneSphere_Initialize(AudioServerPlugInDriverRef inDriver,
                                      AudioServerPlugInHostRef inHost);
static OSStatus ToneSphere_CreateDevice(AudioServerPlugInDriverRef inDriver,
                                        CFDictionaryRef inDescription,
                                        const AudioServerPlugInClientInfo* inClientInfo,
                                        AudioObjectID* outDeviceObjectID);
static OSStatus ToneSphere_DestroyDevice(AudioServerPlugInDriverRef inDriver,
                                         AudioObjectID inDeviceObjectID);
static OSStatus ToneSphere_AddDeviceClient(AudioServerPlugInDriverRef inDriver,
                                           AudioObjectID inDeviceObjectID,
                                           const AudioServerPlugInClientInfo* inClientInfo);
static OSStatus ToneSphere_RemoveDeviceClient(AudioServerPlugInDriverRef inDriver,
                                              AudioObjectID inDeviceObjectID,
                                              const AudioServerPlugInClientInfo* inClientInfo);
static OSStatus ToneSphere_PerformDeviceConfigurationChange(AudioServerPlugInDriverRef inDriver,
                                                            AudioObjectID inDeviceObjectID,
                                                            UInt64 inChangeAction,
                                                            void* inChangeInfo);
static OSStatus ToneSphere_AbortDeviceConfigurationChange(AudioServerPlugInDriverRef inDriver,
                                                          AudioObjectID inDeviceObjectID,
                                                          UInt64 inChangeAction,
                                                          void* inChangeInfo);
static Boolean  ToneSphere_HasProperty(AudioServerPlugInDriverRef inDriver,
                                       AudioObjectID inObjectID,
                                       pid_t inClientProcessID,
                                       const AudioObjectPropertyAddress* inAddress);
static OSStatus ToneSphere_IsPropertySettable(AudioServerPlugInDriverRef inDriver,
                                              AudioObjectID inObjectID,
                                              pid_t inClientProcessID,
                                              const AudioObjectPropertyAddress* inAddress,
                                              Boolean* outIsSettable);
static OSStatus ToneSphere_GetPropertyDataSize(AudioServerPlugInDriverRef inDriver,
                                               AudioObjectID inObjectID,
                                               pid_t inClientProcessID,
                                               const AudioObjectPropertyAddress* inAddress,
                                               UInt32 inQualifierDataSize,
                                               const void* inQualifierData,
                                               UInt32* outDataSize);
static OSStatus ToneSphere_GetPropertyData(AudioServerPlugInDriverRef inDriver,
                                           AudioObjectID inObjectID,
                                           pid_t inClientProcessID,
                                           const AudioObjectPropertyAddress* inAddress,
                                           UInt32 inQualifierDataSize,
                                           const void* inQualifierData,
                                           UInt32 inDataSize,
                                           UInt32* outDataSize,
                                           void* outData);
static OSStatus ToneSphere_SetPropertyData(AudioServerPlugInDriverRef inDriver,
                                           AudioObjectID inObjectID,
                                           pid_t inClientProcessID,
                                           const AudioObjectPropertyAddress* inAddress,
                                           UInt32 inQualifierDataSize,
                                           const void* inQualifierData,
                                           UInt32 inDataSize,
                                           const void* inData);
static OSStatus ToneSphere_StartIO(AudioServerPlugInDriverRef inDriver,
                                   AudioObjectID inDeviceObjectID, UInt32 inClientID);
static OSStatus ToneSphere_StopIO(AudioServerPlugInDriverRef inDriver,
                                  AudioObjectID inDeviceObjectID, UInt32 inClientID);
static OSStatus ToneSphere_GetZeroTimeStamp(AudioServerPlugInDriverRef inDriver,
                                            AudioObjectID inDeviceObjectID,
                                            UInt32 inClientID,
                                            Float64* outSampleTime,
                                            UInt64* outHostTime,
                                            UInt64* outSeed);
static OSStatus ToneSphere_WillDoIOOperation(AudioServerPlugInDriverRef inDriver,
                                             AudioObjectID inDeviceObjectID,
                                             UInt32 inClientID,
                                             UInt32 inOperationID,
                                             Boolean* outWillDo,
                                             Boolean* outWillDoInPlace);
static OSStatus ToneSphere_BeginIOOperation(AudioServerPlugInDriverRef inDriver,
                                            AudioObjectID inDeviceObjectID,
                                            UInt32 inClientID,
                                            UInt32 inOperationID,
                                            UInt32 inIOBufferFrameSize,
                                            const AudioServerPlugInIOCycleInfo* inIOCycleInfo);
static OSStatus ToneSphere_DoIOOperation(AudioServerPlugInDriverRef inDriver,
                                         AudioObjectID inDeviceObjectID,
                                         AudioObjectID inStreamObjectID,
                                         UInt32 inClientID,
                                         UInt32 inOperationID,
                                         UInt32 inIOBufferFrameSize,
                                         const AudioServerPlugInIOCycleInfo* inIOCycleInfo,
                                         void* ioMainBuffer,
                                         void* ioSecondaryBuffer);
static OSStatus ToneSphere_EndIOOperation(AudioServerPlugInDriverRef inDriver,
                                          AudioObjectID inDeviceObjectID,
                                          UInt32 inClientID,
                                          UInt32 inOperationID,
                                          UInt32 inIOBufferFrameSize,
                                          const AudioServerPlugInIOCycleInfo* inIOCycleInfo);

#pragma mark - The driver interface

/*
    Designated initializers, not positional ones: Apple's header names every slot, so
    binding by name makes the one structural detail this file could not verify by
    running it — the order of the function pointers — impossible to get wrong. A typo in
    a member name fails the build instead of dispatching StartIO into StopIO.
*/
static AudioServerPlugInDriverInterface gAudioServerPlugInDriverInterface =
{
    ._reserved                          = NULL,
    .QueryInterface                     = ToneSphere_QueryInterface,
    .AddRef                             = ToneSphere_AddRef,
    .Release                            = ToneSphere_Release,
    .Initialize                         = ToneSphere_Initialize,
    .CreateDevice                       = ToneSphere_CreateDevice,
    .DestroyDevice                      = ToneSphere_DestroyDevice,
    .AddDeviceClient                    = ToneSphere_AddDeviceClient,
    .RemoveDeviceClient                 = ToneSphere_RemoveDeviceClient,
    .PerformDeviceConfigurationChange   = ToneSphere_PerformDeviceConfigurationChange,
    .AbortDeviceConfigurationChange     = ToneSphere_AbortDeviceConfigurationChange,
    .HasProperty                        = ToneSphere_HasProperty,
    .IsPropertySettable                 = ToneSphere_IsPropertySettable,
    .GetPropertyDataSize                = ToneSphere_GetPropertyDataSize,
    .GetPropertyData                    = ToneSphere_GetPropertyData,
    .SetPropertyData                    = ToneSphere_SetPropertyData,
    .StartIO                            = ToneSphere_StartIO,
    .StopIO                             = ToneSphere_StopIO,
    .GetZeroTimeStamp                   = ToneSphere_GetZeroTimeStamp,
    .WillDoIOOperation                  = ToneSphere_WillDoIOOperation,
    .BeginIOOperation                   = ToneSphere_BeginIOOperation,
    .DoIOOperation                      = ToneSphere_DoIOOperation,
    .EndIOOperation                     = ToneSphere_EndIOOperation
};

static AudioServerPlugInDriverInterface*    gAudioServerPlugInDriverInterfacePtr =
                                                &gAudioServerPlugInDriverInterface;
static AudioServerPlugInDriverRef           gAudioServerPlugInDriverRef =
                                                &gAudioServerPlugInDriverInterfacePtr;

/*
    Named in Info.plist's CFPlugInFactories. CFPlugIn resolves it by symbol name, so
    renaming this function means renaming it there too or the bundle loads and produces
    nothing.
*/
void* ToneSphereAudio_Create(CFAllocatorRef inAllocator, CFUUIDRef inRequestedTypeUUID)
{
    #pragma unused(inAllocator)
    void* theAnswer = NULL;
    if(CFEqual(inRequestedTypeUUID, kAudioServerPlugInTypeUUID))
    {
        theAnswer = gAudioServerPlugInDriverRef;
    }
    return theAnswer;
}

#pragma mark - IUnknown

static HRESULT ToneSphere_QueryInterface(void* inDriver, REFIID inUUID, LPVOID* outInterface)
{
    if((inDriver != gAudioServerPlugInDriverRef) || (outInterface == NULL))
    {
        return kAudioHardwareBadObjectError;
    }

    CFUUIDRef theRequestedUUID = CFUUIDCreateFromUUIDBytes(NULL, inUUID);
    if(theRequestedUUID == NULL)
    {
        return kAudioHardwareIllegalOperationError;
    }

    HRESULT theAnswer = E_NOINTERFACE;
    if(CFEqual(theRequestedUUID, IUnknownUUID) ||
       CFEqual(theRequestedUUID, kAudioServerPlugInDriverInterfaceUUID))
    {
        pthread_mutex_lock(&gPlugIn_StateMutex);
        ++gPlugIn_RefCount;
        pthread_mutex_unlock(&gPlugIn_StateMutex);
        *outInterface = gAudioServerPlugInDriverRef;
        theAnswer = 0;
    }

    CFRelease(theRequestedUUID);
    return theAnswer;
}

static ULONG ToneSphere_AddRef(void* inDriver)
{
    if(inDriver != gAudioServerPlugInDriverRef)
    {
        return 0;
    }

    pthread_mutex_lock(&gPlugIn_StateMutex);
    if(gPlugIn_RefCount < UINT32_MAX)
    {
        ++gPlugIn_RefCount;
    }
    ULONG theAnswer = gPlugIn_RefCount;
    pthread_mutex_unlock(&gPlugIn_StateMutex);

    return theAnswer;
}

static ULONG ToneSphere_Release(void* inDriver)
{
    if(inDriver != gAudioServerPlugInDriverRef)
    {
        return 0;
    }

    pthread_mutex_lock(&gPlugIn_StateMutex);
    if(gPlugIn_RefCount > 0)
    {
        --gPlugIn_RefCount;
    }
    ULONG theAnswer = gPlugIn_RefCount;
    pthread_mutex_unlock(&gPlugIn_StateMutex);

    return theAnswer;
}

#pragma mark - Lifecycle

static void ToneSphere_RecalculateHostTicksPerFrame(void)
{
    struct mach_timebase_info theTimeBaseInfo;
    mach_timebase_info(&theTimeBaseInfo);

    Float64 theHostClockFrequency = ((Float64)theTimeBaseInfo.denom / (Float64)theTimeBaseInfo.numer)
                                    * 1000000000.0;
    gDevice_HostTicksPerFrame = theHostClockFrequency / gDevice_SampleRate;
}

static OSStatus ToneSphere_Initialize(AudioServerPlugInDriverRef inDriver,
                                      AudioServerPlugInHostRef inHost)
{
    if(inDriver != gAudioServerPlugInDriverRef)
    {
        return kAudioHardwareBadObjectError;
    }

    gPlugIn_Host = inHost;
    ToneSphere_RecalculateHostTicksPerFrame();

    return 0;
}

/*
    This plug-in publishes one fixed device rather than letting the HAL create devices on
    demand (that is what a "box" is for), so both of these are honest refusals rather than
    stubs that return success and do nothing.
*/
static OSStatus ToneSphere_CreateDevice(AudioServerPlugInDriverRef inDriver,
                                        CFDictionaryRef inDescription,
                                        const AudioServerPlugInClientInfo* inClientInfo,
                                        AudioObjectID* outDeviceObjectID)
{
    #pragma unused(inDriver, inDescription, inClientInfo, outDeviceObjectID)
    return kAudioHardwareUnsupportedOperationError;
}

static OSStatus ToneSphere_DestroyDevice(AudioServerPlugInDriverRef inDriver,
                                         AudioObjectID inDeviceObjectID)
{
    #pragma unused(inDriver, inDeviceObjectID)
    return kAudioHardwareUnsupportedOperationError;
}

static OSStatus ToneSphere_AddDeviceClient(AudioServerPlugInDriverRef inDriver,
                                           AudioObjectID inDeviceObjectID,
                                           const AudioServerPlugInClientInfo* inClientInfo)
{
    #pragma unused(inClientInfo)
    if(inDriver != gAudioServerPlugInDriverRef)
    {
        return kAudioHardwareBadObjectError;
    }
    if(inDeviceObjectID != kObjectID_Device)
    {
        return kAudioHardwareBadObjectError;
    }
    return 0;
}

static OSStatus ToneSphere_RemoveDeviceClient(AudioServerPlugInDriverRef inDriver,
                                              AudioObjectID inDeviceObjectID,
                                              const AudioServerPlugInClientInfo* inClientInfo)
{
    #pragma unused(inClientInfo)
    if(inDriver != gAudioServerPlugInDriverRef)
    {
        return kAudioHardwareBadObjectError;
    }
    if(inDeviceObjectID != kObjectID_Device)
    {
        return kAudioHardwareBadObjectError;
    }
    return 0;
}

/*
    The only configuration change this device has is its sample rate, and the HAL hands it
    back here as the change action after SetPropertyData asked for it. Applying it means
    re-deriving the host-ticks-per-frame the zero timestamps are paced by; forgetting that
    is how a driver ends up running at the right nominal rate and the wrong real one.
*/
static OSStatus ToneSphere_PerformDeviceConfigurationChange(AudioServerPlugInDriverRef inDriver,
                                                            AudioObjectID inDeviceObjectID,
                                                            UInt64 inChangeAction,
                                                            void* inChangeInfo)
{
    #pragma unused(inChangeInfo)
    if(inDriver != gAudioServerPlugInDriverRef)
    {
        return kAudioHardwareBadObjectError;
    }
    if(inDeviceObjectID != kObjectID_Device)
    {
        return kAudioHardwareBadObjectError;
    }

    Boolean theRateIsSupported = false;
    for(UInt32 i = 0; i < kSupportedSampleRateCount; ++i)
    {
        if((Float64)inChangeAction == kSupportedSampleRates[i])
        {
            theRateIsSupported = true;
            break;
        }
    }
    if(!theRateIsSupported)
    {
        return kAudioHardwareBadObjectError;
    }

    pthread_mutex_lock(&gPlugIn_StateMutex);
    gDevice_SampleRate = (Float64)inChangeAction;
    ToneSphere_RecalculateHostTicksPerFrame();
    pthread_mutex_unlock(&gPlugIn_StateMutex);

    pthread_mutex_lock(&gDevice_IOMutex);
    gDevice_NumberTimeStamps = 0;
    gDevice_AnchorHostTime = mach_absolute_time();
    memset(gDevice_RingBuffer, 0, sizeof(gDevice_RingBuffer));
    pthread_mutex_unlock(&gDevice_IOMutex);

    return 0;
}

static OSStatus ToneSphere_AbortDeviceConfigurationChange(AudioServerPlugInDriverRef inDriver,
                                                          AudioObjectID inDeviceObjectID,
                                                          UInt64 inChangeAction,
                                                          void* inChangeInfo)
{
    #pragma unused(inChangeAction, inChangeInfo)
    if(inDriver != gAudioServerPlugInDriverRef)
    {
        return kAudioHardwareBadObjectError;
    }
    if(inDeviceObjectID != kObjectID_Device)
    {
        return kAudioHardwareBadObjectError;
    }
    return 0;
}

#pragma mark - Property helpers

static void ToneSphere_GetStreamFormat(AudioStreamBasicDescription* outFormat)
{
    pthread_mutex_lock(&gPlugIn_StateMutex);
    Float64 theRate = gDevice_SampleRate;
    pthread_mutex_unlock(&gPlugIn_StateMutex);

    memset(outFormat, 0, sizeof(AudioStreamBasicDescription));
    outFormat->mSampleRate = theRate;
    outFormat->mFormatID = kAudioFormatLinearPCM;
    outFormat->mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagsNativeEndian |
                              kAudioFormatFlagIsPacked;
    outFormat->mBytesPerPacket = sizeof(Float32) * kDevice_Channels;
    outFormat->mFramesPerPacket = 1;
    outFormat->mBytesPerFrame = sizeof(Float32) * kDevice_Channels;
    outFormat->mChannelsPerFrame = kDevice_Channels;
    outFormat->mBitsPerChannel = 32;
}

static UInt32 ToneSphere_StreamsInScope(AudioObjectPropertyScope inScope, AudioObjectID* outIDs)
{
    UInt32 theCount = 0;
    if((inScope == kAudioObjectPropertyScopeGlobal) || (inScope == kAudioObjectPropertyScopeInput))
    {
        outIDs[theCount++] = kObjectID_Stream_Input;
    }
    if((inScope == kAudioObjectPropertyScopeGlobal) || (inScope == kAudioObjectPropertyScopeOutput))
    {
        outIDs[theCount++] = kObjectID_Stream_Output;
    }
    return theCount;
}

#pragma mark - Property sizes

static OSStatus ToneSphere_GetPlugInPropertyDataSize(const AudioObjectPropertyAddress* inAddress,
                                                     UInt32* outDataSize)
{
    switch(inAddress->mSelector)
    {
        case kAudioObjectPropertyBaseClass:
        case kAudioObjectPropertyClass:
            *outDataSize = sizeof(AudioClassID);
            return 0;

        case kAudioObjectPropertyOwner:
            *outDataSize = sizeof(AudioObjectID);
            return 0;

        case kAudioObjectPropertyManufacturer:
        case kAudioPlugInPropertyResourceBundle:
            *outDataSize = sizeof(CFStringRef);
            return 0;

        case kAudioObjectPropertyOwnedObjects:
        case kAudioPlugInPropertyDeviceList:
            *outDataSize = sizeof(AudioObjectID);
            return 0;

        case kAudioPlugInPropertyTranslateUIDToDevice:
            *outDataSize = sizeof(AudioObjectID);
            return 0;

        case kAudioObjectPropertyCustomPropertyInfoList:
            *outDataSize = 0;
            return 0;

        default:
            return kAudioHardwareUnknownPropertyError;
    }
}

static OSStatus ToneSphere_GetDevicePropertyDataSize(const AudioObjectPropertyAddress* inAddress,
                                                     UInt32* outDataSize)
{
    AudioObjectID theStreams[2];

    switch(inAddress->mSelector)
    {
        case kAudioObjectPropertyBaseClass:
        case kAudioObjectPropertyClass:
            *outDataSize = sizeof(AudioClassID);
            return 0;

        case kAudioObjectPropertyOwner:
            *outDataSize = sizeof(AudioObjectID);
            return 0;

        case kAudioObjectPropertyName:
        case kAudioObjectPropertyManufacturer:
        case kAudioDevicePropertyDeviceUID:
        case kAudioDevicePropertyModelUID:
            *outDataSize = sizeof(CFStringRef);
            return 0;

        case kAudioDevicePropertyTransportType:
        case kAudioDevicePropertyClockDomain:
        case kAudioDevicePropertyDeviceIsAlive:
        case kAudioDevicePropertyDeviceIsRunning:
        case kAudioDevicePropertyDeviceCanBeDefaultDevice:
        case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice:
        case kAudioDevicePropertyLatency:
        case kAudioDevicePropertySafetyOffset:
        case kAudioDevicePropertyZeroTimeStampPeriod:
        case kAudioDevicePropertyIsHidden:
            *outDataSize = sizeof(UInt32);
            return 0;

        case kAudioObjectPropertyOwnedObjects:
        case kAudioDevicePropertyStreams:
            *outDataSize = ToneSphere_StreamsInScope(inAddress->mScope, theStreams) *
                           (UInt32)sizeof(AudioObjectID);
            return 0;

        case kAudioObjectPropertyControlList:
            *outDataSize = 0;
            return 0;

        case kAudioDevicePropertyRelatedDevices:
            *outDataSize = sizeof(AudioObjectID);
            return 0;

        case kAudioDevicePropertyNominalSampleRate:
            *outDataSize = sizeof(Float64);
            return 0;

        case kAudioDevicePropertyAvailableNominalSampleRates:
            *outDataSize = (UInt32)(kSupportedSampleRateCount * sizeof(AudioValueRange));
            return 0;

        case kAudioDevicePropertyPreferredChannelsForStereo:
            *outDataSize = 2 * sizeof(UInt32);
            return 0;

        default:
            return kAudioHardwareUnknownPropertyError;
    }
}

static OSStatus ToneSphere_GetStreamPropertyDataSize(const AudioObjectPropertyAddress* inAddress,
                                                     UInt32* outDataSize)
{
    switch(inAddress->mSelector)
    {
        case kAudioObjectPropertyBaseClass:
        case kAudioObjectPropertyClass:
            *outDataSize = sizeof(AudioClassID);
            return 0;

        case kAudioObjectPropertyOwner:
            *outDataSize = sizeof(AudioObjectID);
            return 0;

        case kAudioStreamPropertyIsActive:
        case kAudioStreamPropertyDirection:
        case kAudioStreamPropertyTerminalType:
        case kAudioStreamPropertyStartingChannel:
        case kAudioStreamPropertyLatency:
            *outDataSize = sizeof(UInt32);
            return 0;

        case kAudioStreamPropertyVirtualFormat:
        case kAudioStreamPropertyPhysicalFormat:
            *outDataSize = sizeof(AudioStreamBasicDescription);
            return 0;

        case kAudioStreamPropertyAvailableVirtualFormats:
        case kAudioStreamPropertyAvailablePhysicalFormats:
            *outDataSize = (UInt32)(kSupportedSampleRateCount * sizeof(AudioStreamRangedDescription));
            return 0;

        default:
            return kAudioHardwareUnknownPropertyError;
    }
}

#pragma mark - Property data

static OSStatus ToneSphere_GetPlugInPropertyData(const AudioObjectPropertyAddress* inAddress,
                                                 UInt32 inQualifierDataSize,
                                                 const void* inQualifierData,
                                                 UInt32 inDataSize,
                                                 UInt32* outDataSize,
                                                 void* outData)
{
    switch(inAddress->mSelector)
    {
        case kAudioObjectPropertyBaseClass:
            if(inDataSize < sizeof(AudioClassID)) { return kAudioHardwareBadPropertySizeError; }
            *((AudioClassID*)outData) = kAudioObjectClassID;
            *outDataSize = sizeof(AudioClassID);
            return 0;

        case kAudioObjectPropertyClass:
            if(inDataSize < sizeof(AudioClassID)) { return kAudioHardwareBadPropertySizeError; }
            *((AudioClassID*)outData) = kAudioPlugInClassID;
            *outDataSize = sizeof(AudioClassID);
            return 0;

        case kAudioObjectPropertyOwner:
            if(inDataSize < sizeof(AudioObjectID)) { return kAudioHardwareBadPropertySizeError; }
            *((AudioObjectID*)outData) = kAudioObjectUnknown;
            *outDataSize = sizeof(AudioObjectID);
            return 0;

        case kAudioObjectPropertyManufacturer:
            if(inDataSize < sizeof(CFStringRef)) { return kAudioHardwareBadPropertySizeError; }
            *((CFStringRef*)outData) = CFSTR(kManufacturer_Name);
            *outDataSize = sizeof(CFStringRef);
            return 0;

        case kAudioObjectPropertyOwnedObjects:
        case kAudioPlugInPropertyDeviceList:
            if(inDataSize < sizeof(AudioObjectID))
            {
                *outDataSize = 0;
                return 0;
            }
            *((AudioObjectID*)outData) = kObjectID_Device;
            *outDataSize = sizeof(AudioObjectID);
            return 0;

        case kAudioPlugInPropertyTranslateUIDToDevice:
        {
            if(inQualifierDataSize != sizeof(CFStringRef)) { return kAudioHardwareBadPropertySizeError; }
            if(inDataSize < sizeof(AudioObjectID)) { return kAudioHardwareBadPropertySizeError; }

            CFStringRef theUID = *((const CFStringRef*)inQualifierData);
            *((AudioObjectID*)outData) = CFStringCompare(theUID, CFSTR(kDevice_UID), 0) ==
                                         kCFCompareEqualTo ? kObjectID_Device : kAudioObjectUnknown;
            *outDataSize = sizeof(AudioObjectID);
            return 0;
        }

        case kAudioPlugInPropertyResourceBundle:
            if(inDataSize < sizeof(CFStringRef)) { return kAudioHardwareBadPropertySizeError; }
            *((CFStringRef*)outData) = CFSTR("");
            *outDataSize = sizeof(CFStringRef);
            return 0;

        case kAudioObjectPropertyCustomPropertyInfoList:
            *outDataSize = 0;
            return 0;

        default:
            return kAudioHardwareUnknownPropertyError;
    }
}

static OSStatus ToneSphere_GetDevicePropertyData(const AudioObjectPropertyAddress* inAddress,
                                                 UInt32 inDataSize,
                                                 UInt32* outDataSize,
                                                 void* outData)
{
    AudioObjectID theStreams[2];
    UInt32 theStreamCount;
    UInt32 theCapacity;
    UInt32 theWritten;

    switch(inAddress->mSelector)
    {
        case kAudioObjectPropertyBaseClass:
            if(inDataSize < sizeof(AudioClassID)) { return kAudioHardwareBadPropertySizeError; }
            *((AudioClassID*)outData) = kAudioObjectClassID;
            *outDataSize = sizeof(AudioClassID);
            return 0;

        case kAudioObjectPropertyClass:
            if(inDataSize < sizeof(AudioClassID)) { return kAudioHardwareBadPropertySizeError; }
            *((AudioClassID*)outData) = kAudioDeviceClassID;
            *outDataSize = sizeof(AudioClassID);
            return 0;

        case kAudioObjectPropertyOwner:
            if(inDataSize < sizeof(AudioObjectID)) { return kAudioHardwareBadPropertySizeError; }
            *((AudioObjectID*)outData) = kObjectID_PlugIn;
            *outDataSize = sizeof(AudioObjectID);
            return 0;

        case kAudioObjectPropertyName:
            if(inDataSize < sizeof(CFStringRef)) { return kAudioHardwareBadPropertySizeError; }
            *((CFStringRef*)outData) = CFSTR(kDevice_Name);
            *outDataSize = sizeof(CFStringRef);
            return 0;

        case kAudioObjectPropertyManufacturer:
            if(inDataSize < sizeof(CFStringRef)) { return kAudioHardwareBadPropertySizeError; }
            *((CFStringRef*)outData) = CFSTR(kManufacturer_Name);
            *outDataSize = sizeof(CFStringRef);
            return 0;

        case kAudioDevicePropertyDeviceUID:
            if(inDataSize < sizeof(CFStringRef)) { return kAudioHardwareBadPropertySizeError; }
            *((CFStringRef*)outData) = CFSTR(kDevice_UID);
            *outDataSize = sizeof(CFStringRef);
            return 0;

        case kAudioDevicePropertyModelUID:
            if(inDataSize < sizeof(CFStringRef)) { return kAudioHardwareBadPropertySizeError; }
            *((CFStringRef*)outData) = CFSTR(kDevice_ModelUID);
            *outDataSize = sizeof(CFStringRef);
            return 0;

        case kAudioDevicePropertyTransportType:
            if(inDataSize < sizeof(UInt32)) { return kAudioHardwareBadPropertySizeError; }
            *((UInt32*)outData) = kAudioDeviceTransportTypeVirtual;
            *outDataSize = sizeof(UInt32);
            return 0;

        case kAudioDevicePropertyRelatedDevices:
            if(inDataSize < sizeof(AudioObjectID))
            {
                *outDataSize = 0;
                return 0;
            }
            *((AudioObjectID*)outData) = kObjectID_Device;
            *outDataSize = sizeof(AudioObjectID);
            return 0;

        case kAudioDevicePropertyClockDomain:
        case kAudioDevicePropertyLatency:
        case kAudioDevicePropertySafetyOffset:
        case kAudioDevicePropertyIsHidden:
            if(inDataSize < sizeof(UInt32)) { return kAudioHardwareBadPropertySizeError; }
            *((UInt32*)outData) = 0;
            *outDataSize = sizeof(UInt32);
            return 0;

        case kAudioDevicePropertyDeviceIsAlive:
        case kAudioDevicePropertyDeviceCanBeDefaultDevice:
        case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice:
            if(inDataSize < sizeof(UInt32)) { return kAudioHardwareBadPropertySizeError; }
            *((UInt32*)outData) = 1;
            *outDataSize = sizeof(UInt32);
            return 0;

        case kAudioDevicePropertyDeviceIsRunning:
            if(inDataSize < sizeof(UInt32)) { return kAudioHardwareBadPropertySizeError; }
            pthread_mutex_lock(&gDevice_IOMutex);
            *((UInt32*)outData) = (gDevice_IOClientCount > 0) ? 1 : 0;
            pthread_mutex_unlock(&gDevice_IOMutex);
            *outDataSize = sizeof(UInt32);
            return 0;

        case kAudioObjectPropertyOwnedObjects:
        case kAudioDevicePropertyStreams:
            theStreamCount = ToneSphere_StreamsInScope(inAddress->mScope, theStreams);
            theCapacity = inDataSize / (UInt32)sizeof(AudioObjectID);
            theWritten = (theStreamCount < theCapacity) ? theStreamCount : theCapacity;
            for(UInt32 i = 0; i < theWritten; ++i)
            {
                ((AudioObjectID*)outData)[i] = theStreams[i];
            }
            *outDataSize = theWritten * (UInt32)sizeof(AudioObjectID);
            return 0;

        case kAudioObjectPropertyControlList:
            *outDataSize = 0;
            return 0;

        case kAudioDevicePropertyNominalSampleRate:
            if(inDataSize < sizeof(Float64)) { return kAudioHardwareBadPropertySizeError; }
            pthread_mutex_lock(&gPlugIn_StateMutex);
            *((Float64*)outData) = gDevice_SampleRate;
            pthread_mutex_unlock(&gPlugIn_StateMutex);
            *outDataSize = sizeof(Float64);
            return 0;

        case kAudioDevicePropertyAvailableNominalSampleRates:
            theCapacity = inDataSize / (UInt32)sizeof(AudioValueRange);
            theWritten = (kSupportedSampleRateCount < theCapacity) ?
                            (UInt32)kSupportedSampleRateCount : theCapacity;
            for(UInt32 i = 0; i < theWritten; ++i)
            {
                ((AudioValueRange*)outData)[i].mMinimum = kSupportedSampleRates[i];
                ((AudioValueRange*)outData)[i].mMaximum = kSupportedSampleRates[i];
            }
            *outDataSize = theWritten * (UInt32)sizeof(AudioValueRange);
            return 0;

        case kAudioDevicePropertyPreferredChannelsForStereo:
            if(inDataSize < (2 * sizeof(UInt32))) { return kAudioHardwareBadPropertySizeError; }
            ((UInt32*)outData)[0] = 1;
            ((UInt32*)outData)[1] = 2;
            *outDataSize = 2 * sizeof(UInt32);
            return 0;

        case kAudioDevicePropertyZeroTimeStampPeriod:
            if(inDataSize < sizeof(UInt32)) { return kAudioHardwareBadPropertySizeError; }
            *((UInt32*)outData) = kDevice_RingBufferFrames;
            *outDataSize = sizeof(UInt32);
            return 0;

        default:
            return kAudioHardwareUnknownPropertyError;
    }
}

static OSStatus ToneSphere_GetStreamPropertyData(AudioObjectID inObjectID,
                                                 const AudioObjectPropertyAddress* inAddress,
                                                 UInt32 inDataSize,
                                                 UInt32* outDataSize,
                                                 void* outData)
{
    const Boolean theStreamIsInput = (inObjectID == kObjectID_Stream_Input);
    UInt32 theCapacity;
    UInt32 theWritten;

    switch(inAddress->mSelector)
    {
        case kAudioObjectPropertyBaseClass:
            if(inDataSize < sizeof(AudioClassID)) { return kAudioHardwareBadPropertySizeError; }
            *((AudioClassID*)outData) = kAudioObjectClassID;
            *outDataSize = sizeof(AudioClassID);
            return 0;

        case kAudioObjectPropertyClass:
            if(inDataSize < sizeof(AudioClassID)) { return kAudioHardwareBadPropertySizeError; }
            *((AudioClassID*)outData) = kAudioStreamClassID;
            *outDataSize = sizeof(AudioClassID);
            return 0;

        case kAudioObjectPropertyOwner:
            if(inDataSize < sizeof(AudioObjectID)) { return kAudioHardwareBadPropertySizeError; }
            *((AudioObjectID*)outData) = kObjectID_Device;
            *outDataSize = sizeof(AudioObjectID);
            return 0;

        case kAudioStreamPropertyIsActive:
            if(inDataSize < sizeof(UInt32)) { return kAudioHardwareBadPropertySizeError; }
            *((UInt32*)outData) = 1;
            *outDataSize = sizeof(UInt32);
            return 0;

        case kAudioStreamPropertyDirection:
            if(inDataSize < sizeof(UInt32)) { return kAudioHardwareBadPropertySizeError; }
            *((UInt32*)outData) = theStreamIsInput ? 1 : 0;
            *outDataSize = sizeof(UInt32);
            return 0;

        case kAudioStreamPropertyTerminalType:
            if(inDataSize < sizeof(UInt32)) { return kAudioHardwareBadPropertySizeError; }
            *((UInt32*)outData) = theStreamIsInput ? kAudioStreamTerminalTypeMicrophone
                                                   : kAudioStreamTerminalTypeSpeaker;
            *outDataSize = sizeof(UInt32);
            return 0;

        case kAudioStreamPropertyStartingChannel:
            if(inDataSize < sizeof(UInt32)) { return kAudioHardwareBadPropertySizeError; }
            *((UInt32*)outData) = 1;
            *outDataSize = sizeof(UInt32);
            return 0;

        case kAudioStreamPropertyLatency:
            if(inDataSize < sizeof(UInt32)) { return kAudioHardwareBadPropertySizeError; }
            *((UInt32*)outData) = 0;
            *outDataSize = sizeof(UInt32);
            return 0;

        case kAudioStreamPropertyVirtualFormat:
        case kAudioStreamPropertyPhysicalFormat:
            if(inDataSize < sizeof(AudioStreamBasicDescription))
            {
                return kAudioHardwareBadPropertySizeError;
            }
            ToneSphere_GetStreamFormat((AudioStreamBasicDescription*)outData);
            *outDataSize = sizeof(AudioStreamBasicDescription);
            return 0;

        case kAudioStreamPropertyAvailableVirtualFormats:
        case kAudioStreamPropertyAvailablePhysicalFormats:
            theCapacity = inDataSize / (UInt32)sizeof(AudioStreamRangedDescription);
            theWritten = (kSupportedSampleRateCount < theCapacity) ?
                            (UInt32)kSupportedSampleRateCount : theCapacity;
            for(UInt32 i = 0; i < theWritten; ++i)
            {
                AudioStreamRangedDescription* theEntry =
                    &(((AudioStreamRangedDescription*)outData)[i]);
                memset(theEntry, 0, sizeof(AudioStreamRangedDescription));
                theEntry->mFormat.mSampleRate = kSupportedSampleRates[i];
                theEntry->mFormat.mFormatID = kAudioFormatLinearPCM;
                theEntry->mFormat.mFormatFlags = kAudioFormatFlagIsFloat |
                                                 kAudioFormatFlagsNativeEndian |
                                                 kAudioFormatFlagIsPacked;
                theEntry->mFormat.mBytesPerPacket = sizeof(Float32) * kDevice_Channels;
                theEntry->mFormat.mFramesPerPacket = 1;
                theEntry->mFormat.mBytesPerFrame = sizeof(Float32) * kDevice_Channels;
                theEntry->mFormat.mChannelsPerFrame = kDevice_Channels;
                theEntry->mFormat.mBitsPerChannel = 32;
                theEntry->mSampleRateRange.mMinimum = kSupportedSampleRates[i];
                theEntry->mSampleRateRange.mMaximum = kSupportedSampleRates[i];
            }
            *outDataSize = theWritten * (UInt32)sizeof(AudioStreamRangedDescription);
            return 0;

        default:
            return kAudioHardwareUnknownPropertyError;
    }
}

#pragma mark - Property entry points

static OSStatus ToneSphere_GetPropertyDataSize(AudioServerPlugInDriverRef inDriver,
                                               AudioObjectID inObjectID,
                                               pid_t inClientProcessID,
                                               const AudioObjectPropertyAddress* inAddress,
                                               UInt32 inQualifierDataSize,
                                               const void* inQualifierData,
                                               UInt32* outDataSize)
{
    #pragma unused(inClientProcessID, inQualifierDataSize, inQualifierData)
    if(inDriver != gAudioServerPlugInDriverRef) { return kAudioHardwareBadObjectError; }
    if((inAddress == NULL) || (outDataSize == NULL)) { return kAudioHardwareIllegalOperationError; }

    switch(inObjectID)
    {
        case kObjectID_PlugIn:
            return ToneSphere_GetPlugInPropertyDataSize(inAddress, outDataSize);
        case kObjectID_Device:
            return ToneSphere_GetDevicePropertyDataSize(inAddress, outDataSize);
        case kObjectID_Stream_Input:
        case kObjectID_Stream_Output:
            return ToneSphere_GetStreamPropertyDataSize(inAddress, outDataSize);
        default:
            return kAudioHardwareBadObjectError;
    }
}

/*
    Answered by asking GetPropertyDataSize whether the property exists at all, rather than
    by a second parallel switch statement. Two switches over the same selector list is how
    a plug-in ends up claiming a property it cannot then produce — the HAL asks HasProperty
    first and treats a later "unknown property" as a driver bug.
*/
static Boolean ToneSphere_HasProperty(AudioServerPlugInDriverRef inDriver,
                                      AudioObjectID inObjectID,
                                      pid_t inClientProcessID,
                                      const AudioObjectPropertyAddress* inAddress)
{
    UInt32 theSize = 0;
    OSStatus theError = ToneSphere_GetPropertyDataSize(inDriver, inObjectID, inClientProcessID,
                                                       inAddress, 0, NULL, &theSize);
    return theError == 0;
}

static OSStatus ToneSphere_IsPropertySettable(AudioServerPlugInDriverRef inDriver,
                                              AudioObjectID inObjectID,
                                              pid_t inClientProcessID,
                                              const AudioObjectPropertyAddress* inAddress,
                                              Boolean* outIsSettable)
{
    UInt32 theSize = 0;
    OSStatus theError = ToneSphere_GetPropertyDataSize(inDriver, inObjectID, inClientProcessID,
                                                       inAddress, 0, NULL, &theSize);
    if(theError != 0) { return theError; }
    if(outIsSettable == NULL) { return kAudioHardwareIllegalOperationError; }

    switch(inAddress->mSelector)
    {
        case kAudioDevicePropertyNominalSampleRate:
            *outIsSettable = (inObjectID == kObjectID_Device);
            return 0;

        case kAudioStreamPropertyVirtualFormat:
        case kAudioStreamPropertyPhysicalFormat:
            *outIsSettable = (inObjectID == kObjectID_Stream_Input) ||
                             (inObjectID == kObjectID_Stream_Output);
            return 0;

        default:
            *outIsSettable = false;
            return 0;
    }
}

static OSStatus ToneSphere_GetPropertyData(AudioServerPlugInDriverRef inDriver,
                                           AudioObjectID inObjectID,
                                           pid_t inClientProcessID,
                                           const AudioObjectPropertyAddress* inAddress,
                                           UInt32 inQualifierDataSize,
                                           const void* inQualifierData,
                                           UInt32 inDataSize,
                                           UInt32* outDataSize,
                                           void* outData)
{
    #pragma unused(inClientProcessID)
    if(inDriver != gAudioServerPlugInDriverRef) { return kAudioHardwareBadObjectError; }
    if((inAddress == NULL) || (outDataSize == NULL) || (outData == NULL))
    {
        return kAudioHardwareIllegalOperationError;
    }

    switch(inObjectID)
    {
        case kObjectID_PlugIn:
            return ToneSphere_GetPlugInPropertyData(inAddress, inQualifierDataSize,
                                                    inQualifierData, inDataSize,
                                                    outDataSize, outData);
        case kObjectID_Device:
            return ToneSphere_GetDevicePropertyData(inAddress, inDataSize, outDataSize, outData);
        case kObjectID_Stream_Input:
        case kObjectID_Stream_Output:
            return ToneSphere_GetStreamPropertyData(inObjectID, inAddress, inDataSize,
                                                    outDataSize, outData);
        default:
            return kAudioHardwareBadObjectError;
    }
}

/*
    A sample-rate change cannot be applied here: the HAL has to be told first so it can
    stop IO, and it then calls back into PerformDeviceConfigurationChange. Writing
    gDevice_SampleRate directly from this function would change the rate underneath a
    running stream, which is the classic way a virtual device starts producing garbage.
*/
static OSStatus ToneSphere_SetPropertyData(AudioServerPlugInDriverRef inDriver,
                                           AudioObjectID inObjectID,
                                           pid_t inClientProcessID,
                                           const AudioObjectPropertyAddress* inAddress,
                                           UInt32 inQualifierDataSize,
                                           const void* inQualifierData,
                                           UInt32 inDataSize,
                                           const void* inData)
{
    #pragma unused(inClientProcessID, inQualifierDataSize, inQualifierData)
    if(inDriver != gAudioServerPlugInDriverRef) { return kAudioHardwareBadObjectError; }
    if((inAddress == NULL) || (inData == NULL)) { return kAudioHardwareIllegalOperationError; }

    Float64 theRequestedRate = 0.0;

    switch(inAddress->mSelector)
    {
        case kAudioDevicePropertyNominalSampleRate:
            if(inObjectID != kObjectID_Device) { return kAudioHardwareBadObjectError; }
            if(inDataSize != sizeof(Float64)) { return kAudioHardwareBadPropertySizeError; }
            theRequestedRate = *((const Float64*)inData);
            break;

        case kAudioStreamPropertyVirtualFormat:
        case kAudioStreamPropertyPhysicalFormat:
        {
            if((inObjectID != kObjectID_Stream_Input) && (inObjectID != kObjectID_Stream_Output))
            {
                return kAudioHardwareBadObjectError;
            }
            if(inDataSize != sizeof(AudioStreamBasicDescription))
            {
                return kAudioHardwareBadPropertySizeError;
            }

            const AudioStreamBasicDescription* theFormat =
                (const AudioStreamBasicDescription*)inData;

            if((theFormat->mFormatID != kAudioFormatLinearPCM) ||
               ((theFormat->mFormatFlags & kAudioFormatFlagIsFloat) == 0) ||
               (theFormat->mChannelsPerFrame != kDevice_Channels) ||
               (theFormat->mBitsPerChannel != 32) ||
               (theFormat->mFramesPerPacket != 1))
            {
                return kAudioDeviceUnsupportedFormatError;
            }

            theRequestedRate = theFormat->mSampleRate;
            break;
        }

        default:
            return kAudioHardwareUnknownPropertyError;
    }

    Boolean theRateIsSupported = false;
    for(UInt32 i = 0; i < kSupportedSampleRateCount; ++i)
    {
        if(theRequestedRate == kSupportedSampleRates[i])
        {
            theRateIsSupported = true;
            break;
        }
    }
    if(!theRateIsSupported)
    {
        return (inAddress->mSelector == kAudioDevicePropertyNominalSampleRate)
                    ? kAudioHardwareIllegalOperationError
                    : kAudioDeviceUnsupportedFormatError;
    }

    pthread_mutex_lock(&gPlugIn_StateMutex);
    const Boolean theRateIsNew = (theRequestedRate != gDevice_SampleRate);
    pthread_mutex_unlock(&gPlugIn_StateMutex);

    // Outside the lock on purpose: the host is free to call straight back into
    // PerformDeviceConfigurationChange, which takes the same mutex.
    if(theRateIsNew && (gPlugIn_Host != NULL))
    {
        gPlugIn_Host->RequestDeviceConfigurationChange(gPlugIn_Host, kObjectID_Device,
                                                       (UInt64)theRequestedRate, NULL);
    }

    return 0;
}

#pragma mark - IO

static OSStatus ToneSphere_StartIO(AudioServerPlugInDriverRef inDriver,
                                   AudioObjectID inDeviceObjectID, UInt32 inClientID)
{
    #pragma unused(inClientID)
    if(inDriver != gAudioServerPlugInDriverRef) { return kAudioHardwareBadObjectError; }
    if(inDeviceObjectID != kObjectID_Device) { return kAudioHardwareBadObjectError; }

    pthread_mutex_lock(&gDevice_IOMutex);
    // Only the first client resets the clock and clears the loopback: doing it for every
    // client would drop whatever a client that started earlier is already playing.
    if(gDevice_IOClientCount == 0)
    {
        gDevice_NumberTimeStamps = 0;
        gDevice_AnchorHostTime = mach_absolute_time();
        memset(gDevice_RingBuffer, 0, sizeof(gDevice_RingBuffer));
    }
    ++gDevice_IOClientCount;
    pthread_mutex_unlock(&gDevice_IOMutex);

    return 0;
}

static OSStatus ToneSphere_StopIO(AudioServerPlugInDriverRef inDriver,
                                  AudioObjectID inDeviceObjectID, UInt32 inClientID)
{
    #pragma unused(inClientID)
    if(inDriver != gAudioServerPlugInDriverRef) { return kAudioHardwareBadObjectError; }
    if(inDeviceObjectID != kObjectID_Device) { return kAudioHardwareBadObjectError; }

    pthread_mutex_lock(&gDevice_IOMutex);
    if(gDevice_IOClientCount > 0)
    {
        --gDevice_IOClientCount;
    }
    pthread_mutex_unlock(&gDevice_IOMutex);

    return 0;
}

/*
    The device's clock. There is no hardware to read, so time is derived from
    mach_absolute_time(): one "tick" of the timeline is kDevice_RingBufferFrames frames
    long, and the sample time advances by exactly that much each time real time has passed
    the anchor by another buffer's worth of host ticks. This is the timing model from
    Apple's NullAudio sample; getting it wrong shows up as a device that runs fast, slow,
    or not at all, which is precisely what the CI round-trip test measures.
*/
static OSStatus ToneSphere_GetZeroTimeStamp(AudioServerPlugInDriverRef inDriver,
                                            AudioObjectID inDeviceObjectID,
                                            UInt32 inClientID,
                                            Float64* outSampleTime,
                                            UInt64* outHostTime,
                                            UInt64* outSeed)
{
    #pragma unused(inClientID)
    if(inDriver != gAudioServerPlugInDriverRef) { return kAudioHardwareBadObjectError; }
    if(inDeviceObjectID != kObjectID_Device) { return kAudioHardwareBadObjectError; }

    pthread_mutex_lock(&gDevice_IOMutex);

    const UInt64 theCurrentHostTime = mach_absolute_time();
    const Float64 theHostTicksPerRingBuffer = gDevice_HostTicksPerFrame *
                                              ((Float64)kDevice_RingBufferFrames);
    const Float64 theHostTickOffset = ((Float64)(gDevice_NumberTimeStamps + 1)) *
                                      theHostTicksPerRingBuffer;
    const UInt64 theNextHostTime = gDevice_AnchorHostTime + ((UInt64)theHostTickOffset);

    if(theNextHostTime <= theCurrentHostTime)
    {
        ++gDevice_NumberTimeStamps;
    }

    *outSampleTime = (Float64)(gDevice_NumberTimeStamps * kDevice_RingBufferFrames);
    *outHostTime = gDevice_AnchorHostTime +
                   (UInt64)(((Float64)gDevice_NumberTimeStamps) * theHostTicksPerRingBuffer);
    *outSeed = 1;

    pthread_mutex_unlock(&gDevice_IOMutex);

    return 0;
}

static OSStatus ToneSphere_WillDoIOOperation(AudioServerPlugInDriverRef inDriver,
                                             AudioObjectID inDeviceObjectID,
                                             UInt32 inClientID,
                                             UInt32 inOperationID,
                                             Boolean* outWillDo,
                                             Boolean* outWillDoInPlace)
{
    #pragma unused(inClientID)
    if(inDriver != gAudioServerPlugInDriverRef) { return kAudioHardwareBadObjectError; }
    if(inDeviceObjectID != kObjectID_Device) { return kAudioHardwareBadObjectError; }

    Boolean theWillDo = false;
    switch(inOperationID)
    {
        case kAudioServerPlugInIOOperationReadInput:
        case kAudioServerPlugInIOOperationWriteMix:
            theWillDo = true;
            break;
        default:
            theWillDo = false;
            break;
    }

    if(outWillDo != NULL) { *outWillDo = theWillDo; }
    if(outWillDoInPlace != NULL) { *outWillDoInPlace = true; }

    return 0;
}

static OSStatus ToneSphere_BeginIOOperation(AudioServerPlugInDriverRef inDriver,
                                            AudioObjectID inDeviceObjectID,
                                            UInt32 inClientID,
                                            UInt32 inOperationID,
                                            UInt32 inIOBufferFrameSize,
                                            const AudioServerPlugInIOCycleInfo* inIOCycleInfo)
{
    #pragma unused(inClientID, inOperationID, inIOBufferFrameSize, inIOCycleInfo)
    if(inDriver != gAudioServerPlugInDriverRef) { return kAudioHardwareBadObjectError; }
    if(inDeviceObjectID != kObjectID_Device) { return kAudioHardwareBadObjectError; }
    return 0;
}

/*
    The loopback itself.

    Output frames are mixed into the ring at their own sample time; input frames are read
    from the ring at theirs, and the slot is zeroed as it is read. Zeroing matters: without
    it the ring's contents would repeat every kDevice_RingBufferFrames frames forever once
    playback stopped. The cost of zeroing on read is that a second simultaneous *input*
    client gets silence for whatever the first one consumed — a real limitation, listed as
    such in native/coreaudio-plugin/README.md rather than left to be discovered.
*/
static OSStatus ToneSphere_DoIOOperation(AudioServerPlugInDriverRef inDriver,
                                         AudioObjectID inDeviceObjectID,
                                         AudioObjectID inStreamObjectID,
                                         UInt32 inClientID,
                                         UInt32 inOperationID,
                                         UInt32 inIOBufferFrameSize,
                                         const AudioServerPlugInIOCycleInfo* inIOCycleInfo,
                                         void* ioMainBuffer,
                                         void* ioSecondaryBuffer)
{
    #pragma unused(inStreamObjectID, inClientID, ioSecondaryBuffer)
    if(inDriver != gAudioServerPlugInDriverRef) { return kAudioHardwareBadObjectError; }
    if(inDeviceObjectID != kObjectID_Device) { return kAudioHardwareBadObjectError; }
    if((ioMainBuffer == NULL) || (inIOCycleInfo == NULL)) { return 0; }

    Float32* const theBuffer = (Float32*)ioMainBuffer;

    if(inOperationID == kAudioServerPlugInIOOperationWriteMix)
    {
        const SInt64 theBase = (SInt64)inIOCycleInfo->mOutputTime.mSampleTime;
        for(UInt32 theFrame = 0; theFrame < inIOBufferFrameSize; ++theFrame)
        {
            const SInt64 theTime = theBase + (SInt64)theFrame;
            const SInt64 theSlot = ((theTime % kDevice_RingBufferFrames) +
                                    kDevice_RingBufferFrames) % kDevice_RingBufferFrames;
            for(UInt32 theChannel = 0; theChannel < kDevice_Channels; ++theChannel)
            {
                gDevice_RingBuffer[(theSlot * kDevice_Channels) + theChannel] +=
                    theBuffer[(theFrame * kDevice_Channels) + theChannel];
            }
        }
    }
    else if(inOperationID == kAudioServerPlugInIOOperationReadInput)
    {
        const SInt64 theBase = (SInt64)inIOCycleInfo->mInputTime.mSampleTime;
        for(UInt32 theFrame = 0; theFrame < inIOBufferFrameSize; ++theFrame)
        {
            const SInt64 theTime = theBase + (SInt64)theFrame;
            const SInt64 theSlot = ((theTime % kDevice_RingBufferFrames) +
                                    kDevice_RingBufferFrames) % kDevice_RingBufferFrames;
            for(UInt32 theChannel = 0; theChannel < kDevice_Channels; ++theChannel)
            {
                const SInt64 theIndex = (theSlot * kDevice_Channels) + theChannel;
                theBuffer[(theFrame * kDevice_Channels) + theChannel] =
                    gDevice_RingBuffer[theIndex];
                gDevice_RingBuffer[theIndex] = 0.0f;
            }
        }
    }

    return 0;
}

static OSStatus ToneSphere_EndIOOperation(AudioServerPlugInDriverRef inDriver,
                                          AudioObjectID inDeviceObjectID,
                                          UInt32 inClientID,
                                          UInt32 inOperationID,
                                          UInt32 inIOBufferFrameSize,
                                          const AudioServerPlugInIOCycleInfo* inIOCycleInfo)
{
    #pragma unused(inClientID, inOperationID, inIOBufferFrameSize, inIOCycleInfo)
    if(inDriver != gAudioServerPlugInDriverRef) { return kAudioHardwareBadObjectError; }
    if(inDeviceObjectID != kObjectID_Device) { return kAudioHardwareBadObjectError; }
    return 0;
}
