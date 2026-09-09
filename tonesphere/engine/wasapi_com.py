"""
The Windows COM plumbing per-process capture needs, and nothing else.

`ActivateAudioInterfaceAsync` with `VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK` is the only
way to get an `IAudioClient` bound to one process's output rather than to a device, and
PortAudio does not expose it. `docs/VIRTUAL_AUDIO_DRIVER.md` says that means a native
extension; it does not. The function is a plain `__stdcall` export of `mmdevapi.dll`, and
the one interface we have to *implement* rather than call — a completion handler with a
single method beyond `IUnknown` — is four vtable slots. So this is pure `ctypes`, adding
no dependency, matching `app_capture.py`'s existing choice not to pull in
pycaw/comtypes/pywin32.

Everything here is Windows-only, but the module still has to import on Linux and macOS
(`tests/test_imports.py`). `ctypes.wintypes`, `ctypes.WINFUNCTYPE` and `ctypes.HRESULT`
do not exist off Windows, so the structures below are built from plain ctypes primitives
and the Windows-only pieces are touched inside functions, which run nowhere else.

The GUIDs, struct layouts, vtable slot numbers and HRESULTs were read out of the SDK
headers on the machine this was written on (`mmdeviceapi.h`,
`audioclientactivationparams.h`, `Audioclient.h`, `AudioSessionTypes.h`, `mmreg.h`,
`propidlbase.h` under `Windows Kits\\10\\Include`), not recalled. A wrong slot number here
is not a wrong answer, it is a call through a function pointer that is not the function.
"""

import ctypes
import threading

HRESULT = ctypes.c_long
HANDLE = ctypes.c_void_p
REFERENCE_TIME = ctypes.c_longlong

S_OK = 0
S_FALSE = 1
E_NOINTERFACE = 0x80004002
E_FAIL = 0x80004005

# CoInitializeEx: the capture thread is the only thread touching these interfaces, so a
# multithreaded apartment costs nothing and avoids needing a message pump.
COINIT_MULTITHREADED = 0x0
RPC_E_CHANGED_MODE = 0x80010106

WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102

VT_BLOB = 65

VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK = "VAD\\Process_Loopback"

AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE = 0
PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE = 1

AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
AUDCLNT_STREAMFLAGS_EVENTCALLBACK = 0x00040000

AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY = 0x1
AUDCLNT_BUFFERFLAGS_SILENT = 0x2

AUDCLNT_S_BUFFER_EMPTY = 0x08890001

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STILL_ACTIVE = 259
ERROR_INVALID_PARAMETER = 87
ERROR_ACCESS_DENIED = 5

# Vtable slots, counting IUnknown's QueryInterface/AddRef/Release as 0/1/2.
SLOT_QUERY_INTERFACE = 0
SLOT_ADD_REF = 1
SLOT_RELEASE = 2

SLOT_ACTIVATE_COMPLETED = 3

SLOT_GET_ACTIVATE_RESULT = 3

SLOT_AUDIO_CLIENT_INITIALIZE = 3
SLOT_AUDIO_CLIENT_GET_BUFFER_SIZE = 4
SLOT_AUDIO_CLIENT_IS_FORMAT_SUPPORTED = 7
SLOT_AUDIO_CLIENT_GET_MIX_FORMAT = 8
SLOT_AUDIO_CLIENT_START = 10
SLOT_AUDIO_CLIENT_STOP = 11
SLOT_AUDIO_CLIENT_SET_EVENT_HANDLE = 13
SLOT_AUDIO_CLIENT_GET_SERVICE = 14

SLOT_CAPTURE_GET_BUFFER = 3
SLOT_CAPTURE_RELEASE_BUFFER = 4
SLOT_CAPTURE_GET_NEXT_PACKET_SIZE = 5


class GUID(ctypes.Structure):
    _fields_ = [
        ('Data1', ctypes.c_ulong),
        ('Data2', ctypes.c_ushort),
        ('Data3', ctypes.c_ushort),
        ('Data4', ctypes.c_ubyte * 8),
    ]

    def __repr__(self) -> str:
        tail = bytes(self.Data4).hex().upper()
        return (f"{{{self.Data1:08X}-{self.Data2:04X}-{self.Data3:04X}-"
                f"{tail[:4]}-{tail[4:]}}}")


def guid(text: str) -> GUID:
    """Parse the canonical 8-4-4-4-12 form used in the SDK headers."""
    parts = text.strip('{}').split('-')
    tail = bytes.fromhex(parts[3] + parts[4])
    return GUID(
        int(parts[0], 16), int(parts[1], 16), int(parts[2], 16),
        (ctypes.c_ubyte * 8)(*tail),
    )


IID_IUNKNOWN = guid("00000000-0000-0000-C000-000000000046")
# A marker interface with no methods of its own, meaning "safe to call from any
# apartment". mmdevapi asks the completion handler for it; see the class below.
IID_IAGILE_OBJECT = guid("94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90")
IID_IACTIVATE_AUDIO_INTERFACE_COMPLETION_HANDLER = guid("41D949AB-9862-444A-80F6-C261334DA5EB")
IID_IACTIVATE_AUDIO_INTERFACE_ASYNC_OPERATION = guid("72A22D78-CDE4-431D-B8CC-843A71199B6D")
IID_IAUDIO_CLIENT = guid("1CB9AD4C-DBFA-4C32-B178-C2F568A703B2")
IID_IAUDIO_CAPTURE_CLIENT = guid("C8ADBD64-E71E-48A0-A4DE-185C395CD317")


class BLOB(ctypes.Structure):
    _fields_ = [('cbSize', ctypes.c_ulong), ('pBlobData', ctypes.c_void_p)]


class PROPVARIANT(ctypes.Structure):
    """
    Only the `VT_BLOB` arm of the real union is declared.

    A PROPVARIANT's union is as wide as its widest member, but every arm starts at the
    same offset (8, after `vt` and three reserved words) and `BLOB`'s pointer alignment
    reproduces that here. Declaring the one arm we set is therefore laid out identically
    to the real thing for the field we touch, and nothing reads the others.
    """
    _fields_ = [
        ('vt', ctypes.c_ushort),
        ('wReserved1', ctypes.c_ushort),
        ('wReserved2', ctypes.c_ushort),
        ('wReserved3', ctypes.c_ushort),
        ('blob', BLOB),
    ]


class AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS(ctypes.Structure):
    _fields_ = [
        ('TargetProcessId', ctypes.c_ulong),
        ('ProcessLoopbackMode', ctypes.c_int),
    ]


class AUDIOCLIENT_ACTIVATION_PARAMS(ctypes.Structure):
    _fields_ = [
        ('ActivationType', ctypes.c_int),
        ('ProcessLoopbackParams', AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS),
    ]


class WAVEFORMATEX(ctypes.Structure):
    # mmreg.h is inside `pshpack1.h`, so this is byte-packed and 18 bytes, not the 20 that
    # natural alignment would give. Getting this wrong misreads every format the OS hands
    # back.
    _pack_ = 1
    _fields_ = [
        ('wFormatTag', ctypes.c_ushort),
        ('nChannels', ctypes.c_ushort),
        ('nSamplesPerSec', ctypes.c_ulong),
        ('nAvgBytesPerSec', ctypes.c_ulong),
        ('nBlockAlign', ctypes.c_ushort),
        ('wBitsPerSample', ctypes.c_ushort),
        ('cbSize', ctypes.c_ushort),
    ]


def float32_format(sample_rate: int, channels: int) -> WAVEFORMATEX:
    """The format we ask for: the engine's own rate, float32, no extension block."""
    block_align = 4 * channels
    return WAVEFORMATEX(
        wFormatTag=WAVE_FORMAT_IEEE_FLOAT,
        nChannels=channels,
        nSamplesPerSec=sample_rate,
        nAvgBytesPerSec=sample_rate * block_align,
        nBlockAlign=block_align,
        wBitsPerSample=32,
        cbSize=0,
    )


def describe_format(fmt: WAVEFORMATEX) -> str:
    tags = {
        WAVE_FORMAT_PCM: 'PCM',
        WAVE_FORMAT_IEEE_FLOAT: 'float',
        WAVE_FORMAT_EXTENSIBLE: 'extensible',
    }
    tag = tags.get(fmt.wFormatTag, f"tag 0x{fmt.wFormatTag:04X}")
    return (f"{fmt.nSamplesPerSec} Hz, {fmt.nChannels} ch, "
            f"{fmt.wBitsPerSample}-bit {tag}")


_HRESULT_NAMES = {
    0x88890001: "AUDCLNT_E_NOT_INITIALIZED",
    0x88890002: "AUDCLNT_E_ALREADY_INITIALIZED",
    0x88890003: "AUDCLNT_E_WRONG_ENDPOINT_TYPE",
    0x88890004: "AUDCLNT_E_DEVICE_INVALIDATED",
    0x88890005: "AUDCLNT_E_NOT_STOPPED",
    0x88890006: "AUDCLNT_E_BUFFER_TOO_LARGE",
    0x88890007: "AUDCLNT_E_OUT_OF_ORDER",
    0x88890008: "AUDCLNT_E_UNSUPPORTED_FORMAT",
    0x88890009: "AUDCLNT_E_INVALID_SIZE",
    0x8889000A: "AUDCLNT_E_DEVICE_IN_USE",
    0x8889000B: "AUDCLNT_E_BUFFER_OPERATION_PENDING",
    0x8889000C: "AUDCLNT_E_THREAD_NOT_REGISTERED",
    0x8889000E: "AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED",
    0x8889000F: "AUDCLNT_E_ENDPOINT_CREATE_FAILED",
    0x88890010: "AUDCLNT_E_SERVICE_NOT_RUNNING",
    0x88890011: "AUDCLNT_E_EVENTHANDLE_NOT_EXPECTED",
    0x88890012: "AUDCLNT_E_EXCLUSIVE_MODE_ONLY",
    0x88890013: "AUDCLNT_E_BUFDURATION_PERIOD_NOT_EQUAL",
    0x88890014: "AUDCLNT_E_EVENTHANDLE_NOT_SET",
    0x88890015: "AUDCLNT_E_INCORRECT_BUFFER_SIZE",
    0x88890016: "AUDCLNT_E_BUFFER_SIZE_ERROR",
    0x88890018: "AUDCLNT_E_BUFFER_ERROR",
    0x88890019: "AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED",
    0x88890020: "AUDCLNT_E_INVALID_DEVICE_PERIOD",
    0x88890021: "AUDCLNT_E_INVALID_STREAM_FLAG",
    0x88890026: "AUDCLNT_E_RESOURCES_INVALIDATED",
    0x88890027: "AUDCLNT_E_RAW_MODE_UNSUPPORTED",
    0x80004001: "E_NOTIMPL",
    0x80004002: "E_NOINTERFACE",
    0x80004003: "E_POINTER",
    0x80004005: "E_FAIL",
    0x8000000E: "E_ILLEGAL_METHOD_CALL",
    0x8000001C: "RO_E_MUST_BE_AGILE",
    0x8000FFFF: "E_UNEXPECTED",
    0x80070005: "E_ACCESSDENIED",
    0x80070006: "E_HANDLE",
    0x8007000E: "E_OUTOFMEMORY",
    0x80070057: "E_INVALIDARG",
    0x80010106: "RPC_E_CHANGED_MODE",
}

# The failures a user can actually do something about, in the style of
# `AudioHost._explain_open_failure`: name the cause, then the fix.
_HRESULT_HINTS = {
    0x88890004: ("the target process's audio endpoint went away — it exited, or its "
                 "output device was removed or disabled"),
    0x88890008: ("Windows refused the requested capture format. Process loopback only "
                 "offers shared-mode float32; a different rate or bit depth is not "
                 "negotiable here"),
    0x8889000A: "another application holds this endpoint exclusively",
    0x88890010: ("the Windows Audio service is not running — start 'Windows Audio' "
                 "(audiosrv) in services.msc"),
    0x80070005: ("Windows refused access to that process. DRM-protected playback and "
                 "protected processes cannot be captured, and this is a refusal by the "
                 "OS rather than a fault here"),
    0x80070057: ("Windows rejected the activation arguments — most often a process id "
                 "that no longer exists"),
    0x80004001: ("that method is not implemented for this object. Measured: a "
                 "process-loopback IAudioClient returns this for GetMixFormat and "
                 "IsFormatSupported, which is expected rather than a fault"),
    0x8000000E: ("mmdevapi rejected the activation request itself. Measured cause: the "
                 "completion handler did not answer QueryInterface for IAgileObject"),
}


def describe_hresult(hr: int) -> str:
    """
    Turn an HRESULT into something a user can act on.

    A bare `0x88890008` tells nobody anything. The distinction that matters most here is
    "Windows refused" (a protected process, a stopped audio service) versus "we asked for
    something wrong", because only the second is a bug in this code.
    """
    code = hr & 0xFFFFFFFF
    name = _HRESULT_NAMES.get(code)
    label = f"{name} (0x{code:08X})" if name else f"HRESULT 0x{code:08X}"

    hint = _HRESULT_HINTS.get(code)
    if hint:
        return f"{label} — {hint}"

    # FACILITY_WIN32 HRESULTs wrap a plain Win32 error the OS can describe itself.
    if (code >> 16) == 0x8007:
        try:
            return f"{label} — {ctypes.FormatError(code & 0xFFFF).strip()}"
        except (OSError, ValueError, AttributeError):
            return label

    return label


def failed(hr: int) -> bool:
    return (hr & 0x80000000) != 0


def bind(pointer, slot: int, restype, *argtypes):
    """
    Bind one vtable slot of a COM object to a callable.

    A COM interface pointer points at a pointer to its vtable, so the function address is
    `(*pointer)[slot]`. `WINFUNCTYPE` is `__stdcall`, which is what every COM method on
    32-bit Windows is; on x64 there is only one calling convention and it is the same
    thing. Referenced here rather than at module scope because it does not exist off
    Windows.
    """
    vtable = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0]
    entry = ctypes.cast(vtable, ctypes.POINTER(ctypes.c_void_p))[slot]
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return prototype(entry)


def query_interface(pointer, iid: GUID) -> int:
    """Return the requested interface pointer as an int, or 0 if it is not offered."""
    method = bind(pointer, SLOT_QUERY_INTERFACE, HRESULT,
                  ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))
    out = ctypes.c_void_p()
    hr = method(pointer, ctypes.byref(iid), ctypes.byref(out))
    if failed(hr) or not out:
        return 0
    return int(out.value)


def release(pointer) -> None:
    if pointer:
        bind(pointer, SLOT_RELEASE, ctypes.c_ulong)(pointer)


def mmdevapi():
    """`ActivateAudioInterfaceAsync` is a real export of mmdevapi.dll, not a COM class."""
    dll = ctypes.WinDLL('mmdevapi.dll')
    function = dll.ActivateAudioInterfaceAsync
    function.restype = HRESULT
    function.argtypes = [
        ctypes.c_wchar_p,               # deviceInterfacePath
        ctypes.POINTER(GUID),           # riid
        ctypes.POINTER(PROPVARIANT),    # activationParams
        ctypes.c_void_p,                # completionHandler
        ctypes.POINTER(ctypes.c_void_p),  # activationOperation
    ]
    return function


def process_loopback_propvariant(pid: int, include_process_tree: bool) -> tuple[PROPVARIANT, object]:
    """
    Wrap `AUDIOCLIENT_ACTIVATION_PARAMS` for a PID in the PROPVARIANT the API wants.

    The params struct is returned alongside the variant because the variant only holds a
    borrowed pointer to it; dropping it would leave the API reading freed memory.
    """
    params = AUDIOCLIENT_ACTIVATION_PARAMS(
        ActivationType=AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
        ProcessLoopbackParams=AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS(
            TargetProcessId=pid,
            ProcessLoopbackMode=(PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE
                                 if include_process_tree
                                 else PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE),
        ),
    )

    variant = PROPVARIANT()
    variant.vt = VT_BLOB
    variant.blob.cbSize = ctypes.sizeof(params)
    variant.blob.pBlobData = ctypes.cast(ctypes.byref(params), ctypes.c_void_p)

    return variant, params


class ActivationCompletionHandler:
    """
    A hand-rolled `IActivateAudioInterfaceCompletionHandler`.

    Four vtable slots: `IUnknown`'s three plus `ActivateCompleted`. The object is a single
    pointer to that vtable, which is all a COM interface pointer is.

    Two things here are load-bearing and easy to get wrong:

    * The `WINFUNCTYPE` instances must stay referenced for as long as Windows can call
      them. ctypes frees the trampoline with the object, and a freed trampoline called
      from an mmdevapi thread is a hard crash in unowned memory, so they are held on the
      instance.
    * `ActivateCompleted` may run synchronously inside `ActivateAudioInterfaceAsync` *or*
      later on an mmdevapi thread. Signalling an `Event` and having the caller wait on it
      afterwards is correct for both, where "check whether it already fired" is not.
    * `QueryInterface` must answer `IAgileObject`. Measured on this machine: mmdevapi asks
      the handler for it first, and refusing makes the whole activation fail with
      `E_ILLEGAL_METHOD_CALL` ("a method was called at an unexpected time") before an
      operation object even exists — a completely misleading error for the actual cause.
      Microsoft's own sample gets this for free by deriving from `FtmBase`. Claiming it is
      honest here: the handler's only state is an `Event`, which is not apartment-bound.

    `AddRef`/`Release` count but never free: the object's lifetime is this Python object's,
    and Windows only borrows it for the duration of one activation.
    """

    def __init__(self):
        self.completed = threading.Event()
        self._refcount = 1

        query_interface_type = ctypes.WINFUNCTYPE(
            HRESULT, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))
        refcount_type = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
        completed_type = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, ctypes.c_void_p)

        self._callbacks = (
            query_interface_type(self._query_interface),
            refcount_type(self._add_ref),
            refcount_type(self._release),
            completed_type(self._activate_completed),
        )

        self._vtable = (ctypes.c_void_p * 4)(
            *(ctypes.cast(callback, ctypes.c_void_p) for callback in self._callbacks)
        )
        self._interface = ctypes.c_void_p(ctypes.addressof(self._vtable))
        self.pointer = ctypes.addressof(self._interface)

    def _query_interface(self, this, riid, out):
        wanted = riid[0]
        offered = (IID_IUNKNOWN, IID_IAGILE_OBJECT,
                   IID_IACTIVATE_AUDIO_INTERFACE_COMPLETION_HANDLER)

        if any(bytes(wanted) == bytes(candidate) for candidate in offered):
            out[0] = ctypes.c_void_p(self.pointer)
            self._refcount += 1
            return S_OK

        out[0] = ctypes.c_void_p(None)
        return E_NOINTERFACE

    def _add_ref(self, this):
        self._refcount += 1
        return self._refcount

    def _release(self, this):
        self._refcount -= 1
        return max(self._refcount, 0)

    def _activate_completed(self, this, operation):
        self.completed.set()
        return S_OK


def co_initialize() -> bool:
    """
    Join a multithreaded apartment. Returns whether this call is the one that must
    uninitialize — an already-initialized thread must not be torn down from here.
    """
    ole32 = ctypes.WinDLL('ole32.dll')
    ole32.CoInitializeEx.restype = HRESULT
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    hr = ole32.CoInitializeEx(None, COINIT_MULTITHREADED)

    if (hr & 0xFFFFFFFF) == RPC_E_CHANGED_MODE:
        # Something already put this thread in a single-threaded apartment. COM still
        # works; we simply do not own the apartment.
        return False

    return not failed(hr)


def co_uninitialize() -> None:
    ctypes.WinDLL('ole32.dll').CoUninitialize()


def co_task_mem_free(pointer) -> None:
    ole32 = ctypes.WinDLL('ole32.dll')
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole32.CoTaskMemFree(pointer)


def create_event() -> int:
    """An auto-reset, initially unsignalled event for `IAudioClient::SetEventHandle`."""
    kernel32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)
    kernel32.CreateEventW.restype = HANDLE
    kernel32.CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_wchar_p]
    handle = kernel32.CreateEventW(None, 0, 0, None)
    if not handle:
        raise OSError(ctypes.get_last_error(), "CreateEventW failed")
    return int(handle)


def wait_for_event(handle: int, timeout_ms: int) -> int:
    kernel32 = ctypes.WinDLL('kernel32.dll')
    kernel32.WaitForSingleObject.restype = ctypes.c_ulong
    kernel32.WaitForSingleObject.argtypes = [HANDLE, ctypes.c_ulong]
    return int(kernel32.WaitForSingleObject(HANDLE(handle), timeout_ms))


def close_handle(handle: int) -> None:
    if handle:
        ctypes.WinDLL('kernel32.dll').CloseHandle(HANDLE(handle))


def process_liveness(pid: int) -> tuple[bool, str]:
    """
    Whether a PID names a live process, and if not, why not.

    Worth asking before activating, because `ActivateAudioInterfaceAsync` does not
    reliably fail for a process that has exited — it can hand back a working
    `IAudioClient` that produces silence forever. Reporting that as a healthy capture is
    exactly the failure this project exists to avoid, so the check happens here where the
    answer is unambiguous.
    """
    if pid <= 0:
        return False, f"{pid} is not a process id"

    kernel32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)
    kernel32.OpenProcess.restype = HANDLE
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]

    ctypes.set_last_error(0)
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)

    if not handle:
        error = ctypes.get_last_error()
        if error == ERROR_INVALID_PARAMETER:
            return False, f"no process with id {pid} is running"
        if error == ERROR_ACCESS_DENIED:
            # A protected process exists but will not be introspected or captured.
            return False, (f"process {pid} exists but Windows refuses access to it "
                           f"(a protected or higher-integrity process)")
        return False, f"could not open process {pid}: {ctypes.FormatError(error).strip()}"

    try:
        kernel32.GetExitCodeProcess.argtypes = [HANDLE, ctypes.POINTER(ctypes.c_ulong)]
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(HANDLE(handle), ctypes.byref(code)):
            return False, f"could not read the state of process {pid}"
        if code.value != STILL_ACTIVE:
            return False, f"process {pid} has exited (code {code.value})"
    finally:
        close_handle(int(handle))

    return True, ""
