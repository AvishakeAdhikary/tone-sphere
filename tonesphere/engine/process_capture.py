"""
Capturing one application's audio output by process id, on Windows, with no driver.

This is the thing people install a virtual audio cable for: take Spotify's output, or
Discord's, and route it somewhere else. Windows 10 build 20348 and later can do it without
any driver at all, through `ActivateAudioInterfaceAsync` with
`VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK`. `wasapi_com.py` holds the COM plumbing; this
module is the part the rest of the engine talks to.

Measured on the machine this was written on (Windows 11, build 26200), because none of it
is documented in a way you would want to trust blind:

* `IAudioClient::GetMixFormat` and `IsFormatSupported` both return `E_NOTIMPL` on a
  process-loopback client. There is no format to query and no negotiation to do — you pass
  a format to `Initialize` and Windows converts into it. 44.1/48/96 kHz, mono and stereo
  were all accepted and all delivered at the requested rate. `IsFormatSupported` is still
  called first, so a future build that implements it is honoured rather than overridden;
  today it declines and the requested format stands.
* Because nothing can be *queried* back, the delivered rate is verified by measurement
  instead: `statistics()` reports `measured_sample_rate` as frames divided by the time
  between the first and last packet, so a Windows that quietly delivered something else
  shows up rather than being assumed away.
* **`ActivateAudioInterfaceAsync` returns `S_OK` for a process id that has exited, and for
  one that never existed at all.** Activation, `Initialize` and `Start` all succeed, and
  the capture then produces silence forever. That is precisely the "reports healthy, moves
  nothing" failure this project exists to prevent, and it is why `start()` checks that the
  process is alive before activating. The check is not defensive padding; it is the only
  thing between a user and a capture that lies.
"""

import platform
import threading
import time
from collections.abc import Callable
from ctypes import (
    POINTER,
    byref,
    c_float,
    c_int,
    c_uint32,
    c_uint64,
    c_ulong,
    c_void_p,
)
from dataclasses import dataclass

import numpy as np

from tonesphere.engine import wasapi_com as com
from tonesphere.engine.app_capture import process_loopback_supported
from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)

# Advisory: Windows hands back a 10 ms buffer for process loopback regardless (measured
# 480 frames at 48 kHz). The event-driven loop below drains whatever it is given.
_BUFFER_DURATION_HNS = 2_000_000

# Long enough that the loop is not spinning, short enough that stop() is not waiting on it.
_WAIT_TIMEOUT_MS = 200


class ProcessCaptureError(RuntimeError):
    """
    A capture that cannot carry audio.

    Raised for a real, named cause — a dead process id, a refusal by Windows, a failed
    COM call — and never for a capture that is merely quiet. Silence from an application
    that is not playing anything is a correct capture of nothing.
    """


@dataclass(frozen=True)
class CaptureFormat:
    """The format Windows is delivering, not the one that was asked for."""
    sample_rate: int
    channels: int
    sample_format: str = 'float32'

    def describe(self) -> str:
        return f"{self.sample_rate} Hz, {self.channels} ch, {self.sample_format}"


class ProcessCapture:
    """
    One application's audio output, pushed into a sink on a dedicated thread.

    The sink is a callable returning how many frames it accepted, which is how the caller
    finds out it is being outrun rather than having frames silently dropped.
    """

    def __init__(self, pid: int, include_process_tree: bool = True,
                 sample_rate: int = 48000, channels: int = 2):
        self.pid = pid
        self.include_process_tree = include_process_tree
        # What to ask Windows to convert into. A capture has no opinion of its own about
        # what rate the mixer runs at, so the engine passes its own.
        self.sample_rate = sample_rate
        self.channels = channels

        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._negotiated = threading.Event()
        self._released = threading.Event()

        self._format: CaptureFormat | None = None
        self._sink: Callable[[np.ndarray], int] | None = None
        self._error: str | None = None
        self._buffer_frames: int | None = None
        self._handler: com.ActivationCompletionHandler | None = None

        self._frames_captured = 0
        self._glitch_count = 0
        self._silent_packets = 0
        self._frames_rejected = 0
        self._unrouted_writes = 0
        # The delivered rate is measured between the first and last packet, not from
        # Start(): activation latency and a target that has not begun playing yet are not
        # part of the rate, and including them would report a rate nobody delivered.
        self._first_frame_at: float | None = None
        self._last_frame_at: float | None = None
        self._frames_after_first = 0

    # --- Public API ---

    @property
    def is_running(self) -> bool:
        """Derived from the thread, so it cannot claim to be running after dying."""
        return self._thread is not None and self._thread.is_alive()

    def start(self, on_format: Callable[[CaptureFormat], Callable[[np.ndarray], int]],
              timeout: float = 5.0) -> CaptureFormat:
        """
        Activate the capture and begin delivering audio to the sink `on_format` returns.

        `on_format` is handed the real delivered format before anything is captured, so a
        caller sizing a buffer or a bus never has to guess the channel count. It is called
        on *this* thread, not the capture thread: the engine creates a bus inside it while
        holding its own lock, and running it on the capture thread would deadlock against
        the caller that is waiting here.

        Raises `ProcessCaptureError` on any failure. It never returns having started a
        capture that cannot carry audio.
        """
        if self.is_running:
            raise ProcessCaptureError("this capture is already running")

        if not process_loopback_supported():
            # Checked before anything touches ctypes.WinDLL, so callers on Linux and macOS
            # get the real reason rather than an AttributeError from the plumbing.
            raise ProcessCaptureError(
                f"Per-process capture is not supported on {platform.system()}: it needs "
                f"Windows 10 build 20348 or later"
            )

        alive, reason = com.process_liveness(self.pid)
        if not alive:
            # Windows would activate anyway and hand back silence; see the module
            # docstring. Refusing here is the difference between an error and a lie.
            raise ProcessCaptureError(f"Cannot capture process {self.pid}: {reason}")

        self._stopping.clear()
        self._negotiated.clear()
        self._released.clear()
        self._error = None

        self._thread = threading.Thread(
            target=self._run, name=f"process-capture-{self.pid}", daemon=True)
        self._thread.start()

        if not self._negotiated.wait(timeout):
            self._stopping.set()
            raise ProcessCaptureError(
                f"Timed out after {timeout:.1f}s activating capture for process {self.pid}")

        if self._error is not None:
            self._thread.join(timeout=1.0)
            raise ProcessCaptureError(self._error)

        fmt = self._format
        if fmt is None:
            self._stopping.set()
            raise ProcessCaptureError("Capture reported no format")

        try:
            self._sink = on_format(fmt)
        except Exception as e:
            self._stopping.set()
            self._released.set()
            self._thread.join(timeout=1.0)
            raise ProcessCaptureError(f"Could not prepare a sink for the capture: {e}") from e

        self._released.set()
        logger.info(f"Capturing process {self.pid}: {fmt.describe()}")
        return fmt

    def stop(self, timeout: float = 2.0) -> None:
        """Stop and release everything. Safe to call on a capture that already died."""
        self._stopping.set()
        self._released.set()

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(f"Capture thread for process {self.pid} did not stop in "
                               f"{timeout:.1f}s")

    def statistics(self) -> dict:
        """
        What this capture actually did.

        Unmeasured values are None, never 0: `measured_sample_rate` before any audio has
        arrived is not a rate of zero, and `frames_captured` of 0 on a running capture
        means an application that is silent, which is different from a broken one.
        """
        return {
            'pid': self.pid,
            'running': self.is_running,
            'format': self._format.describe() if self._format else None,
            'sample_rate': self._format.sample_rate if self._format else None,
            'channels': self._format.channels if self._format else None,
            'measured_sample_rate': self._measured_sample_rate(),
            'buffer_frames': self._buffer_frames,
            'frames_captured': self._frames_captured,
            'seconds_captured': (round(self._frames_captured / self._format.sample_rate, 3)
                                 if self._format else None),
            'glitch_count': self._glitch_count,
            'silent_packets': self._silent_packets,
            'frames_rejected_by_sink': self._frames_rejected,
            'unrouted_writes': self._unrouted_writes,
            'error': self._error,
        }

    def _measured_sample_rate(self) -> float | None:
        """
        The rate Windows actually delivered, or None before there is enough to divide.

        This is the only real check that the format passed to `Initialize` was honoured:
        `GetMixFormat` and `IsFormatSupported` are both E_NOTIMPL on a loopback client, so
        there is nothing to ask.
        """
        if self._first_frame_at is None or self._last_frame_at is None:
            return None

        span = self._last_frame_at - self._first_frame_at
        if span <= 0 or self._frames_after_first == 0:
            return None

        return round(self._frames_after_first / span, 1)

    # --- The capture thread ---

    def _run(self) -> None:
        """
        Activate, negotiate, capture, tear down — all on one thread in one apartment.

        Everything COM touches lives here, so no interface pointer ever crosses an
        apartment boundary and no marshaling is needed. Any failure sets `self._error` and
        returns; `is_running` then goes false on its own, because it is derived from this
        thread rather than from a flag something has to remember to clear.
        """
        owns_apartment = False
        operation = None
        activated = None
        client = None
        capture = None
        event = 0

        try:
            owns_apartment = com.co_initialize()

            operation, activated = self._activate()
            client = com.query_interface(activated, com.IID_IAUDIO_CLIENT)
            if not client:
                raise ProcessCaptureError(
                    "The activated object is not an IAudioClient")

            fmt = self._negotiate(client)
            event = com.create_event()
            capture = self._initialize(client, fmt, event)

            self._format = CaptureFormat(
                sample_rate=int(fmt.nSamplesPerSec), channels=int(fmt.nChannels))
        except ProcessCaptureError as e:
            self._error = str(e)
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"

        self._negotiated.set()

        if self._error is None:
            # The caller builds its sink now; it cannot be done before the format is
            # known, and it must not be done on this thread. See start().
            self._released.wait(timeout=10.0)

            if not self._stopping.is_set() and self._sink is not None:
                try:
                    self._capture_loop(client, capture, event)
                except ProcessCaptureError as e:
                    self._error = str(e)
                except Exception as e:
                    self._error = f"{type(e).__name__}: {e}"

        if self._error is not None:
            logger.error(f"Process {self.pid} capture: {self._error}")

        com.close_handle(event)
        for pointer in (capture, client, activated, operation):
            com.release(pointer)
        # Only now that the operation is gone can Windows no longer reach the handler.
        self._handler = None
        if owns_apartment:
            com.co_uninitialize()

    def _activate(self) -> tuple[int, int]:
        """Ask mmdevapi for an IAudioClient bound to the process rather than a device."""
        activate = com.mmdevapi()

        # Held on the instance, not in a local. mmdevapi AddRefs the handler and keeps the
        # pointer for as long as the operation object lives; letting Python collect it at
        # the end of this method frees the ctypes vtable underneath Windows, and the crash
        # then lands somewhere unrelated — measured as an access violation inside a later,
        # perfectly correct `Release`.
        handler = com.ActivationCompletionHandler()
        self._handler = handler

        variant, params = com.process_loopback_propvariant(
            self.pid, self.include_process_tree)

        operation = c_void_p()
        hr = activate(
            com.VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
            byref(com.IID_IAUDIO_CLIENT),
            byref(variant),
            c_void_p(handler.pointer),
            byref(operation),
        )
        # `params` is referenced through the variant's blob pointer, so it has to outlive
        # the call; naming it here is what keeps it alive.
        del params

        if com.failed(hr) or not operation:
            raise ProcessCaptureError(
                f"ActivateAudioInterfaceAsync failed: {com.describe_hresult(hr)}")

        if not handler.completed.wait(5.0):
            com.release(operation)
            raise ProcessCaptureError(
                "Windows never signalled that the capture activation completed")

        get_result = com.bind(operation, com.SLOT_GET_ACTIVATE_RESULT, com.HRESULT,
                              POINTER(com.HRESULT), POINTER(c_void_p))
        activate_hr = com.HRESULT()
        activated = c_void_p()
        hr = get_result(operation, byref(activate_hr), byref(activated))

        if com.failed(hr):
            com.release(operation)
            raise ProcessCaptureError(
                f"GetActivateResult failed: {com.describe_hresult(hr)}")

        # This is where a refusal actually surfaces — a protected process, a build without
        # process loopback, a rejected process id.
        if com.failed(activate_hr.value) or not activated:
            com.release(operation)
            raise ProcessCaptureError(
                f"Windows refused to capture process {self.pid}: "
                f"{com.describe_hresult(activate_hr.value)}")

        return int(operation.value), int(activated.value)

    def _negotiate(self, client: int) -> com.WAVEFORMATEX:
        """
        Ask for float32 at the engine's rate and take whatever Windows says instead.

        Measured: process-loopback clients return `E_NOTIMPL` here, so in practice the
        requested format stands and `Initialize` is the authority on whether it is real.
        The call is still made rather than skipped, so that a build which does implement
        it gets the last word instead of being second-guessed.
        """
        wanted = com.float32_format(self.sample_rate, self.channels)

        is_supported = com.bind(client, com.SLOT_AUDIO_CLIENT_IS_FORMAT_SUPPORTED,
                                com.HRESULT, c_int, POINTER(com.WAVEFORMATEX),
                                POINTER(POINTER(com.WAVEFORMATEX)))
        closest = POINTER(com.WAVEFORMATEX)()
        hr = is_supported(client, com.AUDCLNT_SHAREMODE_SHARED, byref(wanted),
                          byref(closest))

        if closest:
            offered = com.WAVEFORMATEX.from_buffer_copy(bytes(closest[0]))
            com.co_task_mem_free(closest)
            if hr == com.S_FALSE:
                logger.info(f"Windows offered {com.describe_format(offered)} instead of "
                            f"{com.describe_format(wanted)}")
                return offered

        return wanted

    def _initialize(self, client: int, fmt: com.WAVEFORMATEX, event: int) -> int:
        """Initialize for loopback capture, then fetch the capture service."""
        initialize = com.bind(
            client, com.SLOT_AUDIO_CLIENT_INITIALIZE, com.HRESULT,
            c_int, c_ulong, com.REFERENCE_TIME, com.REFERENCE_TIME,
            POINTER(com.WAVEFORMATEX), c_void_p)

        flags = (com.AUDCLNT_STREAMFLAGS_LOOPBACK
                 | com.AUDCLNT_STREAMFLAGS_EVENTCALLBACK)
        hr = initialize(client, com.AUDCLNT_SHAREMODE_SHARED, flags,
                        _BUFFER_DURATION_HNS, 0, byref(fmt), None)
        if com.failed(hr):
            raise ProcessCaptureError(
                f"IAudioClient::Initialize failed for {com.describe_format(fmt)}: "
                f"{com.describe_hresult(hr)}")

        buffer_size = c_uint32()
        hr = com.bind(client, com.SLOT_AUDIO_CLIENT_GET_BUFFER_SIZE, com.HRESULT,
                      POINTER(c_uint32))(client, byref(buffer_size))
        if not com.failed(hr):
            self._buffer_frames = int(buffer_size.value)

        hr = com.bind(client, com.SLOT_AUDIO_CLIENT_SET_EVENT_HANDLE,
                      com.HRESULT, com.HANDLE)(client, com.HANDLE(event))
        if com.failed(hr):
            raise ProcessCaptureError(f"SetEventHandle failed: {com.describe_hresult(hr)}")

        capture = c_void_p()
        hr = com.bind(client, com.SLOT_AUDIO_CLIENT_GET_SERVICE, com.HRESULT,
                      POINTER(com.GUID), POINTER(c_void_p))(
            client, byref(com.IID_IAUDIO_CAPTURE_CLIENT), byref(capture))
        if com.failed(hr) or not capture:
            raise ProcessCaptureError(
                f"GetService(IAudioCaptureClient) failed: {com.describe_hresult(hr)}")

        return int(capture.value)

    def _capture_loop(self, client: int, capture: int, event: int) -> None:
        """
        Drain packets until asked to stop.

        The vtable slots are bound once up front: rebinding per packet would build a
        ctypes trampoline several hundred times a second for no reason.
        """
        start = com.bind(client, com.SLOT_AUDIO_CLIENT_START, com.HRESULT)
        stop = com.bind(client, com.SLOT_AUDIO_CLIENT_STOP, com.HRESULT)
        get_buffer = com.bind(
            capture, com.SLOT_CAPTURE_GET_BUFFER, com.HRESULT,
            POINTER(POINTER(c_float)), POINTER(c_uint32), POINTER(c_ulong),
            POINTER(c_uint64), POINTER(c_uint64))
        release_buffer = com.bind(capture, com.SLOT_CAPTURE_RELEASE_BUFFER,
                                  com.HRESULT, c_uint32)
        next_packet_size = com.bind(capture, com.SLOT_CAPTURE_GET_NEXT_PACKET_SIZE,
                                    com.HRESULT, POINTER(c_uint32))

        channels = self._format.channels
        sink = self._sink

        hr = start(client)
        if com.failed(hr):
            raise ProcessCaptureError(f"IAudioClient::Start failed: "
                                      f"{com.describe_hresult(hr)}")

        try:
            while not self._stopping.is_set():
                if com.wait_for_event(event, _WAIT_TIMEOUT_MS) != com.WAIT_OBJECT_0:
                    # A timeout is normal: an application that is not playing produces no
                    # packets and signals nothing.
                    continue

                while not self._stopping.is_set():
                    packet = c_uint32()
                    hr = next_packet_size(capture, byref(packet))
                    if com.failed(hr):
                        raise ProcessCaptureError(
                            f"GetNextPacketSize failed: {com.describe_hresult(hr)}")
                    if packet.value == 0:
                        break

                    data = POINTER(c_float)()
                    frames = c_uint32()
                    flags = c_ulong()
                    hr = get_buffer(capture, byref(data), byref(frames), byref(flags),
                                    None, None)
                    if com.failed(hr):
                        raise ProcessCaptureError(
                            f"GetBuffer failed: {com.describe_hresult(hr)}")

                    count = int(frames.value)

                    if flags.value & com.AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY:
                        # A real dropout in the captured stream. Counting it is the only
                        # way anyone finds out the capture has holes in it.
                        self._glitch_count += 1

                    if count:
                        if flags.value & com.AUDCLNT_BUFFERFLAGS_SILENT:
                            # The docs are explicit that pData is undefined when this is
                            # set, so the zeros are written rather than read.
                            self._silent_packets += 1
                            block = np.zeros((count, channels), dtype=np.float32)
                        else:
                            block = np.ctypeslib.as_array(
                                data, shape=(count, channels)).copy()

                        self._deliver(sink, block, count)
                        self._frames_captured += count
                        self._note_arrival(count)

                    release_buffer(capture, count)
        finally:
            stop(client)

    def _note_arrival(self, count: int) -> None:
        """
        Timestamp packet arrivals so the delivered rate is a measurement.

        The first packet's frames are excluded: they were produced before the clock
        started, so counting them would inflate the rate by one packet's worth.
        """
        now = time.monotonic()

        if self._first_frame_at is None:
            self._first_frame_at = now
        else:
            self._frames_after_first += count

        self._last_frame_at = now

    def _deliver(self, sink, block: np.ndarray, count: int) -> None:
        """
        Hand a block to the sink, counting the two ways it can decline.

        "Nothing is routed anywhere" and "the destination is backed up" both leave audio
        on the floor, but only the second says anything about performance, so they are
        counted apart rather than added together into one meaningless number.
        """
        accepted = sink(block)

        if accepted == 0:
            self._unrouted_writes += 1
        elif accepted < count:
            self._frames_rejected += count - accepted
