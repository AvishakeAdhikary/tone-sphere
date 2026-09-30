"""
MIDI input from a hardware port on Windows (winmm, through ctypes), forwarded to an
instrument's native MIDI queue.

winmm calls back on its own thread, where the documentation allows almost nothing but
posting the message on, so the callback only appends to a queue and a forwarding thread
sends each message to the engine (`AudioEngine.send_midi`). Note on/off, controllers and
the pitch wheel are forwarded; clock, active sensing and SysEx are not.

No MIDI input device was attached to the development machine, so this is exercised only as
far as enumerating none — `docs/IMPLEMENTATION_STATUS.md` records it as UNVERIFIED.
"""

import ctypes
import queue
import sys
import threading
from ctypes import wintypes

from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)

MIM_DATA = 0x3C3
CALLBACK_FUNCTION = 0x30000
MAXPNAMELEN = 32


class MIDIINCAPSW(ctypes.Structure):
    _fields_ = [('wMid', wintypes.WORD), ('wPid', wintypes.WORD), ('vDriverVersion', wintypes.UINT),
                ('szPname', wintypes.WCHAR * MAXPNAMELEN), ('dwSupport', wintypes.DWORD)]


def _winmm():
    if sys.platform != 'win32':
        raise OSError("MIDI input is implemented on Windows only")
    return ctypes.WinDLL('winmm')


def inputs() -> list[str]:
    """The MIDI input ports Windows lists, by index."""
    if sys.platform != 'win32':
        return []
    winmm = _winmm()
    names = []
    for i in range(winmm.midiInGetNumDevs()):
        caps = MIDIINCAPSW()
        if winmm.midiInGetDevCapsW(i, ctypes.byref(caps), ctypes.sizeof(caps)) == 0:
            names.append(caps.szPname)
    return names


_CALLBACK = ctypes.WINFUNCTYPE(None, wintypes.HANDLE, wintypes.UINT, ctypes.c_size_t, ctypes.c_size_t,
                               ctypes.c_size_t)


class MidiInput:
    """One open port. `forward(status, data1, data2)` runs on the forwarding thread."""

    def __init__(self, port: int, forward):
        self._forward = forward
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._handle = wintypes.HANDLE()
        self._callback = _CALLBACK(self._on_message)   # kept alive for as long as the port is open
        winmm = _winmm()
        result = winmm.midiInOpen(ctypes.byref(self._handle), port, self._callback, 0, CALLBACK_FUNCTION)
        if result != 0:
            raise OSError(f"midiInOpen({port}) failed with MMRESULT {result}")
        self.name = inputs()[port]
        self._thread = threading.Thread(target=self._pump, name=f'midi-in-{port}', daemon=True)
        self._thread.start()
        winmm.midiInStart(self._handle)
        self.received = 0

    def _on_message(self, _handle, message, _instance, param1, _param2):
        if message == MIM_DATA:
            self._queue.put(param1)

    def _pump(self):
        while True:
            packed = self._queue.get()
            if packed is None:
                return
            status, data1, data2 = packed & 0xFF, (packed >> 8) & 0x7F, (packed >> 16) & 0x7F
            if status & 0xF0 not in (0x80, 0x90, 0xB0, 0xE0):
                continue
            self.received += 1
            try:
                self._forward(status, data1, data2)
            except Exception as e:   # noqa: BLE001 - one bad message must not end the port
                logger.warning(f"MIDI from {self.name} not delivered: {e}")

    def close(self):
        if self._handle:
            winmm = _winmm()
            winmm.midiInStop(self._handle)
            winmm.midiInClose(self._handle)
            self._handle = wintypes.HANDLE()
            self._queue.put(None)
            self._thread.join(2)
