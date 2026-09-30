"""
The engine's side of an external backend (the ABI the ASIO host attaches through), with a
backend written here: what `ts_engine_stop_backend` reports when a stream fails in stopping —
an ASIO driver abandoned in its own stop() — and what it does not, a stream that had already
failed while running. Its callbacks run on the calling (control) thread: stop, status and
destroy are never called from the audio thread.
"""

import ctypes

import pytest

from tonesphere.native import NativeEngine, NativeError, _abi, available

pytestmark = pytest.mark.skipif(not available(), reason="needs tonesphere_native.dll")

STOP = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
STATUS = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(_abi.StreamStatus), ctypes.c_int32)
DESTROY = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
RUNNING, FAILED, STOPPED = 1, 2, 3


class BackendOps(ctypes.Structure):
    _fields_ = [('context', ctypes.c_void_p), ('stop', STOP), ('status', STATUS), ('destroy', DESTROY)]


class Backend:
    """One stream whose state is scripted: before stop(), and what stop() leaves it in."""

    def __init__(self, before: int, after: int, message: bytes = b''):
        self.state, self.after, self.message, self.calls = before, after, message, []
        self.ops = BackendOps(None, STOP(self._stop), STATUS(self._status), DESTROY(self._destroy))

    def _stop(self, _):
        self.calls.append('stop')
        self.state = self.after

    def _status(self, _, out, capacity):
        if capacity < 1:
            return 0
        out[0].node_id, out[0].state, out[0].error = 20, self.state, self.message
        return 1

    def _destroy(self, _):
        self.calls.append('destroy')


def attach(engine: NativeEngine, backend: Backend):
    prototype = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(BackendOps))
    attach_backend = prototype(('ts_engine_attach_backend', engine._dll))
    assert attach_backend(engine._handle, ctypes.byref(backend.ops)) == 0


def test_a_stream_that_fails_in_stopping_is_reported_and_the_engine_is_still_detached():
    backend = Backend(RUNNING, FAILED, b'ASIO: X; the driver did not return from stop() within 5 s and was abandoned')
    with NativeEngine(48000, 128) as engine:
        attach(engine, backend)
        with pytest.raises(NativeError, match='did not stop cleanly: ASIO: X; the driver did not return'):
            engine.stop_backend()
        assert backend.calls == ['stop', 'destroy']
        assert engine.stream_status() == [], "detached regardless"
        attach(engine, Backend(RUNNING, STOPPED))
        engine.stop_backend()


def test_a_stream_that_had_already_failed_while_running_is_not_news_at_stop():
    """A device removed mid-run leaves its stream failed; stopping the engine is then no failure."""
    backend = Backend(FAILED, FAILED, b'the device was removed or disabled')
    with NativeEngine(48000, 128) as engine:
        attach(engine, backend)
        engine.stop_backend()
        assert backend.calls == ['stop', 'destroy']


def test_a_clean_stop_is_clean():
    backend = Backend(RUNNING, STOPPED)
    with NativeEngine(48000, 128) as engine:
        attach(engine, backend)
        engine.stop_backend()
        assert backend.calls == ['stop', 'destroy']
