"""
Engine calls off the Qt main thread.

Opening a device can take seconds (a WASAPI stream waits for its first event, an ASIO
driver loads a DLL, a plugin scans its presets), and the engine's control lock can be
held by whichever caller got there first. The main thread must never wait on either: it
owns the window, and a window that stops repainting while audio starts looks like a crash.

So the UI never calls the engine itself. `EngineTasks` runs each action on one worker
thread, in the order the user made them — a gain move and then a mute must land in that
order — and hands the result, or the exception, back to the main thread through a queued
signal. `EnginePoller` gathers meters and statistics on a thread of its own at a fixed
rate and hands them over the same way; if a poll is still running when the next is due,
that tick is skipped rather than queued.
"""

import functools
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QCoreApplication, QObject, Qt, Signal

from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)


class _Bridge(QObject):
    """Lives on the main thread; anything emitted to it from a worker runs there."""

    call = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.call.connect(self._run, Qt.ConnectionType.QueuedConnection)

    @staticmethod
    def _run(fn):
        fn()


class EngineTasks(QObject):
    """
    One serial worker for every engine action the UI takes.

    `submit(fn, done=..., failed=...)` returns at once; `fn` runs on the worker and
    `done(result)` or `failed(exception)` runs on the main thread afterwards. `busy`
    reports whether anything is queued or running, for a "working" indicator.
    """

    busy = Signal(bool)

    def __init__(self, parent=None, name: str = 'engine-control'):
        super().__init__(parent)
        self._bridge = _Bridge(self)
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._outstanding = 0          # submitted and not yet delivered to the main thread
        self._closed = False
        self._thread = threading.Thread(target=self._work, name=name, daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[[], Any], done: Callable[[Any], None] | None = None,
               failed: Callable[[BaseException], None] | None = None):
        with self._lock:
            if self._closed:
                return
            self._outstanding += 1
            first = self._outstanding == 1
        if first:
            self.busy.emit(True)
        self._queue.put((fn, done, failed))

    @property
    def pending(self) -> int:
        with self._lock:
            return self._outstanding

    def _work(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            fn, done, failed = item
            try:
                callback, value = done, fn()
            except BaseException as e:   # noqa: BLE001 - every failure is handed back, never lost
                logger.warning(f"engine task failed: {e}")
                callback, value = failed, e
            try:
                self._bridge.call.emit(functools.partial(self._deliver, callback, value))
            except RuntimeError:
                # The window or dialog that asked is gone, and with it its bridge: the
                # result has nobody to go to.
                with self._lock:
                    self._outstanding -= 1

    def _deliver(self, callback, value):
        try:
            if callback is not None:
                callback(value)
        finally:
            with self._lock:
                self._outstanding -= 1
                idle = self._outstanding == 0
            if idle:
                self.busy.emit(False)

    def flush(self, timeout: float = 30.0) -> bool:
        """
        Wait, pumping the event loop, until every submitted task has run and been
        delivered. For tests and for shutdown; the UI itself never waits.
        """
        deadline = time.monotonic() + timeout
        app = QCoreApplication.instance()
        while self.pending and time.monotonic() < deadline:
            if app is not None:
                app.processEvents()
            time.sleep(0.002)
        return self.pending == 0

    def close(self, final: Callable[[], Any] | None = None, timeout: float = 30.0):
        """Run `final` (the engine's cleanup) after everything queued, then stop the worker."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if final is not None:
            self._queue.put((final, None, None))
        self._queue.put(None)
        self._thread.join(timeout)


class EnginePoller(QObject):
    """
    Calls `gather()` every `interval_ms` on a thread of its own and emits what it returns
    as `polled` on the main thread. `gather` may take the engine's lock and wait; the main
    thread never does. Exceptions are logged and the tick is dropped.
    """

    polled = Signal(object)

    def __init__(self, gather: Callable[[], Any], interval_ms: int, parent=None, name: str = 'engine-poll'):
        super().__init__(parent)
        self._gather = gather
        self._interval = interval_ms / 1000.0
        self._stop = threading.Event()
        self._delivered = threading.Event()
        self._delivered.set()
        self._bridge = _Bridge(self)
        self._thread = threading.Thread(target=self._work, name=name, daemon=True)
        self._thread.start()

    def _work(self):
        next_tick = time.monotonic()
        while not self._stop.is_set():
            # Until the main thread has taken the last result, gathering another only
            # builds a queue of stale readings.
            if self._delivered.is_set():
                try:
                    value = self._gather()
                except Exception as e:   # noqa: BLE001 - a failed poll is dropped, not fatal
                    logger.debug(f"poll failed: {e}")
                else:
                    self._delivered.clear()
                    try:
                        self._bridge.call.emit(lambda v=value: self._deliver(v))
                    except RuntimeError:
                        return   # its owner was deleted without stopping it
            next_tick += self._interval
            delay = next_tick - time.monotonic()
            if delay < 0:
                next_tick = time.monotonic()
                delay = 0
            self._stop.wait(delay)

    def _deliver(self, value):
        try:
            if not self._stop.is_set():
                self.polled.emit(value)
        finally:
            self._delivered.set()

    def stop(self, timeout: float = 5.0):
        self._stop.set()
        self._thread.join(timeout)
