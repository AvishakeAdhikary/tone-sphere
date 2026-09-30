"""
Devices that come and go while the engine runs.

Windows reports endpoint arrival, removal and state changes through `IMMNotificationClient`,
which the native layer queues (`native/wasapi.py: watch, poll_events`). This thread drains
that queue and, once the events have stopped arriving for a moment — unplugging a USB
interface raises several, one per endpoint and state — asks the engine to reconcile: to
re-enumerate, and to reopen its streams if a device it routes left, arrived, or failed.

The thread only decides *when*; what happens is `AudioEngine.handle_device_change`, under
the engine's lock. The lock is taken in short attempts that watch for `stop()`, so the
engine can stop this monitor while it holds that lock itself without the two waiting on
each other.
"""

import threading
import time
from collections.abc import Callable

from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)

POLL_S = 0.05
DEBOUNCE_S = 0.3
# Default-device changes do not touch streams opened by endpoint ID, which is all ToneSphere
# opens; they are recorded but reconcile nothing.
RELEVANT = frozenset({'added', 'removed', 'state_changed', 'events_lost'})


class _Hub:
    """
    The native queue is one per process, and draining it is destructive, while a process
    can hold several engines (the GUI and a test harness, or two tests): each subscriber
    gets its own copy of every event, and the watch runs while anyone subscribes.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._queues: list[list[dict]] = []

    def subscribe(self) -> list[dict]:
        from tonesphere.native import wasapi

        with self._lock:
            if not self._queues:
                wasapi.watch(True)
            queue: list[dict] = []
            self._queues.append(queue)
            return queue

    def unsubscribe(self, queue: list[dict]):
        from tonesphere.native import wasapi

        with self._lock:
            if any(q is queue for q in self._queues):
                self._queues = [q for q in self._queues if q is not queue]
                if not self._queues:
                    wasapi.watch(False)

    def poll(self, queue: list[dict]) -> list[dict]:
        from tonesphere.native import wasapi

        with self._lock:
            if self._queues:
                events = wasapi.poll_events()
                for q in self._queues:
                    q.extend(events)
            out = list(queue)
            queue.clear()
            return out


_hub = _Hub()


class WasapiEvents:
    """Windows' endpoint notifications, one subscription to the process-wide queue."""

    def __init__(self):
        self._queue: list[dict] | None = None

    def start(self):
        self._queue = _hub.subscribe()

    def poll(self) -> list[dict]:
        return _hub.poll(self._queue) if self._queue is not None else []

    def stop(self):
        if self._queue is not None:
            _hub.unsubscribe(self._queue)
            self._queue = None


class DeviceMonitor:
    """
    Calls `on_change(events)` with `lock` held, `DEBOUNCE_S` after the last relevant event
    of a burst. `source` provides start/poll/stop; the default is Windows' own.
    """

    def __init__(self, on_change: Callable[[list[dict]], None], lock, source=None,
                 debounce_s: float = DEBOUNCE_S, poll_s: float = POLL_S):
        self._on_change = on_change
        self._lock = lock
        self._source = source or WasapiEvents()
        self._debounce = debounce_s
        self._poll = poll_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.reconciles = 0

    def start(self):
        self._source.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='device-monitor', daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        self._thread = None
        try:
            self._source.stop()
        except Exception as e:   # noqa: BLE001 - stopping must not fail shutdown
            logger.debug(f"device watch stop: {e}")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        pending: list[dict] = []
        last_event = 0.0
        while not self._stop.wait(self._poll):
            try:
                events = self._source.poll()
            except Exception as e:   # noqa: BLE001 - a failed poll is retried, not fatal
                logger.warning(f"device events could not be read: {e}")
                continue
            relevant = [e for e in events if e['kind'] in RELEVANT]
            if relevant:
                pending += relevant
                last_event = time.monotonic()
            if pending and time.monotonic() - last_event >= self._debounce:
                batch, pending = pending, []
                self._reconcile(batch)

    def _reconcile(self, events: list[dict]):
        while not self._stop.is_set():
            if self._lock.acquire(timeout=0.1):
                try:
                    self.reconciles += 1
                    self._on_change(events)
                except Exception as e:   # noqa: BLE001 - one failed reconcile must not end the watch
                    logger.error(f"reconciling a device change failed: {e}")
                finally:
                    self._lock.release()
                return
