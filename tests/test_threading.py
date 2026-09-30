"""
The control plane's threading rules, checked where they can be broken.

- Every public method of the engine and of both hosts takes its object's lock, except the
  named data paths (`utils/threads.py`). A method added later without it fails here.
- The UI never calls the engine on the Qt main thread: every engine method is wrapped to
  record the thread it ran on, and the window is driven through its actions.
- The window keeps repainting while the engine is slow: an engine call made to hold the
  control lock for 1.5 s must not stall the main thread's event loop.

The concurrency storm against real native state is `tests/native/test_control_threads.py`.
"""

import functools
import threading
import time

import pytest

from tonesphere.core.engine import AudioEngine
from tonesphere.core.engine_factory import UnifiedAudioEngine
from tonesphere.engine.host import AudioHost
from tonesphere.engine.native_host import NativeHost
from tonesphere.utils.config import ConfigManager
from tonesphere.utils.threads import is_locked

pytest.importorskip("PySide6")

# Data paths that must never wait on the control lock (the reason is at each decorator).
UNLOCKED = {AudioEngine: {'write_to_bus'}, NativeHost: set(), AudioHost: set()}


@pytest.mark.parametrize('cls', [AudioEngine, NativeHost, AudioHost], ids=lambda c: c.__name__)
def test_every_public_method_takes_the_control_lock(cls):
    missing = []
    for name, member in vars(cls).items():
        if name.startswith('_') or name in UNLOCKED[cls]:
            continue
        if isinstance(member, property) or (callable(member) and not isinstance(member, (staticmethod, type))):
            if not is_locked(member):
                missing.append(name)
    assert not missing, f"{cls.__name__} methods callable from several threads without its lock: {missing}"


def test_the_data_path_exemptions_really_are_unlocked():
    assert not is_locked(vars(AudioEngine)['write_to_bus'])


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _pump(seconds: float):
    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)


def test_the_window_never_calls_the_engine_on_the_main_thread(qt_app, tmp_path, monkeypatch):
    from tonesphere.ui.main_window import MainWindow

    main = threading.main_thread()
    on_main: list[str] = []

    def guard(cls, name, fn):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            if threading.current_thread() is main:
                on_main.append(f"{cls.__name__}.{name}")
            return fn(*args, **kwargs)
        return call

    for cls in (AudioEngine, UnifiedAudioEngine):
        for name, member in list(vars(cls).items()):
            if name.startswith('_'):
                continue
            if isinstance(member, property):
                monkeypatch.setattr(cls, name, property(guard(cls, name, member.fget),
                                                        guard(cls, name, member.fset) if member.fset else None))
            elif callable(member) and not isinstance(member, (staticmethod, type)):
                monkeypatch.setattr(cls, name, guard(cls, name, member))

    window = MainWindow(ConfigManager(db_path=tmp_path / "settings.db"))
    try:
        assert window.tasks.flush()
        window._add_bus()
        window._add_bus()
        assert window.tasks.flush()
        buses = [d['id'] for d in window._view['devices'] if d['origin'] == 'in_process_bus']
        assert len(buses) == 2, window._view['devices']
        window._connect_nodes(buses[0], buses[1])
        window._set_route_gain(buses[0], buses[1], -6.0)
        window._set_route_mute(buses[0], buses[1], True)
        window._set_device_gain(buses[0], -3.0)
        window._set_device_mute(buses[0], True)
        window._set_master_gain(-1.0)
        window._toggle_engine()
        window._rescan()
        window._show_diagnostics()
        assert window.tasks.flush()
        _pump(0.6)   # poller ticks, the diagnostics dialog's too
        window._toggle_engine()
        window._disconnect_nodes(buses[0], buses[1])
        window._clear_routing()
        assert window.tasks.flush()
        _pump(0.2)
    finally:
        window.close()
    assert not on_main, f"engine called on the Qt main thread: {sorted(set(on_main))}"


def test_the_window_keeps_repainting_while_the_engine_is_slow(qt_app, tmp_path, monkeypatch):
    """
    `start_engine` is made to hold the control lock for 1.5 s, as a device that is slow to
    open does. The poller then waits on that lock too; the main thread must not.
    """
    from PySide6.QtCore import QTimer

    from tonesphere.ui.main_window import MainWindow

    held = threading.Event()

    def slow_start(self):
        with self._lock:
            held.set()
            time.sleep(1.5)

    monkeypatch.setattr(AudioEngine, 'start_engine', slow_start)
    window = MainWindow(ConfigManager(db_path=tmp_path / "settings.db"))
    try:
        assert window.tasks.flush()
        gaps, last = [], [time.monotonic()]

        def tick():
            now = time.monotonic()
            gaps.append(now - last[0])
            last[0] = now

        timer = QTimer()
        timer.timeout.connect(tick)
        timer.start(10)
        window._toggle_engine()
        _pump(2.0)
        timer.stop()
        assert held.is_set(), "the slow start never ran"
        assert window.tasks.flush()
        assert max(gaps) < 0.1, f"the main thread stalled for {max(gaps) * 1000:.0f} ms"
    finally:
        window.close()
