"""
Devices that come and go: the monitor's timing and the engine's decision, without hardware.

What these prove is the control logic — a burst of notifications is one reconcile, a
default-device change is none, the monitor stops while the engine holds its lock, and the
engine reopens its streams only when a device its routing uses changed. That audio stops
when a real device is disabled and resumes when it returns, with no user action, is proved
against the ToneSphere virtual cable in the driver VM
(`tests/hardware/test_virtual_driver.py::test_a_device_that_leaves_and_returns_is_reopened`).
"""

import threading
import time

import pytest

from tonesphere.core.engine import AudioEngine
from tonesphere.engine.device_monitor import DeviceMonitor
from tonesphere.engine.devices import DeviceInfo, HostApi


class ScriptedEvents:
    def __init__(self):
        self.queue: list[dict] = []
        self.lock = threading.Lock()
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def push(self, *kinds):
        with self.lock:
            self.queue += [{'kind': k, 'id': f'{{0.0.0.00000000}}.{i}', 'flow': None, 'role': None, 'state': 0}
                           for i, k in enumerate(kinds)]

    def poll(self):
        with self.lock:
            out, self.queue = self.queue, []
        return out


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_a_burst_of_notifications_is_one_reconcile():
    source, calls = ScriptedEvents(), []
    monitor = DeviceMonitor(calls.append, threading.RLock(), source, debounce_s=0.15, poll_s=0.01)
    monitor.start()
    try:
        for kinds in (('removed',), ('state_changed', 'state_changed'), ('removed',)):
            source.push(*kinds)
            time.sleep(0.03)
        assert wait_for(lambda: len(calls) == 1)
        time.sleep(0.3)
        assert len(calls) == 1, "a burst must settle into one reconcile, not one per event"
        assert [e['kind'] for e in calls[0]] == ['removed', 'state_changed', 'state_changed', 'removed']
    finally:
        monitor.stop()
    assert source.started and source.stopped


def test_a_default_device_change_reconciles_nothing():
    source, calls = ScriptedEvents(), []
    monitor = DeviceMonitor(calls.append, threading.RLock(), source, debounce_s=0.05, poll_s=0.01)
    monitor.start()
    try:
        source.push('default_changed', 'default_changed')
        time.sleep(0.3)
        assert calls == []
    finally:
        monitor.stop()


def test_the_monitor_stops_while_the_engine_holds_its_lock():
    """cleanup() stops the monitor with the engine's lock held; a waiting reconcile must give way."""
    lock, source, calls = threading.RLock(), ScriptedEvents(), []
    monitor = DeviceMonitor(calls.append, lock, source, debounce_s=0.02, poll_s=0.01)
    monitor.start()
    with lock:
        source.push('removed')
        time.sleep(0.2)   # the monitor is now waiting for the lock
        started = time.monotonic()
        monitor.stop(timeout=2.0)
        assert time.monotonic() - started < 1.0, "stop waited on the lock this thread holds"
    assert not monitor.running
    assert calls == []


def device(key: str, output: bool) -> DeviceInfo:
    return DeviceInfo(index=0, name=key.title(), host_api=HostApi.WASAPI, host_api_name='Windows WASAPI',
                      max_input_channels=0 if output else 2, max_output_channels=2 if output else 0,
                      default_samplerate=48000, default_low_input_latency_ms=0.0, default_low_output_latency_ms=0.0,
                      default_high_input_latency_ms=0.0, default_high_output_latency_ms=0.0,
                      endpoint_id=f'{{id}}.{key}')


class TestTheEngineReopensOnlyWhatItUses:
    @pytest.fixture
    def engine(self, monkeypatch):
        e = AudioEngine(host_backend='portaudio')
        present = {'speakers': device('speakers', True), 'mic': device('mic', False),
                   'headset': device('headset', True)}
        monkeypatch.setattr(e, '_enumerate', lambda: list(present.values()))
        e.initialize()
        restarts = []
        monkeypatch.setattr(e.host, 'stop', lambda: restarts.append('stop'))
        monkeypatch.setattr(e, 'start_engine', lambda: restarts.append('start'))
        e.present, e.restarts = present, restarts
        yield e

    def ids(self, engine):
        return {d.name.lower(): i for i, d in engine._device_by_id.items()}

    def test_an_unused_device_leaving_changes_the_list_and_reopens_nothing(self, engine):
        ids = self.ids(engine)
        engine.routing_matrix.create_routing(ids['mic'], ids['speakers'])
        engine._started = True
        del engine.present['headset']
        change = engine.handle_device_change()
        assert change['left'] == ['Headset'] and change['reopened'] is False
        assert engine.restarts == []

    def test_a_used_device_leaving_and_returning_reopens_both_times_under_the_same_id(self, engine):
        ids = self.ids(engine)
        engine.routing_matrix.create_routing(ids['mic'], ids['speakers'])
        engine._started = True
        speakers = engine.present.pop('speakers')
        left = engine.handle_device_change()
        assert left['left'] == ['Speakers'] and left['reopened'] is True
        assert engine.restarts == ['stop', 'start']
        engine.present['speakers'] = speakers
        back = engine.handle_device_change()
        assert back['arrived'] == ['Speakers'] and back['reopened'] is True
        assert self.ids(engine)['speakers'] == ids['speakers'], "a returning device keeps its id"
        assert (ids['mic'], ids['speakers']) in engine.routing_matrix.connections, "its route is kept"
        assert back['serial'] == left['serial'] + 1

    def test_a_stopped_engine_is_not_started_by_a_device_change(self, engine):
        ids = self.ids(engine)
        engine.routing_matrix.create_routing(ids['mic'], ids['speakers'])
        engine._started = False
        engine.present.pop('speakers')
        assert engine.handle_device_change()['reopened'] is False
        assert engine.restarts == []
