"""
`AudioEngine` on the native host, with no audio device: routing that touches only buses and
network sends, run by the native engine's own clock. This is the path the PortAudio host
could run without any stream (Python wrote straight into its rings); the native host must
run it too, and carry the audio exactly.
"""

import sys
import time

import numpy as np
import pytest

from tests.signals import RATE, dominant_frequency, sine
from tonesphere.core.engine import AudioEngine
from tonesphere.engine.native_host import NativeHost

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason="the native host is Windows-only")

BLOCK = 1024


@pytest.fixture
def engine():
    e = AudioEngine(sample_rate=RATE, buffer_size=BLOCK)
    assert isinstance(e.host, NativeHost)
    e.initialize()
    yield e
    e.cleanup()


def network_sink(engine, bus_id):
    """Route a bus to a network send and return the (source, dest) node keys of that route."""
    ok, message = engine.send_device_audio_to_network(bus_id, transport='udp')
    assert ok, message
    # Only the ring is under test here: the sender thread is stopped so nothing drains it.
    if engine._send_worker is not None:
        engine._send_worker.stop()
    sink_id = engine.list_network_sends()[0]['sink_id']
    return str(engine._node_for(bus_id)), str(engine._node_for(sink_id))


def collect(engine, source, dest, seconds):
    out = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        block = engine.host.read_available(source, dest, RATE)
        if block is not None and len(block):
            out.append(block)
        time.sleep(0.02)
    return np.concatenate(out) if out else np.zeros((0, 2), np.float32)


def test_a_bus_reaches_a_network_send_sample_for_sample_with_no_device(engine):
    started, message = engine.start_udp_transport('127.0.0.1', 0)
    assert started, message
    bus = engine.create_virtual_input("src", channels=2)
    source, dest = network_sink(engine, bus)
    engine.start_engine()
    assert engine.host.is_running, "a routed bus must run even with no device: the native clock drives it"
    status = engine.host.stream_status()
    assert [s['kind'] for s in status] == [5], "the clock, not a device, runs this plan"

    time.sleep(0.2)  # let the routes' one-block fade-in pass over silence
    tone = sine(int(RATE * 0.5), 1000.0, amplitude=0.4)
    assert engine.write_to_bus(bus, tone) == len(tone)
    received = collect(engine, source, dest, 1.2)

    audible = np.nonzero(np.abs(received[:, 0]) > 1e-6)[0]
    assert len(audible), "nothing arrived"
    start = audible[0] - 1 if audible[0] > 0 and abs(tone[0, 0]) < 1e-6 else audible[0]
    arrived = received[start:start + len(tone)]
    assert np.array_equal(arrived, tone), "the tone must arrive unchanged, with no gap and no fade"
    stats = engine.get_performance_stats()
    assert stats['backend'] == 'native' and stats['audio_thread_allocations'] == 0
    assert dominant_frequency(arrived) == pytest.approx(1000.0, abs=5.0)


def test_writing_to_a_bus_with_no_routes_takes_nothing(engine):
    """As with the PortAudio host: an unrouted write is counted by the caller, not queued."""
    bus = engine.create_virtual_input("idle", channels=2)
    assert engine.write_to_bus(bus, sine(512)) == 0


def test_the_engine_refuses_nothing_it_used_to_accept(engine):
    """A device routed to itself (monitoring) is not a feedback loop; a bus cycle is."""
    a = engine.create_virtual_input("a", channels=2)
    b = engine.create_virtual_output("b", channels=2)
    assert engine.create_routing(a, b)[0] is True
    ok, message = engine.create_routing(b, a)
    assert ok is False and 'feedback' in message.lower()
