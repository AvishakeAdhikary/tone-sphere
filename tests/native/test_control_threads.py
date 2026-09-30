"""
The control plane under concurrent callers, on the native host with no device.

Six threads do at once what the UI worker, the REST threadpool, a network playout thread
and a stats poller do in the real application: write into a bus, drain a network send,
add and remove buses and routes, move gains, read statistics and meters, and rebuild the
native engine by changing the block size — the one operation that swaps the engine every
data path writes into. None may raise, and afterwards a 1 kHz tone must still cross the
graph unchanged: a race that corrupted routing state would show up as a wrong or missing
signal, not only as an exception.
"""

import random
import sys
import threading
import time

import numpy as np
import pytest

from tests.native.test_engine_native_host import collect, network_sink
from tests.signals import RATE, dominant_frequency, sine
from tonesphere.core.engine import AudioEngine
from tonesphere.engine.native_host import NativeHost

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason="the native host is Windows-only")

STORM_SECONDS = 5.0


def test_concurrent_control_and_data_paths_leave_a_working_graph():
    engine = AudioEngine(sample_rate=RATE, buffer_size=1024)
    assert isinstance(engine.host, NativeHost)
    engine.initialize()
    errors: list[str] = []
    counts: dict[str, int] = {}
    stop = threading.Event()
    try:
        started, message = engine.start_udp_transport('127.0.0.1', 0)
        assert started, message
        bus = engine.create_virtual_input("src", channels=2)
        source, dest = network_sink(engine, bus)
        sink = engine.list_network_sends()[0]['sink_id']
        engine.start_engine()
        assert engine.host.is_running

        noise = (np.random.default_rng(1).standard_normal((256, 2)) * 0.01).astype(np.float32)

        def loop(name, body):
            def run():
                rng = random.Random(name)
                n = 0
                while not stop.is_set():
                    try:
                        body(rng)
                    except Exception as e:   # noqa: BLE001 - every failure is the finding
                        errors.append(f"{name}: {type(e).__name__}: {e}")
                        return
                    n += 1
                counts[name] = n
            return threading.Thread(target=run, name=name)

        def churn(rng):
            a = engine.create_virtual_input(f"a{rng.randrange(1000)}", channels=2)
            b = engine.create_virtual_output(f"b{rng.randrange(1000)}", channels=2)
            if a is not None and b is not None:
                engine.create_routing(a, b)
                engine.set_routing_volume(a, b, rng.random())
                engine.remove_routing(a, b)
            for device in (a, b):
                if device is not None:
                    engine.remove_virtual_device(device)

        def rebuild(rng):
            engine.set_buffer_size(rng.choice((512, 1024)))
            time.sleep(0.2)

        threads = [
            loop('writer', lambda rng: (engine.write_to_bus(bus, noise), time.sleep(0.002))),
            loop('reader', lambda rng: (engine.host.read_available(source, dest, RATE), time.sleep(0.002))),
            loop('churn', churn),
            loop('gains', lambda rng: (engine.set_routing_volume(bus, sink, rng.uniform(0.5, 1.0)),
                                       engine.set_channel_volume(bus, 0, rng.uniform(0.5, 1.0)))),
            loop('stats', lambda rng: (engine.get_performance_stats(), engine.get_meters(),
                                       engine.get_routing_matrix(), engine.get_devices(), engine.state)),
            loop('rebuild', rebuild),
        ]
        for t in threads:
            t.start()
        time.sleep(STORM_SECONDS)
        stop.set()
        for t in threads:
            t.join(10)
        assert not any(t.is_alive() for t in threads), "a thread is stuck: a deadlock"
        assert not errors, errors
        assert all(counts.get(t.name, 0) > 0 for t in threads), counts
        print(f"\noperations in {STORM_SECONDS:.0f} s: {counts}")

        # The graph is what the storm left: the one route, and none of the churned buses.
        assert [(r['source_id'], r['destination_id']) for r in engine.get_routing_matrix().values()] == [(bus, sink)]
        engine.set_routing_volume(bus, sink, 1.0)
        engine.set_channel_volume(bus, 0, 1.0)
        engine.set_buffer_size(1024)
        assert engine.host.is_running
        time.sleep(0.3)
        collect(engine, source, dest, 0.3)   # drain what the storm queued

        tone = sine(int(RATE * 0.5), 1000.0, amplitude=0.4)
        assert engine.write_to_bus(bus, tone) == len(tone)
        received = collect(engine, source, dest, 1.2)
        audible = np.nonzero(np.abs(received[:, 0]) > 1e-6)[0]
        assert len(audible), "nothing arrived after the storm"
        start = audible[0] - 1 if audible[0] > 0 else audible[0]
        arrived = received[start:start + len(tone)]
        assert dominant_frequency(arrived) == pytest.approx(1000.0, abs=5.0)
        assert np.sqrt(np.mean(arrived[:, 0] ** 2)) == pytest.approx(0.4 / np.sqrt(2), rel=0.01)
        assert engine.get_performance_stats()['audio_thread_allocations'] == 0
    finally:
        stop.set()
        engine.cleanup()
