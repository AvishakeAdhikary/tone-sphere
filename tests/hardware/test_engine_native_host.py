"""
`AudioEngine` on the native host, against the real default output: the whole chain the UI
and API use — engine -> native host -> VST3 plugin -> WASAPI -> Windows — checked by
capturing this process's own audio through process loopback in a separate native engine.
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest

from tests.signals import dominant_frequency, rms, sine
from tonesphere.core.engine import AudioEngine
from tonesphere.core.presets import PresetManager
from tonesphere.native import NativeEngine, Node, Route, available
from tonesphere.native.wasapi import StreamSpec, default_endpoint
from tonesphere.plugins import classes_in

pytestmark = [pytest.mark.hardware, pytest.mark.skipif(not available(), reason="needs tonesphere_native.dll")]

RATE = 48000
BLOCK = 480
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def test_gain():
    found = sorted(ROOT.glob("native/build/*/VST3/Release/tonesphere_test_gain.vst3"))
    if not found:
        pytest.fail("the test plugin is not built")
    return next(c for c in classes_in(found[0]) if c.name == "ToneSphere Test Gain")


@pytest.fixture
def engine():
    e = AudioEngine(sample_rate=RATE, buffer_size=BLOCK, exclusive=False)
    e.initialize()
    yield e
    e.cleanup()


class Listener:
    """This process's audio, captured by process loopback in its own native engine."""

    def __init__(self):
        self.engine = NativeEngine(RATE, 480)
        self.engine.apply_plan([Node.sink(1, 2), Node.source(2, 2), Node.sink(3, 2, ring_frames=RATE * 8)],
                               [Route(2, 3)])
        clock = default_endpoint('render')
        self.engine.start_wasapi([StreamSpec(1, 'render', 2, clock.id),
                                  StreamSpec(2, 'process_loopback', 2, process_id=os.getpid())])

    def listen(self, seconds: float) -> np.ndarray:
        self.engine.port_read(3, RATE * 8)
        out = []
        deadline = time.time() + seconds
        while time.time() < deadline:
            out.append(self.engine.port_read(3, RATE))
            time.sleep(0.02)
        return np.concatenate(out)

    def close(self):
        self.engine.stop_backend()
        self.engine.close()


def play(engine, bus, seconds):
    """Keep the bus fed in real time for `seconds`."""
    tone = sine(int(RATE * (seconds + 0.5)), 1000.0, amplitude=0.2)
    written = 0
    deadline = time.time() + seconds
    while time.time() < deadline:
        written += engine.write_to_bus(bus, tone[written:written + BLOCK * 4])
        time.sleep(0.01)
    return tone


def steady_level(captured) -> float:
    loud = np.nonzero(np.abs(captured[:, 0]) > 1e-5)[0]
    assert len(loud) > RATE // 2, "nothing of the tone was heard back"
    window = captured[loud[0] + RATE // 5: loud[0] + RATE // 5 + RATE // 2]
    assert dominant_frequency(window) == pytest.approx(1000.0, abs=3.0)
    return rms(window)


def test_the_engine_plays_through_a_plugin_and_keeps_it_across_a_restart(engine, test_gain, tmp_path):
    speaker = engine.default_output_id()
    bus = engine.create_virtual_input("tone", channels=2)
    assert engine.create_routing(bus, speaker)[0]
    ok, message = engine.add_plugin(speaker, test_gain, is_input=False)
    assert ok, message
    engine.set_plugin_parameter(speaker, 0, 0, 0.25, is_input=False)  # the test plugin at x0.5
    engine.start_engine()
    listener = Listener()
    try:
        feeder = [None]

        def run(seconds):
            import threading
            t = threading.Thread(target=lambda: feeder.__setitem__(0, play(engine, bus, seconds)))
            t.start()
            heard = listener.listen(seconds)
            t.join()
            return heard

        first = steady_level(run(1.5))
        expected = rms(sine(RATE, amplitude=0.2)) * 0.5
        assert first == pytest.approx(expected, rel=0.02), "the plugin's gain must be what is heard"

        stats = engine.get_performance_stats()
        assert stats['backend'] == 'native' and stats['xruns'] == 0
        assert stats['audio_thread_allocations'] == 0
        print(f"\nthrough AudioEngine + plugin: {first:.5f} (expected {expected:.5f}); callback mean "
              f"{stats['callback_mean_ms'] * 1000:.1f} us, worst {stats['callback_max_ms'] * 1000:.1f} us, "
              f"reported latency {stats['reported_latency_ms']} ms incl. the plugin's 64 samples")

        engine.set_device_master_volume(speaker, 0.5)
        engine.stop_engine()
        engine.start_engine()
        after = steady_level(run(1.5))
        assert after == pytest.approx(expected * 0.5, rel=0.03), "a restart must keep the fader and the plugin"

        manager = PresetManager(engine, tmp_path)
        saved = manager.capture("with plugin")
        assert saved['plugins'], "the preset must carry the plugin chain"
        engine.remove_plugin(speaker, 0, is_input=False)
        result = manager.apply(saved)
        assert result.restored_plugins == 1 and not result.missing_plugins
        restored = engine.plugin_instances(speaker, is_input=False)[0]
        assert restored.parameters()[0].normalized == pytest.approx(0.25)
    finally:
        listener.close()
