"""
Whole-system loopback as a routable source, against a real output.

Two other processes (PortAudio, not ToneSphere) play 1 kHz and 440 Hz into one output at
once. ToneSphere routes that output's loopback through a bus to a network send's ring and
reads it back: both tones must be there, each at the level its application played — the
system mix, not one stream. The output is TONESPHERE_TEST_INTERFACE (default "AI-04"),
falling back to the default render endpoint.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from tests.native.test_engine_native_host import collect, network_sink
from tonesphere.native import available

pytestmark = [pytest.mark.hardware, pytest.mark.skipif(not available(), reason="needs tonesphere_native.dll")]

PEER = Path(__file__).with_name('cable_peer.py')
RATE = 48000


def level_at(signal: np.ndarray, freq: float) -> float:
    """Amplitude of the sinusoid at `freq`, from a Hann-windowed spectrum."""
    window = np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(signal * window)) / (window.sum() / 2)
    bin_ = int(round(freq * len(signal) / RATE))
    return float(spectrum[bin_ - 2:bin_ + 3].max())


def test_an_outputs_loopback_carries_every_application_it_plays():
    from tonesphere.core.engine import AudioEngine

    engine = AudioEngine(sample_rate=RATE, buffer_size=480, exclusive=False)
    engine.initialize()
    peers = []
    try:
        loops = [d for d in engine.get_devices() if d['origin'] == 'loopback']
        name = os.environ.get('TONESPHERE_TEST_INTERFACE', 'AI-04')
        chosen = next((d for d in loops if name in d['name']), None)
        if chosen is None:
            default = engine.default_output_id()
            chosen = next((d for d in loops if default is not None and d['id'] == default + 30000), None)
        if chosen is None:
            pytest.skip("no WASAPI output to take a loopback of")
        output_name = chosen['name'].removesuffix(' (loopback)')

        started, message = engine.start_udp_transport('127.0.0.1', 0)
        assert started, message
        bus = engine.create_virtual_input('system', channels=2)
        ok, message = engine.create_routing(chosen['id'], bus)
        assert ok, message
        source, dest = network_sink(engine, bus)
        engine.start_engine()
        assert engine.host.is_running
        kinds = sorted(s['kind'] for s in engine.host.stream_status())
        assert kinds == [1, 3], f"a silent render clock and the loopback, got {kinds}"

        for freq, amplitude in ((1000.0, 0.1), (440.0, 0.05)):
            peers.append(subprocess.Popen([sys.executable, str(PEER), 'play', output_name, '3', str(freq),
                                           str(amplitude)]))
        time.sleep(0.8)
        collect(engine, source, dest, 0.2)   # what arrived before both were playing
        heard = collect(engine, source, dest, 1.2)[:, 0].astype(np.float64)
        assert len(heard) > RATE, f"only {len(heard)} frames arrived"
        heard = heard[:RATE]
        a1000, a440 = level_at(heard, 1000.0), level_at(heard, 440.0)
        print(f"\nloopback of {output_name}: 1 kHz at {a1000:.4f} (played 0.1), 440 Hz at {a440:.4f} "
              f"(played 0.05); streams {engine.host.stream_status()[0]['message'] or 'running'}")
        assert a1000 == pytest.approx(0.1, rel=0.05)
        assert a440 == pytest.approx(0.05, rel=0.05)
        assert engine.get_performance_stats()['audio_thread_allocations'] == 0
    finally:
        for p in peers:
            p.wait(10)
        engine.cleanup()
