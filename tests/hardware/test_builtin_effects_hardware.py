"""
A built-in effect on a real output, heard through a cable back into the same interface
(TONESPHERE_TEST_INTERFACE, default "AI-04", with its output cabled to its input).

A sweep goes into a bus routed to the interface's output; the interface's input is routed
to a network send's ring and read back, and cross-correlated with the sweep: once with
nothing on the output, once with the built-in delay set to 100 ms, no feedback, echo at half
level. The delay passes the dry signal and adds the echo, so the second capture holds the
sweep twice: the echo's peak must sit 100 ms after the dry one, to within a sample either
way, and stand clear of everything but the dry peak. (Plain cross-correlation, not the
round-trip tool's GCC-PHAT: whitening a signal that holds its own echo turns the echo into a
train of peaks at every multiple of the delay.)
"""

import os
import time

import numpy as np
import pytest

from tests.native.test_engine_native_host import network_sink
from tonesphere.native import available
from tonesphere.native.roundtrip import sweep

pytestmark = [pytest.mark.hardware, pytest.mark.skipif(not available(), reason="needs tonesphere_native.dll")]

RATE = 48000


def correlation(captured: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Cross-correlation over the first second of lags, whole, so two peaks can be read."""
    n = 1 << int(np.ceil(np.log2(len(captured) + len(reference))))
    return np.fft.irfft(np.fft.rfft(captured, n) * np.conj(np.fft.rfft(reference, n)), n)[:RATE]


def through(engine, bus, source, dest, stimulus) -> np.ndarray:
    engine.host.read_available(source, dest, RATE * 4)
    written, heard = 0, []
    deadline = time.time() + len(stimulus) / RATE + 3.0
    while time.time() < deadline:
        if written < len(stimulus):
            written += engine.write_to_bus(bus, stimulus[written:written + 4096])
        block = engine.host.read_available(source, dest, RATE)
        if block is not None and len(block):
            heard.append(block)
        time.sleep(0.005)
    captured = np.concatenate(heard)[:, 0].astype(np.float64)
    reference = np.zeros(len(captured))
    reference[:len(stimulus)] = stimulus[:, 0]
    return correlation(captured, reference)


def test_a_built_in_delay_is_heard_through_the_cable_exactly_as_set():
    from tonesphere.core.engine import AudioEngine

    name = os.environ.get('TONESPHERE_TEST_INTERFACE', 'AI-04')
    engine = AudioEngine(sample_rate=RATE, buffer_size=480, exclusive=False)
    engine.initialize()
    try:
        devices = engine.get_devices()
        out = next((d for d in devices if d['origin'] == 'hardware' and d['direction'] == 'output'
                    and name in d['name']), None)
        inp = next((d for d in devices if d['origin'] == 'hardware' and d['direction'] == 'input'
                    and name in d['name']), None)
        if out is None or inp is None:
            pytest.skip(f"no interface named '{name}'")
        started, message = engine.start_udp_transport('127.0.0.1', 0)
        assert started, message
        bus = engine.create_virtual_input('sweep', channels=2)
        assert engine.create_routing(bus, out['id'])[0]
        source, dest = network_sink(engine, inp['id'])
        engine.start_engine()
        time.sleep(0.5)

        signal = sweep(RATE, 0.5, 150.0, 12000.0, -18.0)
        stimulus = np.repeat(np.concatenate([signal, np.zeros(RATE // 2, np.float32)]).reshape(-1, 1), 2, axis=1)
        plain = through(engine, bus, source, dest, stimulus)
        dry = int(np.argmax(plain))

        ok, message = engine.add_builtin(out['id'], 'delay', is_input=False, values=[100.0, 0.0, 0.5])
        assert ok, message
        time.sleep(0.3)
        delayed = through(engine, bus, source, dest, stimulus)
        # The dry peak moves with each restart's phase; find it again, then the echo after it.
        now_dry = int(np.argmax(delayed[:dry + RATE // 20]))
        window = slice(now_dry + RATE // 20, now_dry + RATE * 3 // 20)
        echo = window.start + int(np.argmax(delayed[window]))
        guard = np.ones(len(delayed), bool)
        for peak in (now_dry, echo):
            guard[max(0, peak - RATE // 200):peak + RATE // 200] = False   # the sweep's own skirt
        confidence = float(delayed[echo] / np.max(np.abs(delayed[guard])))
        print(f"\n{name}: the sweep through the cable at {dry / RATE * 1000:.2f} ms; with the built-in delay at "
              f"100 ms, its echo {(echo - now_dry) / RATE * 1000:.3f} ms after the dry sweep (confidence "
              f"{confidence:.1f})")
        assert confidence > 4
        assert abs((echo - now_dry) - RATE // 10) <= 1
    finally:
        engine.cleanup()
