"""
The native VST3 host against a real third-party plugin: Surge XT (GPL-3.0, open source,
installed from its official release into the per-user VST3 folder). Marked `hardware`
because it depends on software installed on the machine, not on anything CI has.

Only what runs here is claimed. Surge XT working says nothing about Guitar Rig, Neural DSP
or any other plugin; commercial plugins were not tested.
"""

import time

import numpy as np
import pytest

from tests.signals import dominant_frequency, rms, sine, white_noise
from tonesphere.native import VST3, Insert, NativeEngine, Node, Route, available
from tonesphere.plugins import PluginInstance
from tonesphere.plugins import scan as scanner

pytestmark = [pytest.mark.hardware, pytest.mark.skipif(not available(), reason="needs tonesphere_native.dll")]

RATE = 48000
BLOCK = 256
SRC, OUT = 1, 2


@pytest.fixture(scope="module")
def surge_fx():
    for result in scanner.scan():
        for c in result.effects:
            if c.name == "Surge XT Effects":
                assert result.status == scanner.OK
                return c
    pytest.skip("Surge XT Effects is not installed in a standard VST3 folder")


def process(plugin, signal):
    with NativeEngine(RATE, BLOCK) as engine:
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)],
                          [Insert(SRC, 0, VST3, plugin=plugin.handle)])
        engine.process({SRC: np.zeros((BLOCK, 2), np.float32)}, {OUT: 2})
        out = [engine.process({SRC: signal[b * BLOCK:(b + 1) * BLOCK]}, {OUT: 2})[OUT]
               for b in range(len(signal) // BLOCK)]
        stats = engine.stats()
    return np.concatenate(out), stats


def param(plugin, title):
    return next(p for p in plugin.parameters() if p.title == title)


def test_its_default_delay_echoes_an_impulse_250_ms_later(surge_fx):
    with PluginInstance(surge_fx, RATE, BLOCK, 2) as plugin:
        assert param(plugin, "FX Type").display == "Delay"
        assert param(plugin, "Delay Time Left").display == "250.00 ms"
        impulse = np.zeros((RATE, 2), np.float32)
        impulse[BLOCK * 4] = 0.5
        out, stats = process(plugin, impulse)
    envelope = np.abs(out[:, 0])
    first = int(np.argmax(envelope[:BLOCK * 8]))
    later = envelope[first + RATE // 8:]
    echo = first + RATE // 8 + int(np.argmax(later))
    delay_ms = (echo - first) / RATE * 1000
    print(f"\nSurge XT Effects: dry peak at {first}, first echo {echo - first} samples later ({delay_ms:.2f} ms); "
          f"engine callback mean {stats['callback_ns_mean'] / 1000:.1f} us, "
          f"max {stats['callback_ns_max'] / 1000:.1f} us")
    assert delay_ms == pytest.approx(250.0, abs=2.0)
    assert np.all(np.isfinite(out))


def test_a_tone_comes_through_at_its_own_frequency(surge_fx):
    with PluginInstance(surge_fx, RATE, BLOCK, 2) as plugin:
        tone = sine(RATE * 2, 1000.0, amplitude=0.2)
        out, _ = process(plugin, tone)
    assert dominant_frequency(out[RATE:]) == pytest.approx(1000.0, abs=5.0)
    assert rms(out[RATE:]) > 0.05


def test_its_bypass_parameter_passes_the_dry_signal(surge_fx):
    with PluginInstance(surge_fx, RATE, BLOCK, 2) as plugin:
        bypass = param(plugin, "Bypass")
        assert bypass.is_bypass
        plugin.set_parameter(bypass.id, 1.0)
        signal = white_noise(BLOCK * 96, amplitude=0.2)
        out, _ = process(plugin, signal)
    # Allow the plugin a short crossfade into bypass, then the dry signal must be untouched.
    settled = slice(RATE // 8, len(out))
    assert np.allclose(out[settled], signal[settled], atol=1e-5)


def test_its_state_round_trips_into_a_new_instance(surge_fx):
    with PluginInstance(surge_fx, RATE, BLOCK, 2) as plugin:
        target = param(plugin, "Delay Time Left")
        plugin.set_parameter(target.id, 0.3)
        process(plugin, np.zeros((BLOCK * 2, 2), np.float32))
        saved = plugin.state()
        expected = param(plugin, "Delay Time Left").display
    with PluginInstance(surge_fx, RATE, BLOCK, 2) as fresh:
        fresh.restore(saved)
        assert param(fresh, "Delay Time Left").display == expected
    print(f"\nstate: {len(saved.component)} + {len(saved.controller)} bytes; Delay Time Left restored as {expected}")


def test_its_editor_opens_and_closes(surge_fx):
    with PluginInstance(surge_fx, RATE, BLOCK, 2) as plugin:
        assert plugin.has_editor()
        plugin.open_editor()
        time.sleep(0.5)
        plugin.close_editor()


def test_the_instrument_opens_without_input_and_stays_silent_without_notes(surge_fx):
    synth = next((c for r in scanner.scan() for c in r.effects if c.name == "Surge XT"), None)
    if synth is None:
        pytest.skip("Surge XT (the instrument) is not installed")
    with PluginInstance(synth, RATE, BLOCK, 2) as plugin:
        out, _ = process(plugin, np.zeros((BLOCK * 20, 2), np.float32))
    assert np.all(np.isfinite(out)) and np.max(np.abs(out)) < 1e-3
