"""
Native Instruments Guitar Rig 7 — a commercial amp simulator, the plugin v0.2.0 reported as
"crashed while scanning" while Reaper loaded it — through ToneSphere's scanner and host.

Marked `hardware`: it needs Guitar Rig installed (a single 176 MB legacy-layout module in
the Common Files VST3 folder). What it proves is that this plugin scans, opens, processes
audio, keeps its state and shows its editor in ToneSphere; nothing about any other plugin.
"""

import time

import numpy as np
import pytest

from tests.signals import dominant_frequency, rms, sine
from tonesphere.native import VST3, Insert, NativeEngine, Node, Route, available
from tonesphere.plugins import PluginInstance
from tonesphere.plugins import scan as scanner

pytestmark = [pytest.mark.hardware, pytest.mark.skipif(not available(), reason="needs tonesphere_native.dll")]

RATE = 48000
BLOCK = 256
SRC, OUT = 1, 2


@pytest.fixture(scope="module")
def guitar_rig():
    results = scanner.scan(retry_failed=True)
    for result in results:
        for c in result.effects:
            if c.name.startswith("Guitar Rig"):
                assert result.status == scanner.OK, result.detail
                return c
    pytest.skip("Guitar Rig is not installed in a standard VST3 folder")


def process(plugin, signal):
    with NativeEngine(RATE, BLOCK) as engine:
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)],
                          [Insert(SRC, 0, VST3, plugin=plugin.handle)])
        out = [engine.process({SRC: signal[b * BLOCK:(b + 1) * BLOCK]}, {OUT: 2})[OUT]
               for b in range(len(signal) // BLOCK)]
        stats = engine.stats()
    return np.concatenate(out), stats


def test_it_scans_through_the_frozen_and_source_scanner_alike(guitar_rig):
    result = scanner.scan_module(guitar_rig.path)
    assert result.status == scanner.OK and [c.name for c in result.classes] == [guitar_rig.name]
    print(f"\n{guitar_rig.name} {guitar_rig.version} by {guitar_rig.vendor} ({guitar_rig.sdk_version}): scanned OK")


def test_it_processes_a_guitar_level_tone_into_finite_audible_output(guitar_rig):
    with PluginInstance(guitar_rig, RATE, BLOCK, 2) as plugin:
        tone = sine(RATE * 4, 220.0, amplitude=0.1)
        out, stats = process(plugin, tone)
        status = plugin.status()
    body = out[RATE * 2:]
    assert not status.crashed, status.fault
    assert np.all(np.isfinite(out))
    assert rms(body) > 1e-3, "the plugin made no sound"
    print(f"\nGuitar Rig 7, 220 Hz at -20 dBFS in: out rms {rms(body):.4f} ({20 * np.log10(rms(body)):.1f} dBFS), "
          f"dominant {dominant_frequency(body[:, 0]):.1f} Hz, plugin latency {status.latency_samples} samples, "
          f"engine callback mean {stats['callback_ns_mean'] / 1000:.0f} us, "
          f"max {stats['callback_ns_max'] / 1000:.0f} us")
    # An amp simulator distorts: what it must keep is the note, as the fundamental or a harmonic.
    peak = dominant_frequency(body[:, 0])
    assert min(abs(peak - 220.0 * k) for k in range(1, 9)) < 10.0, f"dominant {peak:.1f} Hz is not the note"


def test_its_own_master_volume_shapes_the_audio_by_what_it_says(guitar_rig):
    """The audio really goes through Guitar Rig's processing, not around it: its rack's
    master volume, set to what it displays as -12 dB, takes the output down 12 dB."""
    with PluginInstance(guitar_rig, RATE, BLOCK, 2) as plugin:
        volume = next(p for p in plugin.parameters() if p.title == 'Rack Master Volume')
        tone = sine(RATE * 3, 220.0, amplitude=0.1)
        plugin.set_parameter(volume.id, 0.6)
        shown = next(p for p in plugin.parameters() if p.id == volume.id).display.strip()
        out, _ = process(plugin, tone)
    assert shown == '-12.0dB', shown
    change_db = 20 * np.log10(rms(out[RATE:]) / rms(tone[RATE:]))
    print(f"\nRack Master Volume at {shown}: the output moved {change_db:+.2f} dB")
    assert change_db == pytest.approx(-12.0, abs=0.2)


def test_a_minute_of_audio_without_a_fault(guitar_rig):
    with PluginInstance(guitar_rig, RATE, BLOCK, 2) as plugin:
        out, stats = process(plugin, sine(RATE * 60, 110.0, amplitude=0.1))
        status = plugin.status()
    assert not status.crashed, status.fault
    assert status.blocks >= RATE * 60 // BLOCK
    assert np.all(np.isfinite(out))
    print(f"\n60 s through Guitar Rig 7: {status.blocks} blocks, 0 faults, callback max "
          f"{stats['callback_ns_max'] / 1000:.0f} us, {stats['rt_allocations']} engine allocations")


def test_its_state_round_trips_into_a_new_instance(guitar_rig):
    with PluginInstance(guitar_rig, RATE, BLOCK, 2) as plugin:
        process(plugin, np.zeros((BLOCK * 4, 2), np.float32))
        saved = plugin.state()
    assert len(saved.component) + len(saved.controller) > 0
    with PluginInstance(guitar_rig, RATE, BLOCK, 2) as fresh:
        fresh.restore(saved)
        out, _ = process(fresh, sine(RATE, 220.0, amplitude=0.1))
        assert not fresh.status().crashed and np.all(np.isfinite(out))
    print(f"\nstate: {len(saved.component)} + {len(saved.controller)} bytes, restored into a new instance")


def test_its_editor_opens_and_closes(guitar_rig):
    with PluginInstance(guitar_rig, RATE, BLOCK, 2) as plugin:
        assert plugin.has_editor()
        plugin.open_editor()
        time.sleep(3.0)
        plugin.close_editor()
