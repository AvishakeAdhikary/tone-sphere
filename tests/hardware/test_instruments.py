"""
VST3 instruments played by MIDI, against real third-party instruments.

Surge XT (GPL-3.0, installed from its official release) and Dexed (GPL-3.0, the DX7
emulation; its official 1.0.1 Windows zip, MD5 matching the project's own checksum file,
unzipped into TONESPHERE_TEST_PLUGINS — nothing installed). Each is opened on a bus of its
own (`AudioEngine.create_instrument`), the bus is routed to a network send's ring, and a
note is sent: from the engine, from the on-screen keyboard, over REST, and finally out of
the AI-04 and back through its cable. What comes back must be at the note's pitch — A4 at
440 Hz, middle C at 261.63 Hz — and fall silent after the note is released.

Marked `hardware`: these depend on software installed here, not on anything CI has.
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest

from tests.native.test_engine_native_host import collect, network_sink
from tests.signals import dominant_frequency, rms
from tonesphere.native import available

pytestmark = [pytest.mark.hardware, pytest.mark.skipif(not available(), reason="needs tonesphere_native.dll")]

RATE = 48000
A4, C4 = 440.0, 261.6256


def find_class(name: str, roots=None):
    from tonesphere.plugins import scan as scanner

    for result in scanner.scan(roots):
        for c in result.effects:
            if c.name == name and result.status == scanner.OK:
                return c
    return None


@pytest.fixture(scope="module")
def surge():
    info = find_class("Surge XT")
    if info is None:
        pytest.skip("Surge XT is not installed in a standard VST3 folder")
    return info


@pytest.fixture(scope="module")
def dexed():
    folder = os.environ.get('TONESPHERE_TEST_PLUGINS', r'C:\ToneSphereVM\downloads\plugins')
    info = find_class("Dexed", [Path(folder)]) if Path(folder).is_dir() else None
    if info is None:
        pytest.skip(f"Dexed is not in {folder} (set TONESPHERE_TEST_PLUGINS)")
    return info


@pytest.fixture
def engine():
    from tonesphere.core.engine import AudioEngine

    e = AudioEngine(sample_rate=RATE, buffer_size=256, exclusive=False)
    e.initialize()
    started, message = e.start_udp_transport('127.0.0.1', 0)
    assert started, message
    yield e
    e.cleanup()


def instrument_on_a_ring(engine, info):
    ok, message, bus = engine.create_instrument(info)
    assert ok, message
    source, dest = network_sink(engine, bus)
    engine.start_engine()
    time.sleep(0.3)
    collect(engine, source, dest, 0.1)
    return bus, source, dest


def play(engine, bus, source, dest, note, press=None, release=None):
    """Hold `note` for a second, release it, listen two more; (held audio, the last quarter second)."""
    (press or (lambda: engine.note_on(bus, note)))()
    held = collect(engine, source, dest, 1.0)[:, 0].astype(np.float64)
    (release or (lambda: engine.note_off(bus, note)))()
    after = collect(engine, source, dest, 2.0)[:, 0].astype(np.float64)
    return held[RATE // 4:], after[-RATE // 4:]


def assert_pitch(name, held, after, expected):
    heard = dominant_frequency(held)
    print(f"\n{name}: {heard:.2f} Hz (expected {expected:.2f}), rms {rms(held):.4f}; after release {rms(after):.2e}")
    assert rms(held) > 0.01, "the instrument made no sound"
    assert heard == pytest.approx(expected, rel=0.01)
    assert rms(after) < 1e-3, "the note did not stop"


def test_surge_xt_plays_a4_at_440_hz(engine, surge):
    bus, source, dest = instrument_on_a_ring(engine, surge)
    held, after = play(engine, bus, source, dest, 69)
    assert_pitch("Surge XT, note 69", held, after, A4)
    assert engine.instruments()[0]['instrument'] == 'Surge XT'


def test_dexed_plays_the_notes_it_is_sent(engine, dexed):
    """
    Dexed's default voice puts its fundamental a whole number of octaves from the key (its
    operators' frequency ratios are the patch's choice); what MIDI decides is the note. So:
    each note at its frequency times the same power of two, and a fifth apart exactly.
    """
    bus, source, dest = instrument_on_a_ring(engine, dexed)
    held_c, after_c = play(engine, bus, source, dest, 60)
    held_g, after_g = play(engine, bus, source, dest, 67)
    c, g = dominant_frequency(held_c), dominant_frequency(held_g)
    octaves = round(np.log2(c / C4))
    print(f"\nDexed: note 60 at {c:.2f} Hz, note 67 at {g:.2f} Hz; the voice sits {octaves:+d} octave(s) from the key; "
          f"after release {rms(after_c):.2e} / {rms(after_g):.2e}")
    assert c == pytest.approx(C4 * 2.0 ** octaves, rel=0.01)
    assert g / c == pytest.approx(2 ** (7 / 12), rel=0.01)
    assert rms(held_c) > 0.01 and rms(after_c) < 1e-3 and rms(after_g) < 1e-3


def test_the_on_screen_keyboard_plays_the_instrument(engine, surge, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from tonesphere.ui.keyboard import KeyboardDialog

    QApplication.instance() or QApplication([])
    bus, source, dest = instrument_on_a_ring(engine, surge)
    dialog = KeyboardDialog(engine, bus, "Surge XT")
    try:
        def press():
            dialog.keys.press(60)
            assert dialog.tasks.flush()

        def release():
            dialog.keys.release(60)
            assert dialog.tasks.flush()

        held, after = play(engine, bus, source, dest, 60, press, release)
        assert dialog.status.text() == ''
    finally:
        dialog.done(0)
    assert_pitch("Surge XT from the keyboard, C4", held, after, C4)


def test_an_instrument_added_and_played_over_rest(engine, surge):
    from fastapi.testclient import TestClient

    import tonesphere.api.server as server
    from tonesphere.core.engine_factory import UnifiedAudioEngine

    unified = UnifiedAudioEngine.__new__(UnifiedAudioEngine)
    unified.engine = engine
    server.audio_engine = unified
    try:
        client = TestClient(server.app)
        response = client.post("/instruments", json={"path": surge.path, "uid": surge.uid})
        assert response.status_code == 200, response.text
        bus = response.json()['bus_id']
        assert client.get("/instruments").json()[0]['id'] == bus
        source, dest = network_sink(engine, bus)
        engine.start_engine()
        time.sleep(0.3)
        collect(engine, source, dest, 0.1)
        held, after = play(engine, bus, source, dest, 69,
                           lambda: client.post(f"/midi/{bus}/note", json={"note": 69}).raise_for_status(),
                           lambda: client.post(f"/midi/{bus}/note", json={"note": 69, "on": False}).raise_for_status())
    finally:
        server.audio_engine = None
    assert_pitch("Surge XT over REST, note 69", held, after, A4)


def test_an_instrument_heard_through_the_interface_cable(engine, surge):
    """Out of the AI-04's output and back in through its cable: an instrument on real hardware."""
    name = os.environ.get('TONESPHERE_TEST_INTERFACE', 'AI-04')
    devices = engine.get_devices()
    out = next((d for d in devices if d['origin'] == 'hardware' and d['direction'] == 'output' and name in d['name']),
               None)
    inp = next((d for d in devices if d['origin'] == 'hardware' and d['direction'] == 'input' and name in d['name']),
               None)
    if out is None or inp is None:
        pytest.skip(f"no interface named '{name}'")
    ok, message, bus = engine.create_instrument(surge)
    assert ok, message
    assert engine.create_routing(bus, out['id'], 0.5)[0]
    source, dest = network_sink(engine, inp['id'])
    engine.start_engine()
    time.sleep(0.5)
    collect(engine, source, dest, 0.2)
    held, after = play(engine, bus, source, dest, 69)
    assert_pitch(f"Surge XT out of {name} and back through the cable, note 69", held, after, A4)
