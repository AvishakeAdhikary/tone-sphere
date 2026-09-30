"""
The built-in effects, driven from the Inserts dialog's own widgets, on the native engine with
no device: a bus feeding a network send's ring, so what the effect did is read back exactly.

The dialog's controls are what is under test — the menu adds the effect, the parameter
panel's slider sets it — and the audio is the proof: a high-pass EQ set to 1 kHz from the
dialog must take 100 Hz down by 20 dB or more and leave 5 kHz within half a dB, a
compressor added from it must reduce a loud tone by what its own gain-reduction readout
says, and both must survive what rebuilds a chain: a block-size change and a preset.
"""

import math
import sys
import time

import numpy as np
import pytest

from tests.native.test_engine_native_host import collect, network_sink
from tests.signals import RATE, sine
from tonesphere.core.engine import AudioEngine

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason="the native host is Windows-only")


@pytest.fixture
def qt_app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def rig():
    engine = AudioEngine(sample_rate=RATE, buffer_size=1024)
    engine.initialize()
    started, message = engine.start_udp_transport('127.0.0.1', 0)
    assert started, message
    bus = engine.create_virtual_input("fx", channels=2)
    source, dest = network_sink(engine, bus)
    engine.start_engine()
    time.sleep(0.2)
    yield engine, bus, source, dest
    engine.cleanup()


def through(rig, tone: np.ndarray) -> np.ndarray:
    engine, bus, source, dest = rig
    collect(engine, source, dest, 0.1)
    # The bus's feed ring holds less than a second, so the tone goes in as the engine takes it.
    written, out = 0, []
    deadline = time.time() + len(tone) / RATE + 2.0
    while written < len(tone) and time.time() < deadline:
        written += engine.write_to_bus(bus, tone[written:written + 8192])
        block = engine.host.read_available(source, dest, RATE)
        if block is not None and len(block):
            out.append(block)
        time.sleep(0.01)
    assert written == len(tone)
    out.append(collect(engine, source, dest, 0.4))
    heard = np.concatenate(out)
    return heard[len(heard) // 4: len(heard) // 4 + RATE // 4, 0].astype(np.float64)


def db(x: np.ndarray) -> float:
    return 20 * math.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-12))


def set_row(dialog, title: str, value: float):
    """Select a parameter by its displayed title and move the panel's slider to `value` (normalized)."""
    from tonesphere.ui.plugin_views import SLIDER_STEPS

    panel = dialog.parameters
    row = next(r for r in range(panel.table.rowCount()) if panel.table.item(r, 0).text() == title)
    panel.table.selectRow(row)
    panel.slider.setValue(round(value * SLIDER_STEPS))
    assert dialog.tasks.flush()
    return panel.table.item(row, 1).text()


def open_dialog(engine, bus):
    from tonesphere.ui.plugin_views import InsertsDialog

    dialog = InsertsDialog(engine, bus, False, "fx")
    assert dialog.tasks.flush()
    return dialog


def test_a_high_pass_set_from_the_dialog_filters_what_the_bus_carries(qt_app, rig):
    from tonesphere.engine.builtins import EQ_TYPES

    engine, bus, *_ = rig
    dialog = open_dialog(engine, bus)
    try:
        dialog.add_builtin('eq')
        assert dialog.tasks.flush() and dialog.tasks.flush()
        assert [e['type'] for e in dialog.entries()] == ['eq']
        assert dialog.tasks.flush()
        shown_type = set_row(dialog, 'Band 1: Type', EQ_TYPES.index('highpass') / (len(EQ_TYPES) - 1))
        shown_freq = set_row(dialog, 'Band 1: Frequency', math.log(1000 / 20) / math.log(20000 / 20))
        assert shown_type == 'High-pass' and shown_freq == '1.00 kHz', (shown_type, shown_freq)
        values = engine.insert_instance(bus, 0, False).values
        assert values[0] == 4 and values[1] == pytest.approx(1000, rel=0.01)

        low = db(through(rig, sine(RATE, 100.0, amplitude=0.25)))
        high = db(through(rig, sine(RATE, 5000.0, amplitude=0.25)))
        reference = 20 * math.log10(0.25 / math.sqrt(2))
        print(f"\nhigh-pass at 1 kHz from the dialog: 100 Hz {low - reference:+.1f} dB, "
              f"5 kHz {high - reference:+.2f} dB")
        assert low - reference <= -20.0
        assert abs(high - reference) <= 0.5

        # A new block size rebuilds the native engine: the EQ must come back as set.
        engine.set_buffer_size(512)
        time.sleep(0.2)
        again = db(through(rig, sine(RATE, 100.0, amplitude=0.25)))
        assert again - reference <= -20.0, "the rebuild lost the EQ's settings"
    finally:
        dialog.done(0)


def test_a_compressor_from_the_dialog_reduces_by_what_it_reports(qt_app, rig):
    engine, bus, *_ = rig
    dialog = open_dialog(engine, bus)
    try:
        dialog.add_builtin('compressor')
        assert dialog.tasks.flush() and dialog.tasks.flush()
        loud = sine(RATE * 2, 1000.0, amplitude=0.7)
        reference = db(loud[:RATE, 0].astype(np.float64))
        heard = db(through(rig, loud))
        reduction = engine.list_inserts(bus, False)[0]['gain_reduction_db']
        print(f"\ncompressor (defaults: -18 dB, 4:1) on a {reference:.1f} dBFS tone: {heard - reference:+.1f} dB "
              f"measured, {reduction} dB reported")
        assert heard - reference <= -3.0
        assert reduction is not None and abs(abs(reduction) - (reference - heard)) <= 1.5
        dialog.bypass_button.setChecked(True)
        assert dialog.tasks.flush()
        assert abs(db(through(rig, loud)) - reference) <= 0.1, "bypassed, the tone passes untouched"
    finally:
        dialog.done(0)


def test_a_preset_brings_the_chain_back_with_its_values(rig, tmp_path):
    from tonesphere.core.presets import PresetManager

    engine, bus, *_ = rig
    assert engine.add_builtin(bus, 'delay', False)[0]
    assert engine.add_builtin(bus, 'eq', False)[0]
    engine.set_builtin_value(bus, 1, 0, 4, False)
    engine.set_builtin_value(bus, 1, 1, 2500.0, False)
    engine.set_insert_bypassed(bus, 0, True, False)
    manager = PresetManager(engine, tmp_path)
    saved = manager.capture('fx')
    while engine.list_inserts(bus, False):
        engine.remove_insert(bus, 0, False)
    result = manager.apply(saved)
    chain = engine.list_inserts(bus, False)
    assert [(e['type'], e['bypassed']) for e in chain] == [('delay', True), ('eq', False)], result.warnings
    assert engine.insert_instance(bus, 1, False).values[:2] == [4, 2500.0]
