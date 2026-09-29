"""
The round-trip measurement against real endpoints.

The digital path (render endpoint -> its own loopback) has a delay fixed by the Windows
audio engine, so it checks the method end to end: the measurement must find it with high
confidence, and repeat to the frame. An acoustic or cable path is measured only where one
exists; where none does, the tool must say so and return nothing.
"""

import os

import pytest

from tonesphere.native import available
from tonesphere.native.roundtrip import CONFIDENCE_THRESHOLD, measure
from tonesphere.native.wasapi import default_endpoint, endpoints

pytestmark = [pytest.mark.hardware, pytest.mark.skipif(not available(), reason="needs tonesphere_native.dll")]


def test_the_digital_loopback_path_is_measured_and_repeatable():
    out = default_endpoint('render')
    if out is None:
        pytest.skip("no default render endpoint")
    first = measure(out.id, out.id, input_kind='loopback')
    second = measure(out.id, out.id, input_kind='loopback')
    assert first.measured_frames is not None, first.note
    assert first.confidence > CONFIDENCE_THRESHOLD
    assert first.nominal_ms < first.measured_ms < 500
    # Between runs the two device clocks wake in a different phase, and the capture side
    # crosses into the output's clock through a cushion that starts on a packet boundary:
    # so a restart may shift the delay by up to one device period (here 480 frames), and
    # no more. Within a run the delay is fixed.
    assert abs(first.measured_frames - second.measured_frames) <= 480, "a restart moved the delay by more than a period"
    print(f"\ndigital render->loopback: {first.measured_ms:.2f} / {second.measured_ms:.2f} ms "
          f"({first.measured_frames} / {second.measured_frames} frames), "
          f"confidence {first.confidence:.1f}, nominal {first.nominal_ms:.1f} ms, reported {first.reported_ms}")


def test_an_acoustic_attempt_either_measures_or_refuses():
    """
    No assertion about whether a path exists — a closed lid or a muted speaker is not a
    failure. The assertion is that the result is never a number without the confidence to
    back it.
    """
    outs = [e for e in endpoints() if e.flow == 'render']
    mics = [e for e in endpoints() if e.flow == 'capture']
    if not outs or not mics:
        pytest.skip("needs a render and a capture endpoint")
    result = measure(outs[0].id, mics[0].id, input_kind='capture')
    if result.measured_ms is None:
        assert result.note
    else:
        assert result.confidence > CONFIDENCE_THRESHOLD
    print(f"\n{result.output} -> {result.input}: measured {result.measured_ms} ms, "
          f"confidence {result.confidence:.1f}; {result.note}")


def test_the_engine_keeps_a_loopback_measurement_apart_from_the_round_trip():
    """As the Diagnostics view runs it: the digital path is recorded, never reported as the round trip."""
    from tonesphere.core.engine import AudioEngine

    engine = AudioEngine(sample_rate=48000, buffer_size=480, exclusive=False)
    engine.initialize()
    try:
        speaker = engine.default_output_id()
        if speaker is None:
            pytest.skip("no default output")
        result = engine.measure_round_trip(speaker)
        assert result['path'] == 'loopback' and result['measured_ms'] is not None, result['note']
        stats = engine.get_performance_stats()
        assert stats['round_trip']['measured_ms'] == result['measured_ms']
        assert stats['measured_round_trip_ms'] is None, "a loopback measurement is not a round trip"
        bus = engine.create_virtual_input("feed", channels=2)
        assert engine.create_routing(bus, speaker)[0]
        engine.start_engine()
        assert engine.host.is_running
        with pytest.raises(ValueError):
            engine.measure_round_trip(speaker)
    finally:
        engine.cleanup()


def test_an_interface_cable_round_trip():
    """
    A cable from an interface's output to its own input (TONESPHERE_TEST_INTERFACE, default
    "AI-04"): the physical round trip, DAC and ADC included, in exclusive mode. Without the
    cable the tool must refuse, and the test skips with its reason instead of passing.
    """
    name = os.environ.get('TONESPHERE_TEST_INTERFACE', 'AI-04')
    found = {e.flow: e for e in endpoints() if name in e.name}
    if 'render' not in found or 'capture' not in found:
        pytest.skip(f"no interface named '{name}' is connected")
    period = round(found['render'].min_period_ms * 48) if found['render'].min_period_ms else 144
    first = measure(found['render'].id, found['capture'].id, exclusive=True, block=period, level_db=-18.0)
    if first.measured_ms is None:
        assert first.note
        pytest.skip(f"MEASURED ROUND TRIP: -- ({name}: {first.note})")
    second = measure(found['render'].id, found['capture'].id, exclusive=True, block=period, level_db=-18.0)
    assert first.confidence > CONFIDENCE_THRESHOLD and second.measured_ms is not None, second.note
    assert first.nominal_ms < first.measured_ms < 200
    assert abs(first.measured_frames - second.measured_frames) <= period, "a restart moved the delay by over a period"
    print(f"\n{name} out -> cable -> in, exclusive, {period}-frame period: {first.measured_ms:.2f} / "
          f"{second.measured_ms:.2f} ms ({first.measured_frames} / {second.measured_frames} frames), confidence "
          f"{first.confidence:.1f}, nominal {first.nominal_ms:.1f} ms, reported {first.reported_ms}")
