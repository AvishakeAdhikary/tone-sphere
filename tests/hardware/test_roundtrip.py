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


def test_the_laptops_own_speaker_to_its_own_microphone():
    """
    The acoustic round trip of this machine: its speakers (TONESPHERE_TEST_SPEAKER, default
    "Realtek") to its microphone array (TONESPHERE_TEST_MICROPHONE, default "Microphone
    Array"), captured in raw mode so no echo cancellation or noise suppression edits the
    sweep. The speakers are unmuted at 60 % for the sweeps only, then put back exactly as
    they were. Ten runs; each is a figure or `--` with its confidence.
    """
    from tests.hardware import endpoint_volume

    speaker_name = os.environ.get('TONESPHERE_TEST_SPEAKER', 'Realtek')
    mic_name = os.environ.get('TONESPHERE_TEST_MICROPHONE', 'Microphone Array')
    speaker = next((e for e in endpoints() if e.flow == 'render' and speaker_name in e.name), None)
    mic = next((e for e in endpoints() if e.flow == 'capture' and mic_name in e.name), None)
    if speaker is None or mic is None:
        pytest.skip(f"no '{speaker_name}' output or '{mic_name}' input")
    before = endpoint_volume.get(speaker.id)
    results = []
    with endpoint_volume.held_at(speaker.id, 0.6):
        for _ in range(10):
            results.append(measure(speaker.id, mic.id, level_db=-12.0))
    assert endpoint_volume.get(speaker.id) == pytest.approx(before, abs=1e-6), "the volume was not restored"
    for r in results:
        figure = f"{r.measured_ms:.2f} ms ({r.measured_frames} frames)" if r.measured_ms is not None else "--"
        print(f"\n{speaker.name} -> {mic.name}: {figure}, confidence {r.confidence:.1f}, peak "
              f"{r.peak_dbfs if r.peak_dbfs is None else round(r.peak_dbfs, 1)} dBFS; {r.note}")
    measured = [r for r in results if r.measured_ms is not None]
    for r in measured:
        assert r.confidence > CONFIDENCE_THRESHOLD and r.nominal_ms < r.measured_ms < 500
    if len(measured) >= 2:
        spread = max(r.measured_frames for r in measured) - min(r.measured_frames for r in measured)
        print(f"{len(measured)} of 10 measured; spread {spread} frames")


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


def interface_endpoints():
    name = os.environ.get('TONESPHERE_TEST_INTERFACE', 'AI-04')
    found = {e.flow: e for e in endpoints() if name in e.name}
    if 'render' not in found or 'capture' not in found:
        pytest.skip(f"no interface named '{name}' is connected")
    return name, found['render'], found['capture']


def report(tag, first, second):
    print(f"\n{tag}: {first.measured_ms:.2f} / {second.measured_ms:.2f} ms ({first.measured_frames} / "
          f"{second.measured_frames} frames), confidence {first.confidence:.1f}, captured peak "
          f"{first.peak_dbfs:.1f} dBFS, nominal {first.nominal_ms:.1f} ms, reported {first.reported_ms}")


def test_an_interface_cable_round_trip():
    """
    A cable from an interface's output to its own input (TONESPHERE_TEST_INTERFACE, default
    "AI-04"): the physical round trip, DAC and ADC included, in exclusive mode. Without the
    cable the tool must refuse, and the test skips with its reason instead of passing.
    """
    name, out, inp = interface_endpoints()
    period = round(out.min_period_ms * 48) if out.min_period_ms else 144
    first = measure(out.id, inp.id, exclusive=True, block=period, level_db=-18.0)
    if first.measured_ms is None:
        assert first.note
        pytest.skip(f"MEASURED ROUND TRIP: -- ({name}: {first.note})")
    second = measure(out.id, inp.id, exclusive=True, block=period, level_db=-18.0)
    assert first.confidence > CONFIDENCE_THRESHOLD and second.measured_ms is not None, second.note
    assert first.nominal_ms < first.measured_ms < 200
    # A peak near full scale means the cable path clipped, and a clipped sweep can still
    # correlate: the figure would be right while the path is not.
    assert first.peak_dbfs < -1.0, f"the captured sweep peaked at {first.peak_dbfs:.1f} dBFS: the path clips"
    # The capture side starts in a different phase against the render period each time;
    # on the AI-04 the delay lands on one of two values a USB packet group apart (96 frames).
    assert abs(first.measured_frames - second.measured_frames) <= period, "a restart moved the delay by over a period"
    report(f"{name} out -> cable -> in, exclusive, {period}-frame period", first, second)


def test_an_interface_cable_round_trip_in_shared_mode():
    name, out, inp = interface_endpoints()
    first = measure(out.id, inp.id, exclusive=False, block=480, level_db=-18.0)
    if first.measured_ms is None:
        pytest.skip(f"MEASURED ROUND TRIP: -- ({name}: {first.note})")
    second = measure(out.id, inp.id, exclusive=False, block=480, level_db=-18.0)
    assert second.measured_ms is not None, second.note
    assert first.peak_dbfs < -1.0
    assert abs(first.measured_frames - second.measured_frames) <= 480
    report(f"{name} out -> cable -> in, shared, 480-frame block", first, second)


def test_an_asio_round_trip_through_the_interface_cable():
    """
    Every registered ASIO driver, run from its own buffer switch with inputs and outputs on
    one clock. Whether a driver reaches the interface depends on its own configuration
    (FlexASIO's default is the default devices, shared), so a driver that finds no path
    records `--` and the reason, and only a confident measurement is printed as one.
    """
    from tonesphere.native import asio
    from tonesphere.native.roundtrip import measure_asio

    interface_endpoints()
    if not asio.available() or not [d for d in asio.drivers() if d.dll_present]:
        pytest.skip("ASIO HARDWARE VERIFICATION: NOT AVAILABLE ON THIS MACHINE (no ASIO driver)")
    measured = 0
    for d in [d for d in asio.drivers() if d.dll_present]:
        info = asio.query(d.name)
        if not info.inputs or not info.outputs or 48000 not in info.sample_rates:
            continue
        result = measure_asio(d.name, inputs=tuple(range(len(info.inputs))),
                              outputs=tuple(range(min(2, len(info.outputs)))), level_db=-18.0)
        if result.measured_ms is None:
            print(f"\n{d.name}: MEASURED ROUND TRIP: -- ({result.note})")
            continue
        assert result.peak_dbfs < -1.0
        measured += 1
        print(f"\n{d.name} (buffer {result.block}): {result.measured_ms:.2f} ms ({result.measured_frames} frames), "
              f"confidence {result.confidence:.1f}, peak {result.peak_dbfs:.1f} dBFS, driver-reported "
              f"{result.reported_ms} ms")
    if not measured:
        pytest.skip("no ASIO driver reached a path from output to input")
