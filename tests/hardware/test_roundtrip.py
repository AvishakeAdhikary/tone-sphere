"""
The round-trip measurement against real endpoints.

The digital path (render endpoint -> its own loopback) has a delay fixed by the Windows
audio engine, so it checks the method end to end: the measurement must find it with high
confidence, and repeat to the frame. An acoustic or cable path is measured only where one
exists; where none does, the tool must say so and return nothing.
"""

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
