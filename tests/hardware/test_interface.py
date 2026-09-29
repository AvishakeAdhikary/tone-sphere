"""
The native WASAPI engine against a physical USB audio interface.

The interface is found by name: TONESPHERE_TEST_INTERFACE (default "AI-04", the Audio Array
AI-04 on the development machine). Every test skips when it is not connected.

What these prove: the interface's own clock drives the engine, in exclusive and shared
mode, with no dropouts and no audio-thread allocation; its input delivers real samples at
the rate asked for; and a monitoring path (input -> gain -> output) carries what the input
heard at the gain set. What they cannot prove on their own: that sound left the output
jack. Nothing here listens to the jack; that needs a cable from the output back to an
input, which is what test_roundtrip.py's cable test uses.

With an instrument plugged in and not played, the input is its noise floor, which on a
guitar pickup is mains hum. It is printed, not asserted: whether a pickup hums is not
ToneSphere's business.
"""

import os
import time

import numpy as np
import pytest

from tests.signals import dominant_frequency, rms
from tonesphere.native import NativeEngine, Node, Route, available
from tonesphere.native.wasapi import StreamSpec, endpoints

pytestmark = [pytest.mark.hardware, pytest.mark.skipif(not available(), reason="needs tonesphere_native.dll")]

NAME = os.environ.get('TONESPHERE_TEST_INTERFACE', 'AI-04')
IN, OUT, CAP, BUS, MON = 1, 2, 3, 4, 5
SECONDS = 20.0


@pytest.fixture(scope='module')
def interface():
    found = {e.flow: e for e in endpoints() if NAME in e.name}
    if 'render' not in found or 'capture' not in found:
        pytest.skip(f"INTERFACE VERIFICATION: no interface named '{NAME}' is connected "
                    f"(set TONESPHERE_TEST_INTERFACE to test another)")
    return found


def run_duplex(interface, rate, block, exclusive, seconds=SECONDS, gain=0.0):
    render, capture = interface['render'], interface['capture']
    with NativeEngine(rate, block) as engine:
        # The input goes through a gain bus to the output; the bus is also tapped into a ring,
        # beside a tap of the raw input, so what the output was given can be compared with what
        # the input delivered sample for sample.
        channels = capture.mix_channels
        engine.apply_plan([Node.source(IN, channels), Node.bus(BUS, channels), Node.sink(OUT, render.mix_channels),
                           Node.sink(CAP, channels, ring_frames=rate * 4),
                           Node.sink(MON, channels, ring_frames=rate * 4)],
                          [Route(IN, BUS), Route(BUS, OUT), Route(IN, CAP), Route(BUS, MON)])
        # Silent by default: the test must not play an instrument's hum into someone's ears.
        engine.set_node_gain(BUS, gain)
        engine.start_wasapi([
            StreamSpec(OUT, 'render', render.mix_channels, render.id, exclusive=exclusive, allow_shared_fallback=False),
            StreamSpec(IN, 'capture', capture.mix_channels, capture.id, exclusive=exclusive,
                       allow_shared_fallback=False),
        ], master=0)
        captured, monitored = [], []
        deadline = time.time() + seconds
        while time.time() < deadline:
            captured.append(engine.port_read(CAP, rate))
            monitored.append(engine.port_read(MON, rate))
            time.sleep(0.05)
        status = engine.stream_status()
        stats = engine.stats()
        engine.stop_backend()
        captured.append(engine.port_read(CAP, rate * 4))
        monitored.append(engine.port_read(MON, rate * 4))
    return np.concatenate(captured), status, stats, np.concatenate(monitored)


def describe(status, stats, captured, rate):
    lines = []
    for s in status:
        lines.append(f"  {'render ' if s['kind'] == 1 else 'capture'} {s['sample_rate']} Hz {s['bits']}-bit "
                     f"{'float' if s['is_float'] else 'int'}, {'exclusive' if s['exclusive'] else 'shared'}, "
                     f"period {s['period_frames']} / buffer {s['buffer_frames']} frames, reported latency "
                     f"{s['reported_latency_ms']} ms, glitches {s['glitches']}, underruns {s['underruns']}, "
                     f"overruns {s['overruns']}")
    lines.append(f"  {stats['blocks']} blocks; callback min {stats['callback_ns_min'] / 1000:.1f} / mean "
                 f"{stats['callback_ns_mean'] / 1000:.1f} / p99 {stats['callback_ns_p99'] / 1000:.1f} / max "
                 f"{stats['callback_ns_max'] / 1000:.1f} us; worst {stats['processing_load']:.1%} of the "
                 f"{stats['period_ns'] / 1e6:.2f} ms period; xruns {stats['xruns']}; audio-thread allocations "
                 f"{stats['rt_allocations']}")
    body = captured[rate:].astype(np.float64)
    for ch in range(body.shape[1]):
        level = rms(body[:, ch])
        lines.append(f"  input {ch + 1}: {20 * np.log10(max(level, 1e-12)):.1f} dBFS rms, peak "
                     f"{np.max(np.abs(body[:, ch])):.4f}, dominant {dominant_frequency(body[:, ch], rate):.2f} Hz")
    return "\n".join(lines)


def assert_clean(status, stats, captured, rate, seconds=SECONDS):
    assert all(s['state'] == 'running' for s in status), status
    assert stats['xruns'] == 0
    assert stats['rt_allocations'] == 0
    for s in status:
        assert s['underruns'] == 0 and s['overruns'] == 0, s
        # WASAPI flags the first capture packet of a stream as a discontinuity; any
        # further glitch is a real dropout.
        assert s['glitches'] <= (1 if s['kind'] != 1 else 0), s
    assert len(captured) >= (seconds - 0.5) * rate, f"only {len(captured)} frames arrived in {seconds} s"
    assert np.any(captured[rate:] != 0.0), "the input delivered digital silence"


def test_the_interface_enumerates(interface):
    for e in interface.values():
        print(f"\n{e.flow}: {e.name} [{e.id}] mix {e.mix_channels} ch {e.mix_sample_rate} Hz {e.mix_bits}-bit "
              f"{'float' if e.mix_is_float else 'int'}; period default {e.default_period_ms} / min "
              f"{e.min_period_ms} ms; low-latency shared {e.shared_min_period_frames} frames; raw "
              f"{e.raw_supported}")
        assert e.mix_channels >= 1 and e.mix_sample_rate > 0


# The AI-04's input at 44.1 kHz runs about 0.2-0.3 % slow against its own output and
# wanders by +-0.15 % over seconds; the drift corrector follows it without ongoing dropouts,
# but a run can lose a few frames (0-65 in 20 s, measured) while it converges. Recorded
# as a known failure rather than hidden (docs/WINDOWS_AUDIO.md).
SLOW_441 = pytest.mark.xfail(strict=False, reason="AI-04 44.1 kHz input clock: a few frames can be lost at start")


@pytest.mark.parametrize('rate', [48000, pytest.param(44100, marks=SLOW_441)])
def test_exclusive_duplex_on_the_interface_clock(interface, rate):
    captured, status, stats, _ = run_duplex(interface, rate, 128, exclusive=True)
    print(f"\n{NAME} exclusive, {rate} Hz, engine block 128:\n{describe(status, stats, captured, rate)}")
    assert all(s['exclusive'] and s['sample_rate'] == rate for s in status), status
    assert_clean(status, stats, captured, rate)


def test_shared_duplex(interface):
    rate = interface['render'].mix_sample_rate
    captured, status, stats, _ = run_duplex(interface, rate, 480, exclusive=False)
    print(f"\n{NAME} shared, {rate} Hz, block 480:\n{describe(status, stats, captured, rate)}")
    assert not any(s['exclusive'] for s in status)
    assert_clean(status, stats, captured, rate)


def test_the_monitoring_path_carries_the_input_at_the_gain_set(interface):
    """Input -> gain bus -> output at -40 dB: the output is given the input times the gain, sample for sample."""
    gain = 0.01
    rate = 48000
    captured, status, stats, monitored = run_duplex(interface, rate, 128, exclusive=True, seconds=5.0, gain=gain)
    assert_clean(status, stats, captured, rate, seconds=5.0)
    assert len(monitored) == len(captured)
    # Skip the first half second: the gain ramps from unity to its target when the plan starts.
    heard, given = captured[rate // 2:].astype(np.float64), monitored[rate // 2:].astype(np.float64)
    if rms(heard) < 1e-4:
        pytest.skip(f"the input is too quiet to compare ({rms(heard):.2e} rms): plug something into it")
    assert np.max(np.abs(given - heard * gain)) < 1e-6
    print(f"\nmonitoring at gain {gain}: input rms {rms(heard):.5f}, output given {rms(given):.7f} "
          f"(ratio {rms(given) / rms(heard):.5f}), dominant {dominant_frequency(heard[:, 0], rate):.2f} Hz")
