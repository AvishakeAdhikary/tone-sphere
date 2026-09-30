"""
The native WASAPI backend against this machine's real endpoints.

The core trick: render a known signal to the default output with raw processing (no
driver enhancements), and capture the same endpoint back — through whole-system loopback
or through this process's own loopback. Both paths go through the real Windows audio
engine, so a sample-accurate match proves the render path, the capture path, the device
threads and the clock-boundary crossing all carry audio intact.

What this does not prove: anything acoustic. Loopback taps the mix before the DAC, so the
speakers and the room are outside it, and the delay measured here is render-to-loopback,
not what a listener hears.
"""

import os
import threading
import time

import numpy as np
import pytest

from tests.signals import dominant_frequency, rms, sine, white_noise
from tonesphere.native import NativeEngine, Node, Route, available
from tonesphere.native.wasapi import StreamSpec, default_endpoint, endpoints, poll_events, watch

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not available(), reason="needs tonesphere_native.dll on Windows"),
]

RATE = 48000
BLOCK = 480
TONE, SPEAKER, LOOP, BACK, MIC = 1, 20, 30, 40, 50


@pytest.fixture
def speaker():
    endpoint = default_endpoint('render')
    if endpoint is None:
        pytest.skip("no default render endpoint")
    return endpoint


def play_and_capture(signal, render: StreamSpec, capture: StreamSpec):
    """Queue `signal` into the engine, run the streams, and return what came back plus status."""
    with NativeEngine(RATE, BLOCK) as engine:
        engine.apply_plan(
            [Node.source(TONE, 2, ring_frames=len(signal) + RATE), Node.sink(SPEAKER, 2),
             Node.source(LOOP, 2), Node.sink(BACK, 2, ring_frames=len(signal) + RATE * 2)],
            [Route(TONE, SPEAKER), Route(LOOP, BACK)],
        )
        engine.port_write(TONE, signal)
        engine.start_wasapi([render, capture], master=0)
        captured = []
        deadline = time.time() + len(signal) / RATE + 0.6
        while time.time() < deadline:
            captured.append(engine.port_read(BACK, RATE))
            time.sleep(0.02)
        status = engine.stream_status()
        stats = engine.stats()
        engine.stop_backend()
        captured.append(engine.port_read(BACK, RATE * 4))
    return np.concatenate(captured), status, stats


def from_onset(captured, threshold=1e-5):
    """Everything from the first audible sample on, kept contiguous."""
    loud = np.nonzero(np.abs(captured[:, 0]) > threshold)[0]
    return captured[loud[0]:] if len(loud) else captured[:0]


def assert_no_gaps(signal, expected_rms, window=RATE // 100):
    """No 10 ms stretch may fall below 90 % of the expected level: a dropout would."""
    levels = [rms(signal[i:i + window]) for i in range(0, len(signal) - window, window)]
    assert levels and min(levels) > 0.9 * expected_rms, f"a {window}-frame window fell to {min(levels):.4f}"


def align(captured, reference):
    """Offset of `reference` inside `captured` by cross-correlation on the first channel."""
    c = captured[:, 0].astype(np.float64)
    r = reference[:, 0].astype(np.float64)
    n = 1 << int(np.ceil(np.log2(len(c) + len(r))))
    corr = np.fft.irfft(np.fft.rfft(c, n) * np.conj(np.fft.rfft(r, n)), n)
    return int(np.argmax(corr[:len(c)]))


def test_endpoints_have_stable_ids_names_and_formats():
    found = endpoints()
    assert any(e.flow == 'render' for e in found) and any(e.flow == 'capture' for e in found)
    for e in found:
        assert e.id.startswith('{0.0.'), f"not an MMDevice endpoint id: {e.id}"
        assert e.name and e.mix_channels >= 1 and e.mix_sample_rate >= 8000
    ids = [e.id for e in found]
    assert len(ids) == len(set(ids))
    assert [e.id for e in endpoints()] == ids, "enumeration must be stable call to call"


def capture_spec(kind, speaker):
    if kind == 'process_loopback':
        return StreamSpec(LOOP, 'process_loopback', 2, process_id=os.getpid())
    return StreamSpec(LOOP, 'loopback', 2, speaker.id)


@pytest.mark.parametrize("capture_kind", ["loopback", "process_loopback"])
def test_raw_render_comes_back_whole_at_the_level_sent(speaker, capture_kind):
    tone = sine(RATE * 2, 1000.0, amplitude=0.1)
    captured, status, stats = play_and_capture(
        tone, StreamSpec(SPEAKER, 'render', 2, speaker.id, raw=True), capture_spec(capture_kind, speaker))
    if not next(s for s in status if s['node_id'] == SPEAKER)['raw']:
        pytest.skip("this endpoint does not honour raw mode; its enhancement effects alter the level")
    assert all(s['state'] == 'running' for s in status), status

    heard = from_onset(captured)[RATE // 10: RATE // 10 + RATE]
    assert len(heard) == RATE, "the tone was cut short"
    assert dominant_frequency(heard) == pytest.approx(1000.0, abs=2.0)
    assert rms(heard) == pytest.approx(rms(tone), rel=0.005)
    assert_no_gaps(heard, rms(tone))
    assert stats['xruns'] == 0 and stats['rt_allocations'] == 0


def test_raw_render_is_bit_exact_through_process_loopback(speaker):
    """
    Noise excites every frequency, so an exact match leaves no room for a wrong gain, a
    dropped block, a channel swap or a conversion error.

    Only process loopback can be exact: it taps this process's stream before the endpoint
    mix. Whole-endpoint loopback taps after the endpoint's own effects, which raw mode on
    one stream cannot bypass (on the development machine: a high shelf, flat to 1 kHz and
    about +1 dB above 16 kHz — see the next test). And it can only be exact where the drift
    resampler held a ratio of exactly 1.0; any correction interpolates, which is correct
    audio but not the same samples. The offset found is the measured render-to-capture delay.
    """
    capture_kind = 'process_loopback'
    signal = white_noise(RATE * 2, amplitude=0.1, seed=11)
    captured, status, stats = play_and_capture(
        signal, StreamSpec(SPEAKER, 'render', 2, speaker.id, raw=True), capture_spec(capture_kind, speaker))
    by_node = {s['node_id']: s for s in status}
    if not by_node[SPEAKER]['raw']:
        pytest.skip("this endpoint does not honour raw mode")
    offset = align(captured, signal)
    window = captured[offset + RATE // 4: offset + RATE // 4 + RATE]
    expected = signal[RATE // 4: RATE // 4 + RATE]
    error = float(np.max(np.abs(window - expected)))
    print(f"\n{capture_kind}: render-to-capture delay {offset} frames ({offset / RATE * 1000:.1f} ms, measured by "
          f"cross-correlation), drift ratio {by_node[LOOP]['drift_ratio']:.6f}, max sample error {error:.2e}, "
          f"callback mean {stats['callback_ns_mean'] / 1000:.1f} us, max {stats['callback_ns_max'] / 1000:.1f} us")
    if by_node[LOOP]['drift_ratio'] != 1.0:
        pytest.skip("the capture side corrected drift, so its samples are interpolated, not copied")
    assert error == 0.0, f"captured audio differs from what was played by up to {error:.2e}"


def test_the_endpoint_mix_response_is_recorded(speaker):
    """
    A record, not a pass/fail on ToneSphere: the transfer function of the endpoint's own
    effects, as whole-endpoint loopback sees them, measured with noise.
    """
    signal = white_noise(RATE * 2, amplitude=0.1, seed=11)
    captured, _, _ = play_and_capture(signal, StreamSpec(SPEAKER, 'render', 2, speaker.id, raw=True),
                                      StreamSpec(LOOP, 'loopback', 2, speaker.id))
    offset = align(captured, signal)
    heard = captured[offset + RATE // 4: offset + RATE // 4 + RATE, 0].astype(np.float64)
    sent = signal[RATE // 4: RATE // 4 + RATE, 0].astype(np.float64)
    response = np.abs(np.fft.rfft(heard) / np.fft.rfft(sent))
    freqs = np.fft.rfftfreq(RATE, 1 / RATE)
    bands = [(20, 1000), (1000, 4000), (4000, 12000), (12000, 20000)]
    summary = ', '.join(f"{lo}-{hi} Hz {20 * np.log10(np.mean(response[(freqs >= lo) & (freqs < hi)])):+.2f} dB"
                        for lo, hi in bands)
    print(f"\nendpoint loopback response: {summary}")
    low = response[(freqs >= 20) & (freqs < 1000)]
    assert 20 * np.log10(np.mean(low)) == pytest.approx(0.0, abs=0.1), "below 1 kHz the path must be flat"


def test_without_raw_mode_the_driver_may_change_the_level(speaker):
    """
    Not a pass/fail on ToneSphere: a record of what the driver's enhancement effects do to
    a stream that does not ask for raw mode. On the development machine (Realtek) they add
    about +10 dB with an automatic-gain rise; on other hardware there may be none.
    """
    tone = sine(RATE * 2, 1000.0, amplitude=0.05)
    captured, status, _ = play_and_capture(tone, StreamSpec(SPEAKER, 'render', 2, speaker.id, raw=False),
                                           StreamSpec(LOOP, 'loopback', 2, speaker.id))
    heard = from_onset(captured)
    assert dominant_frequency(heard[RATE // 2: RATE]) == pytest.approx(1000.0, abs=2.0)
    gain_db = 20 * np.log10(rms(heard[RATE // 2:RATE]) / rms(tone))
    print(f"\nwithout raw mode the endpoint changed the level by {gain_db:+.1f} dB")


def test_exclusive_mode_is_obtained_or_the_fallback_is_reported(speaker):
    tone = sine(RATE, 1000.0, amplitude=0.02)
    with NativeEngine(RATE, BLOCK) as engine:
        engine.apply_plan([Node.source(TONE, 2, ring_frames=RATE * 2), Node.sink(SPEAKER, 2)], [Route(TONE, SPEAKER)])
        engine.port_write(TONE, tone)
        engine.start_wasapi([StreamSpec(SPEAKER, 'render', 2, speaker.id, exclusive=True)])
        time.sleep(1.0)
        status = engine.stream_status()[0]
        stats = engine.stats()
        engine.stop_backend()
    assert status['state'] == 'running', status
    assert status['frames'] > RATE * 0.5, "a running exclusive stream must be moving frames"
    if status['exclusive']:
        assert status['message'] == ''
    else:
        assert 'exclusive mode refused' in status['message'], "a fallback must say why"
    print(f"\nexclusive={status['exclusive']} format={status['bits']}-bit/{status['valid_bits']} valid "
          f"{'float' if status['is_float'] else 'int'} buffer={status['buffer_frames']} frames "
          f"reported latency={status['reported_latency_ms']} ms glitches={status['glitches']} "
          f"xruns={stats['xruns']} message={status['message']!r}")


def test_a_capture_master_drives_render_and_loopback_satellites(speaker):
    """
    Three device clocks: the microphone's (master), the speaker's and the loopback's, each
    on its own thread, meeting through rings and drift resamplers. The tone must cross two
    clock boundaries and arrive whole at the level it was sent.
    """
    mic = default_endpoint('capture')
    if mic is None:
        pytest.skip("no default capture endpoint")
    tone = sine(RATE * 3, 1000.0, amplitude=0.05)
    with NativeEngine(RATE, BLOCK) as engine:
        engine.apply_plan(
            [Node.source(MIC, mic.mix_channels), Node.source(TONE, 2, ring_frames=RATE * 4), Node.sink(SPEAKER, 2),
             Node.source(LOOP, 2), Node.sink(BACK, 2, ring_frames=RATE * 6)],
            [Route(TONE, SPEAKER), Route(LOOP, BACK)],
        )
        engine.port_write(TONE, tone)
        engine.start_wasapi([
            StreamSpec(MIC, 'capture', mic.mix_channels, mic.id),
            StreamSpec(SPEAKER, 'render', 2, speaker.id, raw=True),
            StreamSpec(LOOP, 'loopback', 2, speaker.id),
        ], master=0)
        captured = []
        deadline = time.time() + 3.2
        while time.time() < deadline:
            captured.append(engine.port_read(BACK, RATE))
            time.sleep(0.02)
        status = {s['node_id']: s for s in engine.stream_status()}
        engine.stop_backend()
    back = np.concatenate(captured)
    for node in (MIC, SPEAKER, LOOP):
        assert status[node]['state'] == 'running', status[node]
    assert status[MIC]['is_master'] and not status[SPEAKER]['is_master']
    steady = from_onset(back)[RATE // 2: RATE // 2 + RATE]
    assert len(steady) == RATE
    assert dominant_frequency(steady) == pytest.approx(1000.0, abs=2.0)
    if status[SPEAKER]['raw']:
        assert rms(steady) == pytest.approx(rms(tone), rel=0.01)
        assert_no_gaps(steady, rms(tone))
    assert status[SPEAKER]['underruns'] == 0, "the render satellite ran dry after priming"
    print(f"\ndrift ratios: speaker {status[SPEAKER]['drift_ratio']:.6f}, loopback {status[LOOP]['drift_ratio']:.6f}")


def test_start_stop_cycles_leave_nothing_behind(speaker):
    threads_before = threading.active_count()
    for _ in range(5):
        with NativeEngine(RATE, BLOCK) as engine:
            engine.apply_plan([Node.source(TONE, 2, ring_frames=RATE), Node.sink(SPEAKER, 2)], [Route(TONE, SPEAKER)])
            engine.start_wasapi([StreamSpec(SPEAKER, 'render', 2, speaker.id)])
            time.sleep(0.2)
            assert engine.stream_status()[0]['frames'] > 0
            engine.stop_backend()
    assert threading.active_count() == threads_before


def test_a_bad_endpoint_fails_with_a_reason_not_silence():
    with NativeEngine(RATE, BLOCK) as engine:
        engine.apply_plan([Node.source(TONE, 2), Node.sink(SPEAKER, 2)], [Route(TONE, SPEAKER)])
        with pytest.raises(Exception) as refused:
            nowhere = '{0.0.0.00000000}.{00000000-0000-0000-0000-000000000000}'
            engine.start_wasapi([StreamSpec(SPEAKER, 'render', 2, nowhere)])
        assert 'endpoint' in str(refused.value).lower()


def test_device_notifications_can_be_watched():
    watch(True)
    try:
        assert isinstance(poll_events(), list)
    finally:
        watch(False)
