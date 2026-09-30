"""
The native channel strip and built-in processors, driven with real signals.

Where a Python reference exists (`tonesphere/engine/effects.py`), the native processor is
checked against it sample for sample, so the port is proven to be the same filter rather
than one that merely sounds plausible.
"""

import math

import numpy as np
import pytest

from tests.signals import RATE, assert_finite, dominant_frequency, impulse, peak, rms, sine
from tonesphere.engine.effects import Biquad
from tonesphere.native import COMPRESSOR, DELAY, EQ, LIMITER, Insert, NativeEngine, NativeError, Node, Route, _abi

BLOCK = 256
SRC, OUT = 1, 20


@pytest.fixture
def engine():
    with NativeEngine(RATE, BLOCK) as e:
        yield e


def through(engine, signal, inserts=()):
    """
    Run `signal` through SRC -> OUT, block by block, after one silent block for the route's
    fade-in. Node width comes from the signal: re-applying a plan with a different width
    would (correctly) replace every processor on the node with a fresh one.
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    channels = signal.shape[1]
    sink = Node.sink(OUT, channels)
    engine.apply_plan([Node.source(SRC, channels), sink], [Route(SRC, OUT)], list(inserts))
    engine.process({SRC: np.zeros((BLOCK, channels), np.float32)}, {OUT: channels})
    blocks = len(signal) // BLOCK
    return np.concatenate([
        engine.process({SRC: signal[b * BLOCK:(b + 1) * BLOCK]}, {OUT: channels})[OUT] for b in range(blocks)
    ])


def db(ratio: float) -> float:
    return 20.0 * math.log10(ratio)


class TestChannelStrip:
    def test_trim_scales_one_channel_only(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)])
        engine.set_channel_trim(SRC, 1, 0.25)
        tone = sine(BLOCK * 4)
        out = np.concatenate([engine.process({SRC: tone[i * BLOCK:(i + 1) * BLOCK]}, {OUT: 2})[OUT] for i in range(4)])
        assert np.array_equal(out[2 * BLOCK:, 0], tone[2 * BLOCK:, 0])
        assert np.allclose(out[2 * BLOCK:, 1], tone[2 * BLOCK:, 1] * 0.25, atol=1e-7)

    def test_polarity_flips_one_channel(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)])
        engine.set_channel_inverted(SRC, 0, True)
        out = through(engine, sine(BLOCK * 4))
        tone = sine(BLOCK * 4)
        assert np.allclose(out[BLOCK:, 0], -tone[BLOCK:, 0], atol=1e-7)
        assert np.allclose(out[BLOCK:, 1], tone[BLOCK:, 1], atol=1e-7)

    def test_fader_and_mute(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.bus(10, 2), Node.sink(OUT, 2)], [Route(SRC, 10), Route(10, OUT)])
        engine.set_node_gain(10, 0.5)
        tone = sine(BLOCK * 6)
        outs = [engine.process({SRC: tone[i * BLOCK:(i + 1) * BLOCK]}, {OUT: 2})[OUT] for i in range(3)]
        assert np.allclose(outs[-1], tone[2 * BLOCK:3 * BLOCK] * 0.5, atol=1e-7)
        engine.set_node_muted(10, True)
        engine.process({SRC: tone[3 * BLOCK:4 * BLOCK]}, {OUT: 2})
        assert peak(engine.process({SRC: tone[4 * BLOCK:5 * BLOCK]}, {OUT: 2})[OUT]) == 0.0

    def test_strip_settings_survive_a_plan_change(self, engine):
        """The legacy host dropped channel controls whenever it rebuilt its streams."""
        nodes = [Node.source(SRC, 2), Node.sink(OUT, 2)]
        engine.apply_plan(nodes, [Route(SRC, OUT)])
        engine.set_channel_trim(SRC, 0, 0.5)
        engine.set_node_gain(OUT, 0.5)
        engine.apply_plan(nodes + [Node.bus(10, 2)], [Route(SRC, OUT)])
        tone = sine(BLOCK * 3)
        for i in range(3):
            out = engine.process({SRC: tone[i * BLOCK:(i + 1) * BLOCK]}, {OUT: 2})[OUT]
        assert np.allclose(out[:, 0], tone[2 * BLOCK:, 0] * 0.25, atol=1e-7)
        assert np.allclose(out[:, 1], tone[2 * BLOCK:, 1] * 0.5, atol=1e-7)

    def test_controls_on_unknown_nodes_and_channels_are_refused(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)])
        with pytest.raises(NativeError):
            engine.set_node_gain(999, 0.5)
        with pytest.raises(NativeError):
            engine.set_channel_trim(SRC, 2, 0.5)
        with pytest.raises(NativeError):
            engine.set_node_gain(SRC, -1.0)

    def test_route_pan_moves_a_mono_source(self, engine):
        engine.apply_plan([Node.source(SRC, 1), Node.sink(OUT, 2)], [Route(SRC, OUT)])
        ones = np.ones((BLOCK, 1), np.float32)
        engine.process({SRC: ones}, {OUT: 2})
        engine.set_route_pan(SRC, OUT, 1.0)
        engine.process({SRC: ones}, {OUT: 2})
        out = engine.process({SRC: ones}, {OUT: 2})[OUT]
        assert abs(out[-1, 0]) < 1e-6 and out[-1, 1] == pytest.approx(1.0, abs=1e-6)


class TestEq:
    def eq(self, engine, bands, channels=2):
        """Place an EQ on the source and set `bands` = [(type, freq, q, gain_db), ...]."""
        engine.apply_plan([Node.source(SRC, channels), Node.sink(OUT, channels)], [Route(SRC, OUT)],
                          [Insert(SRC, 0, EQ)])
        for b, (kind, freq, q, gain) in enumerate(bands):
            for p, value in enumerate((kind, freq, q, gain)):
                engine.set_insert_param(SRC, 0, b * 4 + p, value)

    @pytest.mark.parametrize("kind, setter, args", [
        (_abi.EQ_PEAKING, 'set_peaking', (1000.0, 1.0, 6.0)),
        (_abi.EQ_LOW_SHELF, 'set_low_shelf', (200.0, -9.0, 1.0)),
        (_abi.EQ_HIGH_SHELF, 'set_high_shelf', (5000.0, 4.0, 0.8)),
        (_abi.EQ_HIGHPASS, 'set_highpass', (120.0, 0.707)),
        (_abi.EQ_LOWPASS, 'set_lowpass', (3000.0, 2.0)),
    ])
    def test_matches_the_python_reference_sample_for_sample(self, engine, kind, setter, args):
        """
        The same cookbook design, the same input: the native filter must produce the
        samples the proven Python `Biquad` produces. Noise excites every frequency, so
        a wrong coefficient anywhere shows up.
        """
        if setter == 'set_peaking':
            freq, q, gain = args
            native = (kind, freq, q, gain)
        elif setter in ('set_low_shelf', 'set_high_shelf'):
            freq, gain, slope = args
            native = (kind, freq, slope, gain)
        else:
            freq, q = args
            native = (kind, freq, q, 0.0)

        self.eq(engine, [native], channels=1)
        rng = np.random.default_rng(7)
        noise = (0.3 * rng.standard_normal((BLOCK * 16, 1))).astype(np.float32)

        reference = Biquad(channels=1)
        getattr(reference, setter)(RATE, *args)
        expected = reference.process(noise.copy(), len(noise))

        got = through(engine, noise, inserts=[Insert(SRC, 0, EQ)])
        assert np.allclose(got, expected, atol=2e-6), f"max error {np.max(np.abs(got - expected))}"

    def test_a_peaking_boost_raises_its_centre_by_the_set_gain(self, engine):
        self.eq(engine, [(_abi.EQ_PEAKING, 1000.0, 1.0, 6.0)])
        tone = sine(BLOCK * 40, 1000.0, amplitude=0.2)
        out = through(engine, tone, inserts=[Insert(SRC, 0, EQ)])
        settled = out[BLOCK * 20:]
        assert db(rms(settled) / rms(tone[BLOCK * 20:])) == pytest.approx(6.0, abs=0.1)
        assert dominant_frequency(settled) == pytest.approx(1000.0, abs=10.0)

    def test_a_highpass_removes_rumble_and_keeps_the_note(self, engine):
        self.eq(engine, [(_abi.EQ_HIGHPASS, 200.0, 0.707, 0.0)])
        low = through(engine, sine(BLOCK * 40, 30.0), inserts=[Insert(SRC, 0, EQ)])
        engine.apply_plan([], [])
        self.eq(engine, [(_abi.EQ_HIGHPASS, 200.0, 0.707, 0.0)])
        high = through(engine, sine(BLOCK * 40, 2000.0), inserts=[Insert(SRC, 0, EQ)])
        assert db(rms(low[BLOCK * 20:]) / rms(sine(BLOCK * 20, 30.0))) < -30.0
        assert db(rms(high[BLOCK * 20:]) / rms(sine(BLOCK * 20, 2000.0))) == pytest.approx(0.0, abs=0.1)

    def test_filter_memory_survives_a_plan_change(self, engine):
        """Swapping the plan mid-note must not restart the filter — that is a click."""
        nodes = [Node.source(SRC, 1), Node.sink(OUT, 1)]
        inserts = [Insert(SRC, 0, EQ)]
        noise = (0.3 * np.random.default_rng(3).standard_normal((BLOCK * 8, 1))).astype(np.float32)

        def run(swap_at):
            with NativeEngine(RATE, BLOCK) as e:
                e.apply_plan(nodes, [Route(SRC, OUT)], inserts)
                for p, v in enumerate((_abi.EQ_LOWPASS, 500.0, 0.707, 0.0)):
                    e.set_insert_param(SRC, 0, p, v)
                out = []
                for b in range(8):
                    if b == swap_at:
                        e.apply_plan(nodes + [Node.bus(10, 1)], [Route(SRC, OUT)], inserts)
                    out.append(e.process({SRC: noise[b * BLOCK:(b + 1) * BLOCK]}, {OUT: 1})[OUT])
                return np.concatenate(out)

        assert np.array_equal(run(swap_at=None), run(swap_at=4))

    def test_nan_input_does_not_poison_the_filter(self, engine):
        """
        The poisoned block is silenced at the source, so the filter never sees the NaN;
        a few blocks later its output is the clean output again.
        """
        band = [(_abi.EQ_LOWPASS, 500.0, 0.707, 0.0)]
        tone = sine(BLOCK * 12)
        poisoned = tone.copy()
        poisoned[BLOCK * 2 + 5, 0] = np.nan

        self.eq(engine, band)
        out = through(engine, poisoned, inserts=[Insert(SRC, 0, EQ)])
        with NativeEngine(RATE, BLOCK) as clean_engine:
            self.eq(clean_engine, band)
            clean = through(clean_engine, tone, inserts=[Insert(SRC, 0, EQ)])

        assert_finite(out)
        assert np.allclose(out[BLOCK * 8:], clean[BLOCK * 8:], atol=1e-4), "the filter must recover"

    def test_bypass_passes_the_signal_untouched(self, engine):
        self.eq(engine, [(_abi.EQ_PEAKING, 1000.0, 1.0, 12.0)])
        engine.set_insert_bypassed(SRC, 0, True)
        tone = sine(BLOCK * 4)
        out = through(engine, tone, inserts=[Insert(SRC, 0, EQ, bypassed=True)])
        assert np.array_equal(out[BLOCK:], tone[BLOCK:])

    def test_parameters_are_validated_and_read_back(self, engine):
        self.eq(engine, [(_abi.EQ_PEAKING, 1234.0, 1.0, 3.0)])
        assert engine.insert_param(SRC, 0, 1) == pytest.approx(1234.0)
        with pytest.raises(NativeError):
            engine.set_insert_param(SRC, 0, 999, 1.0)
        with pytest.raises(NativeError):
            engine.set_insert_param(SRC, 5, 0, 1.0)


class TestCompressor:
    def test_reduces_a_loud_tone_by_the_ratio_above_threshold(self, engine):
        """
        -6 dBFS peak into threshold -18 dB, ratio 4, hard knee: 12 dB over becomes 3 dB
        over, so 9 dB of reduction and a -15 dBFS peak once the envelope settles.
        """
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)], [Insert(SRC, 0, COMPRESSOR)])
        for p, v in enumerate((-18.0, 4.0, 1.0, 200.0, 0.0, 0.0)):
            engine.set_insert_param(SRC, 0, p, v)
        tone = sine(BLOCK * 80, 1000.0, amplitude=10 ** (-6 / 20))
        out = through(engine, tone, inserts=[Insert(SRC, 0, COMPRESSOR)])
        settled = out[BLOCK * 40:]
        assert db(peak(settled)) == pytest.approx(-15.0, abs=0.75)
        assert engine.insert_readout(SRC, 0) == pytest.approx(-9.0, abs=0.75)
        assert dominant_frequency(settled) == pytest.approx(1000.0, abs=10.0)

    def test_leaves_a_quiet_signal_alone(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)], [Insert(SRC, 0, COMPRESSOR)])
        tone = sine(BLOCK * 20, amplitude=10 ** (-30 / 20))
        out = through(engine, tone, inserts=[Insert(SRC, 0, COMPRESSOR)])
        assert np.allclose(out[BLOCK:], tone[BLOCK:], atol=1e-7)
        assert engine.insert_readout(SRC, 0) == 0.0

    def test_attack_lags_the_signal(self, engine):
        """What makes it a compressor and not a waveshaper: reduction takes time to arrive."""
        engine.apply_plan([Node.source(SRC, 1), Node.sink(OUT, 1)], [Route(SRC, OUT)], [Insert(SRC, 0, COMPRESSOR)])
        for p, v in enumerate((-20.0, 10.0, 20.0, 100.0, 0.0, 0.0)):
            engine.set_insert_param(SRC, 0, p, v)
        step = np.concatenate([np.zeros((BLOCK * 2, 1)), np.full((BLOCK * 20, 1), 0.8)]).astype(np.float32)
        out = through(engine, step, inserts=[Insert(SRC, 0, COMPRESSOR)])
        onset = BLOCK * 2
        assert out[onset + 1, 0] > 0.5, "the first samples of a transient pass before the attack acts"
        assert out[-1, 0] < 0.2


class TestLimiter:
    def test_output_never_exceeds_the_threshold(self, engine):
        """Sample-accurate: not one sample over, even on the first sample of an overload."""
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)], [Insert(SRC, 0, LIMITER)])
        engine.set_insert_param(SRC, 0, 0, 0.5)
        loud = sine(BLOCK * 20, 440.0, amplitude=2.0)
        out = through(engine, loud, inserts=[Insert(SRC, 0, LIMITER)])
        assert peak(out) <= 0.5 + 1e-6
        assert dominant_frequency(out[BLOCK * 4:]) == pytest.approx(440.0, abs=10.0)

    def test_signal_below_threshold_passes_bit_exact(self, engine):
        engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)], [Insert(SRC, 0, LIMITER)])
        tone = sine(BLOCK * 8, amplitude=0.5)
        out = through(engine, tone, inserts=[Insert(SRC, 0, LIMITER)])
        assert np.array_equal(out[BLOCK:], tone[BLOCK:])

    def test_the_sink_safety_limiter_catches_a_routing_mistake(self, engine):
        """Three hot sources summed into one output: a compressed mix, not full-scale noise."""
        nodes = [Node.source(s, 2) for s in (1, 2, 3)] + [Node.sink(OUT, 2, limiter=True)]
        engine.apply_plan(nodes, [Route(s, OUT) for s in (1, 2, 3)])
        tone = sine(BLOCK, amplitude=0.9)
        outs = [engine.process({1: tone, 2: tone, 3: tone}, {OUT: 2})[OUT] for _ in range(10)]
        assert peak(np.concatenate(outs)) <= 0.99 + 1e-6


class TestDelay:
    def test_an_impulse_echoes_at_the_delay_time_with_feedback(self, engine):
        engine.apply_plan([Node.source(SRC, 1), Node.sink(OUT, 1)], [Route(SRC, OUT)], [Insert(SRC, 0, DELAY)])
        delay_ms, feedback, mix = 10.0, 0.5, 0.4
        for p, v in enumerate((delay_ms, feedback, mix)):
            engine.set_insert_param(SRC, 0, p, v)
        d = int(RATE * delay_ms / 1000)
        signal = np.zeros((BLOCK * 8, 1), np.float32)
        signal[BLOCK * 2] = 1.0
        out = through(engine, signal, inserts=[Insert(SRC, 0, DELAY)])[:, 0]
        at = BLOCK * 2
        assert out[at] == pytest.approx(1.0)
        assert out[at + d] == pytest.approx(mix)
        assert out[at + 2 * d] == pytest.approx(mix * feedback)
        assert out[at + 3 * d] == pytest.approx(mix * feedback ** 2)
        echoes = [at + k * d for k in range(8) if at + k * d < len(out)]
        others = np.delete(out, echoes)
        assert peak(others) < 1e-6, "energy must appear only at multiples of the delay"


def test_the_whole_strip_allocates_nothing(engine):
    nodes = [Node.source(SRC, 2), Node.bus(10, 2), Node.sink(OUT, 2, limiter=True)]
    inserts = [Insert(SRC, 0, EQ), Insert(SRC, 1, COMPRESSOR), Insert(10, 0, DELAY), Insert(10, 1, LIMITER)]
    engine.apply_plan(nodes, [Route(SRC, 10), Route(10, OUT)], inserts)
    engine.set_insert_param(SRC, 0, 0, _abi.EQ_PEAKING)
    engine.reset_stats()
    tone = sine(BLOCK, amplitude=0.8)
    for i in range(300):
        engine.set_insert_param(SRC, 0, 3, float(i % 12))
        engine.set_channel_trim(SRC, 0, 0.5 + (i % 2) * 0.5)
        engine.process({SRC: tone}, {OUT: 2})
    stats = engine.stats()
    assert stats['blocks'] == 300 and stats['rt_allocations'] == 0
    assert impulse(4).shape == (4, 2)
