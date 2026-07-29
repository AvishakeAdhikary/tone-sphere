"""
Effects.

The theme is block boundaries and state. The implementation these replace had an EQ that
ignored two of its three bands, a compressor that ignored attack and release, and a
"reverb" built on np.roll — a circular shift within the block, so it wrapped each block's
tail onto its own head instead of delaying anything.

Several tests therefore run a signal across many blocks and check the joins, which is
where stateless DSP gives itself away.
"""

import math

import numpy as np
import pytest

from tonesphere.engine.effects import (
    Biquad,
    Compressor,
    Delay,
    EQBand,
    InsertChain,
    ParametricEQ,
    PluginChain,
    PluginChainUnavailable,
)

RATE = 48000
BLOCK = 256


def sine(frames, freq=1000.0, amplitude=0.5, channels=2, phase=0.0):
    t = (np.arange(frames, dtype=np.float64) + phase) / RATE
    wave = (amplitude * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)
    return np.repeat(wave.reshape(-1, 1), channels, axis=1)


def run_blocks(processor, freq, blocks=8, amplitude=0.5, channels=2):
    """Push a continuous tone through several blocks and return the concatenation."""
    pieces = []
    for i in range(blocks):
        block = sine(BLOCK, freq=freq, amplitude=amplitude,
                     channels=channels, phase=i * BLOCK)
        processor.process(block, BLOCK)
        pieces.append(block[:, 0].copy())
    return np.concatenate(pieces)


def settled_amplitude(signal):
    """Peak of the last quarter, after any filter transient has died away."""
    tail = signal[len(signal) * 3 // 4:]
    return float(np.max(np.abs(tail)))


def max_step(signal):
    return float(np.max(np.abs(np.diff(signal))))


class TestBiquad:
    def test_highpass_removes_content_below_the_corner(self):
        biquad = Biquad(2)
        biquad.set_highpass(RATE, 500.0)

        assert settled_amplitude(run_blocks(biquad, freq=50.0)) < 0.05

    def test_highpass_passes_content_above_the_corner(self):
        biquad = Biquad(2)
        biquad.set_highpass(RATE, 100.0)

        assert settled_amplitude(run_blocks(biquad, freq=4000.0)) == pytest.approx(0.5, abs=0.05)

    def test_lowpass_removes_content_above_the_corner(self):
        biquad = Biquad(2)
        biquad.set_lowpass(RATE, 500.0)

        assert settled_amplitude(run_blocks(biquad, freq=8000.0)) < 0.05

    def test_peaking_boost_raises_its_centre_frequency(self):
        biquad = Biquad(2)
        biquad.set_peaking(RATE, 1000.0, q=1.0, gain_db=12.0)

        boosted = settled_amplitude(run_blocks(biquad, freq=1000.0))
        assert boosted > 0.5 * 3.0   # +12 dB is roughly 4x; allow for Q shaping

    def test_peaking_cut_lowers_its_centre_frequency(self):
        biquad = Biquad(2)
        biquad.set_peaking(RATE, 1000.0, q=1.0, gain_db=-12.0)

        assert settled_amplitude(run_blocks(biquad, freq=1000.0)) < 0.5 * 0.4

    def test_peaking_leaves_distant_frequencies_alone(self):
        biquad = Biquad(2)
        biquad.set_peaking(RATE, 1000.0, q=2.0, gain_db=12.0)

        assert settled_amplitude(run_blocks(biquad, freq=60.0)) == pytest.approx(0.5, abs=0.1)

    def test_low_shelf_lifts_the_bottom_end(self):
        biquad = Biquad(2)
        biquad.set_low_shelf(RATE, 200.0, gain_db=9.0)

        assert settled_amplitude(run_blocks(biquad, freq=60.0)) > 0.5 * 2.0

    def test_high_shelf_lifts_the_top_end(self):
        biquad = Biquad(2)
        biquad.set_high_shelf(RATE, 4000.0, gain_db=9.0)

        assert settled_amplitude(run_blocks(biquad, freq=10000.0)) > 0.5 * 2.0

    def test_state_carries_across_block_boundaries(self):
        """
        A filter that resets each block is not a filter, it is a click generator. Feed a
        continuous sine through many blocks and require the output to stay smooth.
        """
        biquad = Biquad(1)
        biquad.set_lowpass(RATE, 2000.0)

        output = run_blocks(biquad, freq=300.0, blocks=12, channels=1)
        tail = output[BLOCK * 2:]   # skip the initial transient

        # A 300 Hz sine at 48 kHz steps by at most ~0.02 per sample.
        assert max_step(tail) < 0.05

    def test_reset_clears_state(self):
        biquad = Biquad(1)
        biquad.set_lowpass(RATE, 1000.0)
        run_blocks(biquad, freq=500.0, blocks=4, channels=1)

        biquad.reset()

        assert float(np.max(np.abs(biquad._y1))) == 0.0

    def test_magnitude_response_matches_the_design(self):
        biquad = Biquad(2)
        biquad.set_peaking(RATE, 1000.0, q=1.0, gain_db=6.0)

        at_centre = biquad.magnitude_at(RATE, 1000.0)
        assert 20 * math.log10(at_centre) == pytest.approx(6.0, abs=0.2)

    def test_stability_at_extreme_frequencies(self):
        """A corner above Nyquist must be clamped, not produce an exploding filter."""
        biquad = Biquad(1)
        biquad.set_lowpass(RATE, 100_000.0)

        output = run_blocks(biquad, freq=1000.0, blocks=4, channels=1)
        assert np.all(np.isfinite(output))
        assert float(np.max(np.abs(output))) < 10.0


class TestParametricEQ:
    def test_all_bands_take_effect(self):
        """
        Regression: the previous EQ read low_gain and silently discarded mid and high.
        Each band must independently change the output.
        """
        for index, frequency in enumerate((100.0, 1000.0, 8000.0)):
            eq = ParametricEQ(RATE, 2, [
                EQBand('peaking', 100.0, 0.0, 1.0),
                EQBand('peaking', 1000.0, 0.0, 1.0),
                EQBand('peaking', 8000.0, 0.0, 1.0),
            ])
            eq.set_band(index, EQBand('peaking', frequency, 12.0, 1.0))

            boosted = settled_amplitude(run_blocks(eq, freq=frequency))
            assert boosted > 0.5 * 2.0, f"band {index} at {frequency} Hz had no effect"

    def test_flat_eq_is_detected_and_transparent(self):
        eq = ParametricEQ(RATE, 2, [EQBand('peaking', 1000.0, 0.0, 1.0)])

        assert eq.is_flat is True

        block = sine(BLOCK)
        original = block.copy()
        eq.process(block, BLOCK)
        np.testing.assert_array_equal(block, original)

    def test_disabled_band_does_nothing(self):
        eq = ParametricEQ(RATE, 2, [EQBand('peaking', 1000.0, 12.0, 1.0, enabled=False)])

        assert settled_amplitude(run_blocks(eq, freq=1000.0)) == pytest.approx(0.5, abs=0.02)

    def test_response_curve_reflects_the_bands(self):
        eq = ParametricEQ(RATE, 2, [EQBand('peaking', 1000.0, 6.0, 1.0)])

        response = eq.response([100.0, 1000.0, 10000.0])

        assert response[1] == pytest.approx(6.0, abs=0.3)
        assert abs(response[0]) < 1.5
        assert abs(response[2]) < 1.5


class TestCompressor:
    def test_quiet_signal_passes_untouched(self):
        compressor = Compressor(RATE, BLOCK, threshold_db=-6.0, ratio=4.0)
        block = sine(BLOCK, amplitude=0.05)
        original = block.copy()

        compressor.process(block, BLOCK)

        np.testing.assert_allclose(block, original, atol=1e-6)

    def test_loud_signal_is_reduced(self):
        compressor = Compressor(RATE, BLOCK, threshold_db=-20.0, ratio=4.0, attack_ms=1.0)

        for _ in range(60):
            block = sine(BLOCK, amplitude=0.8)
            compressor.process(block, BLOCK)

        assert compressor.reduction_db < -3.0
        assert float(np.max(np.abs(block))) < 0.8

    def test_ratio_controls_how_much_is_reduced(self):
        """A higher ratio must reduce more. The old version ignored ratio entirely."""
        reductions = {}
        for ratio in (2.0, 8.0):
            compressor = Compressor(RATE, BLOCK, threshold_db=-20.0,
                                    ratio=ratio, attack_ms=1.0)
            for _ in range(60):
                compressor.process(sine(BLOCK, amplitude=0.8), BLOCK)
            reductions[ratio] = compressor.reduction_db

        assert reductions[8.0] < reductions[2.0]

    def test_attack_time_controls_how_fast_it_clamps(self):
        """
        Regression: attack and release were accepted and discarded. A slow attack must
        still be letting signal through when a fast one has already clamped.
        """
        results = {}
        for attack in (0.5, 200.0):
            compressor = Compressor(RATE, BLOCK, threshold_db=-20.0,
                                    ratio=8.0, attack_ms=attack)
            for _ in range(3):
                compressor.process(sine(BLOCK, amplitude=0.9), BLOCK)
            results[attack] = compressor.reduction_db

        assert results[0.5] < results[200.0], "fast attack should clamp sooner"

    def test_release_recovers_after_the_signal_drops(self):
        compressor = Compressor(RATE, BLOCK, threshold_db=-20.0, ratio=8.0,
                                attack_ms=1.0, release_ms=20.0)

        for _ in range(40):
            compressor.process(sine(BLOCK, amplitude=0.9), BLOCK)
        clamped = compressor.reduction_db

        for _ in range(200):
            compressor.process(sine(BLOCK, amplitude=0.01), BLOCK)

        assert compressor.reduction_db > clamped
        assert compressor.reduction_db == pytest.approx(0.0, abs=0.5)

    def test_makeup_gain_is_applied(self):
        """Also ignored by the previous implementation."""
        compressor = Compressor(RATE, BLOCK, threshold_db=0.0, ratio=1.0, makeup_db=6.0)

        for _ in range(20):
            block = sine(BLOCK, amplitude=0.1)
            compressor.process(block, BLOCK)

        assert float(np.max(np.abs(block))) == pytest.approx(0.2, abs=0.02)

    def test_soft_knee_engages_gradually(self):
        """A hard knee switches on at an edge, which clicks on material near threshold."""
        soft = Compressor(RATE, BLOCK, threshold_db=-20.0, ratio=4.0, knee_db=12.0)
        hard = Compressor(RATE, BLOCK, threshold_db=-20.0, ratio=4.0, knee_db=0.0)

        just_below = -22.0
        assert soft._gain_reduction_for(just_below) < 0.0, "soft knee acts before threshold"
        assert hard._gain_reduction_for(just_below) == 0.0

    def test_no_discontinuity_between_blocks(self):
        compressor = Compressor(RATE, BLOCK, threshold_db=-24.0, ratio=6.0, attack_ms=2.0)
        output = run_blocks(compressor, freq=200.0, blocks=10, amplitude=0.8, channels=1)

        assert max_step(output[BLOCK:]) < 0.2


class TestDelay:
    def test_delayed_signal_appears_after_the_delay_time(self):
        """
        Regression: the previous implementation used np.roll, a circular shift within the
        block, so the tail of each block wrapped onto its own head — a stutter at the
        block rate, not an echo.
        """
        delay = Delay(RATE, channels=1, delay_ms=10.0, feedback=0.0, mix=1.0)
        delay_samples = int(RATE * 0.01)

        impulse = np.zeros((BLOCK, 1), dtype=np.float32)
        impulse[0, 0] = 1.0
        delay.process(impulse, BLOCK)

        # Not in the first block: the delay is longer than one block.
        assert float(np.max(np.abs(impulse))) == pytest.approx(1.0, abs=1e-6)

        collected = [impulse[:, 0].copy()]
        for _ in range(6):
            block = np.zeros((BLOCK, 1), dtype=np.float32)
            delay.process(block, BLOCK)
            collected.append(block[:, 0].copy())

        signal = np.concatenate(collected)
        echo_index = int(np.argmax(np.abs(signal[BLOCK // 2:]))) + BLOCK // 2

        assert echo_index == pytest.approx(delay_samples, abs=BLOCK)

    def test_no_wraparound_within_a_single_block(self):
        """The specific np.roll failure: energy appearing at the start of its own block."""
        delay = Delay(RATE, channels=1, delay_ms=50.0, feedback=0.0, mix=1.0)

        block = np.zeros((BLOCK, 1), dtype=np.float32)
        block[-1, 0] = 1.0   # energy only at the very end
        delay.process(block, BLOCK)

        assert float(np.max(np.abs(block[:BLOCK - 1]))) < 1e-6

    def test_feedback_produces_repeats(self):
        delay = Delay(RATE, channels=1, delay_ms=5.0, feedback=0.6, mix=1.0)

        impulse = np.zeros((BLOCK, 1), dtype=np.float32)
        impulse[0, 0] = 1.0
        delay.process(impulse, BLOCK)

        peaks = []
        for _ in range(8):
            block = np.zeros((BLOCK, 1), dtype=np.float32)
            delay.process(block, BLOCK)
            peaks.append(float(np.max(np.abs(block))))

        assert sum(1 for p in peaks if p > 0.01) >= 2, "feedback should give repeats"

    def test_zero_mix_is_transparent(self):
        delay = Delay(RATE, channels=2, delay_ms=100.0, mix=0.0)
        block = sine(BLOCK)
        original = block.copy()

        delay.process(block, BLOCK)

        np.testing.assert_array_equal(block, original)

    def test_delay_time_is_clamped_to_the_buffer(self):
        delay = Delay(RATE, channels=1, max_delay_ms=100.0, delay_ms=100.0)
        delay.set_delay_ms(10_000.0)

        assert delay.delay_ms <= 100.0


class TestInsertChain:
    def test_empty_chain_is_transparent(self):
        chain = InsertChain(RATE, BLOCK, 2)

        assert chain.is_transparent is True

        block = sine(BLOCK)
        original = block.copy()
        chain.process(block, BLOCK)
        np.testing.assert_array_equal(block, original)

    def test_disabled_chain_is_bypassed(self):
        chain = InsertChain(RATE, BLOCK, 2)
        chain.enable_highpass(500.0)
        chain.enabled = False

        assert chain.is_transparent is True

        block = sine(BLOCK, freq=50.0)
        original = block.copy()
        chain.process(block, BLOCK)
        np.testing.assert_array_equal(block, original)

    def test_highpass_runs_before_the_compressor(self):
        """
        Order matters: a compressor reacting to sub-sonic rumble makes a channel breathe
        for no visible reason. The high-pass has to remove it before the detector sees it.
        """
        chain = InsertChain(RATE, BLOCK, 1)
        chain.enable_highpass(100.0)
        chain.enable_compressor(threshold_db=-30.0, ratio=8.0, attack_ms=1.0)

        for i in range(40):
            block = sine(BLOCK, freq=20.0, amplitude=0.9, channels=1, phase=i * BLOCK)
            chain.process(block, BLOCK)

        # Rumble is filtered out, so the compressor should barely engage.
        assert chain.compressor.reduction_db > -3.0

    def test_chain_reports_transparency_accurately(self):
        chain = InsertChain(RATE, BLOCK, 2)
        assert chain.is_transparent

        chain.enable_eq([EQBand('peaking', 1000.0, 0.0, 1.0)])
        assert chain.is_transparent, "a flat EQ should not count as work"

        chain.eq.set_band(0, EQBand('peaking', 1000.0, 6.0, 1.0))
        assert not chain.is_transparent

    def test_reset_clears_all_stateful_stages(self):
        chain = InsertChain(RATE, BLOCK, 1)
        chain.enable_highpass(200.0)
        chain.enable_delay(delay_ms=10.0, mix=0.5)

        for i in range(4):
            chain.process(sine(BLOCK, channels=1, phase=i * BLOCK), BLOCK)

        chain.reset()

        assert float(np.max(np.abs(chain.highpass._y1))) == 0.0
        assert float(np.max(np.abs(chain.delay._buffer))) == 0.0


def require_pedalboard():
    """
    Skip the test if pedalboard cannot be used here, otherwise return the module.

    Deliberately not `pytest.importorskip("pedalboard")`. That does a raw import in this
    process, and this project's own CI proved a raw import is not always safe: pedalboard's
    Linux wheel raised SIGILL — Illegal instruction, a fatal signal — on a runner CPU
    missing some instruction its compiled code used unconditionally. A signal like that
    kills the whole pytest process, not just the one test, so nothing downstream even runs.

    `PluginChain.is_available()` answers the same question from inside a disposable
    subprocess, so a crash there cannot take this process down with it. Only once that has
    confirmed importing is safe do we import it directly, to actually use it.
    """
    if not PluginChain.is_available():
        pytest.skip("pedalboard is not usable in this environment")
    import pedalboard
    return pedalboard


class TestPluginHosting:
    def test_availability_is_reported_honestly(self):
        """
        Either pedalboard is importable here or it is not. Reporting availability without
        checking is how you get a UI offering a feature that throws when used — and here,
        worse than throws: this project's CI hit a case where pedalboard imports crash the
        whole process (see `require_pedalboard` above), which is exactly why
        `is_available()` no longer imports it directly to find out.
        """
        available = PluginChain.is_available()
        assert isinstance(available, bool)

        if available:
            # Safe specifically because `is_available()` already proved it in a
            # subprocess: if that survived, importing it here for real is expected to
            # succeed too, since it's the same wheel and the same CPU.
            import pedalboard  # noqa: F401

    def test_availability_is_cached_rather_than_reprobed_every_call(self):
        """The check spawns a subprocess; paying that cost on every call would be wasteful."""
        first = PluginChain.is_available()
        second = PluginChain.is_available()

        assert first == second

    def test_empty_chain_is_transparent(self):
        chain = PluginChain(RATE, BLOCK)

        assert chain.is_empty is True

        block = sine(BLOCK)
        original = block.copy()
        chain.process(block, BLOCK)
        np.testing.assert_array_equal(block, original)

    def test_loading_a_missing_plugin_raises_rather_than_failing_quietly(self):
        """
        Two legitimate outcomes depending on the machine: if pedalboard itself is not
        usable here, `load()` raises PluginChainUnavailable before ever touching it; if it
        is usable, pedalboard raises ImportError for a path that does not exist. Either is
        a real, specific failure — never a silent no-op.
        """
        chain = PluginChain(RATE, BLOCK)

        with pytest.raises((ImportError, PluginChainUnavailable)):
            chain.load("/nonexistent/plugin.vst3")

    def test_discovery_returns_paths_without_loading_anything(self):
        """Loading an unknown plugin can be slow and can crash, so discovery only lists."""
        found = PluginChain.discover()
        assert isinstance(found, list)

    def test_default_search_paths_are_platform_appropriate(self):
        import platform

        paths = PluginChain.scan_default_paths()
        assert isinstance(paths, list)

        if platform.system() == "Windows" and paths:
            assert any("VST" in p.upper() for p in paths)

    def test_bypassed_plugin_is_skipped(self):
        pedalboard = require_pedalboard()

        chain = PluginChain(RATE, BLOCK)
        chain.add_builtin(pedalboard.Gain(gain_db=12.0), "Gain")

        block = sine(BLOCK, amplitude=0.1)
        chain.process(block, BLOCK)
        assert float(np.max(np.abs(block))) > 0.2

        chain.set_bypassed(0, True)
        block = sine(BLOCK, amplitude=0.1)
        original = block.copy()
        chain.process(block, BLOCK)
        np.testing.assert_array_equal(block, original)

    def test_builtin_plugin_processes_audio(self):
        """Proves the pedalboard bridge works end to end, including the transpose."""
        pedalboard = require_pedalboard()

        chain = PluginChain(RATE, BLOCK)
        chain.add_builtin(pedalboard.Gain(gain_db=-6.0), "Gain")

        block = sine(BLOCK, amplitude=0.8)
        chain.process(block, BLOCK)

        assert float(np.max(np.abs(block))) == pytest.approx(0.4, abs=0.02)

    def test_channel_layout_survives_the_round_trip(self):
        """
        pedalboard uses (channels, frames) and we use (frames, channels). A transpose bug
        would silently swap the two and scramble the audio.
        """
        pedalboard = require_pedalboard()

        chain = PluginChain(RATE, BLOCK)
        chain.add_builtin(pedalboard.Gain(gain_db=0.0), "Unity")

        block = np.zeros((BLOCK, 2), dtype=np.float32)
        block[:, 0] = 0.5     # left only
        chain.process(block, BLOCK)

        assert float(np.max(np.abs(block[:, 0]))) == pytest.approx(0.5, abs=0.01)
        assert float(np.max(np.abs(block[:, 1]))) < 0.01, "channels must not swap"

    def test_a_failing_plugin_does_not_kill_the_audio_thread(self):
        """A misbehaving plugin must be bypassed for the block, not propagate."""
        class Exploding:
            latency_samples = 0

            def process(self, *args, **kwargs):
                raise RuntimeError("boom")

        chain = PluginChain(RATE, BLOCK)
        chain.add_builtin(Exploding(), "Exploding")

        block = sine(BLOCK)
        chain.process(block, BLOCK)   # must not raise

    def test_plugin_latency_is_reported(self):
        """
        Plugin latency has to reach the round-trip figure we show, or the number is wrong
        in exactly the direction that flatters us.
        """
        class Delayed:
            latency_samples = 512

            def process(self, audio, *args, **kwargs):
                return audio

        chain = PluginChain(RATE, BLOCK)
        chain.add_builtin(Delayed(), "Lookahead")

        assert chain.reported_latency_samples == 512

        chain.set_bypassed(0, True)
        assert chain.reported_latency_samples == 0
