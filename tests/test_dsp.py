"""
Per-channel processing.

The recurring theme is block boundaries. A processor that treats each block independently
puts a discontinuity every few milliseconds, and a discontinuity is a click — so most of
these tests run several blocks and check what happens at the joins, not just within one.
"""

import math

import numpy as np
import pytest

from tonesphere.engine.dsp import (
    PAN_LAW_LINEAR,
    PAN_LAW_MINUS_3DB,
    PAN_LAW_MINUS_6DB,
    ChannelStrip,
    DriftResampler,
    Limiter,
    Panner,
    SmoothedGain,
    pan_gains,
)

BLOCK = 256
RATE = 48000


def sine(frames, freq=1000.0, amplitude=0.5, channels=2, phase=0.0, rate=RATE):
    t = (np.arange(frames, dtype=np.float64) + phase) / rate
    wave = (amplitude * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)
    return np.repeat(wave.reshape(-1, 1), channels, axis=1)


def max_step(signal):
    """Largest sample-to-sample jump — the measure of whether something clicks."""
    return float(np.max(np.abs(np.diff(signal))))


class TestPanLaw:
    def test_centre_is_equal_on_both_sides(self):
        left, right = pan_gains(0.0)
        assert left == pytest.approx(right)

    def test_constant_power_centre_is_minus_3db(self):
        """
        The whole point of the -3 dB law: a centred source has the same total power as a
        hard-panned one, so a pan sweep sounds even instead of dipping in the middle.
        """
        left, right = pan_gains(0.0, PAN_LAW_MINUS_3DB)

        assert left == pytest.approx(1 / math.sqrt(2), abs=1e-6)
        assert left ** 2 + right ** 2 == pytest.approx(1.0, abs=1e-6)

    def test_power_is_constant_across_the_sweep(self):
        for pan in np.linspace(-1.0, 1.0, 21):
            left, right = pan_gains(float(pan), PAN_LAW_MINUS_3DB)
            assert left ** 2 + right ** 2 == pytest.approx(1.0, abs=1e-6)

    def test_hard_left_silences_the_right(self):
        left, right = pan_gains(-1.0)
        assert left == pytest.approx(1.0, abs=1e-6)
        assert right == pytest.approx(0.0, abs=1e-6)

    def test_hard_right_silences_the_left(self):
        left, right = pan_gains(1.0)
        assert left == pytest.approx(0.0, abs=1e-6)
        assert right == pytest.approx(1.0, abs=1e-6)

    def test_minus_6db_law_reaches_unity_at_centre(self):
        left, right = pan_gains(0.0, PAN_LAW_MINUS_6DB)
        assert left == pytest.approx(1.0, abs=1e-6)
        assert right == pytest.approx(1.0, abs=1e-6)

    def test_linear_law_sums_to_unity(self):
        for pan in (-1.0, -0.5, 0.0, 0.5, 1.0):
            left, right = pan_gains(pan, PAN_LAW_LINEAR)
            assert left + right == pytest.approx(1.0, abs=1e-6)

    def test_pan_is_clamped(self):
        assert pan_gains(-5.0) == pan_gains(-1.0)
        assert pan_gains(5.0) == pan_gains(1.0)


class TestSmoothedGain:
    def test_static_gain_scales_exactly(self):
        gain = SmoothedGain(0.5, BLOCK)
        block = np.ones((BLOCK, 1), dtype=np.float32)

        gain.apply(block, BLOCK)

        np.testing.assert_allclose(block, 0.5, atol=1e-6)

    def test_unity_gain_leaves_the_block_untouched(self):
        gain = SmoothedGain(1.0, BLOCK)
        block = sine(BLOCK, channels=1)
        original = block.copy()

        gain.apply(block, BLOCK)

        np.testing.assert_array_equal(block, original)

    def test_change_ramps_across_the_block(self):
        gain = SmoothedGain(1.0, BLOCK)
        gain.set(0.0)

        block = np.ones((BLOCK, 1), dtype=np.float32)
        gain.apply(block, BLOCK)

        assert block[0, 0] == pytest.approx(1.0, abs=1e-3)
        assert block[-1, 0] == pytest.approx(0.0, abs=1e-2)
        assert max_step(block[:, 0]) < 0.02, "ramp must be gradual"

    def test_gain_settles_after_one_block(self):
        gain = SmoothedGain(1.0, BLOCK)
        gain.set(0.25)

        gain.apply(np.ones((BLOCK, 1), dtype=np.float32), BLOCK)
        assert gain.is_static()

        block = np.ones((BLOCK, 1), dtype=np.float32)
        gain.apply(block, BLOCK)
        np.testing.assert_allclose(block, 0.25, atol=1e-6)

    def test_consecutive_blocks_join_without_a_step(self):
        """The join between blocks is exactly where a naive implementation clicks."""
        gain = SmoothedGain(1.0, BLOCK)
        steady = np.ones((BLOCK, 1), dtype=np.float32)

        gain.set(0.5)
        first = steady.copy()
        gain.apply(first, BLOCK)

        second = steady.copy()
        gain.apply(second, BLOCK)

        joined = np.concatenate([first[:, 0], second[:, 0]])
        assert max_step(joined) < 0.02

    def test_allocates_nothing_per_block(self):
        import gc
        import tracemalloc

        gain = SmoothedGain(1.0, BLOCK)
        block = np.ones((BLOCK, 1), dtype=np.float32)

        for i in range(10):
            gain.set(0.5 if i % 2 else 1.0)
            gain.apply(block, BLOCK)

        gc.collect()
        tracemalloc.start()
        before = tracemalloc.take_snapshot()

        for i in range(200):
            gain.set(0.5 if i % 2 else 1.0)
            gain.apply(block, BLOCK)

        after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        growth = sum(s.size_diff for s in after.compare_to(before, 'filename'))
        assert growth < 100_000, f"ramping allocated {growth} bytes over 200 blocks"


class TestPanner:
    def test_mono_to_stereo_is_centred_by_default(self):
        panner = Panner(BLOCK)
        mono = np.full((BLOCK, 1), 1.0, dtype=np.float32)

        out = panner.process_mono_to_stereo(mono, BLOCK)

        expected = 1 / math.sqrt(2)
        np.testing.assert_allclose(out[:, 0], expected, atol=1e-5)
        np.testing.assert_allclose(out[:, 1], expected, atol=1e-5)

    def test_hard_left_puts_nothing_in_the_right(self):
        panner = Panner(BLOCK)
        panner.set_pan(-1.0)
        mono = np.full((BLOCK, 1), 1.0, dtype=np.float32)

        # First block ramps from centre; the second is settled.
        panner.process_mono_to_stereo(mono, BLOCK)
        out = panner.process_mono_to_stereo(mono, BLOCK)

        np.testing.assert_allclose(out[:, 0], 1.0, atol=1e-4)
        np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-4)

    def test_pan_change_ramps(self):
        panner = Panner(BLOCK)
        mono = np.full((BLOCK, 1), 1.0, dtype=np.float32)
        panner.process_mono_to_stereo(mono, BLOCK)

        panner.set_pan(1.0)
        out = panner.process_mono_to_stereo(mono, BLOCK)

        assert max_step(out[:, 0]) < 0.02, "pan sweep must not click"

    def test_centred_panner_reports_itself_as_a_no_op(self):
        """Lets the mixer skip the work entirely in the common case."""
        assert Panner(BLOCK).is_centred() is True


class TestLimiter:
    def test_quiet_audio_passes_untouched(self):
        limiter = Limiter(RATE, BLOCK)
        block = sine(BLOCK, amplitude=0.3)
        original = block.copy()

        limiter.process(block, BLOCK)

        np.testing.assert_array_equal(block, original)
        assert limiter.reduction_db == 0.0

    def test_overs_are_pulled_below_the_threshold(self):
        """
        Several sources at unity easily exceed 1.0. In float that is harmless, but the
        device's integer converter would clip it hard.
        """
        limiter = Limiter(RATE, BLOCK, attack_ms=0.1)
        loud = sine(BLOCK, amplitude=2.0)

        # Let the envelope settle over a few blocks.
        for _ in range(20):
            block = loud.copy()
            limiter.process(block, BLOCK)

        assert float(np.max(np.abs(block))) <= 1.001
        assert limiter.reduction_db < 0.0

    def test_reduction_is_gradual_not_a_hard_clip(self):
        """
        Hard clipping flattens the waveform and generates broadband distortion. Gain
        reduction keeps the shape, so the result stays a sine.
        """
        limiter = Limiter(RATE, BLOCK, attack_ms=0.1)
        loud = sine(BLOCK, freq=1000.0, amplitude=1.5)

        for _ in range(30):
            block = loud.copy()
            limiter.process(block, BLOCK)

        # A clipped sine grows strong odd harmonics; a gain-reduced one does not.
        spectrum = np.abs(np.fft.rfft(block[:, 0] * np.hanning(BLOCK)))
        freqs = np.fft.rfftfreq(BLOCK, 1.0 / RATE)
        fundamental = spectrum[np.argmin(np.abs(freqs - 1000.0))]
        third = spectrum[np.argmin(np.abs(freqs - 3000.0))]

        assert third < fundamental * 0.05, "limiter should not be clipping"

    def test_release_recovers_after_the_peak(self):
        limiter = Limiter(RATE, BLOCK, attack_ms=0.1, release_ms=5.0)

        for _ in range(10):
            limiter.process(sine(BLOCK, amplitude=2.0), BLOCK)
        reduced = limiter.reduction_db

        for _ in range(200):
            limiter.process(sine(BLOCK, amplitude=0.1), BLOCK)

        assert limiter.reduction_db > reduced
        assert limiter.reduction_db == pytest.approx(0.0, abs=0.5)

    def test_no_discontinuity_at_block_boundaries(self):
        limiter = Limiter(RATE, BLOCK)
        blocks = []

        for i in range(6):
            block = sine(BLOCK, amplitude=1.8, phase=i * BLOCK)
            limiter.process(block, BLOCK)
            blocks.append(block[:, 0])

        joined = np.concatenate(blocks)
        # A 1 kHz sine at 48 kHz moves at most ~0.13 per sample; allow headroom for the
        # envelope but nothing like a step.
        assert max_step(joined) < 0.3


class TestDriftResampler:
    def test_unity_ratio_reproduces_the_input(self):
        resampler = DriftResampler(1, BLOCK)
        source = sine(BLOCK + 8, channels=1)
        out = np.zeros((BLOCK, 1), dtype=np.float32)

        resampler.process(source, out, BLOCK)

        np.testing.assert_allclose(out[:, 0], source[:BLOCK, 0], atol=1e-5)

    def test_ratio_is_clamped_to_stay_inaudible(self):
        """
        A large correction would be audible as pitch movement. The clamp keeps it well
        below that, at the cost of correcting slowly — which is fine, drift is slow.
        """
        resampler = DriftResampler(2, BLOCK)

        resampler.set_ratio(2.0)
        assert resampler.ratio <= 1.0 + DriftResampler.MAX_RATIO_DEVIATION

        resampler.set_ratio(0.1)
        assert resampler.ratio >= 1.0 - DriftResampler.MAX_RATIO_DEVIATION

    def test_position_is_continuous_across_blocks(self):
        """
        The resampler this replaces reset its position every block, so every boundary got
        a discontinuity — a click at exactly the block rate, the most audible kind. Feed a
        continuous sine through several blocks and check the output is still smooth.
        """
        resampler = DriftResampler(1, BLOCK)
        resampler.set_ratio(1.0 + DriftResampler.MAX_RATIO_DEVIATION)

        phase = 0
        pieces = []
        for _ in range(8):
            needed = resampler.input_frames_needed(BLOCK)
            source = sine(needed, freq=500.0, channels=1, phase=phase)
            out = np.zeros((BLOCK, 1), dtype=np.float32)
            consumed = resampler.process(source, out, BLOCK)
            phase += consumed
            pieces.append(out[:, 0])

        joined = np.concatenate(pieces)
        # A 500 Hz sine at 48 kHz steps by at most ~0.065 between samples.
        assert max_step(joined) < 0.1, "block boundary discontinuity"

    def test_reports_frames_consumed(self):
        resampler = DriftResampler(1, BLOCK)
        resampler.set_ratio(1.0)
        source = sine(BLOCK + 8, channels=1)
        out = np.zeros((BLOCK, 1), dtype=np.float32)

        consumed = resampler.process(source, out, BLOCK)

        assert BLOCK - 2 <= consumed <= BLOCK + 2

    def test_faster_ratio_consumes_more_input(self):
        resampler = DriftResampler(1, BLOCK)
        resampler.set_ratio(1.0 + DriftResampler.MAX_RATIO_DEVIATION)

        needed = resampler.input_frames_needed(BLOCK)
        assert needed > BLOCK

    def test_starved_input_yields_silence_not_garbage(self):
        resampler = DriftResampler(1, BLOCK)
        out = np.full((BLOCK, 1), 7.0, dtype=np.float32)

        resampler.process(np.zeros((1, 1), dtype=np.float32), out, BLOCK)

        assert np.all(out == 0.0)


class TestChannelStrip:
    def test_default_strip_is_transparent(self):
        """The common case, and worth skipping entirely in the mixer."""
        assert ChannelStrip(2, BLOCK).is_transparent() is True

    def test_gain_is_applied_per_channel(self):
        strip = ChannelStrip(2, BLOCK)
        strip.set_channel_gain(0, 0.5)
        strip.set_channel_gain(1, 1.0)

        block = np.ones((BLOCK, 2), dtype=np.float32)
        strip.process(block, BLOCK)
        strip.process(block := np.ones((BLOCK, 2), dtype=np.float32), BLOCK)

        np.testing.assert_allclose(block[:, 0], 0.5, atol=1e-4)
        np.testing.assert_allclose(block[:, 1], 1.0, atol=1e-4)

    def test_mute_silences_only_its_channel(self):
        strip = ChannelStrip(2, BLOCK)
        strip.set_channel_mute(0, True)

        for _ in range(3):
            block = np.ones((BLOCK, 2), dtype=np.float32)
            strip.process(block, BLOCK)

        assert float(np.max(np.abs(block[:, 0]))) < 1e-4
        np.testing.assert_allclose(block[:, 1], 1.0, atol=1e-4)

    def test_mute_ramps_rather_than_cutting(self):
        strip = ChannelStrip(2, BLOCK)
        block = np.ones((BLOCK, 2), dtype=np.float32)
        strip.process(block, BLOCK)

        strip.set_channel_mute(0, True)
        block = np.ones((BLOCK, 2), dtype=np.float32)
        strip.process(block, BLOCK)

        assert block[0, 0] == pytest.approx(1.0, abs=1e-2)
        assert max_step(block[:, 0]) < 0.02

    def test_polarity_inversion_flips_the_sign(self):
        strip = ChannelStrip(2, BLOCK)
        strip.set_channel_inverted(0, True)

        block = np.ones((BLOCK, 2), dtype=np.float32)
        strip.process(block, BLOCK)

        np.testing.assert_allclose(block[:, 0], -1.0, atol=1e-4)
        np.testing.assert_allclose(block[:, 1], 1.0, atol=1e-4)

    def test_inverted_channels_cancel_when_summed(self):
        """The practical use: find which of two signals is fighting the other."""
        strip = ChannelStrip(2, BLOCK)
        strip.set_channel_inverted(1, True)

        block = sine(BLOCK, amplitude=0.5)
        strip.process(block, BLOCK)

        assert float(np.max(np.abs(block[:, 0] + block[:, 1]))) < 1e-5

    def test_channel_swap_exchanges_left_and_right(self):
        strip = ChannelStrip(2, BLOCK)
        strip.set_swapped(True)

        block = np.zeros((BLOCK, 2), dtype=np.float32)
        block[:, 0] = 1.0
        block[:, 1] = 2.0
        strip.process(block, BLOCK)

        np.testing.assert_allclose(block[:, 0], 2.0, atol=1e-4)
        np.testing.assert_allclose(block[:, 1], 1.0, atol=1e-4)

    def test_swap_does_not_alias_channels_together(self):
        """An in-place swap on NumPy views would overwrite one channel with the other."""
        strip = ChannelStrip(2, BLOCK)
        strip.set_swapped(True)

        block = np.zeros((BLOCK, 2), dtype=np.float32)
        block[:, 0] = 1.0
        block[:, 1] = 2.0
        strip.process(block, BLOCK)

        assert not np.allclose(block[:, 0], block[:, 1])

    def test_master_gain_applies_to_every_channel(self):
        strip = ChannelStrip(2, BLOCK)
        strip.set_master_gain(0.5)

        for _ in range(3):
            block = np.ones((BLOCK, 2), dtype=np.float32)
            strip.process(block, BLOCK)

        np.testing.assert_allclose(block, 0.5, atol=1e-4)

    def test_allocates_nothing_per_block(self):
        import gc
        import tracemalloc

        strip = ChannelStrip(2, BLOCK)
        strip.set_channel_gain(0, 0.7)
        strip.set_swapped(True)
        block = np.ones((BLOCK, 2), dtype=np.float32)

        for _ in range(10):
            strip.process(block, BLOCK)

        gc.collect()
        tracemalloc.start()
        before = tracemalloc.take_snapshot()

        for _ in range(200):
            strip.process(block, BLOCK)

        after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        growth = sum(s.size_diff for s in after.compare_to(before, 'filename'))
        assert growth < 100_000, f"strip allocated {growth} bytes over 200 blocks"


class TestChannelControlsReachAudio:
    """
    `ChannelControlManager` stored volume, mute, solo, pan and polarity from the first
    commit and none of it ever touched a sample. These prove the connection exists.
    """

    def test_control_state_projects_into_a_strip(self):
        from tonesphere.core.channel_control import DeviceChannelControl

        control = DeviceChannelControl(device_id=1, num_channels=2)
        control.set_channel_volume(0, 0.5)
        control.set_channel_mute(1, True)
        control.set_channel_inverted(0, True)

        strip = ChannelStrip(2, BLOCK)
        control.apply_to_strip(strip)

        assert strip.is_transparent() is False

        for _ in range(3):
            block = np.ones((BLOCK, 2), dtype=np.float32)
            strip.process(block, BLOCK)

        assert block[BLOCK // 2, 0] == pytest.approx(-0.5, abs=1e-3), "gain and invert"
        assert abs(float(block[BLOCK // 2, 1])) < 1e-3, "mute"

    def test_solo_silences_other_channels(self):
        from tonesphere.core.channel_control import DeviceChannelControl

        control = DeviceChannelControl(device_id=1, num_channels=2)
        control.set_channel_solo(0, True)

        strip = ChannelStrip(2, BLOCK)
        control.apply_to_strip(strip)

        for _ in range(3):
            block = np.ones((BLOCK, 2), dtype=np.float32)
            strip.process(block, BLOCK)

        assert block[BLOCK // 2, 0] == pytest.approx(1.0, abs=1e-3)
        assert abs(float(block[BLOCK // 2, 1])) < 1e-3

    def test_master_mute_projects(self):
        from tonesphere.core.channel_control import DeviceChannelControl

        control = DeviceChannelControl(device_id=1, num_channels=2)
        control.set_master_mute(True)

        strip = ChannelStrip(2, BLOCK)
        control.apply_to_strip(strip)

        for _ in range(3):
            block = np.ones((BLOCK, 2), dtype=np.float32)
            strip.process(block, BLOCK)

        assert float(np.max(np.abs(block))) < 1e-3

    def test_offline_and_realtime_pan_laws_agree(self):
        """
        `process_audio` is used offline and the strip runs live. If they disagreed, a
        rendered file would not match what the user heard.
        """
        from tonesphere.core.channel_control import DeviceChannelControl

        control = DeviceChannelControl(device_id=1, num_channels=2)
        control.set_channel_pan(0, 0.5)

        left, _ = pan_gains(0.5)
        block = np.ones((BLOCK, 2), dtype=np.float32)
        processed = control.process_audio(block)

        assert processed[0, 0] == pytest.approx(left, abs=1e-5)
