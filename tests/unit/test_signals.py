"""The shared test signals are themselves asserted on: a wrong stimulus proves nothing."""

import numpy as np
import pytest

from tests.signals import (
    RATE,
    assert_finite,
    channel_impulses,
    dominant_frequency,
    impulse,
    log_sweep,
    peak,
    pink_noise,
    rms,
    sine,
    white_noise,
)


def test_sine_frequency_and_rms():
    tone = sine(RATE, 1000.0, amplitude=0.5)
    assert dominant_frequency(tone) == pytest.approx(1000.0, abs=2.0)
    assert rms(tone) == pytest.approx(0.5 / np.sqrt(2), rel=1e-3)
    assert tone.dtype == np.float32 and tone.shape == (RATE, 2)


def test_sine_blocks_join_without_a_phase_jump():
    joined = np.concatenate([sine(256, 440.0, phase=0), sine(256, 440.0, phase=256)])
    assert np.allclose(joined, sine(512, 440.0), atol=1e-6)


def test_impulse_and_channel_impulses_are_distinguishable_per_channel():
    assert peak(impulse(64, at=10)) == 1.0
    block = channel_impulses(128, 4, spacing=16)
    for channel in range(4):
        assert int(np.argmax(block[:, channel])) == channel * 16
    with pytest.raises(ValueError):
        channel_impulses(32, 4, spacing=16)


def test_log_sweep_moves_from_low_to_high():
    sweep = log_sweep(RATE, start_hz=100.0, end_hz=10000.0)
    quarter = RATE // 4
    assert dominant_frequency(sweep[:quarter]) < dominant_frequency(sweep[-quarter:])
    assert peak(sweep) <= 0.5 + 1e-6


def test_noise_is_seeded_bounded_and_finite():
    assert np.array_equal(white_noise(1024), white_noise(1024))
    assert peak(white_noise(1024, amplitude=0.25)) <= 0.25
    pink = pink_noise(8192, amplitude=0.25)
    assert peak(pink) == pytest.approx(0.25, rel=1e-4)
    assert_finite(pink)
    spectrum = np.abs(np.fft.rfft(pink[:, 0]))
    assert spectrum[10:100].mean() > spectrum[1000:4000].mean(), "pink noise must fall with frequency"


def test_assert_finite_catches_nan():
    block = np.zeros((8, 2), dtype=np.float32)
    block[3, 1] = np.nan
    with pytest.raises(AssertionError, match="non-finite"):
        assert_finite(block)
