"""
The round-trip estimator on synthetic signals, where the true delay is known exactly.

A measurement method is only trustworthy if it finds a delay it was given, through the
kind of damage a real path does — filtering, noise, level loss — and if it refuses when
there is nothing to find.
"""

import numpy as np
import pytest

from tonesphere.native.roundtrip import CONFIDENCE_THRESHOLD, gcc_phat, sweep

RATE = 48000


def monitor_and_capture(delay: int, *, gain=0.3, noise=0.0, lowpass=False, seed=5):
    stimulus = sweep(RATE, 0.5, 150.0, 12000.0, -12.0).astype(np.float64)
    lead = np.zeros(RATE // 4)
    monitor = np.concatenate([lead, stimulus, np.zeros(RATE)])
    captured = np.zeros_like(monitor)
    captured[delay:] = monitor[:len(monitor) - delay] * gain
    if lowpass:
        # A crude one-pole low-pass: the speaker-and-microphone kind of colouring.
        for i in range(1, len(captured)):
            captured[i] = 0.7 * captured[i - 1] + 0.3 * captured[i]
    if noise:
        captured += np.random.default_rng(seed).normal(0, noise, len(captured))
    return monitor, captured


@pytest.mark.parametrize("delay", [0, 1, 480, 2993, 9000])
def test_finds_a_known_delay_exactly(delay):
    monitor, captured = monitor_and_capture(delay)
    lag, confidence = gcc_phat(captured, monitor, RATE // 2)
    assert lag == delay
    assert confidence > CONFIDENCE_THRESHOLD


def test_finds_it_through_filtering_and_noise():
    monitor, captured = monitor_and_capture(3120, gain=0.05, noise=0.01, lowpass=True)
    lag, confidence = gcc_phat(captured, monitor, RATE // 2)
    assert abs(lag - 3120) <= 1, "a one-pole filter may move the peak by at most a sample"
    assert confidence > CONFIDENCE_THRESHOLD


def test_refuses_when_the_input_only_hears_noise():
    monitor, _ = monitor_and_capture(0)
    unrelated = np.random.default_rng(9).normal(0, 0.01, len(monitor))
    _, confidence = gcc_phat(unrelated, monitor, RATE // 2)
    assert confidence < CONFIDENCE_THRESHOLD, "noise must never produce a latency"


def test_the_sweep_starts_and_ends_without_a_click():
    s = sweep(RATE, 0.5, 150.0, 12000.0, -12.0)
    assert abs(s[0]) < 1e-3 and abs(s[-1]) < 1e-3
    assert np.max(np.abs(s)) == pytest.approx(10 ** (-12 / 20), rel=0.01)
