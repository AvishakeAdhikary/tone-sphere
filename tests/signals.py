"""
Test signals and the measurements the suite asserts on.

One copy, because eight files each had their own `sine()`, and a test is only as good as
the signal it feeds: two helpers that disagreed about phase or amplitude would make two
tests that look alike prove different things. Everything past `frames`/`freq` is
keyword-only, since the old copies disagreed about positional order.
"""

import math

import numpy as np

RATE = 48000


def sine(frames: int, freq: float = 1000.0, *, rate: int = RATE, amplitude: float = 0.5,
         channels: int = 2, phase: float = 0.0) -> np.ndarray:
    """
    A test tone, `(frames, channels)` float32. 1 kHz by default because it sits mid-band
    where nothing rolls it off. `phase` is in samples, so consecutive blocks join.
    """
    t = (np.arange(frames, dtype=np.float64) + phase) / rate
    wave = (amplitude * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)
    return np.repeat(wave.reshape(-1, 1), channels, axis=1)


def impulse(frames: int, *, at: int = 0, amplitude: float = 1.0, channels: int = 2) -> np.ndarray:
    block = np.zeros((frames, channels), dtype=np.float32)
    block[at, :] = amplitude
    return block


def channel_impulses(frames: int, channels: int, *, spacing: int = 16, amplitude: float = 1.0) -> np.ndarray:
    """
    One impulse per channel, each at a different offset (`channel * spacing`), so a
    channel-mapping bug shows up as an impulse arriving at the wrong offset rather than
    as an indistinguishable copy.
    """
    if (channels - 1) * spacing >= frames:
        raise ValueError("block too short for the impulse spacing")
    block = np.zeros((frames, channels), dtype=np.float32)
    for channel in range(channels):
        block[channel * spacing, channel] = amplitude
    return block


def log_sweep(frames: int, *, start_hz: float = 20.0, end_hz: float = 20000.0, rate: int = RATE,
              amplitude: float = 0.5, channels: int = 2) -> np.ndarray:
    """Exponential sine sweep (Farina), the standard stimulus for a transfer-function check."""
    duration = frames / rate
    t = np.arange(frames, dtype=np.float64) / rate
    k = math.log(end_hz / start_hz)
    wave = amplitude * np.sin(2.0 * math.pi * start_hz * duration / k * (np.exp(t / duration * k) - 1.0))
    return np.repeat(wave.astype(np.float32).reshape(-1, 1), channels, axis=1)


def white_noise(frames: int, *, amplitude: float = 0.25, channels: int = 2, seed: int = 1234) -> np.ndarray:
    """Seeded, so a failure reproduces exactly."""
    rng = np.random.default_rng(seed)
    return (amplitude * rng.uniform(-1.0, 1.0, size=(frames, channels))).astype(np.float32)


def pink_noise(frames: int, *, amplitude: float = 0.25, channels: int = 2, seed: int = 1234) -> np.ndarray:
    """1/f noise by spectral shaping of seeded white noise, normalised to `amplitude` peak."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((frames, channels))
    spectrum = np.fft.rfft(white, axis=0)
    freqs = np.fft.rfftfreq(frames)
    freqs[0] = freqs[1] if frames > 1 else 1.0
    spectrum /= np.sqrt(freqs)[:, None]
    shaped = np.fft.irfft(spectrum, n=frames, axis=0)
    shaped *= amplitude / np.max(np.abs(shaped))
    return shaped.astype(np.float32)


def dominant_frequency(block: np.ndarray, rate: int = RATE) -> float:
    """Peak bin of the spectrum, used to prove a tone survived unshifted."""
    mono = block[:, 0] if block.ndim > 1 else block
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    return float(np.fft.rfftfreq(len(mono), 1.0 / rate)[int(np.argmax(spectrum))])


def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))


def peak(block: np.ndarray) -> float:
    return float(np.max(np.abs(block))) if block.size else 0.0


def assert_finite(block: np.ndarray) -> None:
    """A diverging filter produces NaN or inf long before anyone hears it; catch it here."""
    bad = ~np.isfinite(block)
    assert not bad.any(), f"{int(bad.sum())} non-finite sample(s), first at {np.argwhere(bad)[0].tolist()}"
