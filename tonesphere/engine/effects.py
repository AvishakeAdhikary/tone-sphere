"""
Effects: built-in DSP and VST3/AU plugin hosting.

Replaces `core/processor.py`, whose "real-time audio effects" were an EQ that ignored two
of its three bands, a compressor that ignored attack and release, and a reverb built on
`np.roll` — which is a circular shift, so it wrapped the end of each block onto the start
instead of delaying anything. None of it was connected to an audio path.

Two kinds of effect live here:

  Built-ins  Biquad filters and a compressor, written to keep their state across blocks.
             A filter that resets each block is not a filter, it is a click generator.

  Plugins    VST3 and AU, hosted through `pedalboard`. This is the point of the whole
             exercise for a guitarist: Guitar Rig, Neural DSP or whatever else you already
             own runs inside ToneSphere, on the same low-latency path as everything else.
"""

import math
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)


class Biquad:
    """
    A second-order IIR section — the building block of every practical EQ band.

    State (two input and two output samples) carries across blocks. Without that the
    filter restarts every block and produces a discontinuity at each boundary, which is
    the single most common bug in block-based DSP.

    Coefficients follow the Audio EQ Cookbook, because they are correct, well understood
    and easy to check against.
    """

    __slots__ = ('_b0', '_b1', '_b2', '_a1', '_a2', '_x1', '_x2', '_y1', '_y2', 'channels')

    def __init__(self, channels: int = 2):
        self.channels = channels
        self._b0, self._b1, self._b2 = 1.0, 0.0, 0.0
        self._a1, self._a2 = 0.0, 0.0
        self.reset()

    def reset(self):
        self._x1 = np.zeros(self.channels, dtype=np.float64)
        self._x2 = np.zeros(self.channels, dtype=np.float64)
        self._y1 = np.zeros(self.channels, dtype=np.float64)
        self._y2 = np.zeros(self.channels, dtype=np.float64)

    # --- Coefficient design ---

    def set_peaking(self, samplerate: int, frequency: float, q: float, gain_db: float):
        """A bell: boost or cut around a centre frequency."""
        amplitude = 10.0 ** (gain_db / 40.0)
        omega = 2.0 * math.pi * min(frequency, samplerate * 0.49) / samplerate
        alpha = math.sin(omega) / (2.0 * max(q, 0.01))
        cos_omega = math.cos(omega)

        b0 = 1.0 + alpha * amplitude
        b1 = -2.0 * cos_omega
        b2 = 1.0 - alpha * amplitude
        a0 = 1.0 + alpha / amplitude
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha / amplitude

        self._normalise(b0, b1, b2, a0, a1, a2)

    def set_low_shelf(self, samplerate: int, frequency: float, gain_db: float, slope: float = 1.0):
        """Lift or cut everything below a corner. What a "bass" control actually is."""
        amplitude = 10.0 ** (gain_db / 40.0)
        omega = 2.0 * math.pi * min(frequency, samplerate * 0.49) / samplerate
        cos_omega = math.cos(omega)
        alpha = (math.sin(omega) / 2.0) * math.sqrt(
            (amplitude + 1.0 / amplitude) * (1.0 / slope - 1.0) + 2.0
        )
        beta = 2.0 * math.sqrt(amplitude) * alpha

        b0 = amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * cos_omega + beta)
        b1 = 2.0 * amplitude * ((amplitude - 1.0) - (amplitude + 1.0) * cos_omega)
        b2 = amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * cos_omega - beta)
        a0 = (amplitude + 1.0) + (amplitude - 1.0) * cos_omega + beta
        a1 = -2.0 * ((amplitude - 1.0) + (amplitude + 1.0) * cos_omega)
        a2 = (amplitude + 1.0) + (amplitude - 1.0) * cos_omega - beta

        self._normalise(b0, b1, b2, a0, a1, a2)

    def set_high_shelf(self, samplerate: int, frequency: float, gain_db: float, slope: float = 1.0):
        amplitude = 10.0 ** (gain_db / 40.0)
        omega = 2.0 * math.pi * min(frequency, samplerate * 0.49) / samplerate
        cos_omega = math.cos(omega)
        alpha = (math.sin(omega) / 2.0) * math.sqrt(
            (amplitude + 1.0 / amplitude) * (1.0 / slope - 1.0) + 2.0
        )
        beta = 2.0 * math.sqrt(amplitude) * alpha

        b0 = amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_omega + beta)
        b1 = -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cos_omega)
        b2 = amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cos_omega - beta)
        a0 = (amplitude + 1.0) - (amplitude - 1.0) * cos_omega + beta
        a1 = 2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cos_omega)
        a2 = (amplitude + 1.0) - (amplitude - 1.0) * cos_omega - beta

        self._normalise(b0, b1, b2, a0, a1, a2)

    def set_highpass(self, samplerate: int, frequency: float, q: float = 0.707):
        """
        Remove everything below a corner.

        The most useful single filter on a guitar or mic channel: it takes out handling
        noise and room rumble that eat headroom without contributing anything audible.
        """
        omega = 2.0 * math.pi * min(frequency, samplerate * 0.49) / samplerate
        alpha = math.sin(omega) / (2.0 * max(q, 0.01))
        cos_omega = math.cos(omega)

        b0 = (1.0 + cos_omega) / 2.0
        b1 = -(1.0 + cos_omega)
        b2 = (1.0 + cos_omega) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha

        self._normalise(b0, b1, b2, a0, a1, a2)

    def set_lowpass(self, samplerate: int, frequency: float, q: float = 0.707):
        omega = 2.0 * math.pi * min(frequency, samplerate * 0.49) / samplerate
        alpha = math.sin(omega) / (2.0 * max(q, 0.01))
        cos_omega = math.cos(omega)

        b0 = (1.0 - cos_omega) / 2.0
        b1 = 1.0 - cos_omega
        b2 = (1.0 - cos_omega) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha

        self._normalise(b0, b1, b2, a0, a1, a2)

    def _normalise(self, b0, b1, b2, a0, a1, a2):
        self._b0, self._b1, self._b2 = b0 / a0, b1 / a0, b2 / a0
        self._a1, self._a2 = a1 / a0, a2 / a0

    # --- Processing ---

    def process(self, block: np.ndarray, frames: int) -> np.ndarray:
        """
        Filter in place, carrying state across the block boundary.

        A direct-form difference equation is inherently sequential, so this is a Python
        loop over samples — the one place in the engine where that is unavoidable. It is
        why EQ is opt-in per channel rather than always on.
        """
        b0, b1, b2 = self._b0, self._b1, self._b2
        a1, a2 = self._a1, self._a2

        x1, x2 = self._x1, self._x2
        y1, y2 = self._y1, self._y2

        channels = min(block.shape[1], self.channels)

        # Read from a separate copy of the input. Writing the output back into the array
        # being read makes `x0` — a *view* of one row — alias the value just written, so
        # the delay line ends up fed by the filter's own output. The feedback term then
        # compounds and the filter diverges to infinity within a block.
        source = block[:frames, :channels].astype(np.float64)
        output = np.empty_like(source)

        for n in range(frames):
            x0 = source[n]
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            output[n] = y0
            x2, x1 = x1, x0
            y2, y1 = y1, y0

        self._x1, self._x2 = x1, x2
        self._y1, self._y2 = y1, y2

        block[:frames, :channels] = output.astype(np.float32)
        return block

    def magnitude_at(self, samplerate: int, frequency: float) -> float:
        """
        The filter's gain at one frequency, for drawing a response curve.

        Evaluates the transfer function on the unit circle, so it reflects the actual
        coefficients rather than the design intent.
        """
        omega = 2.0 * math.pi * frequency / samplerate
        cos1, sin1 = math.cos(omega), math.sin(omega)
        cos2, sin2 = math.cos(2 * omega), math.sin(2 * omega)

        num_real = self._b0 + self._b1 * cos1 + self._b2 * cos2
        num_imag = -(self._b1 * sin1 + self._b2 * sin2)
        den_real = 1.0 + self._a1 * cos1 + self._a2 * cos2
        den_imag = -(self._a1 * sin1 + self._a2 * sin2)

        numerator = math.hypot(num_real, num_imag)
        denominator = math.hypot(den_real, den_imag)
        return numerator / denominator if denominator else 0.0


@dataclass
class EQBand:
    """One band's settings."""
    kind: str = 'peaking'        # peaking | lowshelf | highshelf | highpass | lowpass
    frequency: float = 1000.0
    gain_db: float = 0.0
    q: float = 0.707
    enabled: bool = True


class ParametricEQ:
    """
    A multi-band EQ built from biquads.

    Bands that are flat and not filters are skipped entirely, so an EQ nobody has touched
    costs nothing. The replaced implementation applied `processed *= (1.0 + low_gain * 0.1)`
    and discarded the mid and high controls altogether.
    """

    def __init__(self, samplerate: int, channels: int = 2,
                 bands: Sequence[EQBand] | None = None):
        self.samplerate = samplerate
        self.channels = channels
        self.bands: list[EQBand] = list(bands) if bands else []
        self._filters: list[Biquad] = []
        self._rebuild()

    def _rebuild(self):
        self._filters = []
        for band in self.bands:
            biquad = Biquad(self.channels)
            self._configure(biquad, band)
            self._filters.append(biquad)

    def _configure(self, biquad: Biquad, band: EQBand):
        if band.kind == 'lowshelf':
            biquad.set_low_shelf(self.samplerate, band.frequency, band.gain_db)
        elif band.kind == 'highshelf':
            biquad.set_high_shelf(self.samplerate, band.frequency, band.gain_db)
        elif band.kind == 'highpass':
            biquad.set_highpass(self.samplerate, band.frequency, band.q)
        elif band.kind == 'lowpass':
            biquad.set_lowpass(self.samplerate, band.frequency, band.q)
        else:
            biquad.set_peaking(self.samplerate, band.frequency, band.q, band.gain_db)

    def set_band(self, index: int, band: EQBand):
        while len(self.bands) <= index:
            self.bands.append(EQBand())
            self._filters.append(Biquad(self.channels))

        self.bands[index] = band
        self._configure(self._filters[index], band)

    def add_band(self, band: EQBand) -> int:
        self.bands.append(band)
        biquad = Biquad(self.channels)
        self._configure(biquad, band)
        self._filters.append(biquad)
        return len(self.bands) - 1

    @property
    def is_flat(self) -> bool:
        """Whether the EQ would change anything, so the caller can skip it."""
        return all(
            not band.enabled or (band.gain_db == 0.0
                                 and band.kind in ('peaking', 'lowshelf', 'highshelf'))
            for band in self.bands
        )

    def process(self, block: np.ndarray, frames: int) -> np.ndarray:
        for band, biquad in zip(self.bands, self._filters, strict=True):
            if not band.enabled:
                continue
            if band.gain_db == 0.0 and band.kind in ('peaking', 'lowshelf', 'highshelf'):
                continue
            biquad.process(block, frames)
        return block

    def response(self, frequencies: Sequence[float]) -> list[float]:
        """Combined magnitude response in dB, for plotting."""
        result = []
        for frequency in frequencies:
            magnitude = 1.0
            for band, biquad in zip(self.bands, self._filters, strict=True):
                if band.enabled:
                    magnitude *= biquad.magnitude_at(self.samplerate, frequency)
            result.append(20.0 * math.log10(magnitude) if magnitude > 0 else -120.0)
        return result

    def reset(self):
        for biquad in self._filters:
            biquad.reset()


class Compressor:
    """
    A compressor with attack, release, knee and makeup gain that all do something.

    The version this replaces converted to dB, applied a static curve and converted back,
    ignoring its attack, release and makeup parameters entirely — a waveshaper, not a
    compressor. What makes a compressor a compressor is the envelope follower: gain
    reduction has to lag the signal, or it distorts rather than compresses.

    Detection is per block. At 2.7-5 ms per block that resolves everything but the very
    fastest attack settings, and it keeps the cost to a couple of NumPy reductions instead
    of a per-sample Python loop.
    """

    __slots__ = ('samplerate', 'blocksize', 'threshold_db', 'ratio', 'knee_db',
                 'makeup_db', '_envelope_db', '_attack', '_release', 'reduction_db',
                 '_scratch', '_unit')

    def __init__(self, samplerate: int, blocksize: int,
                 threshold_db: float = -18.0, ratio: float = 4.0,
                 attack_ms: float = 10.0, release_ms: float = 100.0,
                 knee_db: float = 6.0, makeup_db: float = 0.0):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.knee_db = knee_db
        self.makeup_db = makeup_db

        self._envelope_db = 0.0
        self.reduction_db = 0.0

        self.set_timing(attack_ms, release_ms)

        self._scratch = np.zeros((blocksize, 1), dtype=np.float32)
        self._unit = np.linspace(0.0, 1.0, blocksize, dtype=np.float32).reshape(-1, 1)

    def set_timing(self, attack_ms: float, release_ms: float):
        """
        Coefficients per block, not per sample.

        The envelope advances once per callback, so the time constants must be expressed
        in block periods. Getting this wrong scales every time by the block size — the
        same mistake that made the limiter's 1 ms attack behave like 256 ms.
        """
        block_period = self.blocksize / self.samplerate

        def coeff(time_ms: float) -> float:
            return math.exp(-block_period / max(1e-6, time_ms / 1000.0))

        self._attack = coeff(attack_ms)
        self._release = coeff(release_ms)

    def process(self, block: np.ndarray, frames: int) -> np.ndarray:
        peak = float(np.abs(block[:frames]).max())

        if peak <= 1e-7:
            level_db = -120.0
        else:
            level_db = 20.0 * math.log10(peak)

        target_db = self._gain_reduction_for(level_db)

        # Attack when reducing further, release when recovering. A single time constant
        # for both is what makes a naive compressor pump.
        coeff = self._attack if target_db < self._envelope_db else self._release
        start_db = self._envelope_db
        self._envelope_db = target_db + (self._envelope_db - target_db) * coeff
        self.reduction_db = self._envelope_db

        total_start = start_db + self.makeup_db
        total_end = self._envelope_db + self.makeup_db

        if abs(total_start) < 0.01 and abs(total_end) < 0.01:
            return block

        start_gain = 10.0 ** (total_start / 20.0)
        end_gain = 10.0 ** (total_end / 20.0)

        if abs(start_gain - end_gain) < 1e-6:
            block[:frames] *= end_gain
        else:
            ramp = self._scratch[:frames]
            ramp[:] = self._unit[:frames]
            ramp *= (end_gain - start_gain)
            ramp += start_gain
            block[:frames] *= ramp

        return block

    def _gain_reduction_for(self, level_db: float) -> float:
        """
        How much to reduce, in dB, for a given input level.

        A soft knee interpolates the ratio across `knee_db` around the threshold, so
        compression comes in gradually instead of switching on at a hard edge — audible
        as a click on material that hovers near the threshold.
        """
        over = level_db - self.threshold_db
        half_knee = self.knee_db / 2.0

        if over <= -half_knee:
            return 0.0

        if over >= half_knee:
            return -(over - over / self.ratio)

        # Quadratic interpolation through the knee.
        knee_factor = (over + half_knee) ** 2 / (2.0 * self.knee_db)
        return -(knee_factor - knee_factor / self.ratio)

    def reset(self):
        self._envelope_db = 0.0
        self.reduction_db = 0.0


class Delay:
    """
    A delay line with feedback.

    An actual circular buffer with a write cursor, which is what a delay is. The version
    this replaces used `np.roll`, a circular shift *within the block* — so it wrapped the
    end of each block onto its own start, producing a stutter at the block rate rather
    than an echo.
    """

    def __init__(self, samplerate: int, channels: int = 2,
                 max_delay_ms: float = 2000.0,
                 delay_ms: float = 250.0, feedback: float = 0.35, mix: float = 0.25):
        self.samplerate = samplerate
        self.channels = channels
        self.feedback = feedback
        self.mix = mix

        self._capacity = int(samplerate * max_delay_ms / 1000.0) + 1
        self._buffer = np.zeros((self._capacity, channels), dtype=np.float32)
        self._write = 0

        self._delay_samples = 0
        self.set_delay_ms(delay_ms)

    def set_delay_ms(self, delay_ms: float):
        samples = int(self.samplerate * delay_ms / 1000.0)
        self._delay_samples = max(1, min(samples, self._capacity - 1))

    @property
    def delay_ms(self) -> float:
        return self._delay_samples / self.samplerate * 1000.0

    def process(self, block: np.ndarray, frames: int) -> np.ndarray:
        """
        Mix delayed signal into the block, in place.

        Handled in at most two contiguous slices so the wrap costs no per-sample work.
        """
        if self.mix <= 0.0:
            return block

        channels = min(block.shape[1], self.channels)
        read = (self._write - self._delay_samples) % self._capacity

        for offset, count in self._segments(read, frames):
            source = block[offset:offset + count, :channels]
            tap_start = (read + offset) % self._capacity
            tap = self._buffer[tap_start:tap_start + count, :channels]

            # Write input plus feedback of the tap, then blend the tap into the output.
            write_start = (self._write + offset) % self._capacity
            self._buffer[write_start:write_start + count, :channels] = (
                source + tap * self.feedback
            )
            source += tap * self.mix

        self._write = (self._write + frames) % self._capacity
        return block

    def _segments(self, read: int, frames: int):
        """Split the block so neither the read nor write span crosses the wrap point."""
        offset = 0
        while offset < frames:
            remaining = frames - offset
            to_write_wrap = self._capacity - ((self._write + offset) % self._capacity)
            to_read_wrap = self._capacity - ((read + offset) % self._capacity)
            count = min(remaining, to_write_wrap, to_read_wrap)
            yield offset, count
            offset += count

    def reset(self):
        self._buffer[:] = 0.0
        self._write = 0


class PluginChainUnavailable(RuntimeError):
    """`pedalboard` is not usable here, so plugin hosting is not available."""


# Cache for PluginChain.is_available(): None means "not yet checked". The answer is a
# property of the installed wheel and the host CPU, so it cannot change during a run, and
# the check spawns a subprocess — worth paying for once, not on every call.
_pedalboard_available: bool | None = None


class PluginChain:
    """
    A chain of VST3/AU plugins, hosted through `pedalboard`.

    This is the feature the project was originally for: rather than paying for a separate
    router to get a guitar into Guitar Rig, load Guitar Rig here and monitor through it on
    the same low-latency path.

    Plugin processing is C++ inside pedalboard, so the GIL is released for the duration —
    the Python overhead is one call per block, not per sample.

    Latency: many plugins report an internal latency, which has to be added to the round
    trip we report or the number we show the user is a lie. `reported_latency_samples`
    exposes it.
    """

    def __init__(self, samplerate: int, blocksize: int):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self._plugins: list[Any] = []
        self._names: list[str] = []
        self._bypassed: list[bool] = []

    @staticmethod
    def is_available() -> bool:
        """
        Whether `pedalboard` can actually be imported here.

        Checked in a throwaway subprocess rather than by importing it directly in this
        process. A native extension can fail worse than raising ImportError: this
        project's own CI hit exactly that when pedalboard's Linux wheel produced
        `Illegal instruction` (SIGILL) purely from being imported, on a runner CPU
        missing some instruction the compiled code used unconditionally. SIGILL is a
        fatal signal, not a Python exception — no `try/except` in this process can
        survive it, so the only way to ask "can this be imported" without risking the
        process asking the question is to ask a disposable one instead.

        The result is cached: it cannot legitimately change mid-run (same wheel, same
        CPU), and re-probing would mean a subprocess launch on every call.
        """
        global _pedalboard_available

        if _pedalboard_available is not None:
            return _pedalboard_available

        try:
            result = subprocess.run(
                [sys.executable, '-c', 'import pedalboard'],
                capture_output=True, timeout=20,
            )
            available = result.returncode == 0
            if not available:
                stderr = result.stderr.decode('utf-8', errors='replace').strip()
                logger.warning(
                    "pedalboard could not be imported (exit code "
                    f"{result.returncode}); VST3/AU plugin hosting is disabled. "
                    f"{stderr[-300:]}"
                )
        except (subprocess.TimeoutExpired, OSError) as e:
            available = False
            logger.warning(f"Could not probe pedalboard availability: {e}")

        _pedalboard_available = available
        return available

    @staticmethod
    def scan_default_paths() -> list[str]:
        """
        Where VST3 plugins live by convention on each platform.

        Returns paths that exist; the caller enumerates them. We do not attempt to load
        anything here, because loading an unknown plugin can be slow and can crash.
        """
        import platform
        from pathlib import Path

        system = platform.system()
        if system == "Windows":
            candidates = [
                Path("C:/Program Files/Common Files/VST3"),
                Path("C:/Program Files/VSTPlugins"),
                Path("C:/Program Files/Steinberg/VSTPlugins"),
            ]
        elif system == "Darwin":
            candidates = [
                Path("/Library/Audio/Plug-Ins/VST3"),
                Path("/Library/Audio/Plug-Ins/Components"),
                Path.home() / "Library/Audio/Plug-Ins/VST3",
            ]
        else:
            candidates = [
                Path("/usr/lib/vst3"),
                Path("/usr/local/lib/vst3"),
                Path.home() / ".vst3",
            ]

        return [str(path) for path in candidates if path.is_dir()]

    @staticmethod
    def discover(paths: Sequence[str] | None = None) -> list[str]:
        """Plugin files found in the given paths, or the platform defaults."""
        from pathlib import Path

        search = list(paths) if paths else PluginChain.scan_default_paths()
        found: list[str] = []

        for root in search:
            base = Path(root)
            if not base.is_dir():
                continue
            for pattern in ("*.vst3", "*.component", "*.dll"):
                found.extend(str(p) for p in base.glob(pattern))

        return sorted(set(found))

    def load(self, path: str, name: str | None = None) -> int:
        """
        Load a plugin and append it to the chain.

        Raises rather than returning a sentinel: a plugin that failed to load is not
        something to carry on quietly with, because the user's chain would silently differ
        from what they configured.

        Checks `is_available()` first rather than importing directly. That check has
        already proven pedalboard imports cleanly in this exact environment (see its
        docstring for why that matters); importing it here for real, now that we know it
        is safe, is what actually lets us load the plugin.
        """
        if not PluginChain.is_available():
            raise PluginChainUnavailable(
                "pedalboard is not usable in this environment; plugin hosting unavailable"
            )

        import pedalboard

        plugin = pedalboard.load_plugin(path)

        self._plugins.append(plugin)
        self._names.append(name or getattr(plugin, 'name', None) or path)
        self._bypassed.append(False)

        logger.info(f"Loaded plugin: {self._names[-1]}")
        return len(self._plugins) - 1

    def add_builtin(self, plugin: Any, name: str) -> int:
        """Append a pedalboard built-in (Reverb, Chorus, and so on)."""
        self._plugins.append(plugin)
        self._names.append(name)
        self._bypassed.append(False)
        return len(self._plugins) - 1

    def remove(self, index: int):
        if 0 <= index < len(self._plugins):
            del self._plugins[index]
            del self._names[index]
            del self._bypassed[index]

    def set_bypassed(self, index: int, bypassed: bool):
        if 0 <= index < len(self._bypassed):
            self._bypassed[index] = bypassed

    def clear(self):
        self._plugins.clear()
        self._names.clear()
        self._bypassed.clear()

    @property
    def names(self) -> list[str]:
        return list(self._names)

    @property
    def is_empty(self) -> bool:
        return not self._plugins or all(self._bypassed)

    @property
    def reported_latency_samples(self) -> int:
        """
        Total latency the plugins declare.

        Must be added to the measured round trip. A look-ahead limiter can add 5 ms on its
        own, and omitting it would make the latency figure we show wrong.
        """
        total = 0
        for plugin, bypassed in zip(self._plugins, self._bypassed, strict=True):
            if bypassed:
                continue
            total += int(getattr(plugin, 'latency_samples', 0) or 0)
        return total

    def process(self, block: np.ndarray, frames: int) -> np.ndarray:
        """
        Run the chain.

        pedalboard wants (channels, frames), the opposite of our layout, so the block is
        transposed in and out. The transpose is a view, not a copy, and the plugin work
        itself happens in C++ with the GIL released.
        """
        if self.is_empty:
            return block

        audio = np.ascontiguousarray(block[:frames].T)

        for plugin, bypassed in zip(self._plugins, self._bypassed, strict=True):
            if bypassed:
                continue
            try:
                audio = plugin.process(audio, self.samplerate, reset=False)
            except Exception as e:
                # A misbehaving plugin must not take the audio thread down with it.
                logger.error(f"Plugin error, bypassing for this block: {e}")
                return block

        processed = np.ascontiguousarray(audio.T)
        copy_frames = min(frames, processed.shape[0])
        copy_channels = min(block.shape[1], processed.shape[1])
        block[:copy_frames, :copy_channels] = processed[:copy_frames, :copy_channels]

        return block

    def describe(self) -> list[dict[str, Any]]:
        return [
            {'index': i, 'name': name, 'bypassed': bypassed}
            for i, (name, bypassed) in enumerate(zip(self._names, self._bypassed, strict=True))
        ]


class InsertChain:
    """
    Everything that can be inserted on one channel, in a fixed order.

    Order matches a console's signal flow: high-pass first so rumble never reaches the
    compressor's detector, then EQ, then dynamics, then plugins, then delay. A compressor
    reacting to sub-sonic rumble is a classic reason a channel "breathes" for no visible
    reason.

    Each stage is skipped entirely when it would do nothing, so an untouched channel costs
    a handful of boolean checks per block.
    """

    def __init__(self, samplerate: int, blocksize: int, channels: int = 2):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels

        self.highpass: Biquad | None = None
        self.eq: ParametricEQ | None = None
        self.compressor: Compressor | None = None
        self.plugins: PluginChain | None = None
        self.delay: Delay | None = None

        self.enabled = True

    def enable_highpass(self, frequency: float = 80.0, q: float = 0.707):
        self.highpass = Biquad(self.channels)
        self.highpass.set_highpass(self.samplerate, frequency, q)

    def enable_eq(self, bands: Sequence[EQBand] | None = None) -> ParametricEQ:
        self.eq = ParametricEQ(self.samplerate, self.channels, bands)
        return self.eq

    def enable_compressor(self, **kwargs) -> Compressor:
        self.compressor = Compressor(self.samplerate, self.blocksize, **kwargs)
        return self.compressor

    def enable_plugins(self) -> PluginChain:
        self.plugins = PluginChain(self.samplerate, self.blocksize)
        return self.plugins

    def enable_delay(self, **kwargs) -> Delay:
        self.delay = Delay(self.samplerate, self.channels, **kwargs)
        return self.delay

    @property
    def is_transparent(self) -> bool:
        if not self.enabled:
            return True
        if self.highpass is not None:
            return False
        if self.eq is not None and not self.eq.is_flat:
            return False
        if self.compressor is not None:
            return False
        if self.plugins is not None and not self.plugins.is_empty:
            return False
        if self.delay is not None and self.delay.mix > 0:
            return False
        return True

    @property
    def latency_samples(self) -> int:
        return self.plugins.reported_latency_samples if self.plugins else 0

    def process(self, block: np.ndarray, frames: int) -> np.ndarray:
        if not self.enabled:
            return block

        if self.highpass is not None:
            self.highpass.process(block, frames)
        if self.eq is not None:
            self.eq.process(block, frames)
        if self.compressor is not None:
            self.compressor.process(block, frames)
        if self.plugins is not None:
            self.plugins.process(block, frames)
        if self.delay is not None:
            self.delay.process(block, frames)

        return block

    def reset(self):
        for stage in (self.highpass, self.eq, self.compressor, self.delay):
            if stage is not None and hasattr(stage, 'reset'):
                stage.reset()
