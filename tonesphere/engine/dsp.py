"""
Per-channel processing that runs inside the audio callback.

Everything here is written to be called at block rate on the audio thread, so it obeys the
same rules as the mixer: preallocated buffers, no allocation, no branching on Python
objects more than necessary, and correctness at block boundaries.

That last point is what most of this file is about. A block-based processor that treats
each block independently produces a discontinuity every 2.7 ms, and a discontinuity is a
click. Anything with state — a gain ramp, a pan position, a limiter's envelope — has to
carry that state across blocks, which is why these are objects rather than functions.
"""

import math
from typing import Optional, Tuple

import numpy as np

# Pan law. -3 dB at centre keeps total power constant as a source is panned, which is what
# you want for a mono source placed in a stereo field. A linear law would make the centre
# sound quieter than the edges; a -6 dB law keeps amplitude rather than power constant and
# suits already-correlated stereo material.
PAN_LAW_MINUS_3DB = 'minus3'
PAN_LAW_MINUS_6DB = 'minus6'
PAN_LAW_LINEAR = 'linear'


def pan_gains(pan: float, law: str = PAN_LAW_MINUS_3DB) -> Tuple[float, float]:
    """
    Left and right gain for a pan position in [-1, 1].

    At centre with the -3 dB law both channels get 1/sqrt(2), so the summed power matches
    a hard-panned source. This is the difference between a pan control that sounds even
    across its travel and one that dips in the middle.
    """
    pan = min(max(pan, -1.0), 1.0)

    if law == PAN_LAW_LINEAR:
        # Gains sum to unity across the sweep, which keeps amplitude rather than power
        # constant. Quieter at centre than the -3 dB law, and rarely what you want.
        return ((1.0 - pan) * 0.5, (1.0 + pan) * 0.5)

    # Map -1..1 onto a quarter turn, so the two gains trace sine/cosine.
    angle = (pan + 1.0) * (math.pi / 4.0)
    left = math.cos(angle)
    right = math.sin(angle)

    if law == PAN_LAW_MINUS_6DB:
        # Normalise so centre reaches unity on each side rather than 0.707.
        left *= math.sqrt(2.0)
        right *= math.sqrt(2.0)

    return (left, right)


def fill_ramp(out: np.ndarray, start: float, end: float, unit: np.ndarray):
    """
    Write a linear ramp from `start` to `end` into `out`, allocating nothing.

    `np.linspace` has no `out=` parameter, so it would allocate on every block — and an
    allocation in an audio callback risks a garbage collection, which is a dropout. Scaling
    a precomputed 0..1 ramp in place gives the same result for free.
    """
    frames = out.shape[0]
    out[:] = unit[:frames]
    out *= (end - start)
    out += start


class SmoothedGain:
    """
    A gain that moves to its target over time instead of jumping.

    A fader move applied instantly puts a step in the waveform, and a step is a broadband
    click. Ramping across a block hides it completely.
    """

    __slots__ = ('_current', '_target', '_ramp', '_unit', '_blocksize')

    def __init__(self, initial: float = 1.0, blocksize: int = 256):
        self._current = float(initial)
        self._target = float(initial)
        self._blocksize = blocksize
        self._ramp = np.zeros((blocksize, 1), dtype=np.float32)
        self._unit = np.linspace(0.0, 1.0, blocksize, dtype=np.float32).reshape(-1, 1)

    @property
    def current(self) -> float:
        return self._current

    @property
    def target(self) -> float:
        return self._target

    def set(self, target: float):
        """Request a new gain. Takes effect over the next block."""
        self._target = float(target)

    def jump(self, value: float):
        """Set with no ramp. Only safe when the signal is already silent."""
        self._current = self._target = float(value)

    def is_static(self) -> bool:
        return self._current == self._target

    def apply(self, block: np.ndarray, frames: int) -> np.ndarray:
        """
        Scale `block` in place, ramping if the gain is moving.

        Returns the block for chaining. When static and at unity this does nothing at all,
        which is the common case and worth the check.
        """
        if self._current == self._target:
            if self._current != 1.0:
                block *= self._current
            return block

        ramp = self._ramp[:frames]
        # Linear interpolation across the block. Not perceptually ideal for very large
        # jumps, but at 2.7-5 ms per block the difference is inaudible and it is far
        # cheaper than an exponential curve.
        fill_ramp(ramp, self._current, self._target, self._unit)
        block *= ramp

        self._current = self._target
        return block


class ChannelStrip:
    """
    Per-channel gain, mute, pan and phase for one node.

    This exists because `core.channel_control` has always held exactly this state and has
    never been connected to any audio. Its settings were adjustable in the UI and had no
    effect on anything. This is the part that makes them real.
    """

    __slots__ = (
        'channels', 'blocksize', 'pan_law',
        '_gains', '_pans', '_inverted', '_muted', '_master',
        '_swap', '_scratch',
    )

    def __init__(self, channels: int, blocksize: int, pan_law: str = PAN_LAW_MINUS_3DB):
        self.channels = channels
        self.blocksize = blocksize
        self.pan_law = pan_law

        self._gains = [SmoothedGain(1.0, blocksize) for _ in range(channels)]
        self._pans = [0.0] * channels
        self._inverted = [False] * channels
        self._muted = [False] * channels
        self._master = SmoothedGain(1.0, blocksize)
        self._swap = False

        self._scratch = np.zeros((blocksize, max(channels, 2)), dtype=np.float32)

    # --- Control thread ---

    def set_channel_gain(self, channel: int, gain: float):
        if 0 <= channel < self.channels:
            self._gains[channel].set(0.0 if self._muted[channel] else gain)

    def set_channel_mute(self, channel: int, muted: bool):
        """Mute ramps like any other gain change, so it does not click."""
        if 0 <= channel < self.channels:
            self._muted[channel] = muted
            target = 0.0 if muted else self._gains[channel].target
            self._gains[channel].set(0.0 if muted else (target or 1.0))

    def set_channel_pan(self, channel: int, pan: float):
        if 0 <= channel < self.channels:
            self._pans[channel] = min(max(pan, -1.0), 1.0)

    def set_channel_inverted(self, channel: int, inverted: bool):
        """
        Flip polarity.

        Useful for a mic pair where one is behind the source, and for finding which of two
        signals is fighting the other — invert one and the common part cancels.
        """
        if 0 <= channel < self.channels:
            self._inverted[channel] = inverted

    def set_master_gain(self, gain: float):
        self._master.set(gain)

    def set_swapped(self, swapped: bool):
        self._swap = swapped

    def is_transparent(self) -> bool:
        """
        Whether this strip would leave the audio untouched, so the caller can skip it.

        Worth checking: the default state is transparent, and skipping avoids a copy per
        block per node.
        """
        if self._swap:
            return False
        if not self._master.is_static() or self._master.current != 1.0:
            return False
        for i in range(self.channels):
            if self._inverted[i] or self._pans[i] != 0.0:
                return False
            gain = self._gains[i]
            if not gain.is_static() or gain.current != 1.0:
                return False
        return True

    # --- Audio thread ---

    def process(self, block: np.ndarray, frames: int) -> np.ndarray:
        """
        Apply the strip in place.

        Order is deliberate: polarity, then per-channel gain, then swap, then master. Pan
        is not applied here because it changes channel count semantics — it belongs to the
        route, where the destination width is known.
        """
        channels = min(block.shape[1], self.channels)

        for channel in range(channels):
            if self._inverted[channel]:
                block[:frames, channel] *= -1.0

            column = block[:frames, channel:channel + 1]
            self._gains[channel].apply(column, frames)

        if self._swap and block.shape[1] >= 2:
            # Copy through scratch: an in-place swap on NumPy views would alias.
            scratch = self._scratch[:frames, :2]
            scratch[:] = block[:frames, :2]
            block[:frames, 0] = scratch[:, 1]
            block[:frames, 1] = scratch[:, 0]

        self._master.apply(block[:frames], frames)
        return block


class Panner:
    """
    Places a source into a wider destination, with smoothed gains.

    Separate from ChannelStrip because panning is a property of a route: the same source
    can sit centre in one destination and hard left in another.
    """

    __slots__ = ('_left', '_right', '_pan', 'law', '_scratch', 'blocksize')

    def __init__(self, blocksize: int, law: str = PAN_LAW_MINUS_3DB):
        self.blocksize = blocksize
        self.law = law
        self._pan = 0.0
        left, right = pan_gains(0.0, law)
        self._left = SmoothedGain(left, blocksize)
        self._right = SmoothedGain(right, blocksize)
        self._scratch = np.zeros((blocksize, 2), dtype=np.float32)

    @property
    def pan(self) -> float:
        return self._pan

    def set_pan(self, pan: float):
        self._pan = min(max(pan, -1.0), 1.0)
        left, right = pan_gains(self._pan, self.law)
        self._left.set(left)
        self._right.set(right)

    def is_centred(self) -> bool:
        return self._pan == 0.0 and self._left.is_static() and self._right.is_static()

    def process_mono_to_stereo(self, mono: np.ndarray, frames: int) -> np.ndarray:
        """Spread a mono source across a stereo pair. Returns an internal buffer."""
        out = self._scratch[:frames]
        source = mono[:frames, 0:1] if mono.ndim > 1 else mono[:frames].reshape(-1, 1)

        out[:, 0:1] = source
        out[:, 1:2] = source

        self._left.apply(out[:, 0:1], frames)
        self._right.apply(out[:, 1:2], frames)
        return out

    def process_stereo(self, stereo: np.ndarray, frames: int) -> np.ndarray:
        """Balance an existing stereo pair. Operates in place."""
        self._left.apply(stereo[:frames, 0:1], frames)
        self._right.apply(stereo[:frames, 1:2], frames)
        return stereo


class Limiter:
    """
    Catches overs on the way out of the mixer.

    Summing several sources easily exceeds 1.0. In float that is harmless internally, but
    the moment it reaches the device's integer converter it wraps or clips hard, which is
    the worst-sounding kind of distortion. This applies gain reduction with a fast attack
    and a slow release so the mix ducks briefly rather than tearing.

    Not a mastering limiter. It exists so a routing mistake sounds like a compressed mix
    instead of a burst of digital noise.
    """

    __slots__ = ('threshold', '_envelope', '_release_coeff', '_attack_coeff',
                 'blocksize', 'samplerate', 'reduction_db', '_scratch', '_unit')

    def __init__(self, samplerate: int, blocksize: int,
                 threshold: float = 0.99,
                 attack_ms: float = 1.0,
                 release_ms: float = 80.0):
        self.threshold = threshold
        self.samplerate = samplerate
        self.blocksize = blocksize
        self._envelope = 1.0
        self.reduction_db = 0.0

        # One-pole coefficients, per BLOCK.
        #
        # The envelope is updated once per block, so the time constant must be expressed
        # in blocks. Computing it per sample and applying it per block, which is the
        # obvious mistake, makes every time constant wrong by a factor of the block size —
        # a 1 ms attack behaves like a 256 ms one and the limiter fails to catch anything.
        block_period = blocksize / samplerate

        def coeff(time_ms: float) -> float:
            seconds = max(1e-6, time_ms / 1000.0)
            return math.exp(-block_period / seconds)

        self._attack_coeff = coeff(attack_ms)
        self._release_coeff = coeff(release_ms)
        self._scratch = np.zeros((blocksize, 1), dtype=np.float32)
        self._unit = np.linspace(0.0, 1.0, blocksize, dtype=np.float32).reshape(-1, 1)

    def process(self, block: np.ndarray, frames: int) -> np.ndarray:
        """
        Reduce gain where the block exceeds the threshold. In place.

        Gain is computed per block rather than per sample: at 2.7-5 ms per block that is
        fast enough to catch anything the converter would have clipped, and it keeps the
        cost to a couple of NumPy reductions.
        """
        peak = float(np.max(np.abs(block[:frames])))

        if peak <= self.threshold and self._envelope >= 0.999:
            self.reduction_db = 0.0
            return block

        target = self.threshold / peak if peak > self.threshold else 1.0

        # Attack when clamping down, release when letting go.
        coeff = self._attack_coeff if target < self._envelope else self._release_coeff
        start = self._envelope
        self._envelope = target + (self._envelope - target) * coeff

        ramp = self._scratch[:frames]
        fill_ramp(ramp, start, self._envelope, self._unit)
        block[:frames] *= ramp

        self.reduction_db = 20.0 * math.log10(max(self._envelope, 1e-6))
        return block

    def reset(self):
        self._envelope = 1.0
        self.reduction_db = 0.0


class DriftResampler:
    """
    Absorbs clock drift between two devices by resampling slightly.

    Dropping or repeating a block, which is what the ring buffer does on its own, costs an
    audible glitch each time it happens. Two devices at a nominal 48 kHz can differ by
    enough to trigger that every few seconds. Nudging the playback rate by a fraction of a
    percent instead is inaudible.

    Linear interpolation is used deliberately. The correction ratio stays within about
    0.1%, so the interpolation error sits far below the noise floor, and unlike a windowed
    sinc it costs almost nothing per block. It keeps its fractional read position across
    blocks — the thing the resampler this replaces got wrong, producing a discontinuity at
    every boundary.
    """

    __slots__ = ('ratio', '_position', 'channels', 'blocksize', '_scratch',
                 '_indices', '_counter')

    # Never correct faster than this; a larger step would be audible as pitch movement.
    MAX_RATIO_DEVIATION = 0.002

    def __init__(self, channels: int, blocksize: int):
        self.channels = channels
        self.blocksize = blocksize
        self.ratio = 1.0
        self._position = 0.0
        # Generous headroom so a ratio below 1.0 asking for more input still fits.
        self._scratch = np.zeros((blocksize * 2, channels), dtype=np.float32)
        self._indices = np.zeros(blocksize, dtype=np.float64)
        # Precomputed 0,1,2,... — `np.arange` has no `out=` parameter, so recreating it
        # each block would allocate in the audio callback.
        self._counter = np.arange(blocksize, dtype=np.float64)

    def set_ratio(self, ratio: float):
        """Clamp the correction so it can never become audible pitch drift."""
        self.ratio = min(max(ratio, 1.0 - self.MAX_RATIO_DEVIATION),
                         1.0 + self.MAX_RATIO_DEVIATION)

    def input_frames_needed(self, output_frames: int) -> int:
        """How much input to produce `output_frames`, given the current position."""
        return int(math.ceil(self._position + output_frames * self.ratio)) + 1

    def process(self, source: np.ndarray, out: np.ndarray, frames: int) -> int:
        """
        Resample `source` into the first `frames` rows of `out`.

        Returns how many input frames were consumed, so the caller can advance its
        reader. The fractional remainder is kept for the next call, which is what makes
        this continuous across blocks.
        """
        available = source.shape[0]
        if available < 2:
            out[:frames] = 0.0
            return 0

        indices = self._indices[:frames]
        indices[:] = self._counter[:frames]
        indices *= self.ratio
        indices += self._position

        # Clamp so interpolation never reads past the end of what we were given.
        np.clip(indices, 0.0, available - 1.0001, out=indices)

        base = indices.astype(np.int64)
        frac = (indices - base).astype(np.float32).reshape(-1, 1)

        lower = source[base]
        upper = source[base + 1]
        out[:frames] = lower + (upper - lower) * frac

        consumed = int(base[-1]) + 1
        self._position = indices[-1] + self.ratio - consumed
        if self._position < 0.0:
            self._position = 0.0

        return consumed

    def reset(self):
        self._position = 0.0
        self.ratio = 1.0
