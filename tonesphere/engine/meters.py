"""
Level metering.

Two requirements shape this file. First, a meter has to be *read* by the UI at 30 Hz and
*written* by the audio callback at ~375 Hz, without the writer ever blocking: the
callback stores plain floats into preallocated slots and the UI reads whatever is
current. A queue would grow without bound whenever the UI stalled, and eventually the
callback would block on a full queue — a dropout caused by drawing a meter.

Second, a meter has to behave like a meter. Raw per-block peak flickers uselessly. Real
meters attack instantly and release slowly, and hold the peak long enough for a human to
read it. Those constants are here rather than in the UI, because they are audio
behaviour, not decoration.
"""

import math
import time
from dataclasses import dataclass

import numpy as np

# Below this we report -inf rather than a large negative number.
SILENCE_FLOOR_DB = -100.0

# Broadcast-style ballistics. Attack is instant so a transient is never missed;
# release is slow enough to read. Peak hold is roughly the eye's dwell time.
DEFAULT_RELEASE_DB_PER_SEC = 20.0 * 11.8   # ~11.8 dB per 0.05 s, i.e. IEC-ish decay
DEFAULT_PEAK_HOLD_SECONDS = 1.5

# Anything at or above this counts as clipping; float audio can exceed 1.0 without
# wrapping, but it will clip the moment it hits an integer converter downstream.
CLIP_THRESHOLD = 0.999


def amplitude_to_db(amplitude: float) -> float:
    """dBFS for a linear amplitude. Guards log(0) instead of raising or returning -inf."""
    if amplitude <= 0.0:
        return SILENCE_FLOOR_DB
    db = 20.0 * math.log10(amplitude)
    return max(db, SILENCE_FLOOR_DB)


@dataclass
class MeterReading:
    """An immutable snapshot handed to the UI."""
    peak: float = 0.0
    rms: float = 0.0
    peak_hold: float = 0.0
    clipped: bool = False

    @property
    def peak_db(self) -> float:
        return amplitude_to_db(self.peak)

    @property
    def rms_db(self) -> float:
        return amplitude_to_db(self.rms)

    @property
    def peak_hold_db(self) -> float:
        return amplitude_to_db(self.peak_hold)


class ChannelMeter:
    """
    One channel's level state.

    Written only by the audio callback, read only by the UI. Each field is a float
    attribute, so a torn read is impossible under the GIL — the UI may see a value one
    block old, which at 2.7 ms per block is imperceptible.
    """

    __slots__ = (
        'peak', 'rms', 'peak_hold', 'clipped',
        '_release_per_sec', '_hold_seconds', '_hold_until', '_last_update',
    )

    def __init__(
        self,
        release_db_per_sec: float = DEFAULT_RELEASE_DB_PER_SEC,
        peak_hold_seconds: float = DEFAULT_PEAK_HOLD_SECONDS,
    ):
        self.peak = 0.0
        self.rms = 0.0
        self.peak_hold = 0.0
        self.clipped = False

        self._release_per_sec = release_db_per_sec
        self._hold_seconds = peak_hold_seconds
        self._hold_until = 0.0
        self._last_update = 0.0

    def update(self, block_peak: float, block_rms: float, now: float):
        """
        Fold one block into the meter. Called from the audio callback.

        Deliberately free of allocation, logging and exceptions — the caller has already
        reduced the block to two floats.
        """
        # Instant attack, timed release. Without the release the meter would snap to
        # zero between notes and read as a strobe.
        if block_peak >= self.peak:
            self.peak = block_peak
        else:
            elapsed = now - self._last_update if self._last_update else 0.0
            if elapsed > 0.0:
                decay_db = self._release_per_sec * elapsed
                self.peak = max(block_peak, self.peak * (10.0 ** (-decay_db / 20.0)))
            else:
                self.peak = max(block_peak, self.peak)

        self.rms = block_rms
        self._last_update = now

        if block_peak >= self.peak_hold or now >= self._hold_until:
            self.peak_hold = max(block_peak, self.peak)
            self._hold_until = now + self._hold_seconds

        if block_peak >= CLIP_THRESHOLD:
            self.clipped = True

    def read(self) -> MeterReading:
        """Snapshot for the UI."""
        return MeterReading(
            peak=self.peak,
            rms=self.rms,
            peak_hold=self.peak_hold,
            clipped=self.clipped,
        )

    def clear_clip(self):
        """Reset the clip latch — the user clicked the indicator."""
        self.clipped = False

    def reset(self):
        self.peak = 0.0
        self.rms = 0.0
        self.peak_hold = 0.0
        self.clipped = False
        self._hold_until = 0.0
        self._last_update = 0.0


class MeterBank:
    """
    Meters for one node's channels, plus the reduction from block to floats.

    Channel meters are allocated when the bank is created, never during a callback.
    """

    __slots__ = ('name', 'channels', '_meters', '_scratch')

    def __init__(self, name: str, channels: int):
        self.name = name
        self.channels = channels
        self._meters: list[ChannelMeter] = [ChannelMeter() for _ in range(channels)]
        # Preallocated so measure() allocates nothing per block.
        self._scratch = np.zeros(channels, dtype=np.float64)

    def measure(self, block: np.ndarray, now: float | None = None):
        """
        Measure a (frames, channels) block.

        Allocation-free on purpose. The obvious spelling —
        `np.sqrt(np.mean(column.astype(np.float64) ** 2))` — allocates two temporary
        arrays per channel per block, which at 375 blocks a second is a steady stream of
        garbage created on the audio thread. A collection triggered there is a dropout.

        `np.dot(column, column)` gives the sum of squares as a scalar with no temporary,
        and `np.abs(...).max()` is fused by NumPy into a single pass.
        """
        if block.size == 0:
            return

        if now is None:
            now = time.monotonic()

        frames = block.shape[0]
        channels = min(block.shape[1], self.channels)

        for channel in range(channels):
            column = block[:, channel]

            peak = float(np.abs(column).max())
            # Sum of squares without materialising the squared array.
            rms = math.sqrt(float(np.dot(column, column)) / frames) if frames else 0.0

            self._meters[channel].update(peak, rms, now)

    def read(self) -> list[MeterReading]:
        return [meter.read() for meter in self._meters]

    def read_summary(self) -> MeterReading:
        """Loudest channel, for a single-meter display."""
        readings = self.read()
        if not readings:
            return MeterReading()

        return MeterReading(
            peak=max(r.peak for r in readings),
            rms=max(r.rms for r in readings),
            peak_hold=max(r.peak_hold for r in readings),
            clipped=any(r.clipped for r in readings),
        )

    def clear_clip(self):
        for meter in self._meters:
            meter.clear_clip()

    def reset(self):
        for meter in self._meters:
            meter.reset()

    def resize(self, channels: int):
        """Change channel count. Control thread only — this allocates."""
        if channels == self.channels:
            return
        self._meters = [ChannelMeter() for _ in range(channels)]
        self._scratch = np.zeros(channels, dtype=np.float64)
        self.channels = channels


class MeterRegistry:
    """
    Every meter bank in the engine, keyed by node.

    Banks are created up front by the host when it builds its streams, so the callback
    only ever looks one up. `get` returning None is correct and expected during
    reconfiguration; the callback skips metering rather than allocating a bank.
    """

    def __init__(self):
        self._banks: dict[str, MeterBank] = {}

    def ensure(self, key: str, channels: int) -> MeterBank:
        """Create or resize a bank. Control thread only."""
        bank = self._banks.get(key)
        if bank is None:
            bank = MeterBank(key, channels)
            self._banks[key] = bank
        elif bank.channels != channels:
            bank.resize(channels)
        return bank

    def get(self, key: str) -> MeterBank | None:
        """Look up a bank. Safe from the callback; never allocates."""
        return self._banks.get(key)

    def remove(self, key: str):
        self._banks.pop(key, None)

    def keys(self) -> list[str]:
        return list(self._banks.keys())

    def read_all(self) -> dict[str, list[MeterReading]]:
        return {key: bank.read() for key, bank in self._banks.items()}

    def read_summaries(self) -> dict[str, MeterReading]:
        return {key: bank.read_summary() for key, bank in self._banks.items()}

    def clear_clips(self):
        for bank in self._banks.values():
            bank.clear_clip()

    def reset(self):
        for bank in self._banks.values():
            bank.reset()

    def clear(self):
        self._banks.clear()
