"""
Single-producer / single-consumer ring buffer for audio frames.

Independent audio devices run on independent clocks. Two devices nominally at 48 kHz
drift apart by tens of samples per minute, so a router cannot hand a block straight
from one device's callback to another's — it needs elastic storage between them. This
is that storage.

Thread safety rests on the SPSC discipline, not on locks: exactly one thread advances
`_write_index` and exactly one advances `_read_index`. Each index is a plain int
attribute, so loads and stores of it are atomic under the GIL, and neither thread ever
writes the other's index. A lock here would be worse than useless — blocking inside an
audio callback is what causes dropouts in the first place.
"""

from typing import Optional

import numpy as np


class RingBufferOverflow(Exception):
    """Raised only by the non-realtime helpers; callbacks count instead of raising."""


class AudioRingBuffer:
    """
    Fixed-capacity FIFO of interleaved audio frames.

    One slot is always left empty so a full buffer is distinguishable from an empty one
    without a separate count that both threads would have to write.
    """

    __slots__ = (
        '_buffer', '_capacity', '_channels', '_read_index', '_write_index',
        'frames_written', 'frames_read', 'overflow_count', 'underflow_count',
    )

    def __init__(self, capacity_frames: int, channels: int):
        if capacity_frames < 2:
            raise ValueError("capacity_frames must be at least 2")
        if channels < 1:
            raise ValueError("channels must be at least 1")

        # One extra frame is the full/empty sentinel.
        self._capacity = capacity_frames + 1
        self._channels = channels
        self._buffer = np.zeros((self._capacity, channels), dtype=np.float32)

        self._read_index = 0
        self._write_index = 0

        # Diagnostics. Each counter is only ever incremented by one side.
        self.frames_written = 0
        self.frames_read = 0
        self.overflow_count = 0
        self.underflow_count = 0

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def capacity(self) -> int:
        """Usable capacity in frames."""
        return self._capacity - 1

    @property
    def available(self) -> int:
        """Frames ready to read. Safe to call from either thread."""
        return (self._write_index - self._read_index) % self._capacity

    @property
    def space(self) -> int:
        """Frames that can be written without overwriting unread data."""
        return self.capacity - self.available

    def write(self, frames: np.ndarray) -> int:
        """
        Append frames. Producer side only.

        Returns the number of frames written, which is less than requested when the
        consumer has fallen behind. Callers in a callback must not raise on a short
        write — they count it and carry on, because the show has to go on.
        """
        count = frames.shape[0]
        if count == 0:
            return 0

        space = self.space
        if count > space:
            self.overflow_count += 1
            count = space
            if count == 0:
                return 0
            frames = frames[:count]

        write_index = self._write_index
        first = min(count, self._capacity - write_index)

        self._buffer[write_index:write_index + first] = frames[:first]
        if first < count:
            self._buffer[:count - first] = frames[first:]

        # Publish last: the consumer must never see an index pointing at a frame we
        # have not finished writing.
        self._write_index = (write_index + count) % self._capacity
        self.frames_written += count
        return count

    def read_into(self, out: np.ndarray) -> int:
        """
        Fill `out` with the oldest frames. Consumer side only.

        Any part of `out` we cannot fill is zeroed — silence is the correct sound for
        "no data", and leaving the caller's stale buffer contents would repeat the last
        block as a click or buzz.
        """
        count = out.shape[0]
        if count == 0:
            return 0

        available = self.available
        got = min(count, available)

        if got < count:
            self.underflow_count += 1
            out[got:] = 0.0

        if got == 0:
            return 0

        read_index = self._read_index
        first = min(got, self._capacity - read_index)

        out[:first] = self._buffer[read_index:read_index + first]
        if first < got:
            out[first:got] = self._buffer[:got - first]

        self._read_index = (read_index + got) % self._capacity
        self.frames_read += got
        return got

    def peek_latest(self, out: np.ndarray) -> int:
        """
        Copy the most recent frames without consuming them.

        Used by metering, which must not steal audio from the mixer.
        """
        count = out.shape[0]
        available = self.available
        got = min(count, available)

        if got == 0:
            out[:] = 0.0
            return 0

        start = (self._write_index - got) % self._capacity
        first = min(got, self._capacity - start)

        out[:first] = self._buffer[start:start + first]
        if first < got:
            out[first:got] = self._buffer[:got - first]
        if got < count:
            out[got:] = 0.0

        return got

    def unread(self, frames: int) -> int:
        """
        Rewind the read cursor, giving frames back. Consumer side only.

        The drift resampler must read a whole number of frames but may consume a
        fractional amount of them, so it hands the remainder back rather than dropping it.
        Without this, one sample would be lost at every block boundary — a periodic click
        at exactly the block rate, which is the most audible artefact there is.

        Safe because the data is still in the buffer; the producer cannot have overwritten
        it while it counted as read, since it only ever writes into free space.
        """
        rewind = min(frames, self._capacity - 1 - self.available)
        if rewind <= 0:
            return 0

        self._read_index = (self._read_index - rewind) % self._capacity
        self.frames_read -= rewind
        return rewind

    def discard(self, frames: int) -> int:
        """
        Drop the oldest frames. Consumer side only.

        This is the crude half of drift correction: when a source's clock runs faster
        than the sink's, its buffer fills without bound and latency grows forever. We
        drop to bring it back, which is audible as a tiny glitch but bounded. Phase 2
        replaces this with adaptive resampling, which is inaudible.
        """
        dropped = min(frames, self.available)
        self._read_index = (self._read_index + dropped) % self._capacity
        return dropped

    def clear(self):
        """Reset to empty. Only safe when neither side is running."""
        self._read_index = 0
        self._write_index = 0
        self._buffer[:] = 0.0

    def statistics(self) -> dict:
        return {
            'capacity': self.capacity,
            'available': self.available,
            'channels': self._channels,
            'frames_written': self.frames_written,
            'frames_read': self.frames_read,
            'overflow_count': self.overflow_count,
            'underflow_count': self.underflow_count,
        }
