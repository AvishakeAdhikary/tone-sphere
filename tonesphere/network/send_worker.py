"""
The send side: one paced thread that drains route ring buffers into datagrams.

Why a thread at all, rather than sending from the audio callback: sending is a syscall
that can block, and blocking in a PortAudio callback is what causes dropouts. The callback
already writes every route's audio into a ring buffer, so the sender is an ordinary
consumer of that ring — the same relationship a second sound card has to the first.

Why packets are not blocks
--------------------------
`MAX_PAYLOAD_BYTES` is about 1170, and one 256-frame float32 stereo block is 2048, so
there is no block size at which one read equals one packet. Two independent quantisations
have to be reconciled: the engine's block size, which the audio path chooses, and the
packet's frame count, which the MTU chooses. `PacketAccumulator` is the join between them
— it collects whatever the reads produce and hands out exactly the packet size asked for,
which is also what an Opus encoder would need if one were ever added, since Opus accepts
only a fixed set of frame durations.

Why each read takes whatever is there, rather than a fixed block
----------------------------------------------------------------
`threading.Event.wait` is not a high-resolution timer. On Windows it is backed by
`WaitForSingleObject`, whose timeouts round to the system tick — measured here at roughly
8-15 ms against a 5.3 ms block interval — so this thread wakes late far more often than
it wakes on time.

That makes a fixed-size read actively wrong: `read_route` zero-fills whatever had not
arrived, so a late wake asking for a full block would splice silence into the stream at
exactly the moments the sender was already behind. Reading only what the ring holds
removes the timer from the correctness argument entirely — the producer sets the rate, a
late wake simply reads more, and the interval decides only how often to look. The ring is
four blocks deep, so it absorbs the observed lateness with room to spare, and anything
worse than that is already counted as `overflow_count` by the ring itself.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tonesphere.network.audio_router import NetworkQuality
from tonesphere.network.udp_transport import (
    UdpAudioTransport,
    codec_for_quality,
    frames_per_packet,
)
from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)


class PacketAccumulator:
    """
    Re-chunks a stream of arbitrary-length blocks into fixed-size ones.

    Kept as its own object rather than inlined into the worker loop so the reconciliation
    of two block sizes — the part with the off-by-one potential — can be tested on its own
    without a socket or a thread anywhere near it.
    """

    def __init__(self, channels: int):
        if channels < 1:
            raise ValueError("channels must be at least 1")
        self.channels = channels
        self._blocks: list[np.ndarray] = []
        self._pending = 0

    @property
    def pending(self) -> int:
        return self._pending

    def push(self, block: np.ndarray) -> int:
        """Append frames. Returns how many were taken, which is zero for a mismatch."""
        frames = np.asarray(block, dtype=np.float32)
        if frames.ndim == 1:
            frames = frames.reshape(-1, 1)
        if frames.shape[1] != self.channels or frames.shape[0] == 0:
            return 0

        self._blocks.append(frames)
        self._pending += frames.shape[0]
        return frames.shape[0]

    def take(self, frames: int) -> np.ndarray | None:
        """
        Exactly `frames` frames, or None if that many are not held yet.

        Never a short block: a datagram whose frame count does not match its payload is a
        packet the receiver will reject, so waiting is right and padding would be a lie
        about what the source produced.
        """
        if frames < 1 or self._pending < frames:
            return None

        out = np.empty((frames, self.channels), dtype=np.float32)
        filled = 0

        while filled < frames:
            head = self._blocks[0]
            wanted = frames - filled
            if head.shape[0] <= wanted:
                out[filled:filled + head.shape[0]] = head
                filled += head.shape[0]
                self._blocks.pop(0)
            else:
                out[filled:] = head[:wanted]
                self._blocks[0] = head[wanted:]
                filled = frames

        self._pending -= frames
        return out

    def reset(self):
        self._blocks.clear()
        self._pending = 0


@dataclass
class SendRoute:
    """One registered source -> network route, and what has actually happened on it."""

    sink_id: int
    device_id: int
    source_key: str
    dest_key: str
    target: str | None = None

    accumulator: PacketAccumulator | None = None
    frames_per_packet: int = 0

    reads: int = 0
    empty_reads: int = 0
    missing_route_reads: int = 0
    frames_read: int = 0
    packets_sent: int = 0
    packets_undeliverable: int = 0
    encode_errors: int = 0
    last_error: str | None = None

    def statistics(self) -> dict:
        return {
            'sink_id': self.sink_id,
            'device_id': self.device_id,
            'source': self.source_key,
            'dest': self.dest_key,
            'target': self.target,
            'channels': self.accumulator.channels if self.accumulator else None,
            'frames_per_packet': self.frames_per_packet or None,
            'reads': self.reads,
            'empty_reads': self.empty_reads,
            'missing_route_reads': self.missing_route_reads,
            'frames_read': self.frames_read,
            'packets_sent': self.packets_sent,
            'packets_undeliverable': self.packets_undeliverable,
            'encode_errors': self.encode_errors,
            'pending_frames': self.accumulator.pending if self.accumulator else 0,
            'last_error': self.last_error,
        }


@dataclass
class _WorkerStats:
    ticks: int = 0
    late_ticks: int = 0
    reader_errors: int = 0


class NetworkSendWorker:
    """
    Reads registered routes on a fixed interval and sends what it gets.

    Pacing is against `time.monotonic()` with an accumulating deadline rather than
    `sleep(interval)` per iteration, so the time spent reading and sending does not add to
    the interval and slowly drift the send rate below the audio rate.
    """

    def __init__(
        self,
        transport: UdpAudioTransport,
        reader: Callable[[str, str, int], np.ndarray | None],
        sample_rate: int = 48000,
        frames_per_read: int = 256,
        quality: NetworkQuality | None = None,
    ):
        self.transport = transport
        self.reader = reader
        self.sample_rate = sample_rate
        self.frames_per_read = frames_per_read
        self.quality = quality

        self._routes: dict[int, SendRoute] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._stats = _WorkerStats()

    @property
    def interval(self) -> float:
        return self.frames_per_read / self.sample_rate

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def route_count(self) -> int:
        with self._lock:
            return len(self._routes)

    # --- Routes ---

    def add_route(
        self,
        sink_id: int,
        device_id: int,
        source_key: str,
        dest_key: str,
        target: str | None = None,
    ) -> SendRoute:
        route = SendRoute(
            sink_id=sink_id, device_id=device_id,
            source_key=source_key, dest_key=dest_key, target=target,
        )
        with self._lock:
            self._routes[sink_id] = route
        return route

    def remove_route(self, sink_id: int) -> bool:
        with self._lock:
            return self._routes.pop(sink_id, None) is not None

    def route(self, sink_id: int) -> SendRoute | None:
        with self._lock:
            return self._routes.get(sink_id)

    # --- Lifecycle ---

    def start(self):
        with self._lock:
            if self.is_running:
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name='udp-audio-send', daemon=True
            )
            self._thread.start()
        logger.info(f"Network send worker running every {self.interval * 1000:.1f} ms")

    def stop(self):
        self._stop.set()
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _run(self):
        deadline = time.monotonic()

        while not self._stop.is_set():
            # One interval elapses before the first read, not after it. At the instant the
            # worker starts, every route's ring is empty by definition — nothing has had
            # time to produce a block yet — so reading immediately would put a packet of
            # guaranteed silence at the head of the stream, ahead of the audio the user
            # actually routed.
            deadline += self.interval
            remaining = deadline - time.monotonic()

            if remaining > 0:
                # Wait on the stop event rather than sleeping, so stop() is immediate.
                if self._stop.wait(remaining):
                    break
            else:
                # Woke late, which on a coarse-timer OS is the common case rather than the
                # exception. No catch-up burst is needed or wanted: each read already takes
                # everything the ring holds, so one late read carries the audio two
                # on-time reads would have. Only the schedule is re-based, and the lateness
                # is counted so a persistently overloaded machine is visible.
                self._stats.late_ticks += 1
                deadline = time.monotonic()

            self._stats.ticks += 1
            self.tick()

    def tick(self):
        """
        One pass over every route. Public so a test can drive it without the thread.

        Driving pacing from a test thread is the difference between a test that proves
        packetisation and one that proves the machine was not busy.
        """
        with self._lock:
            routes = list(self._routes.values())

        for route in routes:
            self._service(route)

    def _service(self, route: SendRoute):
        try:
            block = self.reader(route.source_key, route.dest_key, self.frames_per_read)
        except Exception as e:
            self._stats.reader_errors += 1
            route.last_error = f"read failed: {e}"
            return

        route.reads += 1

        if block is None:
            # The route's ring has gone away — the graph no longer contains it, which is
            # a real problem and not the same thing as a quiet source. Counted rather than
            # logged: this happens once per interval, and a log line per interval is a
            # denial of service on the log file.
            route.missing_route_reads += 1
            return

        if block.shape[0] == 0:
            # Nothing had been produced since the last read. Ordinary — a source that is
            # not playing anything reads empty — so it is counted, not treated as a fault.
            route.empty_reads += 1
            return

        if route.accumulator is None:
            # Channel count comes from the audio, not from a guess. The first real block
            # is the earliest point at which it is known.
            channels = block.shape[1]
            codec, _level = codec_for_quality(self._quality())
            route.accumulator = PacketAccumulator(channels)
            route.frames_per_packet = frames_per_packet(codec, channels)

        route.frames_read += route.accumulator.push(block)

        while True:
            chunk = route.accumulator.take(route.frames_per_packet)
            if chunk is None:
                break

            try:
                delivered = self.transport.send_block(
                    route.device_id, chunk, route.target, self._quality()
                )
            except Exception as e:
                route.encode_errors += 1
                route.last_error = f"send failed: {e}"
                break

            if delivered:
                route.packets_sent += 1
            else:
                route.packets_undeliverable += 1

    def _quality(self) -> NetworkQuality:
        return self.transport.quality if self.quality is None else self.quality

    # --- Statistics ---

    def statistics(self) -> dict:
        with self._lock:
            routes = {str(sink_id): r.statistics() for sink_id, r in self._routes.items()}
        return {
            'running': self.is_running,
            'interval_ms': round(self.interval * 1000.0, 3),
            'frames_per_read': self.frames_per_read,
            'ticks': self._stats.ticks,
            'late_ticks': self._stats.late_ticks,
            'reader_errors': self._stats.reader_errors,
            'routes': routes,
        }
