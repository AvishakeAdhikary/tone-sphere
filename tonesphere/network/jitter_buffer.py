"""
The jitter buffer that sits between socket-receive and the audio path.

The model this replaces is "write each packet into the bus as it arrives", which sounds
harmless and is not: packets arrive in bursts, out of order, and sometimes not at all, so
writing on arrival hands the audio path the network's timing instead of the clock's. The
result is audible as stutter even on a link losing nothing.

So arrival and playout are separated. `push()` inserts into a sparse map keyed by sequence
number; `pull()` is called once per packet duration by a paced thread and takes the packet
whose slot is due. A packet that has not arrived by its slot is *concealed* — and counted
as lost, because it was.

Why a dict rather than a pre-sized circular array: the contents are sparse by nature.
Reordering and loss leave holes, and the natural operations are "is sequence N here yet"
and "how far ahead is the nearest packet I do have", both of which are a dict lookup and
a small scan rather than index arithmetic against a moving base.

Priming
-------
Nothing drains until `target_latency_ms` of audio is held. Until then `pull()` returns
None, and that is counted as priming, never as loss — a buffer that has not started is
not a buffer that is dropping audio, and conflating the two would make a healthy startup
look like a broken link.

Playout that has overtaken the sender primes once more. A sender that stalls for longer
than the buffer holds leaves its slots to be concealed, and playout moves on past audio
that is still coming; both then run at the audio rate, so playout never falls back, and
every packet from then on arrives just after its slot and is dropped as late. One stall
lost every packet after it that way (macOS CI, Opus over UDP: 200 received, 200 lost).
A straggler, or jitter, drops a late packet now and then between packets that play; an
overtaken playout drops nothing but. So after `OVERTAKEN_PACKETS` late packets in a row
with nothing held and nothing played between them, playout moves back to the latest and
the buffer fills to its target again before playing — counted as an *underrun*.

Adaptive depth
--------------
`mode='fixed'` holds `target_latency_ms` and nothing else. `mode='adaptive'` measures how
late packets actually arrive and sizes the buffer to that. Each packet's transit is its
arrival time (this machine's clock) minus its send time (the sender's timestamp, or its
sequence number times the packet duration when there is none — the same thing for a sender
that sends at the audio rate); a clock offset between the machines cancels, because only
differences are used. Its *delay* is how much later than the fastest packet of the last
two seconds it arrived: a buffer that deep would have had it in time. The target is the
99th percentile of those delays plus one packet, kept between `min_latency_ms` and
`max_latency_ms`. RFC 3550's interarrival jitter is kept too, for the statistics.

The buffer then moves towards the target by whole packets, and never silently: when its
smoothed depth falls half a packet below the target, one pull holds the stream for a slot
(a concealment, counted as a *stretch*, not as a loss; at most one in four pulls); when it
sits more than one and a
half packets above, one due packet is dropped (a *shrink*), at most once a second, so the
latency comes down slowly and only when the link has been calm. The estimator is driven by
an injectable clock, so `tests/test_jitter_buffer.py` runs it against scripted arrival
schedules, deterministically.
"""

import threading
import time
from collections import deque

import numpy as np

SEQUENCE_MODULUS = 2 ** 32

CONCEAL_SILENCE = 'silence'
CONCEAL_REPEAT = 'repeat'

MODE_FIXED = 'fixed'
MODE_ADAPTIVE = 'adaptive'

DELAY_WINDOW_S = 2.0
DELAY_QUANTILE = 0.99
SHRINK_INTERVAL_S = 1.0
DEPTH_SMOOTHING = 1.0 / 8.0
# A held slot shows in the depth only once the next packet lands, so stretches are spaced
# to let the smoothed depth see each one before deciding on another.
STRETCH_SPACING_PULLS = 4
# Late packets in a row, into an empty buffer, that mean playout has overtaken the sender.
OVERTAKEN_PACKETS = 8


def sequence_distance(origin: int, other: int) -> int:
    """
    Signed distance from `origin` to `other` across the u32 sequence wrap.

    Sequence numbers wrap every 2**32 packets, which at a 3 ms packet is about 400 years —
    except that a sender restarting, or one deliberately starting near the top of the
    range, reaches it immediately. A plain `other - origin` would then read as a jump of
    four billion packets and either look like catastrophic loss or trigger an endless
    resync, so the comparison is modular: results are interpreted in the nearer half of
    the ring, so 1 is "one ahead of" 0xFFFFFFFF.
    """
    diff = (other - origin) % SEQUENCE_MODULUS
    if diff >= SEQUENCE_MODULUS // 2:
        diff -= SEQUENCE_MODULUS
    return diff


class JitterBuffer:
    """
    Reorders, conceals and paces one incoming stream.

    Single stream per instance: mixing two senders' sequence numbers in one buffer would
    make every packet of each look like a reorder of the other.

    Thread safety: `push()` runs on the receive thread and `pull()` on the playout thread,
    so both take a lock. This is not the audio callback — the callback reads the bus that
    playout writes — so a lock here costs nothing that matters.
    """

    def __init__(
        self,
        frames_per_packet: int,
        channels: int,
        sample_rate: int = 48000,
        target_latency_ms: float = 40.0,
        max_buffered_packets: int = 64,
        max_concealment_repeats: int = 3,
        resync_gap_packets: int = 32,
        conceal: str = CONCEAL_SILENCE,
        mode: str = MODE_FIXED,
        decoder=None,
        min_latency_ms: float = 10.0,
        max_latency_ms: float = 200.0,
        clock=time.monotonic,
    ):
        if frames_per_packet < 1:
            raise ValueError("frames_per_packet must be at least 1")
        if channels < 1:
            raise ValueError("channels must be at least 1")
        if conceal not in (CONCEAL_SILENCE, CONCEAL_REPEAT):
            raise ValueError(f"conceal must be {CONCEAL_SILENCE!r} or {CONCEAL_REPEAT!r}")
        if mode not in (MODE_FIXED, MODE_ADAPTIVE):
            raise ValueError(f"mode must be {MODE_FIXED!r} or {MODE_ADAPTIVE!r}")

        self.frames_per_packet = frames_per_packet
        self.channels = channels
        self.sample_rate = sample_rate
        self.target_latency_ms = target_latency_ms
        self.max_buffered_packets = max(1, max_buffered_packets)
        self.max_concealment_repeats = max(0, max_concealment_repeats)
        self.resync_gap_packets = max(2, resync_gap_packets)
        self.conceal = conceal

        self.packet_duration_ms = frames_per_packet / sample_rate * 1000.0

        # Round up: at a 40 ms target and a 3.04 ms packet, holding 14 packets is the
        # first depth that actually covers the target, so rounding down would leave the
        # buffer permanently shallower than the number it reports.
        #
        # Done in integer milliseconds-times-frames rather than on the float packet
        # duration. 40 ms of 128-frame packets is exactly 15 of them, but 40.0 / (128 /
        # 48000 * 1000) evaluates to 15.000000000000002, so a float ceiling asks for 16 —
        # a whole extra packet of latency conjured out of a rounding error.
        self.target_packets = max(
            1,
            -(-int(round(target_latency_ms * sample_rate)) // (frames_per_packet * 1000)),
        )

        self.mode = mode
        # A stateful codec's decoder (Opus): packets are held encoded and decoded as they
        # play, in sequence, and a missing one is concealed by the codec rather than here.
        self._decoder = decoder
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max(max_latency_ms, min_latency_ms)
        self._clock = clock
        self._delays: deque[tuple[float, float]] = deque()   # (arrival, transit)
        self._last_transit: float | None = None
        self._jitter_s = 0.0
        self._depth = float(self.target_packets)
        self._last_shrink = -1e9
        self._pulls_since_stretch = STRETCH_SPACING_PULLS

        self._packets: dict[int, np.ndarray] = {}
        self._next_sequence: int | None = None
        self._priming = True
        self._last_block: np.ndarray | None = None
        self._concealment_run = 0
        self._late_run = 0
        self._silence = np.zeros((frames_per_packet, channels), dtype=np.float32)
        self._lock = threading.Lock()

        self._stats = {
            'packets_pushed': 0,
            'packets_played_on_time': 0,
            'packets_lost': 0,
            'packets_late_dropped': 0,
            'packets_duplicate': 0,
            'packets_evicted': 0,
            'packets_wrong_shape': 0,
            'concealment_repeats': 0,
            'silence_insertions': 0,
            'resync_events': 0,
            'resynced_packets_skipped': 0,
            'priming_pulls': 0,
            'underruns': 0,
            'peak_buffered_packets': 0,
            'stretches': 0,
            'shrinks': 0,
            'codec_concealments': 0,
            'decode_errors': 0,
        }

    # --- Producer side ---

    def push(self, sequence: int, frames: np.ndarray, timestamp_us: int | None = None) -> bool:
        """
        Insert one arrived packet. Returns whether it was kept.

        Rejects, each counted separately because they mean different things: a packet whose
        playout slot has already passed is *late*, and playing it now would put audio in
        the stream out of order; a sequence already held is a *duplicate*, which a
        retransmitting middlebox or a doubled peer entry produces; a block of the wrong
        geometry is a *mismatch*, and writing it into a bus sized for something else would
        be noise.
        """
        encoded = self._decoder is not None and isinstance(frames, (bytes, bytearray))
        if encoded:
            block = bytes(frames)
        else:
            block = np.ascontiguousarray(frames, dtype=np.float32)
            if block.ndim == 1:
                block = block.reshape(-1, 1)

        with self._lock:
            self._stats['packets_pushed'] += 1

            if not encoded and block.shape != (self.frames_per_packet, self.channels):
                self._stats['packets_wrong_shape'] += 1
                return False

            sequence %= SEQUENCE_MODULUS
            if self.mode == MODE_ADAPTIVE:
                self._measure(sequence, timestamp_us)

            if self._next_sequence is None:
                # The first packet ever seen defines where playout starts. Any earlier
                # sequence arriving afterwards is genuinely late.
                self._next_sequence = sequence
            elif sequence_distance(self._next_sequence, sequence) < 0:
                self._late_run = 0 if self._packets else self._late_run + 1
                if self._late_run < OVERTAKEN_PACKETS:
                    self._stats['packets_late_dropped'] += 1
                    return False
                self._stats['underruns'] += 1
                self._late_run = 0
                self._next_sequence = sequence
                self._priming = True

            if sequence in self._packets:
                self._stats['packets_duplicate'] += 1
                return False

            self._packets[sequence] = block

            # A sender that has run away, or a peer flooding us, must not be able to grow
            # this map without limit. Evict from the oldest end, since that is the audio
            # closest to being needed and therefore the audio a stalled playout has
            # already fallen behind.
            #
            # Playout is moved past each eviction rather than left pointing at it. An
            # evicted slot is definitively never going to be filled, so waiting for it
            # would make playout conceal its way through audio it has already thrown
            # away — and count each of those slots a second time as a loss, reporting one
            # overrun as an overrun plus a burst of packet loss that never happened.
            while len(self._packets) > self.max_buffered_packets:
                oldest = min(
                    self._packets,
                    key=lambda s: sequence_distance(self._next_sequence, s),
                )
                del self._packets[oldest]
                self._stats['packets_evicted'] += 1

                if sequence_distance(self._next_sequence, oldest) >= 0:
                    self._next_sequence = (oldest + 1) % SEQUENCE_MODULUS

            self._stats['peak_buffered_packets'] = max(
                self._stats['peak_buffered_packets'], len(self._packets)
            )
            return True

    def _measure(self, sequence: int, timestamp_us: int | None):
        """One arrival's transit into the delay window and the RFC 3550 jitter estimate."""
        arrival = self._clock()
        sent = timestamp_us / 1e6 if timestamp_us else sequence * self.packet_duration_ms / 1000.0
        transit = arrival - sent
        if self._last_transit is not None:
            self._jitter_s += (abs(transit - self._last_transit) - self._jitter_s) / 16.0
        self._last_transit = transit
        self._delays.append((arrival, transit))
        while self._delays and arrival - self._delays[0][0] > DELAY_WINDOW_S:
            self._delays.popleft()

    def _adaptive_target_packets(self) -> int:
        """The 99th-percentile delay of the window, plus a packet, in whole packets."""
        transits = [t for _, t in self._delays]
        if len(transits) < 8:
            target_ms = self.target_latency_ms
        else:
            fastest = min(transits)
            late = np.quantile(np.array(transits) - fastest, DELAY_QUANTILE) * 1000.0
            target_ms = late + self.packet_duration_ms
        target_ms = min(max(target_ms, self.min_latency_ms), self.max_latency_ms)
        return max(1, int(np.ceil(target_ms / self.packet_duration_ms - 1e-9)))

    def _adapt(self) -> str | None:
        """Called once per pull after priming: 'stretch', 'shrink' or None."""
        self.target_packets = self._adaptive_target_packets()
        self._depth += (len(self._packets) - self._depth) * DEPTH_SMOOTHING
        self._pulls_since_stretch += 1
        if self._depth < self.target_packets - 0.5 and self._pulls_since_stretch >= STRETCH_SPACING_PULLS:
            self._pulls_since_stretch = 0
            return 'stretch'
        now = self._clock()
        if self._depth > self.target_packets + 1.5 and now - self._last_shrink >= SHRINK_INTERVAL_S:
            self._last_shrink = now
            return 'shrink'
        return None

    # --- Consumer side ---

    def pull(self) -> np.ndarray | None:
        """
        The block due now, or None while priming.

        None means "not started yet" and nothing else. Every other outcome returns audio:
        the packet that arrived, a concealment, or silence. The caller writes what it gets
        and skips on None, so a priming buffer inserts nothing rather than inserting
        silence that would later have to be distinguished from real silence.
        """
        with self._lock:
            if self._next_sequence is None:
                self._stats['priming_pulls'] += 1
                return None

            if self._priming:
                if self.mode == MODE_ADAPTIVE:
                    self.target_packets = self._adaptive_target_packets()
                if len(self._packets) < self.target_packets:
                    self._stats['priming_pulls'] += 1
                    return None
                self._priming = False
                self._depth = float(len(self._packets))

            if self.mode == MODE_ADAPTIVE:
                action = self._adapt()
                if action == 'stretch':
                    # Hold the stream one slot: what plays is a concealment, but no packet
                    # is skipped and none is counted lost.
                    self._stats['stretches'] += 1
                    if self._decoder is not None:
                        return self._decode(None)
                    if self.conceal == CONCEAL_REPEAT and self._last_block is not None:
                        return self._last_block
                    return self._silence
                if action == 'shrink' and self._next_sequence in self._packets:
                    del self._packets[self._next_sequence]
                    self._next_sequence = (self._next_sequence + 1) % SEQUENCE_MODULUS
                    self._stats['shrinks'] += 1

            block = self._packets.pop(self._next_sequence, None)
            if isinstance(block, bytes):
                block = self._decode(block)

            if block is not None:
                self._stats['packets_played_on_time'] += 1
                self._next_sequence = (self._next_sequence + 1) % SEQUENCE_MODULUS
                self._last_block = block
                self._concealment_run = 0
                self._late_run = 0
                return block

            resynced = self._resync_if_far_behind()
            if resynced is not None:
                return resynced

            return self._conceal_one()

    def _resync_if_far_behind(self) -> np.ndarray | None:
        """
        Jump forward when the nearest held packet is a long way ahead.

        Without this, a two-second outage would be concealed one packet at a time — several
        hundred separate "losses" reported for a single event, and several hundred pulls
        spent emitting silence before reaching audio that has been sitting in the buffer
        the whole time. One resync is both the honest count and the faster recovery.
        """
        if not self._packets:
            return None

        nearest = min(
            self._packets, key=lambda s: sequence_distance(self._next_sequence, s)
        )
        gap = sequence_distance(self._next_sequence, nearest)
        if gap < self.resync_gap_packets:
            return None

        self._stats['resync_events'] += 1
        self._stats['resynced_packets_skipped'] += gap

        block = self._packets.pop(nearest)
        if isinstance(block, bytes):
            block = self._decode(block)
        self._stats['packets_played_on_time'] += 1
        self._next_sequence = (nearest + 1) % SEQUENCE_MODULUS
        self._last_block = block
        self._concealment_run = 0
        return block

    def _conceal_one(self) -> np.ndarray:
        """
        Fill one missing slot, and count it as the loss it is.

        Repeat-last-block is capped rather than open-ended. Repeating indefinitely would
        synthesise an alive-sounding signal out of a dead link — a sender that has been
        unplugged for a minute would still be "playing" its last 3 ms forever, which is
        both a lie and a buzz. After the cap it falls back to silence, which is what the
        link is actually delivering.
        """
        self._stats['packets_lost'] += 1
        self._next_sequence = (self._next_sequence + 1) % SEQUENCE_MODULUS

        if self._decoder is not None:
            self._stats['codec_concealments'] += 1
            return self._decode(None)

        if (self.conceal == CONCEAL_REPEAT
                and self._last_block is not None
                and self._concealment_run < self.max_concealment_repeats):
            self._concealment_run += 1
            self._stats['concealment_repeats'] += 1
            return self._last_block

        self._stats['silence_insertions'] += 1
        return self._silence

    def _decode(self, payload: bytes | None) -> np.ndarray:
        """One packet through the codec, or its concealment of a lost one; silence if it fails."""
        try:
            block = self._decoder.decode(payload)
        except Exception:   # noqa: BLE001 - a corrupt packet must not stop playout
            self._stats['decode_errors'] += 1
            return self._silence
        if block.shape != (self.frames_per_packet, self.channels):
            self._stats['decode_errors'] += 1
            return self._silence
        return block

    def reset(self):
        """Forget everything and prime again — for a peer that has restarted."""
        with self._lock:
            self._packets.clear()
            self._next_sequence = None
            self._priming = True
            self._last_block = None
            self._concealment_run = 0
            self._late_run = 0

    # --- Statistics ---

    @property
    def buffered_packets(self) -> int:
        return len(self._packets)

    @property
    def priming(self) -> bool:
        return self._priming

    def statistics(self) -> dict:
        """
        Counts of things that happened, plus the buffer depth as it stands.

        `buffered_latency_ms` is arithmetic over a real measured depth, and is labelled as
        buffered rather than as a network latency: it is what this buffer is holding, not
        the path delay, which we have not measured.
        """
        with self._lock:
            buffered = len(self._packets)
            return {
                'frames_per_packet': self.frames_per_packet,
                'channels': self.channels,
                'packet_duration_ms': round(self.packet_duration_ms, 3),
                'mode': self.mode,
                'target_latency_ms': (round(self.target_packets * self.packet_duration_ms, 3)
                                      if self.mode == MODE_ADAPTIVE else self.target_latency_ms),
                'target_packets': self.target_packets,
                # Interarrival jitter (RFC 3550), measured in adaptive mode only; None, not 0,
                # where nothing was measured.
                'jitter_ms': round(self._jitter_s * 1000.0, 3) if self.mode == MODE_ADAPTIVE and self._delays
                else None,
                'buffered_packets': buffered,
                'buffered_latency_ms': round(buffered * self.packet_duration_ms, 3),
                'priming': self._priming,
                'conceal': self.conceal,
                'next_sequence': self._next_sequence,
                **self._stats,
            }
