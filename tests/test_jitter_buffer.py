"""
The jitter buffer, driven by hand through the scenarios a real network produces.

Nothing here is mocked, including the buffer itself. The whole value of this class is what
it does about reordering, loss and late arrival, so a test that substituted any of that
would be asserting the substitute. Instead each scenario is constructed exactly: packets
are pushed in a chosen order, with chosen gaps, and `pull()` is called the number of times
playout would call it.

Every packet carries a distinguishable block — its sequence number as a DC value — so an
assertion can say *which* packet came out, not merely that something did. That is what
catches a reorder that "worked" by playing the wrong block.
"""

import numpy as np
import pytest

from tonesphere.network.jitter_buffer import (
    CONCEAL_REPEAT,
    CONCEAL_SILENCE,
    SEQUENCE_MODULUS,
    JitterBuffer,
    sequence_distance,
)

FRAMES = 128
CHANNELS = 2
RATE = 48000

# 128 frames at 48 kHz is 2.667 ms, so 8 packets is a hair over a 20 ms target.
PACKET_MS = FRAMES / RATE * 1000.0


def block(sequence: int) -> np.ndarray:
    """A block identifiable by which packet it came from."""
    return np.full((FRAMES, CHANNELS), float(sequence), dtype=np.float32)


def which(pulled: np.ndarray) -> float:
    return float(pulled[0, 0])


def make(target_latency_ms: float = 0.0, **kwargs) -> JitterBuffer:
    """
    A buffer that plays immediately unless a test asks for priming.

    A zero target still primes one packet — `target_packets` has a floor of 1 — which is
    what "start as soon as anything is here" means.
    """
    return JitterBuffer(
        frames_per_packet=FRAMES, channels=CHANNELS, sample_rate=RATE,
        target_latency_ms=target_latency_ms, **kwargs
    )


class TestSequenceDistance:
    def test_plain_forward_and_backward(self):
        assert sequence_distance(10, 13) == 3
        assert sequence_distance(13, 10) == -3
        assert sequence_distance(7, 7) == 0

    def test_across_the_wrap(self):
        """
        The case a plain subtraction gets catastrophically wrong: it would read one packet
        forward as four billion packets backward.
        """
        assert sequence_distance(SEQUENCE_MODULUS - 1, 0) == 1
        assert sequence_distance(SEQUENCE_MODULUS - 2, 2) == 4
        assert sequence_distance(0, SEQUENCE_MODULUS - 1) == -1


class TestPriming:
    def test_pull_returns_none_before_anything_arrives(self):
        buffer = make(target_latency_ms=20.0)

        assert buffer.pull() is None
        assert buffer.pull() is None

    def test_priming_pulls_are_not_counted_as_loss(self):
        """
        A buffer that has not started is not a buffer dropping audio. Conflating the two
        would make every healthy startup look like a broken link.
        """
        buffer = make(target_latency_ms=20.0)

        for _ in range(5):
            assert buffer.pull() is None

        stats = buffer.statistics()
        assert stats['priming_pulls'] == 5
        assert stats['packets_lost'] == 0
        assert stats['silence_insertions'] == 0

    def test_nothing_drains_until_the_target_depth_is_held(self):
        buffer = make(target_latency_ms=20.0)
        assert buffer.target_packets == 8

        for sequence in range(7):
            buffer.push(sequence, block(sequence))
            assert buffer.pull() is None, "drained while still shallower than the target"
            assert buffer.priming is True

        buffer.push(7, block(7))
        assert which(buffer.pull()) == 0.0
        assert buffer.priming is False

    def test_target_packets_rounds_up_to_cover_the_target(self):
        """Rounding down would leave the buffer permanently shallower than it reports."""
        buffer = make(target_latency_ms=40.0)

        assert buffer.target_packets * PACKET_MS >= 40.0
        assert (buffer.target_packets - 1) * PACKET_MS < 40.0

    def test_priming_happens_once_not_after_every_gap(self):
        buffer = make(target_latency_ms=0.0)
        buffer.push(0, block(0))

        assert which(buffer.pull()) == 0.0
        # Slot 1 is missing. That is a loss, not a return to priming.
        assert buffer.pull() is not None
        assert buffer.statistics()['packets_lost'] == 1
        assert buffer.statistics()['priming_pulls'] == 0


class TestInOrderPlayback:
    def test_packets_come_out_in_order_with_nothing_lost(self):
        buffer = make()
        for sequence in range(20):
            buffer.push(sequence, block(sequence))

        played = [which(buffer.pull()) for _ in range(20)]

        assert played == [float(s) for s in range(20)]

        stats = buffer.statistics()
        assert stats['packets_played_on_time'] == 20
        assert stats['packets_lost'] == 0
        assert stats['buffered_packets'] == 0

    def test_the_first_packet_seen_defines_where_playout_starts(self):
        """A sender starting at an arbitrary sequence is normal, not an error."""
        buffer = make()
        buffer.push(5000, block(5000))
        buffer.push(5001, block(5001))

        assert which(buffer.pull()) == 5000.0
        assert which(buffer.pull()) == 5001.0

    def test_buffered_latency_reflects_the_real_depth(self):
        buffer = make()
        for sequence in range(6):
            buffer.push(sequence, block(sequence))

        stats = buffer.statistics()
        assert stats['buffered_packets'] == 6
        assert stats['buffered_latency_ms'] == pytest.approx(6 * PACKET_MS, abs=0.01)


class TestReordering:
    def test_a_single_swapped_pair_is_played_in_order(self):
        buffer = make()
        buffer.push(0, block(0))
        buffer.push(2, block(2))
        buffer.push(1, block(1))

        assert [which(buffer.pull()) for _ in range(3)] == [0.0, 1.0, 2.0]
        assert buffer.statistics()['packets_lost'] == 0

    def test_a_reversed_burst_is_fully_reassembled(self):
        buffer = make()
        buffer.push(0, block(0))
        for sequence in reversed(range(1, 9)):
            buffer.push(sequence, block(sequence))

        played = [which(buffer.pull()) for _ in range(9)]

        assert played == [float(s) for s in range(9)]
        assert buffer.statistics()['packets_lost'] == 0

    def test_reordering_across_the_sequence_wrap(self):
        start = SEQUENCE_MODULUS - 2
        buffer = make()
        buffer.push(start, block(1))
        buffer.push((start + 2) % SEQUENCE_MODULUS, block(3))
        buffer.push((start + 1) % SEQUENCE_MODULUS, block(2))

        assert [which(buffer.pull()) for _ in range(3)] == [1.0, 2.0, 3.0]
        assert buffer.statistics()['packets_lost'] == 0


class TestLateArrival:
    def test_a_packet_whose_slot_has_passed_is_dropped_not_played(self):
        """
        Playing it now would put audio in the stream out of order — worse than the
        concealment that already covered its slot.
        """
        buffer = make()
        buffer.push(0, block(0))
        buffer.push(1, block(1))

        assert which(buffer.pull()) == 0.0
        assert which(buffer.pull()) == 1.0

        assert buffer.push(0, block(0)) is False

        stats = buffer.statistics()
        assert stats['packets_late_dropped'] == 1
        assert stats['buffered_packets'] == 0

    def test_a_packet_late_but_still_ahead_of_playout_is_kept(self):
        buffer = make()
        buffer.push(0, block(0))
        buffer.push(3, block(3))

        assert which(buffer.pull()) == 0.0
        # Slot 1 is now due. Packet 2 has not been played yet, so it is still useful.
        assert buffer.push(2, block(2)) is True

    def test_a_duplicate_is_counted_separately_from_a_late_packet(self):
        buffer = make()
        buffer.push(4, block(4))

        assert buffer.push(4, block(4)) is False

        stats = buffer.statistics()
        assert stats['packets_duplicate'] == 1
        assert stats['packets_late_dropped'] == 0

    def test_a_block_of_the_wrong_shape_is_refused(self):
        """
        Writing a wrongly-shaped block into a bus sized for something else is noise, so
        the mismatch is refused and counted rather than reshaped hopefully.
        """
        buffer = make()

        assert buffer.push(0, np.zeros((FRAMES, CHANNELS + 1), dtype=np.float32)) is False
        assert buffer.push(1, np.zeros((FRAMES // 2, CHANNELS), dtype=np.float32)) is False
        assert buffer.statistics()['packets_wrong_shape'] == 2
        assert buffer.statistics()['buffered_packets'] == 0


class TestLossAndConcealment:
    def test_a_single_gap_is_concealed_with_silence_and_counted(self):
        buffer = make()
        buffer.push(0, block(0))
        buffer.push(2, block(2))

        assert which(buffer.pull()) == 0.0

        concealed = buffer.pull()
        assert concealed is not None
        np.testing.assert_array_equal(concealed, np.zeros((FRAMES, CHANNELS), np.float32))

        assert which(buffer.pull()) == 2.0

        stats = buffer.statistics()
        assert stats['packets_lost'] == 1
        assert stats['silence_insertions'] == 1
        assert stats['packets_played_on_time'] == 2

    def test_repeat_concealment_repeats_the_last_block_up_to_its_cap(self):
        buffer = make(conceal=CONCEAL_REPEAT, max_concealment_repeats=2)
        buffer.push(0, block(7))

        assert which(buffer.pull()) == 7.0

        assert which(buffer.pull()) == 7.0
        assert which(buffer.pull()) == 7.0

        stats = buffer.statistics()
        assert stats['concealment_repeats'] == 2
        assert stats['packets_lost'] == 2

    def test_repeat_concealment_falls_back_to_silence_after_the_cap(self):
        """
        Repeating forever would synthesise an alive signal out of a dead link — a sender
        unplugged for a minute would still be "playing" its last 3 ms, which is both a
        lie and a buzz.
        """
        buffer = make(conceal=CONCEAL_REPEAT, max_concealment_repeats=2)
        buffer.push(0, block(7))
        buffer.pull()

        buffer.pull()
        buffer.pull()
        third = buffer.pull()

        np.testing.assert_array_equal(third, np.zeros((FRAMES, CHANNELS), np.float32))

        stats = buffer.statistics()
        assert stats['concealment_repeats'] == 2
        assert stats['silence_insertions'] == 1
        assert stats['packets_lost'] == 3

    def test_repeat_concealment_resets_after_real_audio_returns(self):
        buffer = make(conceal=CONCEAL_REPEAT, max_concealment_repeats=1)
        buffer.push(0, block(1))
        buffer.push(2, block(3))
        buffer.push(4, block(5))

        assert which(buffer.pull()) == 1.0
        assert which(buffer.pull()) == 1.0, "concealed by repeating"
        assert which(buffer.pull()) == 3.0
        assert which(buffer.pull()) == 3.0, "cap applies again, not once for the stream"

        assert buffer.statistics()['concealment_repeats'] == 2

    def test_a_burst_loss_is_counted_once_per_missing_packet(self):
        buffer = make()
        buffer.push(0, block(0))
        for sequence in range(6, 10):
            buffer.push(sequence, block(sequence))

        played = [which(buffer.pull()) for _ in range(10)]

        assert played[0] == 0.0
        assert played[1:6] == [0.0] * 5, "five silent slots"
        assert played[6:10] == [6.0, 7.0, 8.0, 9.0]
        assert buffer.statistics()['packets_lost'] == 5

    def test_concealment_never_returns_none(self):
        """
        None means "priming" and only that. A caller distinguishes the two by that alone,
        so a loss returning None would be read as a healthy not-yet-started buffer.
        """
        buffer = make()
        buffer.push(0, block(0))
        buffer.pull()

        for _ in range(5):
            assert buffer.pull() is not None


class TestResync:
    def test_a_large_gap_jumps_forward_instead_of_a_wall_of_losses(self):
        buffer = make(resync_gap_packets=16)
        buffer.push(0, block(0))
        buffer.push(500, block(500))

        assert which(buffer.pull()) == 0.0
        assert which(buffer.pull()) == 500.0, "should have jumped, not concealed 499 slots"

        stats = buffer.statistics()
        assert stats['resync_events'] == 1
        assert stats['resynced_packets_skipped'] == 499
        assert stats['packets_lost'] == 0, "a resync is one event, not 499 losses"

    def test_a_small_gap_is_concealed_rather_than_resynced(self):
        buffer = make(resync_gap_packets=16)
        buffer.push(0, block(0))
        buffer.push(4, block(4))

        buffer.pull()
        for _ in range(3):
            buffer.pull()
        assert which(buffer.pull()) == 4.0

        stats = buffer.statistics()
        assert stats['resync_events'] == 0
        assert stats['packets_lost'] == 3

    def test_playout_continues_from_the_resync_point(self):
        buffer = make(resync_gap_packets=8)
        buffer.push(0, block(0))
        buffer.push(100, block(100))
        buffer.push(101, block(101))

        assert [which(buffer.pull()) for _ in range(3)] == [0.0, 100.0, 101.0]

    def test_an_outage_then_a_restart_recovers_by_resync(self):
        """
        The realistic sequence: a link drops, playout conceals its way through the empty
        buffer, and audio resumes far ahead. It must recover, not conceal indefinitely.
        """
        buffer = make(resync_gap_packets=8)
        buffer.push(0, block(0))
        assert which(buffer.pull()) == 0.0

        for _ in range(5):
            buffer.pull()

        buffer.push(400, block(400))
        assert which(buffer.pull()) == 400.0

        stats = buffer.statistics()
        assert stats['packets_lost'] == 5
        assert stats['resync_events'] == 1


class TestOverflowDefence:
    def test_a_flooding_sender_cannot_grow_the_buffer_without_limit(self):
        buffer = make(max_buffered_packets=10)

        for sequence in range(200):
            buffer.push(sequence, block(sequence))

        stats = buffer.statistics()
        assert stats['buffered_packets'] == 10
        assert stats['packets_evicted'] == 190
        assert stats['peak_buffered_packets'] == 10

    def test_eviction_keeps_the_newest_audio(self):
        """
        Evicting from the oldest end is what lets a stalled playout catch up: the oldest
        packets are the ones it has already fallen behind.
        """
        buffer = make(max_buffered_packets=4)
        for sequence in range(10):
            buffer.push(sequence, block(sequence))

        assert which(buffer.pull()) == 6.0

    def test_eviction_is_not_also_counted_as_packet_loss(self):
        """
        One overrun is one event. Leaving playout pointing at an evicted slot would make
        it conceal through audio already thrown away and report a burst of loss that never
        happened on the wire — an overrun dressed up as a bad link.
        """
        buffer = make(max_buffered_packets=4)
        for sequence in range(10):
            buffer.push(sequence, block(sequence))

        played = [which(buffer.pull()) for _ in range(4)]

        assert played == [6.0, 7.0, 8.0, 9.0]

        stats = buffer.statistics()
        assert stats['packets_evicted'] == 6
        assert stats['packets_lost'] == 0
        assert stats['silence_insertions'] == 0
        assert stats['packets_played_on_time'] == 4

    def test_peak_depth_is_the_high_water_mark_not_the_current_depth(self):
        buffer = make()
        for sequence in range(5):
            buffer.push(sequence, block(sequence))
        for _ in range(5):
            buffer.pull()

        stats = buffer.statistics()
        assert stats['buffered_packets'] == 0
        assert stats['peak_buffered_packets'] == 5


class TestWraparound:
    def test_playout_continues_across_the_sequence_wrap(self):
        start = SEQUENCE_MODULUS - 3
        buffer = make()

        for offset in range(6):
            sequence = (start + offset) % SEQUENCE_MODULUS
            buffer.push(sequence, block(offset))

        played = [which(buffer.pull()) for _ in range(6)]

        assert played == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        assert buffer.statistics()['packets_lost'] == 0
        assert buffer.statistics()['next_sequence'] == (start + 6) % SEQUENCE_MODULUS

    def test_a_late_packet_across_the_wrap_is_still_recognised_as_late(self):
        buffer = make()
        buffer.push(SEQUENCE_MODULUS - 1, block(1))
        buffer.push(0, block(2))

        assert which(buffer.pull()) == 1.0
        assert which(buffer.pull()) == 2.0

        assert buffer.push(SEQUENCE_MODULUS - 1, block(1)) is False
        assert buffer.statistics()['packets_late_dropped'] == 1


class TestConstruction:
    def test_a_bad_geometry_is_refused(self):
        with pytest.raises(ValueError):
            JitterBuffer(frames_per_packet=0, channels=2)
        with pytest.raises(ValueError):
            JitterBuffer(frames_per_packet=128, channels=0)

    def test_an_unknown_concealment_mode_is_refused(self):
        with pytest.raises(ValueError, match="conceal must be"):
            JitterBuffer(frames_per_packet=128, channels=2, conceal='invent-something')

    def test_the_default_target_is_the_documented_forty_milliseconds(self):
        buffer = JitterBuffer(frames_per_packet=FRAMES, channels=CHANNELS, sample_rate=RATE)

        assert buffer.target_latency_ms == 40.0
        assert buffer.conceal == CONCEAL_SILENCE

    def test_reset_primes_again(self):
        buffer = make(target_latency_ms=20.0)
        for sequence in range(8):
            buffer.push(sequence, block(sequence))
        assert buffer.pull() is not None

        buffer.reset()

        assert buffer.priming is True
        assert buffer.pull() is None
        assert buffer.buffered_packets == 0
