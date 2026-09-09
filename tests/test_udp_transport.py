"""
The UDP wire format, and real datagrams over a real loopback socket.

Half of this is the same shape as `test_network.py`'s framing tests: pack, unpack, and
refuse anything that does not add up. The other half deliberately uses actual
`AF_INET`/`SOCK_DGRAM` sockets rather than a fake, because a fake socket cannot lose,
reorder or duplicate a packet, and those are precisely the behaviours this transport
exists to survive — a mocked socket would pass every one of these tests against code
that could not work at all.

Everything binds `127.0.0.1` explicitly. Binding `0.0.0.0` raises a Windows firewall
prompt, which is not a thing a test run should do to somebody.
"""

import math
import socket
import struct
import time

import numpy as np
import pytest

from tonesphere.network.audio_router import NetworkQuality
from tonesphere.network.send_worker import PacketAccumulator
from tonesphere.network.udp_transport import (
    CODEC_OPUS,
    CODEC_PCM_FLOAT32,
    CODEC_PCM_INT16,
    FLAG_ZLIB,
    HEADER_FORMAT,
    HEADER_SIZE,
    MAX_DATAGRAM_BYTES,
    MAX_PAYLOAD_BYTES,
    PROTOCOL_VERSION,
    SEQUENCE_MODULUS,
    CodecUnavailable,
    MalformedPacket,
    UdpAudioTransport,
    UdpPacket,
    codec_for_quality,
    encode_block,
    frames_per_packet,
    opus_available,
)

RATE = 48000


def sine(frames: int, freq: float = 1000.0, rate: int = RATE,
         amplitude: float = 0.5, channels: int = 2, phase: float = 0.0) -> np.ndarray:
    """The same test tone the rest of the suite uses, so results are comparable."""
    t = (np.arange(frames, dtype=np.float64) + phase) / rate
    wave = (amplitude * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)
    return np.repeat(wave.reshape(-1, 1), channels, axis=1)


def dominant_frequency(block: np.ndarray, rate: int = RATE) -> float:
    mono = block[:, 0] if block.ndim > 1 else block
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    return float(np.fft.rfftfreq(len(mono), 1.0 / rate)[int(np.argmax(spectrum))])


def receiver() -> socket.socket:
    """A plain UDP socket bound to loopback, for reading what the transport really sent."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 0))
    sock.settimeout(2.0)
    return sock


@pytest.fixture
def transport():
    tx = UdpAudioTransport(sample_rate=RATE)
    yield tx
    tx.stop()


class TestWireFormat:
    def test_header_is_thirty_bytes(self):
        """The size the whole packetisation budget is calculated against."""
        assert HEADER_SIZE == 30
        assert struct.calcsize(HEADER_FORMAT) == 30

    def test_round_trips_every_field(self):
        packet = UdpPacket(
            codec=CODEC_PCM_INT16, channels=4, flags=FLAG_ZLIB, sample_rate=44100,
            device_id=10007, sequence=123456, timestamp_us=987654321,
            frame_count=292, payload=b"payload bytes",
        )

        restored = UdpPacket.unpack(packet.pack())

        assert restored == packet

    def test_declared_payload_length_matches_what_is_written(self):
        packet = UdpPacket(
            codec=CODEC_PCM_FLOAT32, channels=2, flags=0, sample_rate=RATE,
            device_id=1, sequence=0, timestamp_us=0, frame_count=8,
            payload=b"x" * 64,
        )
        raw = packet.pack()

        declared = struct.unpack(HEADER_FORMAT, raw[:HEADER_SIZE])[-1]

        assert declared == 64
        assert len(raw) == HEADER_SIZE + 64

    def test_a_full_packet_stays_within_the_mtu_budget(self):
        """
        The reason packet size is decoupled from block size in the first place.

        A fragmented audio datagram is worse than a lost one — losing any one fragment
        loses the whole packet — so an over-budget packet is a real defect, not a tuning
        preference.
        """
        for channels in (1, 2, 4, 8):
            for codec in (CODEC_PCM_FLOAT32, CODEC_PCM_INT16):
                frames = frames_per_packet(codec, channels)
                # Random, not a sine: zlib shrinks a sine and would hide an over-budget
                # raw payload behind a lucky compression ratio.
                rng = np.random.default_rng(7)
                audio = rng.uniform(-1.0, 1.0, (frames, channels)).astype(np.float32)

                packet = encode_block(1, audio, RATE, 0, NetworkQuality.LOSSLESS, codec=codec)

                assert len(packet.pack()) <= MAX_DATAGRAM_BYTES, (
                    f"{channels}ch {codec} packet is {len(packet.pack())} bytes"
                )

    def test_frames_per_packet_is_what_actually_fits(self):
        assert frames_per_packet(CODEC_PCM_FLOAT32, 2) == MAX_PAYLOAD_BYTES // 8
        assert frames_per_packet(CODEC_PCM_INT16, 2) == MAX_PAYLOAD_BYTES // 4
        # Never zero, or the sender would loop forever on an accumulator it cannot drain.
        assert frames_per_packet(CODEC_PCM_FLOAT32, 8, limit=4) == 1

    def test_one_packet_holds_less_than_one_engine_block(self):
        """
        The fact that forces an accumulator to exist.

        If this ever stopped being true, `send_worker`'s re-chunking would be dead weight —
        it is here so the assumption is checked rather than remembered.
        """
        assert frames_per_packet(CODEC_PCM_FLOAT32, 2) < 256


class TestMalformedInput:
    def test_datagram_shorter_than_the_header_is_refused(self):
        with pytest.raises(MalformedPacket, match="shorter than"):
            UdpPacket.unpack(b"\x01" * (HEADER_SIZE - 1))

    def test_empty_datagram_is_refused(self):
        with pytest.raises(MalformedPacket):
            UdpPacket.unpack(b"")

    def test_payload_length_mismatch_is_refused(self):
        """
        A header that disagrees with its own datagram cannot be trusted at all.

        Reshaping whatever arrived to fit the declared geometry is how you get noise out
        of a speaker instead of an error in a log.
        """
        header = struct.pack(
            HEADER_FORMAT, PROTOCOL_VERSION, CODEC_PCM_FLOAT32, 2, 0,
            RATE, 1, 0, 0, 8, 64,
        )

        with pytest.raises(MalformedPacket, match="declares 64 payload bytes"):
            UdpPacket.unpack(header + b"x" * 32)

    def test_absurd_payload_length_is_refused_rather_than_allocated(self):
        header = struct.pack(
            HEADER_FORMAT, PROTOCOL_VERSION, CODEC_PCM_FLOAT32, 2, 0,
            RATE, 1, 0, 0, 8, 4_000_000_000,
        )

        with pytest.raises(MalformedPacket):
            UdpPacket.unpack(header + b"x" * 32)

    def test_wrong_protocol_version_is_refused_not_guessed(self):
        header = struct.pack(
            HEADER_FORMAT, PROTOCOL_VERSION + 1, CODEC_PCM_FLOAT32, 2, 0,
            RATE, 1, 0, 0, 1, 8,
        )

        with pytest.raises(MalformedPacket, match="protocol version"):
            UdpPacket.unpack(header + b"x" * 8)

    def test_zero_channels_and_zero_frames_are_refused(self):
        for channels, frames in ((0, 8), (2, 0)):
            header = struct.pack(
                HEADER_FORMAT, PROTOCOL_VERSION, CODEC_PCM_FLOAT32, channels, 0,
                RATE, 1, 0, 0, frames, 8,
            )
            with pytest.raises(MalformedPacket):
                UdpPacket.unpack(header + b"x" * 8)

    def test_payload_that_does_not_match_the_geometry_is_refused(self):
        packet = UdpPacket(
            codec=CODEC_PCM_FLOAT32, channels=2, flags=0, sample_rate=RATE,
            device_id=1, sequence=0, timestamp_us=0, frame_count=64,
            payload=b"x" * 16,
        )

        with pytest.raises(MalformedPacket, match="expected 512"):
            UdpPacket.unpack(packet.pack()).decode()

    def test_corrupt_compressed_payload_is_refused(self):
        packet = UdpPacket(
            codec=CODEC_PCM_FLOAT32, channels=2, flags=FLAG_ZLIB, sample_rate=RATE,
            device_id=1, sequence=0, timestamp_us=0, frame_count=4,
            payload=b"not zlib at all",
        )

        with pytest.raises(MalformedPacket, match="zlib"):
            UdpPacket.unpack(packet.pack()).decode()


class TestOpusIsRefusedNotFaked:
    """
    Opus is not implemented, so nothing may behave as though it were.

    A silent fallback to PCM would be the exact failure this project's honesty rule is
    about: the caller asked for Opus, got PCM, and was told it worked.
    """

    def test_availability_is_reported_as_false(self):
        assert opus_available() is False

    def test_encoding_with_opus_raises_rather_than_substituting_pcm(self):
        audio = sine(64)

        with pytest.raises(CodecUnavailable, match="not implemented"):
            encode_block(1, audio, RATE, 0, NetworkQuality.HIGH, codec=CODEC_OPUS)

    def test_an_opus_datagram_from_a_peer_is_refused(self):
        header = struct.pack(
            HEADER_FORMAT, PROTOCOL_VERSION, CODEC_OPUS, 2, 0, RATE, 1, 0, 0, 960, 8,
        )

        with pytest.raises(MalformedPacket, match="Opus"):
            UdpPacket.unpack(header + b"x" * 8)

    def test_no_quality_preset_selects_opus(self):
        for quality in NetworkQuality:
            codec, _level = codec_for_quality(quality)
            assert codec != CODEC_OPUS


class TestCodecRoundTrip:
    @pytest.mark.parametrize("quality", list(NetworkQuality))
    @pytest.mark.parametrize("channels", [1, 2, 4])
    def test_audio_survives_encode_and_decode(self, quality, channels):
        codec, _level = codec_for_quality(quality)
        frames = frames_per_packet(codec, channels)
        audio = sine(frames, channels=channels)

        packet = encode_block(7, audio, RATE, 42, quality)
        restored = packet.decode()

        assert restored.shape == audio.shape
        assert packet.device_id == 7
        assert packet.sequence == 42
        assert packet.frame_count == frames

        # float32 is exact; int16 quantises at about 3e-5.
        tolerance = 1e-6 if codec == CODEC_PCM_FLOAT32 else 1e-4
        np.testing.assert_allclose(restored, audio, atol=tolerance)

    def test_the_tone_comes_back_at_the_same_frequency(self):
        frames = frames_per_packet(CODEC_PCM_INT16, 2)
        audio = sine(frames, freq=1000.0)

        restored = encode_block(1, audio, RATE, 0, NetworkQuality.LOW).decode()

        # A 292-frame window at 48 kHz has 164 Hz bins, so this is the nearest bin to
        # 1 kHz rather than 1000.0 exactly.
        assert abs(dominant_frequency(restored) - dominant_frequency(audio)) < 1.0

    def test_compression_is_only_claimed_when_it_helped(self):
        """
        zlib can enlarge incompressible data, and audio is nearly incompressible.

        Sending an enlarged payload while flagging it compressed would be both a lie and
        a way to blow past the MTU budget the packet size was calculated for.
        """
        rng = np.random.default_rng(11)
        frames = frames_per_packet(CODEC_PCM_FLOAT32, 2)
        noise = rng.uniform(-1.0, 1.0, (frames, 2)).astype(np.float32)

        packet = encode_block(1, noise, RATE, 0, NetworkQuality.HIGH)

        raw_bytes = frames * 2 * 4
        assert len(packet.payload) <= raw_bytes
        if not packet.compressed:
            assert len(packet.payload) == raw_bytes
        np.testing.assert_allclose(packet.decode(), noise, atol=1e-6)

    def test_int16_clips_rather_than_wrapping(self):
        """
        The cast that goes wrong quietly. int16 wraps on overflow, so +1.2 would arrive
        as a large negative sample — a loud click, not a soft clip.
        """
        loud = np.full((16, 2), 1.5, dtype=np.float32)
        loud[8:] = -1.5

        restored = encode_block(1, loud, RATE, 0, NetworkQuality.MEDIUM).decode()

        assert restored[:8].min() > 0.99
        assert restored[8:].max() < -0.99

    def test_an_empty_block_is_refused(self):
        with pytest.raises(ValueError, match="empty"):
            encode_block(1, np.zeros((0, 2), dtype=np.float32), RATE, 0)


class TestRealSockets:
    """Actual datagrams over 127.0.0.1. No fakes anywhere in this class."""

    def test_bound_address_is_reported_not_assumed(self, transport):
        host, port = transport.start('127.0.0.1', 0)

        assert host == '127.0.0.1'
        assert port > 0
        assert transport.bound_address == (host, port)
        assert transport.statistics()['bound_port'] == port

    def test_nothing_is_claimed_bound_before_start(self, transport):
        stats = transport.statistics()

        assert stats['running'] is False
        # Not '0.0.0.0:0', which would read as a real address we are listening on.
        assert stats['bound_host'] is None
        assert stats['bound_port'] is None

    def test_a_known_tone_arrives_over_a_real_socket(self, transport):
        sock = receiver()
        try:
            transport.start('127.0.0.1', 0)
            transport.add_peer('peer', *sock.getsockname())

            frames = frames_per_packet(CODEC_PCM_FLOAT32, 2)
            audio = sine(frames, freq=1000.0)

            assert transport.send_block(4242, audio, quality=NetworkQuality.LOSSLESS) == 1

            datagram, _address = sock.recvfrom(65535)
        finally:
            sock.close()

        packet = UdpPacket.unpack(datagram)

        assert packet.device_id == 4242
        assert packet.sample_rate == RATE
        np.testing.assert_allclose(packet.decode(), audio, atol=1e-6)

    def test_sequence_numbers_increment_per_device(self, transport):
        sock = receiver()
        try:
            transport.start('127.0.0.1', 0)
            transport.add_peer('peer', *sock.getsockname())

            audio = sine(64)
            for _ in range(5):
                transport.send_block(1, audio)
                transport.send_block(2, audio)

            sequences = {1: [], 2: []}
            for _ in range(10):
                packet = UdpPacket.unpack(sock.recvfrom(65535)[0])
                sequences[packet.device_id].append(packet.sequence)
        finally:
            sock.close()

        assert sequences[1] == [0, 1, 2, 3, 4]
        assert sequences[2] == [0, 1, 2, 3, 4], "each device numbers its own stream"

    def test_broadcast_reaches_every_peer(self, transport):
        first, second = receiver(), receiver()
        try:
            transport.start('127.0.0.1', 0)
            transport.add_peer('a', *first.getsockname())
            transport.add_peer('b', *second.getsockname())

            assert transport.send_block(1, sine(64)) == 2

            assert UdpPacket.unpack(first.recvfrom(65535)[0]).device_id == 1
            assert UdpPacket.unpack(second.recvfrom(65535)[0]).device_id == 1
        finally:
            first.close()
            second.close()

    def test_a_named_target_reaches_only_that_peer(self, transport):
        wanted, other = receiver(), receiver()
        other.settimeout(0.3)
        try:
            transport.start('127.0.0.1', 0)
            transport.add_peer('wanted', *wanted.getsockname())
            transport.add_peer('other', *other.getsockname())

            assert transport.send_block(1, sine(64), target='wanted') == 1

            assert UdpPacket.unpack(wanted.recvfrom(65535)[0]).device_id == 1
            with pytest.raises(TimeoutError):
                other.recvfrom(65535)
        finally:
            wanted.close()
            other.close()

    def test_sending_with_no_peers_is_counted_not_hidden(self, transport):
        """
        The difference between streaming and encoding audio into a void.

        Returning 0 and counting it is what lets the CLI say "no peers registered yet"
        instead of showing a healthy packet count for audio nobody received.
        """
        transport.start('127.0.0.1', 0)

        assert transport.send_block(1, sine(64)) == 0
        assert transport.statistics()['sends_with_no_peer'] == 1
        assert transport.statistics()['packets_sent'] == 0

    def test_sending_while_stopped_is_an_error_not_a_silent_no_op(self, transport):
        assert transport.send_block(1, sine(64)) == 0

        stats = transport.statistics()
        assert stats['send_errors'] == 1
        assert 'not running' in stats['last_error']

    def test_received_packets_are_dispatched_by_device_id(self, transport):
        for_one: list[UdpPacket] = []
        for_two: list[UdpPacket] = []
        transport.register_handler(1, for_one.append)
        transport.register_handler(2, for_two.append)

        host, port = transport.start('127.0.0.1', 0)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            audio = sine(64)
            for device_id in (1, 2, 2, 3):
                packet = encode_block(device_id, audio, RATE, 0, NetworkQuality.LOSSLESS)
                sender.sendto(packet.pack(), (host, port))

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and transport.stats['packets_received'] < 4:
                time.sleep(0.01)
        finally:
            sender.close()

        assert transport.stats['packets_received'] == 4
        assert len(for_one) == 1
        assert len(for_two) == 2, "device 3 has no handler and must not reach another one"
        np.testing.assert_allclose(for_one[0].decode(), audio, atol=1e-6)

    def test_a_garbage_datagram_is_rejected_and_the_receiver_survives(self, transport):
        """
        One corrupt datagram must not take the receive thread with it — the next good
        packet has to still arrive.
        """
        arrived: list[UdpPacket] = []
        transport.register_handler(9, arrived.append)

        host, port = transport.start('127.0.0.1', 0)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(b"this is not an audio packet", (host, port))
            sender.sendto(b"\x00" * 80, (host, port))

            good = encode_block(9, sine(64), RATE, 0, NetworkQuality.LOSSLESS)
            sender.sendto(good.pack(), (host, port))

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not arrived:
                time.sleep(0.01)
        finally:
            sender.close()

        assert len(arrived) == 1, "the receive thread died on the malformed datagram"
        assert transport.stats['packets_rejected'] == 2
        assert transport.statistics()['last_error'] is not None

    def test_a_raising_handler_is_counted_and_does_not_kill_the_thread(self, transport):
        calls = {'n': 0}

        def handler(_packet):
            calls['n'] += 1
            raise RuntimeError("handler is broken")

        transport.register_handler(5, handler)
        host, port = transport.start('127.0.0.1', 0)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for sequence in range(3):
                packet = encode_block(5, sine(64), RATE, sequence, NetworkQuality.LOSSLESS)
                sender.sendto(packet.pack(), (host, port))

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and calls['n'] < 3:
                time.sleep(0.01)
        finally:
            sender.close()

        assert calls['n'] == 3
        assert transport.stats['handler_errors'] == 3

    def test_stop_is_idempotent_and_leaves_nothing_bound(self, transport):
        transport.start('127.0.0.1', 0)
        transport.stop()
        transport.stop()

        assert transport.is_running is False
        assert transport.bound_address is None

    def test_sequence_wraps_rather_than_overflowing_the_field(self, transport):
        """A u32 field cannot carry 2**32, so the sequence has to wrap before packing."""
        transport._sequences[1] = SEQUENCE_MODULUS - 1

        assert transport.next_sequence(1) == SEQUENCE_MODULUS - 1
        assert transport.next_sequence(1) == 0


class TestPacketAccumulator:
    """
    The join between two block sizes the audio path and the MTU each chose independently.

    Tested on its own, with no socket and no thread, because this is where an off-by-one
    would silently drop or duplicate a frame at every packet boundary — a defect that
    would show up end-to-end as a faint periodic tick and nowhere else.
    """

    def test_it_holds_frames_back_until_a_whole_packet_is_available(self):
        accumulator = PacketAccumulator(channels=2)

        assert accumulator.push(sine(100)) == 100
        assert accumulator.pending == 100
        assert accumulator.take(146) is None, "handed out a short packet"

        accumulator.push(sine(100, phase=100))
        assert accumulator.take(146) is not None
        assert accumulator.pending == 54

    def test_re_chunking_loses_and_duplicates_nothing(self):
        """
        The whole 256-into-146 reconciliation, checked sample by sample against the
        original stream rather than by counting frames.
        """
        accumulator = PacketAccumulator(channels=2)
        source = sine(256 * 40)
        packet_frames = frames_per_packet(CODEC_PCM_FLOAT32, 2)

        collected = []
        for index in range(40):
            accumulator.push(source[index * 256:(index + 1) * 256])
            while (chunk := accumulator.take(packet_frames)) is not None:
                collected.append(chunk)

        rebuilt = np.concatenate(collected)

        assert len(rebuilt) == len(collected) * packet_frames
        np.testing.assert_array_equal(rebuilt, source[:len(rebuilt)])
        assert accumulator.pending == len(source) - len(rebuilt)

    def test_a_packet_can_span_three_pushes(self):
        accumulator = PacketAccumulator(channels=2)
        source = sine(90)

        for start in (0, 30, 60):
            accumulator.push(source[start:start + 30])

        np.testing.assert_array_equal(accumulator.take(90), source)

    def test_a_mismatched_channel_count_is_refused(self):
        accumulator = PacketAccumulator(channels=2)

        assert accumulator.push(sine(64, channels=4)) == 0
        assert accumulator.pending == 0

    def test_a_mono_block_is_accepted_by_a_mono_accumulator(self):
        accumulator = PacketAccumulator(channels=1)

        assert accumulator.push(np.zeros(64, dtype=np.float32)) == 64
        assert accumulator.take(64).shape == (64, 1)

    def test_reset_discards_everything_pending(self):
        accumulator = PacketAccumulator(channels=2)
        accumulator.push(sine(200))

        accumulator.reset()

        assert accumulator.pending == 0
        assert accumulator.take(1) is None

    def test_a_bad_channel_count_is_refused_at_construction(self):
        with pytest.raises(ValueError):
            PacketAccumulator(channels=0)
