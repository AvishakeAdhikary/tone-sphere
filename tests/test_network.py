"""
Network packet framing.

The bug these pin down: the wire format carried no payload length, so a receiver had to
guess where a compressed packet ended by retrying zlib.decompress until it worked. That
consumes bytes belonging to the next packet, and once the stream is desynchronised it
never recovers.
"""

import json
import struct

import numpy as np
import pytest

from tonesphere.network.audio_router import (
    MAX_PACKET_BYTES, AudioCodec, AudioPacket, NetworkAudioRouter, NetworkQuality,
)


def make_packet(frames=256, channels=2, compressed=False, payload=b"abcdef"):
    return AudioPacket(
        device_id=1, channels=channels, sample_rate=48000, frames=frames,
        codec=AudioCodec.PCM_FLOAT32.value, timestamp=0.0,
        data=payload, compressed=compressed,
    )


class FakeSocket:
    """A socket that hands back a fixed byte stream, so framing can be tested offline."""

    def __init__(self, data: bytes):
        self._data = data
        self.position = 0

    def recv(self, size: int) -> bytes:
        chunk = self._data[self.position:self.position + size]
        self.position += len(chunk)
        return chunk

    def settimeout(self, value):
        pass

    @property
    def remaining(self) -> bytes:
        return self._data[self.position:]


class TestWireFormat:
    def test_header_declares_the_payload_length(self):
        """The field whose absence broke the protocol."""
        packet = make_packet(payload=b"0123456789")
        raw = packet.to_bytes()

        header_length = struct.unpack('!I', raw[:4])[0]
        header = json.loads(raw[4:4 + header_length])

        assert header['data_length'] == 10

    def test_round_trips(self):
        packet = make_packet(payload=b"payload bytes")
        restored = AudioPacket.from_bytes(packet.to_bytes())

        assert restored.data == b"payload bytes"
        assert restored.channels == packet.channels
        assert restored.frames == packet.frames

    def test_serialised_length_is_exactly_accounted_for(self):
        """No slack: header length + header + payload and nothing else."""
        packet = make_packet(payload=b"x" * 100)
        raw = packet.to_bytes()

        header_length = struct.unpack('!I', raw[:4])[0]
        assert len(raw) == 4 + header_length + 100


class TestFramingDoesNotOverrun:
    def test_two_packets_back_to_back_are_read_separately(self):
        """
        The regression. Concatenate two packets and read the first: the reader must stop at
        its declared end, leaving the second intact.
        """
        router = NetworkAudioRouter()

        first = make_packet(payload=b"FIRSTPACKET", compressed=True)
        second = make_packet(payload=b"SECONDPACKET", compressed=True)

        sock = FakeSocket(first.to_bytes() + second.to_bytes())

        received = router._receive_packet(sock)
        assert received is not None
        assert received.data == b"FIRSTPACKET"

        followed = router._receive_packet(sock)
        assert followed is not None
        assert followed.data == b"SECONDPACKET", "reader consumed into the next packet"

    def test_uncompressed_packets_also_frame_correctly(self):
        router = NetworkAudioRouter()

        first = make_packet(payload=b"A" * 32)
        second = make_packet(payload=b"B" * 16)
        sock = FakeSocket(first.to_bytes() + second.to_bytes())

        assert router._receive_packet(sock).data == b"A" * 32
        assert router._receive_packet(sock).data == b"B" * 16

    def test_stream_stays_aligned_over_many_packets(self):
        router = NetworkAudioRouter()

        payloads = [bytes([i]) * (10 + i) for i in range(20)]
        stream = b"".join(
            make_packet(payload=p, compressed=bool(i % 2)).to_bytes()
            for i, p in enumerate(payloads)
        )
        sock = FakeSocket(stream)

        for expected in payloads:
            packet = router._receive_packet(sock)
            assert packet is not None
            assert packet.data == expected

        assert sock.remaining == b""


class TestMalformedInput:
    def test_absurd_length_is_rejected_rather_than_allocated(self):
        """
        A hostile or corrupt header must not be able to make us allocate until the process
        dies.
        """
        router = NetworkAudioRouter()

        header = json.dumps({
            'device_id': 1, 'channels': 2, 'sample_rate': 48000, 'frames': 256,
            'codec': AudioCodec.PCM_FLOAT32.value, 'timestamp': 0.0,
            'compressed': False, 'data_length': MAX_PACKET_BYTES * 100,
        }).encode()

        sock = FakeSocket(struct.pack('!I', len(header)) + header)

        assert router._receive_packet(sock) is None

    def test_truncated_stream_returns_none(self):
        router = NetworkAudioRouter()
        raw = make_packet(payload=b"x" * 64).to_bytes()

        sock = FakeSocket(raw[:len(raw) // 2])

        assert router._receive_packet(sock) is None

    def test_empty_stream_returns_none(self):
        router = NetworkAudioRouter()

        assert router._receive_packet(FakeSocket(b"")) is None

    def test_old_sender_with_compression_is_refused_not_guessed(self):
        """
        A peer using the old framing gives no payload length. For compressed data the
        boundary is unknowable, so refusing is correct — guessing is what corrupted the
        stream before.
        """
        router = NetworkAudioRouter()

        header = json.dumps({
            'device_id': 1, 'channels': 2, 'sample_rate': 48000, 'frames': 256,
            'codec': AudioCodec.PCM_FLOAT32.value, 'timestamp': 0.0,
            'compressed': True,
        }).encode()

        sock = FakeSocket(struct.pack('!I', len(header)) + header + b"junk")

        assert router._receive_packet(sock) is None


class TestCodecRoundTrip:
    @pytest.mark.parametrize("quality", [
        NetworkQuality.LOSSLESS, NetworkQuality.HIGH,
        NetworkQuality.MEDIUM, NetworkQuality.LOW,
    ])
    def test_audio_survives_encode_and_decode(self, quality):
        router = NetworkAudioRouter(quality=quality)

        audio = np.sin(
            np.linspace(0, 20, 256, dtype=np.float32)
        ).reshape(-1, 1).repeat(2, axis=1).astype(np.float32)

        packet = router._encode_audio(1, audio, 48000)
        restored = router._decode_audio(packet)

        assert restored.shape == audio.shape

        # int16 codecs quantise; float32 should be exact.
        tolerance = 1e-6 if quality == NetworkQuality.LOSSLESS else 1e-3
        np.testing.assert_allclose(restored, audio, atol=tolerance)

    def test_encoded_packet_frames_and_channels_match_the_audio(self):
        router = NetworkAudioRouter()
        audio = np.zeros((128, 2), dtype=np.float32)

        packet = router._encode_audio(7, audio, 48000)

        assert packet.frames == 128
        assert packet.channels == 2
        assert packet.device_id == 7
