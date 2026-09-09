"""
Realtime UDP audio transport.

Why this exists alongside `audio_router.py` rather than replacing it: TCP retransmits and
blocks the head of the line, so one lost packet stalls every packet queued behind it. That
is the wrong trade for monitoring, which would rather drop a block than wait for it. This
module makes the opposite trade, and the TCP path stays as-is for the bulk transfers it
was scoped for.

Wire format
-----------
A fixed 30-byte binary header, not JSON. UDP has no framing problem for a length-prefixed
JSON header to solve — a datagram arrives whole or it does not arrive — and a JSON header
costs 150-250 bytes against a payload of about 1170, which spends a sixth of the bandwidth
describing the audio instead of carrying it.

===================  =====  ==========================================================
version              u8     protocol version; a mismatch is refused, never guessed at
codec                u8     CODEC_PCM_FLOAT32 / CODEC_PCM_INT16
channels             u8
flags                u8     FLAG_ZLIB
sample_rate          u32
device_id            u32
sequence             u32    wraps at 2**32; `jitter_buffer` handles the wrap
timestamp_us         u64    sender's monotonic clock, in microseconds
frame_count          u16    frames per channel carried in the payload
payload_length       u32    byte count following the header
===================  =====  ==========================================================

Datagram size
-------------
The whole datagram is kept at or below `MAX_DATAGRAM_BYTES`. That sits under the smallest
MTU worth designing for on a real path — a 1492-byte PPPoE link, or a VPN that takes
another 60-80 bytes for its own encapsulation — and an IP-fragmented audio datagram is
worse than a lost one: losing any single fragment loses the whole packet, so fragmenting
multiplies the effective loss rate by the fragment count.

The resulting 1170-byte payload is *smaller* than one 256-frame float32 stereo block
(2048 bytes), so packet size cannot be tied to the engine's block size. `frames_per_packet`
computes what actually fits and `send_worker.py` accumulates across reads to fill it.

Opus
----
Deliberately not implemented, and refused rather than silently substituted.

`CODEC_OPUS` is reserved in the wire format so a future encoder does not need a protocol
bump, and both encode and decode raise `CodecUnavailable` for it. The reason is a verified
packaging fact rather than a preference: neither Python binding ships a libopus binary for
all three platforms this project tests on. `PyOgg` publishes only `win32`/`win_amd64`
wheels (checked against PyPI's own file list for every release) — on Linux and macOS it
installs from its sdist and falls back to `ctypes.util.find_library`, exactly the system
package step it was supposed to avoid — and its released version does not expose an
encoder class at all. `opuslib` publishes no wheels whatsoever and needs a system libopus
everywhere. Claiming Opus off a codec path that could only be exercised on one of three
CI platforms is the kind of untested capability claim this project exists to avoid, so it
is left undone and said so.
"""

import socket
import struct
import threading
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tonesphere.network.audio_router import NetworkQuality
from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)

PROTOCOL_VERSION = 1

HEADER_FORMAT = '!BBBBIIIQHI'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

CODEC_PCM_FLOAT32 = 0
CODEC_PCM_INT16 = 1
# Reserved so adding Opus later needs no protocol version bump. Refused, not substituted.
CODEC_OPUS = 2

CODEC_NAMES = {
    CODEC_PCM_FLOAT32: 'pcm_float32',
    CODEC_PCM_INT16: 'pcm_int16',
    CODEC_OPUS: 'opus',
}

_CODEC_DTYPES = {
    CODEC_PCM_FLOAT32: np.dtype(np.float32),
    CODEC_PCM_INT16: np.dtype(np.int16),
}

FLAG_ZLIB = 0x01

# Total datagram budget, header included. See the module docstring for why this is well
# under a 1500-byte Ethernet MTU rather than at it.
MAX_DATAGRAM_BYTES = 1200
MAX_PAYLOAD_BYTES = MAX_DATAGRAM_BYTES - HEADER_SIZE

# Largest payload we will accept from a peer, regardless of what we ourselves send. A
# peer is free to use a larger MTU; nothing can exceed one UDP datagram.
MAX_ACCEPTED_PAYLOAD_BYTES = 65507 - HEADER_SIZE

# A burst of unpaced sends (a test flushing many blocks at once, or a real sender that has
# been starved and is catching up) can hand the kernel more datagrams than it can hold
# before this process's receive thread gets scheduled again. Below this size, some OS
# defaults are small enough for that to mean real, silent packet loss at the socket layer
# rather than anything the jitter buffer ever sees to count: macOS's default SO_RCVBUF is a
# few tens of KB, well under one burst of test-scale traffic (found the hard way — CI on
# macOS dropped packets a Windows run of the same test never did). Requesting more is
# best-effort: some sandboxes refuse `setsockopt` outright, and a refusal here must not be
# fatal to opening the socket.
SOCKET_BUFFER_BYTES = 1 << 20  # 1 MiB

SEQUENCE_MODULUS = 2 ** 32

OPUS_UNAVAILABLE_REASON = (
    "Opus is not implemented. No Python binding ships a libopus binary for Windows, "
    "Linux and macOS alike (PyOgg publishes Windows-only wheels; opuslib publishes none), "
    "so the codec could only be tested on one of three platforms. Use a PCM quality."
)


class MalformedPacket(ValueError):
    """A datagram that cannot be trusted to be one of our packets."""


class CodecUnavailable(RuntimeError):
    """A codec named in the wire format that this build cannot actually perform."""


def bytes_per_sample(codec: int) -> int:
    dtype = _CODEC_DTYPES.get(codec)
    if dtype is None:
        raise CodecUnavailable(
            OPUS_UNAVAILABLE_REASON if codec == CODEC_OPUS else f"Unknown codec {codec}"
        )
    return dtype.itemsize


def frames_per_packet(codec: int, channels: int, limit: int = MAX_PAYLOAD_BYTES) -> int:
    """
    How many frames of `channels`-wide audio fit in one datagram.

    Never zero: a single frame of 8-channel float32 is 32 bytes, so the floor is only
    reachable with an absurd `limit`, and returning 0 would make the sender loop forever
    on an accumulator it can never drain.
    """
    frame_bytes = bytes_per_sample(codec) * max(channels, 1)
    return max(1, limit // frame_bytes)


def codec_for_quality(quality: NetworkQuality) -> tuple[int, int]:
    """
    (codec, zlib level) for a quality preset.

    Deliberately the same mapping `NetworkAudioRouter` uses, so LOW/MEDIUM/HIGH/LOSSLESS
    mean the same thing to a user whichever transport they picked.
    """
    if quality == NetworkQuality.LOSSLESS:
        return CODEC_PCM_FLOAT32, 0
    if quality == NetworkQuality.HIGH:
        return CODEC_PCM_FLOAT32, 3
    if quality == NetworkQuality.MEDIUM:
        return CODEC_PCM_INT16, 6
    return CODEC_PCM_INT16, 9


def opus_available() -> bool:
    """False, always, in this build. See OPUS_UNAVAILABLE_REASON and the module docstring."""
    return False


@dataclass(frozen=True)
class UdpPacket:
    """One datagram's worth of audio, header fields parsed out."""

    codec: int
    channels: int
    flags: int
    sample_rate: int
    device_id: int
    sequence: int
    timestamp_us: int
    frame_count: int
    payload: bytes
    version: int = PROTOCOL_VERSION

    @property
    def compressed(self) -> bool:
        return bool(self.flags & FLAG_ZLIB)

    @property
    def codec_name(self) -> str:
        return CODEC_NAMES.get(self.codec, f"unknown({self.codec})")

    def pack(self) -> bytes:
        return struct.pack(
            HEADER_FORMAT,
            self.version, self.codec, self.channels, self.flags,
            self.sample_rate, self.device_id, self.sequence,
            self.timestamp_us, self.frame_count, len(self.payload),
        ) + self.payload

    @classmethod
    def unpack(cls, datagram: bytes) -> "UdpPacket":
        """
        Parse a datagram, or raise `MalformedPacket` saying what was wrong with it.

        Every field that can contradict the datagram's actual length is checked. The
        alternative — trusting a declared length — is how the TCP path's original framing
        bug worked, and on UDP the same trust would let one corrupt header produce a
        reshape error inside the receive thread on every packet that followed.
        """
        if len(datagram) < HEADER_SIZE:
            raise MalformedPacket(
                f"datagram is {len(datagram)} bytes, shorter than the "
                f"{HEADER_SIZE}-byte header"
            )

        (version, codec, channels, flags, sample_rate, device_id,
         sequence, timestamp_us, frame_count, payload_length) = struct.unpack(
            HEADER_FORMAT, datagram[:HEADER_SIZE]
        )

        if version != PROTOCOL_VERSION:
            raise MalformedPacket(
                f"protocol version {version}, this build speaks {PROTOCOL_VERSION}"
            )

        actual = len(datagram) - HEADER_SIZE
        if payload_length != actual:
            raise MalformedPacket(
                f"header declares {payload_length} payload bytes, datagram carries {actual}"
            )

        if payload_length > MAX_ACCEPTED_PAYLOAD_BYTES:
            raise MalformedPacket(f"payload of {payload_length} bytes cannot fit a datagram")

        if channels < 1:
            raise MalformedPacket("channel count of 0")

        if frame_count < 1:
            raise MalformedPacket("frame count of 0")

        if codec not in _CODEC_DTYPES:
            reason = (
                "sender used Opus, which this build cannot decode"
                if codec == CODEC_OPUS else f"unknown codec {codec}"
            )
            raise MalformedPacket(reason)

        return cls(
            codec=codec, channels=channels, flags=flags, sample_rate=sample_rate,
            device_id=device_id, sequence=sequence, timestamp_us=timestamp_us,
            frame_count=frame_count, payload=datagram[HEADER_SIZE:], version=version,
        )

    def decode(self) -> np.ndarray:
        """
        The audio this packet carries, as float32 (frames, channels).

        A payload whose byte count does not match the declared frame/channel geometry is a
        malformed packet, not something to reshape as best we can — a partial block written
        into a bus is a click, and a wrongly-shaped one is noise.
        """
        data = self.payload

        if self.compressed:
            try:
                data = zlib.decompress(data)
            except zlib.error as e:
                raise MalformedPacket(f"payload is not valid zlib: {e}") from e

        dtype = _CODEC_DTYPES[self.codec]
        expected = self.frame_count * self.channels * dtype.itemsize
        if len(data) != expected:
            raise MalformedPacket(
                f"{len(data)} bytes of audio for {self.frame_count} frames x "
                f"{self.channels} channels of {CODEC_NAMES[self.codec]} (expected {expected})"
            )

        samples = np.frombuffer(data, dtype=dtype)

        if self.codec == CODEC_PCM_INT16:
            audio = samples.astype(np.float32) / 32767.0
        else:
            audio = samples.astype(np.float32, copy=True)

        return audio.reshape((self.frame_count, self.channels))


def encode_block(
    device_id: int,
    audio: np.ndarray,
    sample_rate: int,
    sequence: int,
    quality: NetworkQuality = NetworkQuality.HIGH,
    codec: int | None = None,
    timestamp_us: int | None = None,
) -> UdpPacket:
    """
    Pack one block of audio into a packet.

    Compression is attempted and then *checked*: zlib can enlarge incompressible data, and
    audio is close to incompressible, so a blind compress could push a payload sized to fit
    the MTU back over it. If the result is not smaller the raw bytes go out with the flag
    clear, which the receiver handles by reading the flag rather than by assuming.
    """
    block = np.atleast_2d(np.asarray(audio, dtype=np.float32))
    frames, channels = block.shape

    if frames < 1:
        raise ValueError("cannot send an empty block")
    if frames > 0xFFFF:
        raise ValueError(f"{frames} frames exceeds the u16 frame_count field")

    quality_codec, level = codec_for_quality(quality)
    if codec is None:
        codec = quality_codec

    if codec == CODEC_OPUS:
        raise CodecUnavailable(OPUS_UNAVAILABLE_REASON)
    if codec not in _CODEC_DTYPES:
        raise CodecUnavailable(f"Unknown codec {codec}")

    if codec == CODEC_PCM_INT16:
        # Clipped before the cast, not after: int16 wraps on overflow, so a +1.2 sample
        # would come out as a large negative one — a loud click rather than a soft clip.
        raw = (np.clip(block, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    else:
        raw = np.ascontiguousarray(block, dtype=np.float32).tobytes()

    flags = 0
    payload = raw
    if level:
        squeezed = zlib.compress(raw, level=level)
        if len(squeezed) < len(raw):
            payload = squeezed
            flags |= FLAG_ZLIB

    return UdpPacket(
        codec=codec,
        channels=channels,
        flags=flags,
        sample_rate=sample_rate,
        device_id=device_id,
        sequence=sequence % SEQUENCE_MODULUS,
        timestamp_us=time.monotonic_ns() // 1000 if timestamp_us is None else timestamp_us,
        frame_count=frames,
        payload=payload,
    )


class UdpAudioTransport:
    """
    One UDP socket, a peer list, and a receive thread that dispatches by device id.

    Composed alongside `NetworkAudioRouter` rather than derived from it: the two share no
    socket model (connectionless datagrams versus accepted streams) and no wire format, so
    a common base class would hold nothing but the quality enum, which is imported instead.

    Binds `127.0.0.1` by default. Binding `0.0.0.0` raises a Windows firewall prompt the
    first time, which is an unhelpful thing for a test suite to do; a caller that wants to
    be reachable from another machine passes its own address in.
    """

    def __init__(self, quality: NetworkQuality = NetworkQuality.HIGH, sample_rate: int = 48000):
        self.quality = quality
        self.sample_rate = sample_rate

        self._socket: socket.socket | None = None
        self._bound: tuple[str, int] | None = None
        self._peers: dict[str, tuple[str, int]] = {}
        self._handlers: dict[int, Callable[[UdpPacket], None]] = {}
        self._sequences: dict[int, int] = {}

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()

        # Every one of these is a count of something that happened. Nothing here is
        # derived, estimated, or initialised to a plausible-looking value.
        self.stats = {
            'packets_sent': 0,
            'packets_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'packets_rejected': 0,
            'send_errors': 0,
            'sends_with_no_peer': 0,
            'handler_errors': 0,
        }
        self.last_error: str | None = None

    # --- Lifecycle ---

    @property
    def is_running(self) -> bool:
        return self._socket is not None

    @property
    def bound_address(self) -> tuple[str, int] | None:
        """Where we are actually listening, or None. Never a guess at the requested port."""
        return self._bound

    def start(self, bind_host: str = '127.0.0.1', bind_port: int = 0) -> tuple[str, int]:
        """
        Open the socket and start receiving. Returns the address actually bound.

        Port 0 asks the OS for an ephemeral port, which is what a send-only peer wants —
        the returned tuple is the truth, so a caller never has to assume it got the port
        it asked for.
        """
        with self._lock:
            if self._socket is not None:
                return self._bound

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

                # Best-effort: widening the kernel's receive/send buffers is what stands
                # between a burst and a silent drop before our own code ever sees the
                # datagram (see SOCKET_BUFFER_BYTES). A platform or sandbox that refuses
                # this still gets a working socket at its own default size.
                for option in (socket.SO_RCVBUF, socket.SO_SNDBUF):
                    try:
                        sock.setsockopt(socket.SOL_SOCKET, option, SOCKET_BUFFER_BYTES)
                    except OSError as e:
                        logger.debug(f"Could not widen UDP socket buffer ({option}): {e}")

                sock.bind((bind_host, bind_port))
                # A timeout rather than a blocking recv, so `stop()` is noticed promptly
                # without needing to poke the socket from another thread.
                sock.settimeout(0.2)
            except OSError:
                sock.close()
                raise

            self._socket = sock
            self._bound = sock.getsockname()
            self._stop.clear()

            self._thread = threading.Thread(
                target=self._receive_loop, name='udp-audio-receive', daemon=True
            )
            self._thread.start()

            logger.info(f"UDP audio transport listening on {self._bound[0]}:{self._bound[1]}")
            return self._bound

    def stop(self):
        with self._lock:
            sock = self._socket
            thread = self._thread
            self._socket = None
            self._thread = None

        self._stop.set()

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass  # already closed or broken; nothing left to release

        with self._lock:
            self._bound = None

        logger.info("UDP audio transport stopped")

    # --- Peers ---

    def add_peer(self, name: str, host: str, port: int) -> str:
        with self._lock:
            self._peers[name] = (host, port)
        logger.info(f"UDP peer '{name}' = {host}:{port}")
        return name

    def remove_peer(self, name: str) -> bool:
        with self._lock:
            return self._peers.pop(name, None) is not None

    def peers(self) -> dict[str, tuple[str, int]]:
        with self._lock:
            return dict(self._peers)

    # --- Receive ---

    def register_handler(self, device_id: int, handler: Callable[[UdpPacket], None]):
        """Route incoming packets for one device id to `handler`, on the receive thread."""
        with self._lock:
            self._handlers[device_id] = handler

    def unregister_handler(self, device_id: int) -> bool:
        with self._lock:
            return self._handlers.pop(device_id, None) is not None

    def _receive_loop(self):
        while not self._stop.is_set():
            sock = self._socket
            if sock is None:
                break

            try:
                datagram, _address = sock.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError as e:
                if not self._stop.is_set():
                    # A dead socket is a real, reportable condition. Looping on it would
                    # spin a thread forever while claiming to be receiving.
                    self.last_error = f"receive failed: {e}"
                    logger.error(f"UDP receive stopped: {e}")
                break

            self.stats['bytes_received'] += len(datagram)

            try:
                packet = UdpPacket.unpack(datagram)
            except MalformedPacket as e:
                self.stats['packets_rejected'] += 1
                self.last_error = f"rejected datagram: {e}"
                continue

            self.stats['packets_received'] += 1

            handler = self._handlers.get(packet.device_id)
            if handler is None:
                continue

            try:
                handler(packet)
            except Exception as e:
                self.stats['handler_errors'] += 1
                self.last_error = f"receive handler for device {packet.device_id}: {e}"

    # --- Send ---

    def next_sequence(self, device_id: int) -> int:
        with self._lock:
            sequence = self._sequences.get(device_id, 0)
            self._sequences[device_id] = (sequence + 1) % SEQUENCE_MODULUS
            return sequence

    def send_packet(self, packet: UdpPacket, target: str | None = None) -> int:
        """
        Send to one named peer, or to every peer when `target` is None.

        Returns how many peers it actually reached. Zero is reported, not hidden: a send
        with no peers registered is the difference between "streaming" and "encoding audio
        into a void", and the caller deserves to know which.
        """
        sock = self._socket
        if sock is None:
            self.stats['send_errors'] += 1
            self.last_error = "send attempted while the transport was not running"
            return 0

        with self._lock:
            if target is None:
                addresses = list(self._peers.values())
            else:
                address = self._peers.get(target)
                addresses = [address] if address is not None else []

        if not addresses:
            self.stats['sends_with_no_peer'] += 1
            return 0

        datagram = packet.pack()
        delivered = 0

        for address in addresses:
            try:
                sock.sendto(datagram, address)
            except OSError as e:
                self.stats['send_errors'] += 1
                self.last_error = f"send to {address[0]}:{address[1]} failed: {e}"
                continue
            delivered += 1
            self.stats['bytes_sent'] += len(datagram)

        if delivered:
            self.stats['packets_sent'] += 1

        return delivered

    def send_block(
        self,
        device_id: int,
        audio: np.ndarray,
        target: str | None = None,
        quality: NetworkQuality | None = None,
    ) -> int:
        packet = encode_block(
            device_id=device_id,
            audio=audio,
            sample_rate=self.sample_rate,
            sequence=self.next_sequence(device_id),
            quality=self.quality if quality is None else quality,
        )
        return self.send_packet(packet, target)

    # --- Statistics ---

    def statistics(self) -> dict:
        """Measured counts only. Addresses are None when nothing is bound, not 0.0.0.0:0."""
        bound = self._bound
        return {
            'running': self.is_running,
            'bound_host': bound[0] if bound else None,
            'bound_port': bound[1] if bound else None,
            'quality': self.quality.value,
            'codec': CODEC_NAMES[codec_for_quality(self.quality)[0]],
            'peers': {name: f"{host}:{port}" for name, (host, port) in self.peers().items()},
            'registered_receivers': sorted(self._handlers),
            'opus_available': opus_available(),
            **self.stats,
            'last_error': self.last_error,
        }
