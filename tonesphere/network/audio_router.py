"""
Network audio streaming.

Scope, stated plainly: this is TCP with zlib, which is fine for moving audio between
machines where a little buffering is acceptable, and wrong for low-latency monitoring.
TCP retransmits and blocks the head of the line, so one lost packet stalls everything
behind it — exactly the wrong failure mode for realtime audio, which would rather drop a
packet than wait for it. A realtime version needs UDP with sequence numbers, a jitter
buffer and Opus. That is not what this is.

It is kept because it works for its actual use, and it is labelled so nobody plugs a
guitar through it and wonders why the latency is terrible.

The framing was broken until now: the header carried no payload length, so the receiver
retried zlib.decompress until it succeeded, consuming bytes from the following packet and
desynchronising the stream permanently.
"""

import json
import queue
import socket
import struct
import threading
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from tonesphere.utils.logger import logger

# Largest payload we will accept from a peer. A malformed or hostile header claiming a
# huge length would otherwise make the receiver allocate until the process dies.
MAX_PACKET_BYTES = 8 * 1024 * 1024


class AudioCodec(Enum):
    """Audio codec types"""
    PCM_FLOAT32 = "pcm_float32"
    PCM_INT16 = "pcm_int16"
    COMPRESSED = "compressed"


class NetworkQuality(Enum):
    """Network quality presets"""
    LOW = "low"          # High compression, low quality
    MEDIUM = "medium"    # Balanced
    HIGH = "high"        # Low compression, high quality
    LOSSLESS = "lossless"  # No compression


@dataclass
class AudioPacket:
    """Network audio packet"""
    device_id: int
    channels: int
    sample_rate: int
    frames: int
    codec: str
    timestamp: float
    data: bytes
    compressed: bool = False

    def to_bytes(self) -> bytes:
        """
        Serialise to the wire format.

        `data_length` is the important field. Without it the receiver cannot know where a
        packet ends: the original format wrote only the header length, so the reader had to
        guess the payload boundary by retrying zlib.decompress until it succeeded — which
        consumes bytes belonging to the next packet and desynchronises the stream for good.

        Wire format: [4-byte header length][JSON header][payload]
        """
        header = {
            'device_id': self.device_id,
            'channels': self.channels,
            'sample_rate': self.sample_rate,
            'frames': self.frames,
            'codec': self.codec,
            'timestamp': self.timestamp,
            'compressed': self.compressed,
            'data_length': len(self.data),
        }
        header_json = json.dumps(header).encode('utf-8')

        return struct.pack('!I', len(header_json)) + header_json + self.data

    @staticmethod
    def from_bytes(data: bytes) -> 'AudioPacket':
        """Deserialize packet from bytes"""
        header_length = struct.unpack('!I', data[:4])[0]
        header_json = data[4:4+header_length]
        header = json.loads(header_json.decode('utf-8'))
        audio_data = data[4+header_length:]

        return AudioPacket(
            device_id=header['device_id'],
            channels=header['channels'],
            sample_rate=header['sample_rate'],
            frames=header['frames'],
            codec=header['codec'],
            timestamp=header['timestamp'],
            data=audio_data,
            compressed=header.get('compressed', False)
        )


class NetworkAudioRouter:
    """
    Advanced network audio router
    Handles sending and receiving audio over network with routing
    """

    def __init__(self, port: int = 9001, quality: NetworkQuality = NetworkQuality.HIGH):
        self.port = port
        self.quality = quality
        self.is_running = False

        # Server components
        self.server_socket: socket.socket | None = None
        self.clients: dict[str, socket.socket] = {}
        self.client_info: dict[str, dict] = {}

        # Client components (for connecting to other instances)
        self.connections: dict[str, socket.socket] = {}

        # Audio routing
        self.send_queues: dict[int, queue.Queue] = {}  # device_id -> queue
        self.receive_callbacks: dict[int, Callable] = {}  # device_id -> callback

        # Statistics
        self.stats = {
            'packets_sent': 0,
            'packets_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'compression_ratio': 0.0
        }

        # Threading
        self.threads: list[threading.Thread] = []

    def start_server(self):
        """Start network audio server"""
        if self.is_running:
            return

        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(10)
            self.is_running = True

            # Start server thread
            server_thread = threading.Thread(target=self._server_loop, daemon=True)
            server_thread.start()
            self.threads.append(server_thread)

            logger.info(f"Network audio router started on port {self.port}")

        except Exception as e:
            logger.error(f"Failed to start network audio router: {e}")
            raise

    def stop_server(self):
        """Stop network audio server"""
        self.is_running = False

        # Close all client connections
        for client_id in list(self.clients.keys()):
            self._disconnect_client(client_id)

        # Close all outgoing connections
        for conn_id in list(self.connections.keys()):
            self.disconnect_from(conn_id)

        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except OSError:
                pass  # already closed or broken; nothing left to clean up

        logger.info("Network audio router stopped")

    def _server_loop(self):
        """Main server loop"""
        while self.is_running:
            try:
                client_socket, address = self.server_socket.accept()
                client_id = f"{address[0]}:{address[1]}"

                self.clients[client_id] = client_socket
                self.client_info[client_id] = {
                    'address': address,
                    'connected_at': time.time(),
                    'packets_received': 0
                }

                # Start client handler thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_id, client_socket),
                    daemon=True
                )
                client_thread.start()
                self.threads.append(client_thread)

                logger.info(f"Client connected: {client_id}")

            except Exception as e:
                if self.is_running:
                    logger.error(f"Error accepting client: {e}")

    def _handle_client(self, client_id: str, client_socket: socket.socket):
        """Handle individual client connection"""
        try:
            while self.is_running:
                # Receive packet
                packet = self._receive_packet(client_socket)
                if packet is None:
                    break

                # Update stats
                self.stats['packets_received'] += 1
                self.client_info[client_id]['packets_received'] += 1

                # Decode audio
                audio_data = self._decode_audio(packet)

                # Call receive callback if registered
                if packet.device_id in self.receive_callbacks:
                    self.receive_callbacks[packet.device_id](audio_data, packet)

        except Exception as e:
            logger.warning(f"Client {client_id} error: {e}")
        finally:
            self._disconnect_client(client_id)

    def _disconnect_client(self, client_id: str):
        """Disconnect a client"""
        if client_id in self.clients:
            try:
                self.clients[client_id].close()
            except OSError:
                pass  # already closed or broken; nothing left to clean up
            del self.clients[client_id]

        if client_id in self.client_info:
            del self.client_info[client_id]

        logger.info(f"Client disconnected: {client_id}")

    def connect_to(self, host: str, port: int) -> bool:
        """Connect to another ToneSphere instance"""
        conn_id = f"{host}:{port}"

        if conn_id in self.connections:
            logger.warning(f"Already connected to {conn_id}")
            return True

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            self.connections[conn_id] = sock

            # Start receive thread for this connection
            receive_thread = threading.Thread(
                target=self._receive_loop,
                args=(conn_id, sock),
                daemon=True
            )
            receive_thread.start()
            self.threads.append(receive_thread)

            logger.info(f"Connected to {conn_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to {conn_id}: {e}")
            return False

    def disconnect_from(self, conn_id: str):
        """Disconnect from a remote instance"""
        if conn_id in self.connections:
            try:
                self.connections[conn_id].close()
            except OSError:
                pass  # already closed or broken; nothing left to clean up
            del self.connections[conn_id]
            logger.info(f"Disconnected from {conn_id}")

    def _receive_loop(self, conn_id: str, sock: socket.socket):
        """Receive loop for outgoing connections"""
        try:
            while self.is_running and conn_id in self.connections:
                packet = self._receive_packet(sock)
                if packet is None:
                    break

                self.stats['packets_received'] += 1
                audio_data = self._decode_audio(packet)

                if packet.device_id in self.receive_callbacks:
                    self.receive_callbacks[packet.device_id](audio_data, packet)

        except Exception as e:
            logger.warning(f"Connection {conn_id} error: {e}")
        finally:
            self.disconnect_from(conn_id)

    def send_audio(self, device_id: int, audio_data: np.ndarray,
                   sample_rate: int, target: str | None = None):
        """
        Send audio over network

        Args:
            device_id: Source device ID
            audio_data: Audio data (frames, channels)
            sample_rate: Sample rate
            target: Target client/connection ID (None = broadcast)
        """
        try:
            # Create packet
            packet = self._encode_audio(device_id, audio_data, sample_rate)
            packet_bytes = packet.to_bytes()

            # Update stats
            self.stats['packets_sent'] += 1
            self.stats['bytes_sent'] += len(packet_bytes)

            # Send to target or broadcast
            if target:
                if target in self.connections:
                    self._send_packet(self.connections[target], packet_bytes)
            else:
                # Broadcast to all clients
                for client_socket in self.clients.values():
                    self._send_packet(client_socket, packet_bytes)

                # Broadcast to all connections
                for conn_socket in self.connections.values():
                    self._send_packet(conn_socket, packet_bytes)

        except Exception as e:
            logger.error(f"Error sending audio: {e}")

    def _encode_audio(self, device_id: int, audio_data: np.ndarray,
                     sample_rate: int) -> AudioPacket:
        """Encode audio data into packet"""
        frames, channels = audio_data.shape

        # Convert based on quality
        if self.quality == NetworkQuality.LOSSLESS:
            # Float32 PCM
            data_bytes = audio_data.astype(np.float32).tobytes()
            codec = AudioCodec.PCM_FLOAT32.value
            compressed = False
        elif self.quality == NetworkQuality.HIGH:
            # Float32 with light compression
            data_bytes = audio_data.astype(np.float32).tobytes()
            data_bytes = zlib.compress(data_bytes, level=3)
            codec = AudioCodec.PCM_FLOAT32.value
            compressed = True
        elif self.quality == NetworkQuality.MEDIUM:
            # Int16 with compression
            audio_int16 = (audio_data * 32767).astype(np.int16)
            data_bytes = audio_int16.tobytes()
            data_bytes = zlib.compress(data_bytes, level=6)
            codec = AudioCodec.PCM_INT16.value
            compressed = True
        else:  # LOW
            # Int16 with high compression
            audio_int16 = (audio_data * 32767).astype(np.int16)
            data_bytes = audio_int16.tobytes()
            data_bytes = zlib.compress(data_bytes, level=9)
            codec = AudioCodec.PCM_INT16.value
            compressed = True

        return AudioPacket(
            device_id=device_id,
            channels=channels,
            sample_rate=sample_rate,
            frames=frames,
            codec=codec,
            timestamp=time.time(),
            data=data_bytes,
            compressed=compressed
        )

    def _decode_audio(self, packet: AudioPacket) -> np.ndarray:
        """Decode audio packet"""
        data = packet.data

        # Decompress if needed
        if packet.compressed:
            data = zlib.decompress(data)

        # Decode based on codec
        if packet.codec == AudioCodec.PCM_FLOAT32.value:
            audio_data = np.frombuffer(data, dtype=np.float32)
        elif packet.codec == AudioCodec.PCM_INT16.value:
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_data = audio_int16.astype(np.float32) / 32767.0
        else:
            raise ValueError(f"Unknown codec: {packet.codec}")

        # Reshape
        audio_data = audio_data.reshape((packet.frames, packet.channels))
        return audio_data

    def _send_packet(self, sock: socket.socket, packet_bytes: bytes):
        """Send packet over socket"""
        try:
            sock.sendall(packet_bytes)
        except Exception as e:
            logger.error(f"Error sending packet: {e}")

    def _receive_packet(self, sock: socket.socket) -> AudioPacket | None:
        """Receive packet from socket"""
        try:
            # Receive header length
            header_length_bytes = self._recv_exact(sock, 4)
            if not header_length_bytes:
                return None

            header_length = struct.unpack('!I', header_length_bytes)[0]

            # Receive header
            header_bytes = self._recv_exact(sock, header_length)
            if not header_bytes:
                return None

            header = json.loads(header_bytes.decode('utf-8'))

            # Read exactly the number of bytes the sender declared.
            data_length = header.get('data_length')

            if data_length is None:
                # A sender predating the length field. Its payload boundary is
                # unknowable for compressed data, so refuse rather than corrupt the
                # stream by guessing.
                if header.get('compressed'):
                    logger.error(
                        "Peer is using the old framing, which had no payload length. "
                        "Compressed packets cannot be read reliably; upgrade the sender."
                    )
                    return None

                dtype_size = 4 if header['codec'] == AudioCodec.PCM_FLOAT32.value else 2
                data_length = header['frames'] * header['channels'] * dtype_size

            if data_length < 0 or data_length > MAX_PACKET_BYTES:
                # Guards against a malformed or hostile header claiming a huge payload,
                # which would otherwise make us allocate until the process dies.
                logger.error(f"Rejecting packet claiming {data_length} bytes")
                return None

            data_bytes = self._recv_exact(sock, data_length) if data_length else b''
            if data_length and not data_bytes:
                return None

            return AudioPacket(
                device_id=header['device_id'],
                channels=header['channels'],
                sample_rate=header['sample_rate'],
                frames=header['frames'],
                codec=header['codec'],
                timestamp=header['timestamp'],
                data=data_bytes,
                compressed=header.get('compressed', False)
            )

        except Exception as e:
            logger.error(f"Error receiving packet: {e}")
            return None

    def _recv_exact(self, sock: socket.socket, length: int) -> bytes | None:
        """Receive exact number of bytes"""
        data = b''
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def register_receive_callback(self, device_id: int, callback: Callable):
        """Register callback for receiving audio for a device"""
        self.receive_callbacks[device_id] = callback
        logger.info(f"Registered receive callback for device {device_id}")

    def unregister_receive_callback(self, device_id: int):
        """Unregister receive callback"""
        if device_id in self.receive_callbacks:
            del self.receive_callbacks[device_id]

    def get_connected_clients(self) -> list[str]:
        """Get list of connected clients"""
        return list(self.clients.keys())

    def get_connections(self) -> list[str]:
        """Get list of outgoing connections"""
        return list(self.connections.keys())

    def get_statistics(self) -> dict:
        """Get network statistics"""
        return {
            **self.stats,
            'connected_clients': len(self.clients),
            'outgoing_connections': len(self.connections),
            'quality': self.quality.value
        }
