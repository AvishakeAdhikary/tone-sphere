from typing import List
from tonesphere.utils.logger import logger
import socket
import threading
import queue
import struct
import time

class NetworkAudioStreamer:
    """Handles audio streaming over network"""
    
    def __init__(self, port: int = 9001):
        self.port = port
        self.server_socket = None
        self.clients = {}  # client_id: socket
        self.is_streaming = False
        self.audio_queue = queue.Queue(maxsize=100)
        
    def start_server(self):
        """Start network audio server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(10)
            self.is_streaming = True
            
            # Start client handler thread
            threading.Thread(target=self._handle_clients, daemon=True).start()
            logger.info(f"Network audio server started on port {self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start network audio server: {e}")
    
    def _handle_clients(self):
        """Handle incoming client connections"""
        while self.is_streaming:
            try:
                client_socket, address = self.server_socket.accept()
                client_id = f"{address[0]}:{address[1]}"
                self.clients[client_id] = client_socket
                
                # Start client thread
                threading.Thread(
                    target=self._handle_client, 
                    args=(client_id, client_socket), 
                    daemon=True
                ).start()
                
                logger.info(f"Client connected: {client_id}")
                
            except Exception as e:
                if self.is_streaming:  # Only log if we're supposed to be running
                    logger.error(f"Error accepting client: {e}")
    
    def _handle_client(self, client_id: str, client_socket: socket.socket):
        """Handle individual client"""
        try:
            while self.is_streaming:
                # Send audio data to client
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get_nowait()
                    
                    # Send data length first, then data
                    data_length = len(audio_data)
                    client_socket.send(struct.pack('!I', data_length))
                    client_socket.send(audio_data)
                
                time.sleep(0.001)  # Small delay to prevent tight loop
                
        except Exception as e:
            logger.warning(f"Client {client_id} disconnected: {e}")
        finally:
            self._remove_client(client_id)
    
    def _remove_client(self, client_id: str):
        """Remove disconnected client"""
        if client_id in self.clients:
            try:
                self.clients[client_id].close()
            except:
                pass
            del self.clients[client_id]
            logger.info(f"Client removed: {client_id}")
    
    def broadcast_audio(self, audio_data: bytes):
        """Broadcast audio data to all connected clients"""
        if not self.audio_queue.full():
            self.audio_queue.put(audio_data)
    
    def stop_server(self):
        """Stop network audio server"""
        self.is_streaming = False
        
        # Close all client connections
        for client_id in list(self.clients.keys()):
            self._remove_client(client_id)
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
        
        logger.info("Network audio server stopped")
    
    def get_connected_clients(self) -> List[str]:
        """Get list of connected clients"""
        return list(self.clients.keys())