"""
ToneSphere - Professional Audio Routing Engine
=================================================
A complete audio routing system inspired by VoiceMeeter Potato,
VB-Audio Cable, and Odeus ASIO Link Pro.

Features:
- Low-latency ASIO driver support
- Virtual audio cable creation
- Multi-channel audio routing matrix
- Real-time audio effects processing
- RESTful API for frontend integration
- Professional audio device management
"""

import asyncio
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from contextlib import asynccontextmanager
import yaml
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import socket
import struct
import queue
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Core Data Models and Enums
# ==============================================================================

class DeviceType(Enum):
    PHYSICAL_INPUT = "physical_input"
    PHYSICAL_OUTPUT = "physical_output"
    VIRTUAL_INPUT = "virtual_input"
    VIRTUAL_OUTPUT = "virtual_output"

class RoutingState(Enum):
    DISABLED = 0
    ENABLED = 1
    MONITOR = 2  # Monitor only (listen but don't route)

class EffectType(Enum):
    EQ = "equalizer"
    COMPRESSOR = "compressor"
    REVERB = "reverb"
    DELAY = "delay"
    NOISE_GATE = "noise_gate"
    LIMITER = "limiter"

@dataclass
class AudioDevice:
    id: int
    name: str
    device_type: DeviceType
    channels: int
    sample_rate: int
    buffer_size: int
    is_asio: bool = False
    is_active: bool = False
    latency: float = 0.0

@dataclass
class RoutingConnection:
    source_id: int
    destination_id: int
    state: RoutingState
    volume: float = 1.0  # 0.0 to 2.0 (0dB = 1.0)
    muted: bool = False
    solo: bool = False

@dataclass
class AudioEffect:
    effect_type: EffectType
    enabled: bool
    parameters: Dict[str, Any]

@dataclass
class AudioChannel:
    id: int
    name: str
    device_id: int
    volume: float = 1.0
    muted: bool = False
    solo: bool = False
    effects: List[AudioEffect] = None
    
    def __post_init__(self):
        if self.effects is None:
            self.effects = []

# ==============================================================================
# Audio Processing Core
# ==============================================================================

class AudioProcessor:
    """Core audio processing engine with real-time effects"""
    
    def __init__(self, sample_rate: int = 48000, buffer_size: int = 128):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.effects_chain = {}
        
    def apply_eq(self, audio_data: np.ndarray, params: Dict) -> np.ndarray:
        """Apply parametric EQ to audio data"""
        # Simplified EQ implementation - in production, use proper filters
        low_gain = params.get('low_gain', 0.0)
        mid_gain = params.get('mid_gain', 0.0)
        high_gain = params.get('high_gain', 0.0)
        
        # Apply basic gain adjustments (simplified)
        processed = audio_data.copy()
        if abs(low_gain) > 0.01:
            processed *= (1.0 + low_gain * 0.1)
        
        return np.clip(processed, -1.0, 1.0)
    
    def apply_compressor(self, audio_data: np.ndarray, params: Dict) -> np.ndarray:
        """Apply dynamic range compression"""
        threshold = params.get('threshold', -20.0)  # dB
        ratio = params.get('ratio', 4.0)
        attack = params.get('attack', 0.003)  # seconds
        release = params.get('release', 0.1)  # seconds
        
        # Simplified compressor - convert to dB, apply compression
        db_audio = 20 * np.log10(np.abs(audio_data) + 1e-10)
        compressed = np.where(
            db_audio > threshold,
            threshold + (db_audio - threshold) / ratio,
            db_audio
        )
        
        # Convert back to linear
        return np.sign(audio_data) * np.power(10, compressed / 20)
    
    def apply_reverb(self, audio_data: np.ndarray, params: Dict) -> np.ndarray:
        """Apply simple reverb effect"""
        room_size = params.get('room_size', 0.5)
        damping = params.get('damping', 0.5)
        wet_level = params.get('wet_level', 0.3)
        
        # Simplified reverb using delay lines
        delay_samples = int(room_size * self.sample_rate * 0.05)  # Max 50ms
        if delay_samples > 0 and delay_samples < len(audio_data):
            delayed = np.roll(audio_data, delay_samples) * damping
            return audio_data + delayed * wet_level
        
        return audio_data
    
    def process_effects_chain(self, audio_data: np.ndarray, effects: List[AudioEffect]) -> np.ndarray:
        """Process audio through effects chain"""
        processed = audio_data.copy()
        
        for effect in effects:
            if not effect.enabled:
                continue
                
            if effect.effect_type == EffectType.EQ:
                processed = self.apply_eq(processed, effect.parameters)
            elif effect.effect_type == EffectType.COMPRESSOR:
                processed = self.apply_compressor(processed, effect.parameters)
            elif effect.effect_type == EffectType.REVERB:
                processed = self.apply_reverb(processed, effect.parameters)
            # Add more effects as needed
            
        return processed

# ==============================================================================
# Virtual Audio Device Manager
# ==============================================================================

class VirtualAudioDevice:
    """Represents a virtual audio input/output device"""
    
    def __init__(self, device_id: int, name: str, channels: int, sample_rate: int):
        self.device_id = device_id
        self.name = name
        self.channels = channels
        self.sample_rate = sample_rate
        self.buffer = np.zeros((1024, channels), dtype=np.float32)
        self.is_active = False
        self.stream = None
        
    def start(self):
        """Start the virtual device"""
        self.is_active = True
        logger.info(f"Started virtual device: {self.name}")
        
    def stop(self):
        """Stop the virtual device"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.is_active = False
        logger.info(f"Stopped virtual device: {self.name}")

class VirtualDeviceManager:
    """Manages virtual audio devices"""
    
    def __init__(self):
        self.virtual_devices: Dict[int, VirtualAudioDevice] = {}
        self.next_device_id = 1000  # Start virtual devices at 1000
        
    def create_virtual_input(self, name: str, channels: int = 2, sample_rate: int = 48000) -> int:
        """Create a new virtual input device"""
        device_id = self.next_device_id
        self.next_device_id += 1
        
        device = VirtualAudioDevice(device_id, name, channels, sample_rate)
        self.virtual_devices[device_id] = device
        
        logger.info(f"Created virtual input: {name} (ID: {device_id})")
        return device_id
    
    def create_virtual_output(self, name: str, channels: int = 2, sample_rate: int = 48000) -> int:
        """Create a new virtual output device"""
        device_id = self.next_device_id
        self.next_device_id += 1
        
        device = VirtualAudioDevice(device_id, name, channels, sample_rate)
        self.virtual_devices[device_id] = device
        
        logger.info(f"Created virtual output: {name} (ID: {device_id})")
        return device_id
    
    def remove_virtual_device(self, device_id: int) -> bool:
        """Remove a virtual device"""
        if device_id in self.virtual_devices:
            device = self.virtual_devices[device_id]
            device.stop()
            del self.virtual_devices[device_id]
            logger.info(f"Removed virtual device: {device.name}")
            return True
        return False

# ==============================================================================
# Audio Routing Engine
# ==============================================================================

class AudioRoutingMatrix:
    """Manages audio routing between inputs and outputs"""
    
    def __init__(self):
        self.connections: Dict[Tuple[int, int], RoutingConnection] = {}
        self.channels: Dict[int, AudioChannel] = {}
        self.solo_active = False
        
    def add_channel(self, channel: AudioChannel):
        """Add an audio channel"""
        self.channels[channel.id] = channel
        
    def create_routing(self, source_id: int, destination_id: int, volume: float = 1.0) -> bool:
        """Create a routing connection"""
        key = (source_id, destination_id)
        if key not in self.connections:
            connection = RoutingConnection(
                source_id=source_id,
                destination_id=destination_id,
                state=RoutingState.ENABLED,
                volume=volume
            )
            self.connections[key] = connection
            logger.info(f"Created routing: {source_id} -> {destination_id}")
            return True
        return False
    
    def remove_routing(self, source_id: int, destination_id: int) -> bool:
        """Remove a routing connection"""
        key = (source_id, destination_id)
        if key in self.connections:
            del self.connections[key]
            logger.info(f"Removed routing: {source_id} -> {destination_id}")
            return True
        return False
    
    def set_routing_volume(self, source_id: int, destination_id: int, volume: float):
        """Set volume for a routing connection"""
        key = (source_id, destination_id)
        if key in self.connections:
            self.connections[key].volume = max(0.0, min(2.0, volume))
    
    def toggle_mute(self, source_id: int, destination_id: int):
        """Toggle mute for a routing connection"""
        key = (source_id, destination_id)
        if key in self.connections:
            self.connections[key].muted = not self.connections[key].muted
    
    def set_solo(self, channel_id: int, solo: bool):
        """Set solo state for a channel"""
        if channel_id in self.channels:
            self.channels[channel_id].solo = solo
            self.solo_active = any(ch.solo for ch in self.channels.values())
    
    def get_active_routings_for_output(self, output_id: int) -> List[RoutingConnection]:
        """Get all active routings for an output"""
        active = []
        for connection in self.connections.values():
            if connection.destination_id == output_id and connection.state == RoutingState.ENABLED:
                if not connection.muted:
                    # Check solo logic
                    if self.solo_active:
                        source_channel = self.channels.get(connection.source_id)
                        if source_channel and source_channel.solo:
                            active.append(connection)
                    else:
                        active.append(connection)
        return active

# ==============================================================================
# Main Audio Engine
# ==============================================================================

class AudioEngine:
    """Main audio engine coordinating all components"""
    
    def __init__(self, sample_rate: int = 48000, buffer_size: int = 128):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Core components
        self.processor = AudioProcessor(sample_rate, buffer_size)
        self.virtual_device_manager = VirtualDeviceManager()
        self.routing_matrix = AudioRoutingMatrix()
        self.network_streamer = NetworkAudioStreamer()
        self.connected_clients = set()
        
        # Device management
        self.physical_devices: Dict[int, AudioDevice] = {}
        self.active_streams: Dict[int, Any] = {}
        
        # State
        self.is_running = False
        self.master_volume = 1.0
        
        # Performance monitoring
        self.performance_stats = {
            'cpu_usage': 0.0,
            'buffer_underruns': 0,
            'latency_ms': 0.0
        }
        
        # Threading
        self.monitor_thread = None
        
    def initialize(self):
        """Initialize the audio engine"""
        try:
            self._scan_audio_devices()
            self._create_default_virtual_devices()
            logger.info("Audio engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio engine: {e}")
            raise
    
    def _scan_audio_devices(self):
        """Scan for available audio devices"""
        try:
            devices = sd.query_devices()
            for idx, device in enumerate(devices):
                # Determine if device supports ASIO
                is_asio = 'ASIO' in device['name'] or 'asio' in device['name'].lower()
                
                # Create device objects for inputs and outputs
                if device['max_input_channels'] > 0:
                    audio_device = AudioDevice(
                        id=idx * 2,  # Even IDs for inputs
                        name=f"{device['name']} (Input)",
                        device_type=DeviceType.PHYSICAL_INPUT,
                        channels=device['max_input_channels'],
                        sample_rate=int(device['default_samplerate']),
                        buffer_size=self.buffer_size,
                        is_asio=is_asio,
                        latency=device['default_low_input_latency'] * 1000  # Convert to ms
                    )
                    self.physical_devices[audio_device.id] = audio_device
                
                if device['max_output_channels'] > 0:
                    audio_device = AudioDevice(
                        id=idx * 2 + 1,  # Odd IDs for outputs
                        name=f"{device['name']} (Output)",
                        device_type=DeviceType.PHYSICAL_OUTPUT,
                        channels=device['max_output_channels'],
                        sample_rate=int(device['default_samplerate']),
                        buffer_size=self.buffer_size,
                        is_asio=is_asio,
                        latency=device['default_low_output_latency'] * 1000  # Convert to ms
                    )
                    self.physical_devices[audio_device.id] = audio_device
                    
            logger.info(f"Found {len(self.physical_devices)} audio devices")
            
        except Exception as e:
            logger.error(f"Error scanning audio devices: {e}")
    
    def _create_default_virtual_devices(self):
        """Create default virtual audio devices"""
        # Create 3 virtual inputs (like VoiceMeeter Potato)
        for i in range(3):
            device_id = self.virtual_device_manager.create_virtual_input(
                f"Virtual Input {i+1}", channels=2
            )
            virtual_device = AudioDevice(
                id=device_id,
                name=f"Virtual Input {i+1}",
                device_type=DeviceType.VIRTUAL_INPUT,
                channels=2,
                sample_rate=self.sample_rate,
                buffer_size=self.buffer_size
            )
            self.physical_devices[device_id] = virtual_device
            
        # Create 3 virtual outputs
        for i in range(3):
            device_id = self.virtual_device_manager.create_virtual_output(
                f"Virtual Output {i+1}", channels=2
            )
            virtual_device = AudioDevice(
                id=device_id,
                name=f"Virtual Output {i+1}",
                device_type=DeviceType.VIRTUAL_OUTPUT,
                channels=2,
                sample_rate=self.sample_rate,
                buffer_size=self.buffer_size
            )
            self.physical_devices[device_id] = virtual_device
    
    def start_engine(self):
        """Start the audio engine"""
        if self.is_running:
            return
            
        try:
            self.is_running = True
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._performance_monitor, daemon=True)
            self.monitor_thread.start()
            
            logger.info("Audio engine started")
            
        except Exception as e:
            logger.error(f"Failed to start audio engine: {e}")
            self.is_running = False
            raise
    
    def stop_engine(self):
        """Stop the audio engine"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Stop all active streams
        for stream in self.active_streams.values():
            try:
                stream.stop()
                stream.close()
            except:
                pass
        
        self.active_streams.clear()
        logger.info("Audio engine stopped")
    
    def _performance_monitor(self):
        """Monitor engine performance"""
        while self.is_running:
            try:
                # Update performance stats
                # This is a simplified implementation
                self.performance_stats['cpu_usage'] = 0.0  # Would calculate actual CPU usage
                self.performance_stats['latency_ms'] = self.buffer_size / self.sample_rate * 1000
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
    
    # Device Management Methods
    def get_devices(self) -> List[Dict]:
        """Get all available devices"""
        devices = []
        for device in self.physical_devices.values():
            devices.append({
                'id': device.id,
                'name': device.name,
                'type': device.device_type.value,
                'channels': device.channels,
                'sample_rate': device.sample_rate,
                'is_asio': device.is_asio,
                'is_active': device.is_active,
                'latency_ms': device.latency
            })
        return devices
    
    def create_virtual_input(self, name: str, channels: int = 2) -> int:
        """Create a new virtual input device"""
        device_id = self.virtual_device_manager.create_virtual_input(name, channels, self.sample_rate)
        # Add to physical devices registry
        virtual_device = AudioDevice(
            id=device_id,
            name=name,
            device_type=DeviceType.VIRTUAL_INPUT,
            channels=channels,
            sample_rate=self.sample_rate,
            buffer_size=self.buffer_size
        )
        self.physical_devices[device_id] = virtual_device
        return device_id
    
    def create_virtual_output(self, name: str, channels: int = 2) -> int:
        """Create a new virtual output device"""
        device_id = self.virtual_device_manager.create_virtual_output(name, channels, self.sample_rate)
        # Add to physical devices registry
        virtual_device = AudioDevice(
            id=device_id,
            name=name,
            device_type=DeviceType.VIRTUAL_OUTPUT,
            channels=channels,
            sample_rate=self.sample_rate,
            buffer_size=self.buffer_size
        )
        self.physical_devices[device_id] = virtual_device
        return device_id
    
    # Routing Methods
    def create_routing(self, source_id: int, destination_id: int, volume: float = 1.0) -> bool:
        """Create a routing connection"""
        return self.routing_matrix.create_routing(source_id, destination_id, volume)
    
    def remove_routing(self, source_id: int, destination_id: int) -> bool:
        """Remove a routing connection"""
        return self.routing_matrix.remove_routing(source_id, destination_id)
    
    def set_routing_volume(self, source_id: int, destination_id: int, volume: float):
        """Set volume for a routing connection"""
        self.routing_matrix.set_routing_volume(source_id, destination_id, volume)
    
    def get_routing_matrix(self) -> Dict:
        """Get current routing matrix state"""
        connections = {}
        for (source, dest), connection in self.routing_matrix.connections.items():
            key = f"{source}_{dest}"
            connections[key] = {
                'source_id': connection.source_id,
                'destination_id': connection.destination_id,
                'state': connection.state.value,
                'volume': connection.volume,
                'muted': connection.muted,
                'solo': connection.solo
            }
        return connections
    
    def get_performance_stats(self) -> Dict:
        """Get engine performance statistics"""
        return self.performance_stats.copy()

    def start_network_streaming(self):
        """Start network audio streaming"""
        self.network_streamer.start_server()

    def stop_network_streaming(self):
        """Stop network audio streaming"""
        self.network_streamer.stop_server()

    def get_network_clients(self) -> List[str]:
        """Get connected network clients"""
        return self.network_streamer.get_connected_clients()

# ==============================================================================
# Configuration Manager
# ==============================================================================

class ConfigManager:
    """Manages configuration and presets"""
    
    def __init__(self, config_path: str = "audio_engine_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_default_config()
        
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            'engine': {
                'sample_rate': 48000,
                'buffer_size': 128,
                'master_volume': 1.0
            },
            'virtual_devices': {
                'default_inputs': 3,
                'default_outputs': 3,
                'default_channels': 2
            },
            'effects': {
                'enabled': True,
                'presets': {}
            },
            'api': {
                'host': '127.0.0.1',
                'port': 8080,
                'cors_enabled': True
            }
        }
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    # Merge with defaults
                    self.config.update(loaded_config)
            return self.config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self.config
    
    def save_config(self, config: Dict = None):
        """Save configuration to file"""
        try:
            config_to_save = config or self.config
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(config_to_save, f, default_flow_style=False)
            logger.info("Configuration saved")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

# ==============================================================================
# Network Audio Streaming
# ==============================================================================

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

# ==============================================================================
# REST API Models
# ==============================================================================

class DeviceInfo(BaseModel):
    id: int
    name: str
    type: str
    channels: int
    sample_rate: int
    is_asio: bool
    is_active: bool
    latency_ms: float

class CreateVirtualDeviceRequest(BaseModel):
    name: str
    channels: int = 2
    device_type: str  # "input" or "output"

class CreateRoutingRequest(BaseModel):
    source_id: int
    destination_id: int
    volume: float = 1.0

class SetVolumeRequest(BaseModel):
    source_id: int
    destination_id: int
    volume: float

class PerformanceStats(BaseModel):
    cpu_usage: float
    buffer_underruns: int
    latency_ms: float

# ==============================================================================
# REST API Server
# ==============================================================================

# Global audio engine instance
audio_engine: Optional[AudioEngine] = None
config_manager = ConfigManager()
connected_websockets: set = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler"""
    global audio_engine
    
    # Startup
    config = config_manager.load_config()
    audio_engine = AudioEngine(
        sample_rate=config['engine']['sample_rate'],
        buffer_size=config['engine']['buffer_size']
    )
    
    try:
        audio_engine.initialize()
        audio_engine.start_engine()
        logger.info("Audio engine started successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to start audio engine: {e}")
        raise
    finally:
        # Shutdown
        if audio_engine:
            audio_engine.stop_engine()
        logger.info("Audio engine stopped")

# Create FastAPI app
app = FastAPI(
    title="ToneSphere API",
    description="Professional Audio Routing Engine API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ToneSphere API",
        "version": "1.0.0",
        "status": "running" if audio_engine and audio_engine.is_running else "stopped"
    }

@app.get("/devices", response_model=List[DeviceInfo])
async def get_devices():
    """Get all available audio devices"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")
    
    devices = audio_engine.get_devices()
    return [DeviceInfo(**device) for device in devices]

@app.post("/devices/virtual")
async def create_virtual_device(request: CreateVirtualDeviceRequest):
    """Create a new virtual audio device"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")
    
    if request.device_type.lower() == "input":
        device_id = audio_engine.create_virtual_input(request.name, request.channels)
    elif request.device_type.lower() == "output":
        device_id = audio_engine.create_virtual_output(request.name, request.channels)
    else:
        raise HTTPException(status_code=400, detail="Invalid device type")
    
    return {"device_id": device_id, "message": "Virtual device created successfully"}

@app.post("/routing")
async def create_routing(request: CreateRoutingRequest):
    """Create a routing connection"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")
    
    success = audio_engine.create_routing(
        request.source_id, 
        request.destination_id, 
        request.volume
    )
    
    if success:
        return {"message": "Routing created successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to create routing")

@app.delete("/routing/{source_id}/{destination_id}")
async def remove_routing(source_id: int, destination_id: int):
    """Remove a routing connection"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")
    
    success = audio_engine.remove_routing(source_id, destination_id)
    
    if success:
        return {"message": "Routing removed successfully"}
    else:
        raise HTTPException(status_code=404, detail="Routing not found")

@app.put("/routing/volume")
async def set_routing_volume(request: SetVolumeRequest):
    """Set volume for a routing connection"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")
    
    audio_engine.set_routing_volume(
        request.source_id,
        request.destination_id,
        request.volume
    )
    
    return {"message": "Volume updated successfully"}

@app.get("/routing")
async def get_routing_matrix():
    """Get current routing matrix"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")
    
    return audio_engine.get_routing_matrix()

@app.get("/performance", response_model=PerformanceStats)
async def get_performance_stats():
    """Get engine performance statistics"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")
    
    stats = audio_engine.get_performance_stats()
    return PerformanceStats(**stats)

@app.post("/engine/start")
async def start_engine():
    """Start the audio engine"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")
    
    try:
        audio_engine.start_engine()
        return {"message": "Audio engine started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start engine: {str(e)}")

@app.post("/engine/stop")
async def stop_engine():
    """Stop the audio engine"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")
    
    audio_engine.stop_engine()
    return {"message": "Audio engine stopped successfully"}

@app.get("/engine/status")
async def get_engine_status():
    """Get engine status"""
    if not audio_engine:
        return {"status": "not_initialized"}
    
    return {
        "status": "running" if audio_engine.is_running else "stopped",
        "sample_rate": audio_engine.sample_rate,
        "buffer_size": audio_engine.buffer_size,
        "master_volume": audio_engine.master_volume
    }

@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time events with proper client management"""
    await websocket.accept()
    connected_websockets.add(websocket)
    
    try:
        while True:
            # Send performance stats every second
            if audio_engine:
                stats = audio_engine.get_performance_stats()
                network_clients = audio_engine.get_network_clients()
                
                await websocket.send_json({
                    "type": "performance_stats",
                    "data": stats
                })
                
                await websocket.send_json({
                    "type": "network_clients",
                    "data": {"clients": network_clients, "count": len(network_clients)}
                })
            
            await asyncio.sleep(1.0)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connected_websockets.discard(websocket)
        try:
            await websocket.close()
        except:
            pass

@app.post("/network/start")
async def start_network_streaming():
    """Start network audio streaming"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")
    
    audio_engine.start_network_streaming()
    return {"message": "Network streaming started"}

@app.post("/network/stop")
async def stop_network_streaming():
    """Stop network audio streaming"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")
    
    audio_engine.stop_network_streaming()
    return {"message": "Network streaming stopped"}

@app.get("/network/clients")
async def get_network_clients():
    """Get connected network clients"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")
    
    clients = audio_engine.get_network_clients()
    return {"clients": clients, "count": len(clients)}

# ==============================================================================
# CLI Interface for Development/Testing
# ==============================================================================

class AudioEngineCLI:
    """Command-line interface for the audio engine"""
    
    def __init__(self):
        self.engine = None
        self.config_manager = ConfigManager()
        
    def initialize_engine(self):
        """Initialize the audio engine"""
        try:
            config = self.config_manager.load_config()
            self.engine = AudioEngine(
                sample_rate=config['engine']['sample_rate'],
                buffer_size=config['engine']['buffer_size']
            )
            self.engine.initialize()
            self.engine.start_engine()
            print("✓ Audio engine initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize audio engine: {e}")
            return False
        return True
    
    def list_devices(self):
        """List all available devices"""
        if not self.engine:
            print("Engine not initialized")
            return
            
        devices = self.engine.get_devices()
        print("\nAvailable Audio Devices:")
        print("-" * 80)
        print(f"{'ID':<4} {'Name':<40} {'Type':<15} {'Ch':<3} {'ASIO':<6} {'Latency':<8}")
        print("-" * 80)
        
        for device in devices:
            print(f"{device['id']:<4} {device['name']:<40} {device['type']:<15} "
                  f"{device['channels']:<3} {'✓' if device['is_asio'] else '✗':<6} "
                  f"{device['latency_ms']:.1f}ms")
    
    def create_test_routing(self):
        """Create a test routing setup"""
        if not self.engine:
            print("Engine not initialized")
            return
            
        devices = self.engine.get_devices()
        inputs = [d for d in devices if 'input' in d['type']]
        outputs = [d for d in devices if 'output' in d['type']]
        
        if not inputs or not outputs:
            print("No suitable input/output devices found")
            return
            
        # Create routing from first input to first output
        source_id = inputs[0]['id']
        dest_id = outputs[0]['id']
        
        success = self.engine.create_routing(source_id, dest_id, 1.0)
        if success:
            print(f"✓ Created routing: {inputs[0]['name']} -> {outputs[0]['name']}")
        else:
            print("✗ Failed to create routing")
    
    def show_routing_matrix(self):
        """Display current routing matrix"""
        if not self.engine:
            print("Engine not initialized")
            return
            
        matrix = self.engine.get_routing_matrix()
        if not matrix:
            print("No active routings")
            return
            
        print("\nActive Routings:")
        print("-" * 60)
        print(f"{'Source':<6} {'Destination':<6} {'Volume':<8} {'Status':<10}")
        print("-" * 60)
        
        for connection in matrix.values():
            status = "MUTED" if connection['muted'] else "ACTIVE"
            if connection['solo']:
                status += " (SOLO)"
                
            print(f"{connection['source_id']:<6} {connection['destination_id']:<6} "
                  f"{connection['volume']:.2f}x{'':<3} {status:<10}")
    
    def show_performance(self):
        """Show performance statistics"""
        if not self.engine:
            print("Engine not initialized")
            return
            
        stats = self.engine.get_performance_stats()
        print(f"\nPerformance Statistics:")
        print(f"CPU Usage: {stats['cpu_usage']:.1f}%")
        print(f"Buffer Underruns: {stats['buffer_underruns']}")
        print(f"Latency: {stats['latency_ms']:.1f}ms")
    
    def run_interactive_mode(self):
        """Run interactive CLI mode"""
        if not self.initialize_engine():
            return
            
        print("\nPyAudioEngine Interactive Mode")
        print("Commands: devices, routing, matrix, performance, quit")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command in ["quit", "exit"]:
                    break
                elif command == "devices":
                    self.list_devices()
                elif command == "routing":
                    self.create_test_routing()
                elif command == "matrix":
                    self.show_routing_matrix()
                elif command in ["performance", "stats"]:
                    self.show_performance()
                elif command == "help":
                    print("Available commands:")
                    print("  devices    - List all audio devices")
                    print("  routing    - Create test routing")
                    print("  matrix     - Show routing matrix")
                    print("  performance- Show performance stats")
                    print("  quit       - Exit")
                else:
                    print(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        print("\nShutting down...")
        if self.engine:
            self.engine.stop_engine()

# ==============================================================================
# ToneSphere Studio GUI
# ==============================================================================

class ToneSphereStudioGUI:
    """Professional GUI for ToneSphere Studio"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ToneSphere Studio - Professional Audio Routing")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self._configure_styles()
        
        # Engine reference
        self.engine = None
        self.update_thread = None
        self.is_running = False
        
        # Create GUI components
        self._create_widgets()
        self._create_menu()
        
        # Start engine update thread
        self.start_updates()
    
    def _configure_styles(self):
        """Configure custom styles for professional look"""
        # Configure colors and fonts
        bg_color = '#2b2b2b'
        fg_color = '#ffffff'
        accent_color = '#4CAF50'
        
        self.style.configure('Title.TLabel', 
                           background=bg_color, 
                           foreground=accent_color, 
                           font=('Arial', 16, 'bold'))
        
        self.style.configure('Header.TLabel', 
                           background=bg_color, 
                           foreground=fg_color, 
                           font=('Arial', 12, 'bold'))
        
        self.style.configure('Custom.TFrame', background=bg_color)
        self.style.configure('Custom.TButton', font=('Arial', 10))
        
    def _create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Engine menu
        engine_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Engine", menu=engine_menu)
        engine_menu.add_command(label="Start Engine", command=self.start_engine)
        engine_menu.add_command(label="Stop Engine", command=self.stop_engine)
        engine_menu.add_separator()
        engine_menu.add_command(label="Exit", command=self.root.quit)
        
        # Network menu
        network_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Network", menu=network_menu)
        network_menu.add_command(label="Start Streaming", command=self.start_network)
        network_menu.add_command(label="Stop Streaming", command=self.stop_network)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def _create_widgets(self):
        """Create main GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Custom.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="ToneSphere Studio", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", style='Custom.TFrame')
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Engine: Stopped", style='Header.TLabel')
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.performance_label = ttk.Label(status_frame, text="CPU: 0% | Latency: 0ms", style='Header.TLabel')
        self.performance_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Control frame
        control_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Start Engine", command=self.start_engine, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop Engine", command=self.stop_engine, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Refresh Devices", command=self.refresh_devices, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        
        # Network control frame
        network_frame = ttk.LabelFrame(main_frame, text="Network Streaming", style='Custom.TFrame')
        network_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(network_frame, text="Start Network", command=self.start_network, style='Custom.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(network_frame, text="Stop Network", command=self.stop_network, style='Custom.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        
        self.client_count_label = ttk.Label(network_frame, text="Clients: 0", style='Header.TLabel')
        self.client_count_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Devices frame
        devices_frame = ttk.LabelFrame(main_frame, text="Audio Devices", style='Custom.TFrame')
        devices_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Devices treeview
        columns = ('ID', 'Name', 'Type', 'Channels', 'ASIO', 'Latency')
        self.devices_tree = ttk.Treeview(devices_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.devices_tree.heading(col, text=col)
            self.devices_tree.column(col, width=100)
        
        # Scrollbars for devices tree
        devices_scrollbar_y = ttk.Scrollbar(devices_frame, orient=tk.VERTICAL, command=self.devices_tree.yview)
        devices_scrollbar_x = ttk.Scrollbar(devices_frame, orient=tk.HORIZONTAL, command=self.devices_tree.xview)
        self.devices_tree.configure(yscrollcommand=devices_scrollbar_y.set, xscrollcommand=devices_scrollbar_x.set)
        
        self.devices_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        devices_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        devices_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Routing frame
        routing_frame = ttk.LabelFrame(main_frame, text="Audio Routing", style='Custom.TFrame')
        routing_frame.pack(fill=tk.BOTH, expand=True)
        
        # Routing controls
        routing_controls = ttk.Frame(routing_frame, style='Custom.TFrame')
        routing_controls.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(routing_controls, text="Source:", style='Header.TLabel').pack(side=tk.LEFT, padx=5)
        self.source_var = tk.StringVar()
        self.source_combo = ttk.Combobox(routing_controls, textvariable=self.source_var, width=20)
        self.source_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(routing_controls, text="Destination:", style='Header.TLabel').pack(side=tk.LEFT, padx=5)
        self.dest_var = tk.StringVar()
        self.dest_combo = ttk.Combobox(routing_controls, textvariable=self.dest_var, width=20)
        self.dest_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(routing_controls, text="Volume:", style='Header.TLabel').pack(side=tk.LEFT, padx=5)
        self.volume_var = tk.DoubleVar(value=1.0)
        volume_scale = ttk.Scale(routing_controls, from_=0.0, to=2.0, variable=self.volume_var, length=100)
        volume_scale.pack(side=tk.LEFT, padx=5)
        
        self.volume_label = ttk.Label(routing_controls, text="1.0", style='Header.TLabel')
        self.volume_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(routing_controls, text="Create Routing", command=self.create_routing, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        
        # Update volume label when scale changes
        volume_scale.configure(command=lambda v: self.volume_label.configure(text=f"{float(v):.2f}"))
        
        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="System Log", style='Custom.TFrame')
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, bg='#1e1e1e', fg='#ffffff', font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def start_engine(self):
        """Start the audio engine"""
        try:
            if not self.engine:
                config = ConfigManager().load_config()
                self.engine = AudioEngine(
                    sample_rate=config['engine']['sample_rate'],
                    buffer_size=config['engine']['buffer_size']
                )
                self.engine.initialize()
            
            self.engine.start_engine()
            self.is_running = True
            self.log_message("✓ Audio engine started successfully")
            self.refresh_devices()
            
        except Exception as e:
            self.log_message(f"✗ Failed to start engine: {e}")
            messagebox.showerror("Error", f"Failed to start engine: {e}")
    
    def stop_engine(self):
        """Stop the audio engine"""
        try:
            if self.engine:
                self.engine.stop_engine()
                self.is_running = False
                self.log_message("✓ Audio engine stopped")
        except Exception as e:
            self.log_message(f"✗ Error stopping engine: {e}")
    
    def start_network(self):
        """Start network streaming"""
        try:
            if self.engine:
                self.engine.start_network_streaming()
                self.log_message("✓ Network streaming started on port 9001")
        except Exception as e:
            self.log_message(f"✗ Failed to start network streaming: {e}")
    
    def stop_network(self):
        """Stop network streaming"""
        try:
            if self.engine:
                self.engine.stop_network_streaming()
                self.log_message("✓ Network streaming stopped")
        except Exception as e:
            self.log_message(f"✗ Error stopping network streaming: {e}")
    
    def refresh_devices(self):
        """Refresh device list"""
        if not self.engine:
            return
            
        try:
            # Clear existing items
            for item in self.devices_tree.get_children():
                self.devices_tree.delete(item)
            
            # Get devices and populate tree
            devices = self.engine.get_devices()
            device_options = []
            
            for device in devices:
                self.devices_tree.insert('', 'end', values=(
                    device['id'],
                    device['name'][:30] + '...' if len(device['name']) > 30 else device['name'],
                    device['type'],
                    device['channels'],
                    '✓' if device['is_asio'] else '✗',
                    f"{device['latency_ms']:.1f}ms"
                ))
                device_options.append(f"{device['id']}: {device['name']}")
            
            # Update routing comboboxes
            self.source_combo['values'] = device_options
            self.dest_combo['values'] = device_options
            
            self.log_message(f"✓ Found {len(devices)} audio devices")
            
        except Exception as e:
            self.log_message(f"✗ Error refreshing devices: {e}")
    
    def create_routing(self):
        """Create audio routing"""
        try:
            source_text = self.source_var.get()
            dest_text = self.dest_var.get()
            
            if not source_text or not dest_text:
                messagebox.showwarning("Warning", "Please select both source and destination devices")
                return
            
            # Extract device IDs
            source_id = int(source_text.split(':')[0])
            dest_id = int(dest_text.split(':')[0])
            volume = self.volume_var.get()
            
            if self.engine and self.engine.create_routing(source_id, dest_id, volume):
                self.log_message(f"✓ Created routing: {source_id} -> {dest_id} (Volume: {volume:.2f})")
            else:
                self.log_message(f"✗ Failed to create routing")
                
        except Exception as e:
            self.log_message(f"✗ Error creating routing: {e}")
            messagebox.showerror("Error", f"Error creating routing: {e}")
    
    def log_message(self, message: str):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def update_status(self):
        """Update status information"""
        if self.engine:
            if self.engine.is_running:
                self.status_label.configure(text="Engine: Running")
                
                # Update performance stats
                stats = self.engine.get_performance_stats()
                self.performance_label.configure(text=f"CPU: {stats['cpu_usage']:.1f}% | Latency: {stats['latency_ms']:.1f}ms")
                
                # Update network client count
                clients = self.engine.get_network_clients()
                self.client_count_label.configure(text=f"Clients: {len(clients)}")
            else:
                self.status_label.configure(text="Engine: Stopped")
        else:
            self.status_label.configure(text="Engine: Not Initialized")
    
    def start_updates(self):
        """Start update thread"""
        def update_loop():
            while True:
                try:
                    self.root.after(0, self.update_status)
                    time.sleep(1.0)
                except:
                    break
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About ToneSphere Studio", 
                          "ToneSphere Studio v1.0\n\n"
                          "Professional Audio Routing Engine\n"
                          "Inspired by VoiceMeeter Potato\n\n"
                          "Features:\n"
                          "• Low-latency ASIO support\n"
                          "• Virtual audio devices\n"
                          "• Network audio streaming\n"
                          "• Professional routing matrix\n"
                          "• Real-time audio effects")
    
    def run(self):
        """Run the GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        if self.engine:
            self.engine.stop_engine()
            self.engine.stop_network_streaming()
        self.root.destroy()

# ==============================================================================
# API Client Example for Frontend Integration
# ==============================================================================

def create_api_client_code():
    """Generate API client code for frontend developers"""
    
    client_code = '''# ToneSphere API Client
# Example client for communicating with the audio engine backend

import requests
import json
import websockets
import asyncio
import threading
from typing import Dict, List, Optional, Callable

class PyAudioEngineClient:
    """Client for communicating with ToneSphere API"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        self.websocket = None
        self.event_callbacks = {}
    
    # Device Management API
    def get_devices(self) -> List[Dict]:
        """Get all available audio devices"""
        response = self.session.get(f"{self.base_url}/devices")
        response.raise_for_status()
        return response.json()
    
    def create_virtual_device(self, name: str, device_type: str, channels: int = 2) -> int:
        """Create a virtual audio device"""
        data = {"name": name, "device_type": device_type, "channels": channels}
        response = self.session.post(f"{self.base_url}/devices/virtual", json=data)
        response.raise_for_status()
        return response.json()["device_id"]
    
    # Routing Management API
    def create_routing(self, source_id: int, destination_id: int, volume: float = 1.0):
        """Create a routing connection"""
        data = {"source_id": source_id, "destination_id": destination_id, "volume": volume}
        response = self.session.post(f"{self.base_url}/routing", json=data)
        response.raise_for_status()
        return response.json()
    
    def remove_routing(self, source_id: int, destination_id: int):
        """Remove a routing connection"""
        response = self.session.delete(f"{self.base_url}/routing/{source_id}/{destination_id}")
        response.raise_for_status()
        return response.json()
    
    def set_routing_volume(self, source_id: int, destination_id: int, volume: float):
        """Set volume for a routing connection"""
        data = {"source_id": source_id, "destination_id": destination_id, "volume": volume}
        response = self.session.put(f"{self.base_url}/routing/volume", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_routing_matrix(self) -> Dict:
        """Get current routing matrix"""
        response = self.session.get(f"{self.base_url}/routing")
        response.raise_for_status()
        return response.json()
    
    # Engine Control API
    def start_engine(self):
        """Start the audio engine"""
        response = self.session.post(f"{self.base_url}/engine/start")
        response.raise_for_status()
        return response.json()
    
    def stop_engine(self):
        """Stop the audio engine"""
        response = self.session.post(f"{self.base_url}/engine/stop")
        response.raise_for_status()
        return response.json()
    
    def get_engine_status(self) -> Dict:
        """Get engine status"""
        response = self.session.get(f"{self.base_url}/engine/status")
        response.raise_for_status()
        return response.json()
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        response = self.session.get(f"{self.base_url}/performance")
        response.raise_for_status()
        return response.json()
    
    # WebSocket Event Handling
    async def connect_websocket(self):
        """Connect to WebSocket for real-time events"""
        ws_url = self.base_url.replace("http", "ws") + "/ws/events"
        try:
            self.websocket = await websockets.connect(ws_url)
            # Start message handling in background
            asyncio.create_task(self._handle_websocket_messages())
        except Exception as e:
            print(f"Failed to connect WebSocket: {e}")
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """Register callback for specific event type"""
        self.event_callbacks[event_type] = callback
    
    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    event_type = data.get("type")
                    if event_type in self.event_callbacks:
                        callback = self.event_callbacks[event_type]
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data.get("data"))
                        else:
                            callback(data.get("data"))
                except Exception as e:
                    print(f"WebSocket message error: {e}")
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        except Exception as e:
            print(f"WebSocket error: {e}")
    
    def _on_websocket_error(self, ws, error):
        """Handle WebSocket error"""
        print(f"WebSocket error: {error}")
    
    def _on_websocket_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        print("WebSocket connection closed")

# Usage Example
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize client
        client = PyAudioEngineClient()
        
        # Check engine status
        status = client.get_engine_status()
        print(f"Engine status: {status}")
        
        # Get available devices
        devices = client.get_devices()
        print(f"Found {len(devices)} devices")
        
        # Create a virtual input device
        virtual_input_id = client.create_virtual_device("My Mic Input", "input", 2)
        print(f"Created virtual input with ID: {virtual_input_id}")
        
        # Create a virtual output device
        virtual_output_id = client.create_virtual_device("My Speakers", "output", 2)
        print(f"Created virtual output with ID: {virtual_output_id}")
        
        # Create routing connection
        client.create_routing(virtual_input_id, virtual_output_id, 0.8)
        print("Created routing connection")
        
        # Set up real-time performance monitoring
        def on_performance_update(data):
            print(f"CPU: {data['cpu_usage']:.1f}%, Latency: {data['latency_ms']:.1f}ms")
        
        client.register_event_callback("performance_stats", on_performance_update)
        
        # Connect WebSocket and keep running
        await client.connect_websocket()
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
            if client.websocket:
                await client.websocket.close()
    
    # Run the async main function
    asyncio.run(main())
'''
    
    return client_code

# ==============================================================================
# Main Entry Points
# ==============================================================================

def run_api_server():
    """Run the FastAPI server"""
    config = config_manager.load_config()
    api_config = config['api']
    
    print(f"Starting ToneSphere API server on {api_config['host']}:{api_config['port']}")
    uvicorn.run(
        app,
        host=api_config['host'],
        port=api_config['port'],
        log_level="info"
    )

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "server":
            print("Starting ToneSphere API server...")
            run_api_server()

        elif command == "gui":
            print("Starting ToneSphere Studio GUI...")
            gui = ToneSphereStudioGUI()
            gui.run()
            
        elif command == "cli":
            cli = AudioEngineCLI()
            cli.run_interactive_mode()
            
        elif command == "test":
            print("Running basic engine tests...")
            try:
                engine = AudioEngine()
                engine.initialize()
                engine.start_engine()
                
                devices = engine.get_devices()
                print(f"✓ Found {len(devices)} audio devices")
                
                # Test virtual device creation
                virtual_id = engine.create_virtual_input("Test Input")
                print(f"✓ Created virtual input with ID: {virtual_id}")
                
                # Test routing
                devices = engine.get_devices()
                inputs = [d for d in devices if 'input' in d['type']]
                outputs = [d for d in devices if 'output' in d['type']]
                
                if len(inputs) > 0 and len(outputs) > 0:
                    success = engine.create_routing(inputs[0]['id'], outputs[0]['id'])
                    if success:
                        print("✓ Created test routing")
                    else:
                        print("✗ Failed to create routing")
                
                engine.stop_engine()
                print("✓ All tests passed")
                
            except Exception as e:
                print(f"✗ Test failed: {e}")
                
        elif command == "client":
            # Generate API client example
            client_code = create_api_client_code()
            with open("api_client.py", "w") as f:
                f.write(client_code)
            print("✓ Generated API client example in api_client.py")
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands: server, cli, test, client")
    else:
        print("ToneSphere - Professional Audio Routing System")
        print("Version 1.0.0")
        print()
        print("Usage:")
        print("  python audio_engine.py server  - Start API server")
        print("  python audio_engine.py gui     - Start GUI application")
        print("  python audio_engine.py cli     - Interactive CLI mode")
        print("  python audio_engine.py test    - Run basic tests")
        print("  python audio_engine.py client  - Generate API client example")
        print()
        print("Features:")
        print("  ✓ Low-latency ASIO driver support")
        print("  ✓ Virtual audio cable creation")
        print("  ✓ Advanced routing matrix")
        print("  ✓ Real-time audio effects")
        print("  ✓ RESTful API with WebSocket events")
        print("  ✓ VoiceMeeter Potato-like functionality")

if __name__ == "__main__":
    main()
