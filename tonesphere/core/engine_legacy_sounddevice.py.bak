from tonesphere.core.routing import AudioRoutingMatrix
from tonesphere.devices.virtual import VirtualDeviceManager
from tonesphere.network.streamer import NetworkAudioStreamer
from tonesphere.utils.logger import logger
from tonesphere.core.models import DeviceType, AudioDevice
from tonesphere.core.processor import AudioProcessor
from typing import Dict, Any, List
import sounddevice as sd
import threading
import time

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
    def create_routing(self, source_id: int, destination_id: int, volume: float = 1.0) -> tuple[bool, str]:
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