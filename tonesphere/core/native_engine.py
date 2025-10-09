"""
Native Audio Engine with full driver support
Replaces sounddevice with native audio drivers
"""

import numpy as np
import threading
import time
from typing import Dict, Any, List, Optional
from tonesphere.core.routing import AudioRoutingMatrix
from tonesphere.devices.native_virtual import NativeVirtualDeviceManager
from tonesphere.devices.virtual_device_manager import VirtualDeviceManager
from tonesphere.network.audio_router import NetworkAudioRouter, NetworkQuality
from tonesphere.utils.logger import get_logger
from tonesphere.core.models import DeviceType, AudioDevice
from tonesphere.core.processor import AudioProcessor
from tonesphere.core.stream_manager import AudioStreamManager
from tonesphere.core.channel_control import ChannelControlManager
from tonesphere.core.sample_rate_converter import SampleRateManager
from tonesphere.drivers.manager import AudioDriverManager
from tonesphere.drivers.base import AudioDriverType, AudioStreamConfig

logger = get_logger(__name__)


class NativeAudioEngine:
    """
    Native audio engine with full driver support
    Supports ASIO, WASAPI, DirectSound, ALSA, PulseAudio, JACK, PipeWire, CoreAudio
    """
    
    def __init__(self, sample_rate: int = 48000, buffer_size: int = 128,
                 preferred_driver: AudioDriverType = AudioDriverType.AUTO,
                 max_virtual_inputs: int = 10, max_virtual_outputs: int = 10):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Core components
        self.driver_manager = AudioDriverManager(preferred_driver)
        self.stream_manager = AudioStreamManager(self.driver_manager)
        self.processor = AudioProcessor(sample_rate, buffer_size)
        self.virtual_device_manager = NativeVirtualDeviceManager(sample_rate, buffer_size)
        self.virtual_manager = VirtualDeviceManager(sample_rate, buffer_size, max_virtual_inputs, max_virtual_outputs)
        self.routing_matrix = AudioRoutingMatrix()
        self.network_router = NetworkAudioRouter(quality=NetworkQuality.HIGH)
        self.channel_control_manager = ChannelControlManager()
        self.sample_rate_manager = SampleRateManager(sample_rate)
        
        # Device management
        self.all_devices: Dict[int, AudioDevice] = {}
        
        # State
        self.is_running = False
        self.master_volume = 1.0
        
        # Performance monitoring
        self.performance_stats = {
            'cpu_usage': 0.0,
            'buffer_underruns': 0,
            'latency_ms': 0.0,
            'active_driver': None,
            'total_devices': 0,
            'virtual_devices': 0,
            'active_streams': 0
        }
        
        # Threading
        self.monitor_thread: Optional[threading.Thread] = None
        self.audio_thread: Optional[threading.Thread] = None
        
    def initialize(self):
        """Initialize the native audio engine"""
        try:
            # Initialize driver manager
            if not self.driver_manager.initialize():
                raise Exception("Failed to initialize audio driver")
            
            # Scan for audio devices
            self._scan_audio_devices()
            
            # Create default virtual devices
            self._create_default_virtual_devices()
            
            # Initialize channel controls for all devices
            for device_id, device in self.all_devices.items():
                self.channel_control_manager.add_device(device_id, device.channels)
            
            # Update performance stats
            active_driver = self.driver_manager.get_active_driver()
            if active_driver:
                self.performance_stats['active_driver'] = active_driver.driver_type.value
            
            logger.info(f"Native audio engine initialized with {len(self.all_devices)} devices")
            
        except Exception as e:
            logger.error(f"Failed to initialize native audio engine: {e}")
            raise
    
    def _scan_audio_devices(self):
        """Scan for available audio devices using native drivers"""
        try:
            # Get devices from driver manager
            device_infos = self.driver_manager.enumerate_devices()
            
            for device_info in device_infos:
                # Determine device type
                if device_info.max_input_channels > 0 and device_info.max_output_channels == 0:
                    device_type = DeviceType.PHYSICAL_INPUT
                elif device_info.max_output_channels > 0 and device_info.max_input_channels == 0:
                    device_type = DeviceType.PHYSICAL_OUTPUT
                else:
                    # Device supports both - create two entries
                    if device_info.max_input_channels > 0:
                        input_device = AudioDevice(
                            id=device_info.id * 2,
                            name=f"{device_info.name} (Input)",
                            device_type=DeviceType.PHYSICAL_INPUT,
                            channels=device_info.max_input_channels,
                            sample_rate=device_info.default_sample_rate,
                            buffer_size=device_info.default_buffer_size,
                            is_asio=device_info.is_asio,
                            latency=device_info.latency_input_ms
                        )
                        self.all_devices[input_device.id] = input_device
                    
                    if device_info.max_output_channels > 0:
                        output_device = AudioDevice(
                            id=device_info.id * 2 + 1,
                            name=f"{device_info.name} (Output)",
                            device_type=DeviceType.PHYSICAL_OUTPUT,
                            channels=device_info.max_output_channels,
                            sample_rate=device_info.default_sample_rate,
                            buffer_size=device_info.default_buffer_size,
                            is_asio=device_info.is_asio,
                            latency=device_info.latency_output_ms
                        )
                        self.all_devices[output_device.id] = output_device
                    continue
                
                # Create single device
                audio_device = AudioDevice(
                    id=device_info.id,
                    name=device_info.name,
                    device_type=device_type,
                    channels=max(device_info.max_input_channels, device_info.max_output_channels),
                    sample_rate=device_info.default_sample_rate,
                    buffer_size=device_info.default_buffer_size,
                    is_asio=device_info.is_asio,
                    latency=max(device_info.latency_input_ms, device_info.latency_output_ms)
                )
                self.all_devices[device_info.id] = audio_device
            
            self.performance_stats['total_devices'] = len(self.all_devices)
            logger.info(f"Scanned {len(self.all_devices)} physical audio devices")
            
        except Exception as e:
            logger.error(f"Error scanning audio devices: {e}")
    
    def _create_default_virtual_devices(self):
        """Create default virtual audio devices using the new manager"""
        # Get config for default counts
        from tonesphere.utils.config import ConfigManager
        config = ConfigManager().load_config()
        vdev_config = config.get('virtual_devices', {})
        default_inputs = vdev_config.get('default_inputs', 3)
        default_outputs = vdev_config.get('default_outputs', 3)
        
        # Create default virtual inputs using new manager
        for i in range(default_inputs):
            device_id = self.virtual_manager.create_input(channels=2)
            if device_id:
                virtual_device = AudioDevice(
                    id=device_id,
                    name=f"ToneSphere Input {i+1}",
                    device_type=DeviceType.VIRTUAL_INPUT,
                    channels=2,
                    sample_rate=self.sample_rate,
                    buffer_size=self.buffer_size,
                    is_active=True
                )
                self.all_devices[device_id] = virtual_device
        
        # Create default virtual outputs using new manager
        for i in range(default_outputs):
            device_id = self.virtual_manager.create_output(channels=2)
            if device_id:
                virtual_device = AudioDevice(
                    id=device_id,
                    name=f"ToneSphere Output {i+1}",
                    device_type=DeviceType.VIRTUAL_OUTPUT,
                    channels=2,
                    sample_rate=self.sample_rate,
                    buffer_size=self.buffer_size,
                    is_active=True
                )
                self.all_devices[device_id] = virtual_device
        
        total_created = default_inputs + default_outputs
        self.performance_stats['virtual_devices'] = total_created
        logger.info(f"Created {total_created} default ToneSphere virtual devices")
    
    def start_engine(self):
        """Start the native audio engine"""
        if self.is_running:
            return
        
        try:
            self.is_running = True
            
            # Start stream manager
            self.stream_manager.start_processing()
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._performance_monitor, daemon=True)
            self.monitor_thread.start()
            
            # Start audio processing thread
            self.audio_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
            self.audio_thread.start()
            
            logger.info("Native audio engine started")
            
        except Exception as e:
            logger.error(f"Failed to start native audio engine: {e}")
            self.is_running = False
            raise
    
    def stop_engine(self):
        """Stop the native audio engine"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop stream manager
        self.stream_manager.stop_processing()
        
        # Stop virtual devices
        self.virtual_device_manager.stop_all()
        
        # Wait for threads
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
        
        logger.info("Native audio engine stopped")
    
    def _performance_monitor(self):
        """Monitor engine performance"""
        while self.is_running:
            try:
                # Update performance stats
                stream_stats = self.stream_manager.get_statistics()
                
                self.performance_stats['active_streams'] = stream_stats['running_streams']
                self.performance_stats['latency_ms'] = self.buffer_size / self.sample_rate * 1000
                
                # Calculate CPU usage from streams
                total_cpu = 0.0
                for stream_info in stream_stats['streams'].values():
                    total_cpu += stream_info.get('cpu_load', 0.0)
                
                self.performance_stats['cpu_usage'] = total_cpu / max(1, stream_stats['total_streams'])
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
    
    def _audio_processing_loop(self):
        """Main audio processing loop"""
        while self.is_running:
            try:
                # Process virtual device routing
                self.virtual_device_manager.process_routing()
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
    
    # Device Management Methods
    def get_devices(self) -> List[Dict]:
        """Get all available devices"""
        devices = []
        for device in self.all_devices.values():
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
    
    def create_virtual_input(self, name: str, channels: int = 2) -> Optional[int]:
        """Create a new virtual input device using new manager"""
        # Use new manager which handles limits and auto-naming
        device_id = self.virtual_manager.create_input(channels)
        
        if device_id:
            # Get the auto-generated name from the manager
            device_info = self.virtual_manager.get_device_info(device_id)
            actual_name = device_info.name if device_info else f"ToneSphere Input {len(self.virtual_manager.input_devices)}"
            
            virtual_device = AudioDevice(
                id=device_id,
                name=actual_name,
                device_type=DeviceType.VIRTUAL_INPUT,
                channels=channels,
                sample_rate=self.sample_rate,
                buffer_size=self.buffer_size,
                is_active=True
            )
            self.all_devices[device_id] = virtual_device
            self.performance_stats['virtual_devices'] += 1
            logger.info(f"Created virtual input: {actual_name} (ID: {device_id})")
        
        return device_id
    
    def create_virtual_output(self, name: str, channels: int = 2) -> Optional[int]:
        """Create a new virtual output device using new manager"""
        # Use new manager which handles limits and auto-naming
        device_id = self.virtual_manager.create_output(channels)
        
        if device_id:
            # Get the auto-generated name from the manager
            device_info = self.virtual_manager.get_device_info(device_id)
            actual_name = device_info.name if device_info else f"ToneSphere Output {len(self.virtual_manager.output_devices)}"
            
            virtual_device = AudioDevice(
                id=device_id,
                name=actual_name,
                device_type=DeviceType.VIRTUAL_OUTPUT,
                channels=channels,
                sample_rate=self.sample_rate,
                buffer_size=self.buffer_size,
                is_active=True
            )
            self.all_devices[device_id] = virtual_device
            self.performance_stats['virtual_devices'] += 1
            logger.info(f"Created virtual output: {actual_name} (ID: {device_id})")
        
        return device_id
    
    def remove_virtual_device(self, device_id: int) -> bool:
        """Remove a virtual device using new manager"""
        if device_id not in self.all_devices:
            return False
        
        device = self.all_devices[device_id]
        if device.device_type not in [DeviceType.VIRTUAL_INPUT, DeviceType.VIRTUAL_OUTPUT]:
            return False
        
        # Use new manager to delete
        if self.virtual_manager.delete_device(device_id):
            del self.all_devices[device_id]
            self.performance_stats['virtual_devices'] -= 1
            logger.info(f"Removed virtual device: {device.name} (ID: {device_id})")
            return True
        
        return False
    
    # Virtual Device Manager Methods
    def list_virtual_devices(self) -> List[Dict]:
        """List all virtual devices"""
        return [
            {
                'id': info.id,
                'name': info.name,
                'type': info.device_type,
                'channels': info.channels,
                'sample_rate': info.sample_rate,
                'is_running': info.is_running
            }
            for info in self.virtual_manager.list_all()
        ]
    
    def get_virtual_device_counts(self) -> Dict:
        """Get virtual device counts and limits"""
        return self.virtual_manager.get_counts()
    
    def delete_virtual_device(self, device_id: int) -> bool:
        """Delete a virtual device using the manager"""
        success = self.virtual_manager.delete_device(device_id)
        if success and device_id in self.all_devices:
            del self.all_devices[device_id]
            self.performance_stats['virtual_devices'] -= 1
        return success
    
    def update_virtual_device_sample_rate(self, device_id: int, sample_rate: int) -> bool:
        """Update virtual device sample rate"""
        return self.virtual_manager.update_device_sample_rate(device_id, sample_rate)
    
    def update_virtual_device_channels(self, device_id: int, channels: int) -> bool:
        """Update virtual device channels"""
        return self.virtual_manager.update_device_channels(device_id, channels)
    
    # Routing Methods
    def create_routing(self, source_id: int, destination_id: int, volume: float = 1.0) -> tuple[bool, str]:
        """Create a routing connection"""
        # Create routing in matrix
        success, message = self.routing_matrix.create_routing(source_id, destination_id, volume)
        
        if success:
            # Also route in virtual device manager if both are virtual
            self.virtual_device_manager.route_audio(source_id, destination_id, volume)
        
        return success, message
    
    def remove_routing(self, source_id: int, destination_id: int) -> bool:
        """Remove a routing connection"""
        # Remove from routing matrix
        success = self.routing_matrix.remove_routing(source_id, destination_id)
        
        if success:
            # Also disconnect in virtual device manager
            source_device = self.virtual_device_manager.get_device(source_id)
            if source_device:
                source_device.disconnect_from(destination_id)
        
        return success
    
    def set_routing_volume(self, source_id: int, destination_id: int, volume: float):
        """Set volume for a routing connection"""
        self.routing_matrix.set_routing_volume(source_id, destination_id, volume)
        
        # Update virtual device routing
        source_device = self.virtual_device_manager.get_device(source_id)
        if source_device and destination_id in source_device.connected_devices:
            source_device.connected_devices[destination_id] = volume
    
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
    
    # Stream Management
    def open_device_stream(self, device_id: int, is_input: bool = True) -> int:
        """Open a stream for a physical device"""
        device = self.all_devices.get(device_id)
        if not device:
            logger.error(f"Device {device_id} not found")
            return -1
        
        config = AudioStreamConfig(
            device_id=device_id,
            sample_rate=self.sample_rate,
            buffer_size=self.buffer_size,
            channels=device.channels,
            input_channels=device.channels if is_input else 0,
            output_channels=device.channels if not is_input else 0
        )
        
        return self.stream_manager.create_stream(device_id, config)
    
    def close_device_stream(self, stream_id: int) -> bool:
        """Close a device stream"""
        return self.stream_manager.destroy_stream(stream_id)
    
    # Network Streaming
    def start_network_streaming(self):
        """Start network audio streaming"""
        self.network_router.start_server()
    
    def stop_network_streaming(self):
        """Stop network audio streaming"""
        self.network_router.stop_server()
    
    def get_network_clients(self) -> List[str]:
        """Get connected network clients"""
        return self.network_router.get_connected_clients()
    
    def connect_to_network(self, host: str, port: int) -> bool:
        """Connect to another ToneSphere instance"""
        return self.network_router.connect_to(host, port)
    
    def disconnect_from_network(self, conn_id: str):
        """Disconnect from network instance"""
        self.network_router.disconnect_from(conn_id)
    
    def get_network_connections(self) -> List[str]:
        """Get outgoing network connections"""
        return self.network_router.get_connections()
    
    def send_device_audio_to_network(self, device_id: int, target: Optional[str] = None):
        """Send device audio over network"""
        device = self.all_devices.get(device_id)
        if not device:
            return
        
        # Get audio from virtual device
        virtual_device = self.virtual_device_manager.get_device(device_id)
        if virtual_device:
            audio_data = virtual_device.read_audio()
            self.network_router.send_audio(device_id, audio_data, self.sample_rate, target)
    
    def register_network_receive(self, device_id: int):
        """Register device to receive network audio"""
        def receive_callback(audio_data: np.ndarray, packet):
            virtual_device = self.virtual_device_manager.get_device(device_id)
            if virtual_device:
                virtual_device.write_audio(audio_data)
        
        self.network_router.register_receive_callback(device_id, receive_callback)
    
    def get_network_statistics(self) -> Dict:
        """Get network statistics"""
        return self.network_router.get_statistics()
    
    # Driver Management
    def get_driver_info(self) -> Dict[str, Any]:
        """Get information about the audio driver"""
        return self.driver_manager.get_driver_info()
    
    def get_available_drivers(self) -> List[str]:
        """Get list of available drivers"""
        return [dt.value for dt in self.driver_manager.get_available_drivers()]
    
    def switch_driver(self, driver_type: str) -> bool:
        """Switch to a different audio driver"""
        try:
            dt = AudioDriverType(driver_type)
            success = self.driver_manager.switch_driver(dt)
            
            if success:
                # Rescan devices
                self.all_devices.clear()
                self._scan_audio_devices()
                self.performance_stats['active_driver'] = driver_type
            
            return success
        except ValueError:
            logger.error(f"Invalid driver type: {driver_type}")
            return False
    
    def cleanup(self):
        """Cleanup and terminate engine"""
        self.stop_engine()
        self.stream_manager.cleanup()
        self.driver_manager.terminate()
        logger.info("Native audio engine cleanup complete")
