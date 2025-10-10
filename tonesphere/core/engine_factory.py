"""
Audio Engine Factory
Creates appropriate engine based on configuration
"""

from typing import Optional
from tonesphere.utils.logger import logger
from tonesphere.utils.config import ConfigManager
from tonesphere.drivers.base import AudioDriverType


def create_audio_engine(config_manager: Optional[ConfigManager] = None):
    """
    Create audio engine based on configuration
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        Audio engine instance (NativeAudioEngine or AudioEngine)
    """
    if config_manager is None:
        config_manager = ConfigManager()
    
    config = config_manager.load_config()
    engine_config = config.get('engine', {})
    
    sample_rate = engine_config.get('sample_rate', 48000)
    buffer_size = engine_config.get('buffer_size', 128)
    use_native = engine_config.get('use_native_drivers', True)
    preferred_driver_str = engine_config.get('preferred_driver', 'auto')
    
    # Parse preferred driver
    try:
        preferred_driver = AudioDriverType(preferred_driver_str.lower())
    except ValueError:
        logger.warning(f"Invalid driver type '{preferred_driver_str}', using AUTO")
        preferred_driver = AudioDriverType.AUTO
    
    if use_native:
        try:
            from tonesphere.core.native_engine import NativeAudioEngine
            logger.info(f"Creating native audio engine with {preferred_driver.value} driver")
            return NativeAudioEngine(
                sample_rate=sample_rate,
                buffer_size=buffer_size,
                preferred_driver=preferred_driver
            )
        except Exception as e:
            logger.error(f"Failed to create native audio engine: {e}")
            logger.info("Falling back to sounddevice engine")
    
    # Fallback to sounddevice engine
    from tonesphere.core.engine import AudioEngine
    logger.info("Creating sounddevice audio engine")
    return AudioEngine(sample_rate=sample_rate, buffer_size=buffer_size)


class UnifiedAudioEngine:
    """
    Unified audio engine wrapper
    Provides consistent interface regardless of backend
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.engine = create_audio_engine(config_manager)
        self.is_native = hasattr(self.engine, 'driver_manager')
    
    def initialize(self):
        """Initialize the audio engine"""
        return self.engine.initialize()
    
    def start_engine(self):
        """Start the audio engine"""
        return self.engine.start_engine()
    
    def stop_engine(self):
        """Stop the audio engine"""
        return self.engine.stop_engine()
    
    def get_devices(self):
        """Get all available devices"""
        return self.engine.get_devices()
    
    def refresh_devices(self):
        """Refresh device list to detect newly launched applications"""
        if hasattr(self.engine, 'refresh_devices'):
            return self.engine.refresh_devices()
        return False
    
    def create_virtual_input(self, name: str, channels: int = 2) -> int:
        """Create a virtual input device"""
        return self.engine.create_virtual_input(name, channels)
    
    def create_virtual_output(self, name: str, channels: int = 2) -> int:
        """Create a virtual output device"""
        return self.engine.create_virtual_output(name, channels)
    
    def create_routing(self, source_id: int, destination_id: int, volume: float = 1.0):
        """Create a routing connection"""
        return self.engine.create_routing(source_id, destination_id, volume)
    
    def remove_routing(self, source_id: int, destination_id: int) -> bool:
        """Remove a routing connection"""
        return self.engine.remove_routing(source_id, destination_id)
    
    def set_routing_volume(self, source_id: int, destination_id: int, volume: float):
        """Set routing volume"""
        return self.engine.set_routing_volume(source_id, destination_id, volume)
    
    def get_routing_matrix(self):
        """Get routing matrix"""
        return self.engine.get_routing_matrix()
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return self.engine.get_performance_stats()
    
    def start_network_streaming(self):
        """Start network streaming"""
        return self.engine.start_network_streaming()
    
    def stop_network_streaming(self):
        """Stop network streaming"""
        return self.engine.stop_network_streaming()
    
    def get_network_clients(self):
        """Get network clients"""
        return self.engine.get_network_clients()
    
    def get_driver_info(self):
        """Get driver information (native engine only)"""
        if self.is_native:
            return self.engine.get_driver_info()
        return {'type': 'sounddevice', 'native': False}
    
    def get_available_drivers(self):
        """Get available drivers (native engine only)"""
        if self.is_native:
            return self.engine.get_available_drivers()
        return ['sounddevice']
    
    def switch_driver(self, driver_type: str) -> bool:
        """Switch driver (native engine only)"""
        if self.is_native:
            return self.engine.switch_driver(driver_type)
        return False
    
    @property
    def is_running(self):
        """Check if engine is running"""
        return self.engine.is_running
    
    @property
    def sample_rate(self):
        """Get sample rate"""
        return self.engine.sample_rate
    
    @property
    def buffer_size(self):
        """Get buffer size"""
        return self.engine.buffer_size
    
    @property
    def master_volume(self):
        """Get master volume"""
        return self.engine.master_volume
    
    @master_volume.setter
    def master_volume(self, value: float):
        """Set master volume"""
        self.engine.master_volume = value
    
    # Channel Control Methods
    def get_device_channels(self, device_id: int):
        """Get channel info for device"""
        if hasattr(self.engine, 'channel_control_manager'):
            return self.engine.channel_control_manager.get_device_info(device_id)
        return None
    
    def set_channel_volume(self, device_id: int, channel: int, volume: float):
        """Set channel volume"""
        if hasattr(self.engine, 'channel_control_manager'):
            self.engine.channel_control_manager.set_device_channel_volume(device_id, channel, volume)
    
    def set_channel_mute(self, device_id: int, channel: int, muted: bool):
        """Set channel mute"""
        if hasattr(self.engine, 'channel_control_manager'):
            self.engine.channel_control_manager.set_device_channel_mute(device_id, channel, muted)
    
    def set_channel_solo(self, device_id: int, channel: int, solo: bool):
        """Set channel solo"""
        if hasattr(self.engine, 'channel_control_manager'):
            self.engine.channel_control_manager.set_device_channel_solo(device_id, channel, solo)
    
    def set_channel_pan(self, device_id: int, channel: int, pan: float):
        """Set channel pan"""
        if hasattr(self.engine, 'channel_control_manager'):
            self.engine.channel_control_manager.set_device_channel_pan(device_id, channel, pan)
    
    def swap_channels(self, device_id: int):
        """Swap device channels"""
        if hasattr(self.engine, 'channel_control_manager'):
            self.engine.channel_control_manager.swap_device_channels(device_id)
    
    def set_device_master_volume(self, device_id: int, volume: float):
        """Set device master volume"""
        if hasattr(self.engine, 'channel_control_manager'):
            self.engine.channel_control_manager.set_device_master_volume(device_id, volume)
    
    def set_device_master_mute(self, device_id: int, muted: bool):
        """Set device master mute"""
        if hasattr(self.engine, 'channel_control_manager'):
            self.engine.channel_control_manager.set_device_master_mute(device_id, muted)
    
    # Sample Rate Control
    def set_sample_rate(self, sample_rate: int):
        """Set sample rate"""
        if hasattr(self.engine, 'sample_rate_manager'):
            self.engine.sample_rate_manager.set_master_sample_rate(sample_rate)
    
    # Network Methods
    def connect_to_network(self, host: str, port: int) -> bool:
        """Connect to network instance"""
        if hasattr(self.engine, 'connect_to_network'):
            return self.engine.connect_to_network(host, port)
        return False
    
    def disconnect_from_network(self, conn_id: str):
        """Disconnect from network"""
        if hasattr(self.engine, 'disconnect_from_network'):
            self.engine.disconnect_from_network(conn_id)
    
    def get_network_connections(self):
        """Get network connections"""
        if hasattr(self.engine, 'get_network_connections'):
            return self.engine.get_network_connections()
        return []
    
    def send_device_to_network(self, device_id: int, target=None):
        """Send device audio to network"""
        if hasattr(self.engine, 'send_device_audio_to_network'):
            self.engine.send_device_audio_to_network(device_id, target)
    
    def register_network_receive(self, device_id: int):
        """Register device for network receive"""
        if hasattr(self.engine, 'register_network_receive'):
            self.engine.register_network_receive(device_id)
    
    def get_network_statistics(self):
        """Get network statistics"""
        if hasattr(self.engine, 'get_network_statistics'):
            return self.engine.get_network_statistics()
        return {}
    
    # Virtual Device Manager Methods
    def list_virtual_devices(self):
        """List all virtual devices"""
        if hasattr(self.engine, 'list_virtual_devices'):
            return self.engine.list_virtual_devices()
        return []
    
    def get_virtual_device_counts(self):
        """Get virtual device counts and limits"""
        if hasattr(self.engine, 'get_virtual_device_counts'):
            return self.engine.get_virtual_device_counts()
        return {}
    
    def delete_virtual_device(self, device_id: int) -> bool:
        """Delete a virtual device"""
        if hasattr(self.engine, 'delete_virtual_device'):
            return self.engine.delete_virtual_device(device_id)
        return False
    
    def update_virtual_device_sample_rate(self, device_id: int, sample_rate: int) -> bool:
        """Update virtual device sample rate"""
        if hasattr(self.engine, 'update_virtual_device_sample_rate'):
            return self.engine.update_virtual_device_sample_rate(device_id, sample_rate)
        return False
    
    def update_virtual_device_channels(self, device_id: int, channels: int) -> bool:
        """Update virtual device channels"""
        if hasattr(self.engine, 'update_virtual_device_channels'):
            return self.engine.update_virtual_device_channels(device_id, channels)
        return False
