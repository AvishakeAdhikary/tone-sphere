from typing import Dict
import numpy as np
from tonesphere.utils.logger import logger

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