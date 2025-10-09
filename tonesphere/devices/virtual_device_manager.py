"""
Virtual Device Manager
CRUD operations for virtual audio devices with limits
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from tonesphere.devices.native_virtual import NativeVirtualDevice
from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VirtualDeviceInfo:
    """Virtual device information"""
    id: int
    name: str
    device_type: str  # 'input' or 'output'
    channels: int
    sample_rate: int
    buffer_size: int
    is_running: bool


class VirtualDeviceManager:
    """
    Manages virtual audio devices with CRUD operations
    Enforces limits on number of devices
    """
    
    def __init__(self, sample_rate: int = 48000, buffer_size: int = 128,
                 max_inputs: int = 10, max_outputs: int = 10):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.max_inputs = max_inputs
        self.max_outputs = max_outputs
        
        # Device storage
        self.input_devices: Dict[int, NativeVirtualDevice] = {}
        self.output_devices: Dict[int, NativeVirtualDevice] = {}
        
        # ID counters
        self.next_input_id = 10000
        self.next_output_id = 20000
        
        logger.info(f"Virtual Device Manager initialized (max inputs: {max_inputs}, max outputs: {max_outputs})")
    
    def create_input(self, channels: int = 2) -> Optional[int]:
        """
        Create a new virtual input device
        
        Args:
            channels: Number of audio channels
            
        Returns:
            Device ID if successful, None if limit reached
        """
        if len(self.input_devices) >= self.max_inputs:
            logger.warning(f"Cannot create input: limit of {self.max_inputs} reached")
            return None
        
        device_id = self.next_input_id
        self.next_input_id += 1
        
        # Generate name
        device_num = len(self.input_devices) + 1
        name = f"ToneSphere Input {device_num}"
        
        # Create device
        device = NativeVirtualDevice(
            device_id=device_id,
            name=name,
            channels=channels,
            sample_rate=self.sample_rate,
            buffer_size=self.buffer_size,
            is_input=True
        )
        
        device.start()
        self.input_devices[device_id] = device
        
        logger.info(f"Created virtual input: {name} (ID: {device_id}, channels: {channels})")
        return device_id
    
    def create_output(self, channels: int = 2) -> Optional[int]:
        """
        Create a new virtual output device
        
        Args:
            channels: Number of audio channels
            
        Returns:
            Device ID if successful, None if limit reached
        """
        if len(self.output_devices) >= self.max_outputs:
            logger.warning(f"Cannot create output: limit of {self.max_outputs} reached")
            return None
        
        device_id = self.next_output_id
        self.next_output_id += 1
        
        # Generate name
        device_num = len(self.output_devices) + 1
        name = f"ToneSphere Output {device_num}"
        
        # Create device
        device = NativeVirtualDevice(
            device_id=device_id,
            name=name,
            channels=channels,
            sample_rate=self.sample_rate,
            buffer_size=self.buffer_size,
            is_input=False
        )
        
        device.start()
        self.output_devices[device_id] = device
        
        logger.info(f"Created virtual output: {name} (ID: {device_id}, channels: {channels})")
        return device_id
    
    def delete_device(self, device_id: int) -> bool:
        """
        Delete a virtual device
        
        Args:
            device_id: Device ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        # Check input devices
        if device_id in self.input_devices:
            device = self.input_devices[device_id]
            device.stop()
            del self.input_devices[device_id]
            logger.info(f"Deleted virtual input device {device_id}")
            return True
        
        # Check output devices
        if device_id in self.output_devices:
            device = self.output_devices[device_id]
            device.stop()
            del self.output_devices[device_id]
            logger.info(f"Deleted virtual output device {device_id}")
            return True
        
        logger.warning(f"Device {device_id} not found")
        return False
    
    def get_device(self, device_id: int) -> Optional[NativeVirtualDevice]:
        """Get a virtual device by ID"""
        if device_id in self.input_devices:
            return self.input_devices[device_id]
        if device_id in self.output_devices:
            return self.output_devices[device_id]
        return None
    
    def get_device_info(self, device_id: int) -> Optional[VirtualDeviceInfo]:
        """Get information about a virtual device"""
        device = self.get_device(device_id)
        if not device:
            return None
        
        device_type = 'input' if device.is_input else 'output'
        
        return VirtualDeviceInfo(
            id=device.device_id,
            name=device.name,
            device_type=device_type,
            channels=device.channels,
            sample_rate=device.sample_rate,
            buffer_size=device.buffer_size,
            is_running=device.is_active
        )
    
    def list_inputs(self) -> List[VirtualDeviceInfo]:
        """List all virtual input devices"""
        return [
            VirtualDeviceInfo(
                id=device.device_id,
                name=device.name,
                device_type='input',
                channels=device.channels,
                sample_rate=device.sample_rate,
                buffer_size=device.buffer_size,
                is_running=device.is_active
            )
            for device in self.input_devices.values()
        ]
    
    def list_outputs(self) -> List[VirtualDeviceInfo]:
        """List all virtual output devices"""
        return [
            VirtualDeviceInfo(
                id=device.device_id,
                name=device.name,
                device_type='output',
                channels=device.channels,
                sample_rate=device.sample_rate,
                buffer_size=device.buffer_size,
                is_running=device.is_active
            )
            for device in self.output_devices.values()
        ]
    
    def list_all(self) -> List[VirtualDeviceInfo]:
        """List all virtual devices"""
        return self.list_inputs() + self.list_outputs()
    
    def get_counts(self) -> Dict[str, int]:
        """Get device counts and limits"""
        return {
            'input_count': len(self.input_devices),
            'output_count': len(self.output_devices),
            'total_count': len(self.input_devices) + len(self.output_devices),
            'max_inputs': self.max_inputs,
            'max_outputs': self.max_outputs,
            'inputs_available': self.max_inputs - len(self.input_devices),
            'outputs_available': self.max_outputs - len(self.output_devices)
        }
    
    def can_create_input(self) -> bool:
        """Check if can create more input devices"""
        return len(self.input_devices) < self.max_inputs
    
    def can_create_output(self) -> bool:
        """Check if can create more output devices"""
        return len(self.output_devices) < self.max_outputs
    
    def update_device_channels(self, device_id: int, channels: int) -> bool:
        """
        Update device channel count (requires restart)
        
        Args:
            device_id: Device ID
            channels: New channel count
            
        Returns:
            True if successful
        """
        device = self.get_device(device_id)
        if not device:
            return False
        
        # Stop device
        was_running = device.is_running
        if was_running:
            device.stop()
        
        # Update channels
        device.channels = channels
        device.buffer = device.buffer[:, :channels] if device.buffer.shape[1] > channels else device.buffer
        
        # Restart if it was running
        if was_running:
            device.start()
        
        logger.info(f"Updated device {device_id} to {channels} channels")
        return True
    
    def update_device_sample_rate(self, device_id: int, sample_rate: int) -> bool:
        """
        Update device sample rate (requires restart)
        
        Args:
            device_id: Device ID
            sample_rate: New sample rate
            
        Returns:
            True if successful
        """
        device = self.get_device(device_id)
        if not device:
            return False
        
        # Stop device
        was_running = device.is_running
        if was_running:
            device.stop()
        
        # Update sample rate
        device.sample_rate = sample_rate
        
        # Restart if it was running
        if was_running:
            device.start()
        
        logger.info(f"Updated device {device_id} to {sample_rate}Hz")
        return True
    
    def stop_all(self):
        """Stop all virtual devices"""
        for device in list(self.input_devices.values()):
            device.stop()
        for device in list(self.output_devices.values()):
            device.stop()
        logger.info("Stopped all virtual devices")
    
    def clear_all(self):
        """Delete all virtual devices"""
        self.stop_all()
        self.input_devices.clear()
        self.output_devices.clear()
        logger.info("Cleared all virtual devices")
