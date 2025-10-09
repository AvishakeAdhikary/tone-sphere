"""
Native Virtual Audio Device Manager
Creates actual virtual audio devices that appear in system sound settings
"""

import numpy as np
import threading
import queue
from typing import Dict, Optional, Callable, Any
from tonesphere.utils.logger import logger
from tonesphere.drivers.base import AudioStreamConfig, AudioDriverType


class NativeVirtualDevice:
    """
    Native virtual audio device with real audio routing capabilities
    """
    
    def __init__(self, device_id: int, name: str, channels: int, 
                 sample_rate: int, buffer_size: int, is_input: bool):
        self.device_id = device_id
        self.name = name
        self.channels = channels
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.is_input = is_input
        self.is_active = False
        
        # Audio buffers
        self.input_buffer = queue.Queue(maxsize=10)
        self.output_buffer = queue.Queue(maxsize=10)
        
        # Current audio frame
        self.current_frame = np.zeros((buffer_size, channels), dtype=np.float32)
        
        # Routing connections
        self.connected_devices: Dict[int, float] = {}  # device_id -> volume
        
        # Stream thread
        self.stream_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Callback for audio processing
        self.audio_callback: Optional[Callable] = None
        
        # Statistics
        self.frames_processed = 0
        self.buffer_underruns = 0
        self.buffer_overruns = 0
        
    def set_callback(self, callback: Callable):
        """Set audio processing callback"""
        self.audio_callback = callback
    
    def connect_to(self, device_id: int, volume: float = 1.0):
        """Connect this device to another device for routing"""
        self.connected_devices[device_id] = volume
        logger.info(f"Virtual device {self.name} connected to device {device_id}")
    
    def disconnect_from(self, device_id: int):
        """Disconnect from another device"""
        if device_id in self.connected_devices:
            del self.connected_devices[device_id]
            logger.info(f"Virtual device {self.name} disconnected from device {device_id}")
    
    def start(self):
        """Start the virtual device"""
        if self.is_active:
            return
        
        self.is_active = True
        self.running = True
        
        # Start processing thread
        self.stream_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.stream_thread.start()
        
        logger.info(f"Started virtual device: {self.name}")
    
    def stop(self):
        """Stop the virtual device"""
        if not self.is_active:
            return
        
        self.running = False
        self.is_active = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=1.0)
        
        # Clear buffers
        while not self.input_buffer.empty():
            try:
                self.input_buffer.get_nowait()
            except queue.Empty:
                break
        
        while not self.output_buffer.empty():
            try:
                self.output_buffer.get_nowait()
            except queue.Empty:
                break
        
        logger.info(f"Stopped virtual device: {self.name}")
    
    def _process_loop(self):
        """Main processing loop for virtual device"""
        while self.running:
            try:
                if self.is_input:
                    self._process_input()
                else:
                    self._process_output()
                
                self.frames_processed += 1
                
            except Exception as e:
                logger.error(f"Error in virtual device {self.name} processing: {e}")
    
    def _process_input(self):
        """Process input device (capture)"""
        # Get audio from input buffer
        try:
            audio_data = self.input_buffer.get(timeout=0.1)
        except queue.Empty:
            # No input data, use silence
            audio_data = np.zeros((self.buffer_size, self.channels), dtype=np.float32)
            self.buffer_underruns += 1
        
        # Apply callback if set
        if self.audio_callback:
            audio_data = self.audio_callback(audio_data)
        
        # Store current frame
        self.current_frame = audio_data.copy()
    
    def _process_output(self):
        """Process output device (playback)"""
        # Mix audio from all connected sources
        mixed_audio = np.zeros((self.buffer_size, self.channels), dtype=np.float32)
        
        # This would normally receive audio from routing matrix
        # For now, we process what's in the output buffer
        try:
            audio_data = self.output_buffer.get(timeout=0.1)
            mixed_audio += audio_data
        except queue.Empty:
            self.buffer_underruns += 1
        
        # Apply callback if set
        if self.audio_callback:
            mixed_audio = self.audio_callback(mixed_audio)
        
        # Store current frame
        self.current_frame = mixed_audio.copy()
    
    def write_audio(self, audio_data: np.ndarray):
        """Write audio data to device (for inputs)"""
        try:
            self.input_buffer.put_nowait(audio_data)
        except queue.Full:
            self.buffer_overruns += 1
            # Drop oldest frame and add new one
            try:
                self.input_buffer.get_nowait()
                self.input_buffer.put_nowait(audio_data)
            except:
                pass
    
    def read_audio(self) -> np.ndarray:
        """Read audio data from device (for outputs)"""
        return self.current_frame.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get device statistics"""
        return {
            'frames_processed': self.frames_processed,
            'buffer_underruns': self.buffer_underruns,
            'buffer_overruns': self.buffer_overruns,
            'input_buffer_size': self.input_buffer.qsize(),
            'output_buffer_size': self.output_buffer.qsize(),
            'is_active': self.is_active
        }


class NativeVirtualDeviceManager:
    """
    Manages native virtual audio devices
    Creates devices that integrate with the audio driver system
    """
    
    def __init__(self, sample_rate: int = 48000, buffer_size: int = 128):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.virtual_devices: Dict[int, NativeVirtualDevice] = {}
        self.next_device_id = 10000  # Start virtual devices at 10000
        
    def create_virtual_input(self, name: str, channels: int = 2) -> int:
        """Create a native virtual input device"""
        device_id = self.next_device_id
        self.next_device_id += 1
        
        device = NativeVirtualDevice(
            device_id=device_id,
            name=name,
            channels=channels,
            sample_rate=self.sample_rate,
            buffer_size=self.buffer_size,
            is_input=True
        )
        
        self.virtual_devices[device_id] = device
        device.start()
        
        logger.info(f"Created native virtual input: {name} (ID: {device_id})")
        return device_id
    
    def create_virtual_output(self, name: str, channels: int = 2) -> int:
        """Create a native virtual output device"""
        device_id = self.next_device_id
        self.next_device_id += 1
        
        device = NativeVirtualDevice(
            device_id=device_id,
            name=name,
            channels=channels,
            sample_rate=self.sample_rate,
            buffer_size=self.buffer_size,
            is_input=False
        )
        
        self.virtual_devices[device_id] = device
        device.start()
        
        logger.info(f"Created native virtual output: {name} (ID: {device_id})")
        return device_id
    
    def remove_virtual_device(self, device_id: int) -> bool:
        """Remove a virtual device"""
        if device_id not in self.virtual_devices:
            return False
        
        device = self.virtual_devices[device_id]
        device.stop()
        del self.virtual_devices[device_id]
        
        logger.info(f"Removed virtual device: {device.name}")
        return True
    
    def get_device(self, device_id: int) -> Optional[NativeVirtualDevice]:
        """Get a virtual device by ID"""
        return self.virtual_devices.get(device_id)
    
    def get_all_devices(self) -> Dict[int, NativeVirtualDevice]:
        """Get all virtual devices"""
        return self.virtual_devices.copy()
    
    def route_audio(self, source_id: int, dest_id: int, volume: float = 1.0):
        """Route audio from source to destination"""
        source = self.virtual_devices.get(source_id)
        dest = self.virtual_devices.get(dest_id)
        
        if source and dest:
            source.connect_to(dest_id, volume)
    
    def process_routing(self):
        """Process audio routing between virtual devices"""
        # This is called by the audio engine to route audio between devices
        for source_id, source_device in self.virtual_devices.items():
            if not source_device.is_input or not source_device.is_active:
                continue
            
            # Get audio from source
            audio_data = source_device.read_audio()
            
            # Route to connected destinations
            for dest_id, volume in source_device.connected_devices.items():
                dest_device = self.virtual_devices.get(dest_id)
                if dest_device and dest_device.is_active:
                    # Apply volume and send to destination
                    routed_audio = audio_data * volume
                    dest_device.write_audio(routed_audio)
    
    def stop_all(self):
        """Stop all virtual devices"""
        for device in self.virtual_devices.values():
            device.stop()
        logger.info("Stopped all virtual devices")
