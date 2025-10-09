"""
Base audio driver interface for ToneSphere
Provides abstraction layer for all audio backends
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import numpy as np


class AudioDriverType(Enum):
    """Supported audio driver types"""
    ASIO = "asio"
    WASAPI = "wasapi"
    DIRECTSOUND = "directsound"
    ALSA = "alsa"
    PULSEAUDIO = "pulseaudio"
    JACK = "jack"
    PIPEWIRE = "pipewire"
    COREAUDIO = "coreaudio"
    AUTO = "auto"  # Auto-detect best available


class StreamState(Enum):
    """Audio stream states"""
    STOPPED = 0
    RUNNING = 1
    PAUSED = 2
    ERROR = 3


class ChannelLayout(Enum):
    """Audio channel layouts"""
    MONO = 1
    STEREO = 2
    QUAD = 4
    SURROUND_5_1 = 6
    SURROUND_7_1 = 8


@dataclass
class AudioDeviceInfo:
    """Information about an audio device"""
    id: int
    name: str
    driver_type: AudioDriverType
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: int
    supported_sample_rates: List[int]
    default_buffer_size: int
    supported_buffer_sizes: List[int]
    is_default_input: bool
    is_default_output: bool
    latency_input_ms: float
    latency_output_ms: float
    is_asio: bool
    host_api: str
    
    # Additional capabilities
    supports_exclusive_mode: bool = False
    supports_shared_mode: bool = True
    supports_callback_mode: bool = True
    supports_blocking_mode: bool = True


@dataclass
class AudioStreamConfig:
    """Configuration for an audio stream"""
    device_id: int
    sample_rate: int
    buffer_size: int
    channels: int
    input_channels: int = 0
    output_channels: int = 0
    dtype: np.dtype = np.float32
    exclusive_mode: bool = False
    callback: Optional[Callable] = None
    latency: str = "low"  # "low", "high", "default"


class AudioCallback:
    """Base class for audio callbacks"""
    
    def __init__(self):
        self.user_callback: Optional[Callable] = None
    
    def set_callback(self, callback: Callable):
        """Set user callback function"""
        self.user_callback = callback
    
    def process(self, input_data: Optional[np.ndarray], 
                output_data: np.ndarray, 
                frames: int, 
                time_info: Dict[str, float],
                status_flags: int) -> np.ndarray:
        """
        Process audio callback
        
        Args:
            input_data: Input audio buffer (None if output-only)
            output_data: Output audio buffer to fill
            frames: Number of frames to process
            time_info: Timing information
            status_flags: Status flags (underrun, overflow, etc.)
        
        Returns:
            Processed output data
        """
        if self.user_callback:
            return self.user_callback(input_data, output_data, frames, time_info, status_flags)
        return output_data


class AudioDriverBase(ABC):
    """
    Abstract base class for audio drivers
    All platform-specific drivers must inherit from this
    """
    
    def __init__(self, driver_type: AudioDriverType):
        self.driver_type = driver_type
        self.is_initialized = False
        self.active_streams: Dict[int, Any] = {}
        self.devices_cache: List[AudioDeviceInfo] = []
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the audio driver
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def terminate(self):
        """Terminate the audio driver and cleanup resources"""
        pass
    
    @abstractmethod
    def enumerate_devices(self) -> List[AudioDeviceInfo]:
        """
        Enumerate all available audio devices
        
        Returns:
            List of audio device information
        """
        pass
    
    @abstractmethod
    def get_device_info(self, device_id: int) -> Optional[AudioDeviceInfo]:
        """
        Get information about a specific device
        
        Args:
            device_id: Device identifier
            
        Returns:
            Device information or None if not found
        """
        pass
    
    @abstractmethod
    def open_stream(self, config: AudioStreamConfig) -> int:
        """
        Open an audio stream
        
        Args:
            config: Stream configuration
            
        Returns:
            Stream handle/ID
        """
        pass
    
    @abstractmethod
    def close_stream(self, stream_id: int) -> bool:
        """
        Close an audio stream
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def start_stream(self, stream_id: int) -> bool:
        """
        Start an audio stream
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def stop_stream(self, stream_id: int) -> bool:
        """
        Stop an audio stream
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def get_stream_state(self, stream_id: int) -> StreamState:
        """
        Get current stream state
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Current stream state
        """
        pass
    
    @abstractmethod
    def read_stream(self, stream_id: int, frames: int) -> Optional[np.ndarray]:
        """
        Read audio data from input stream (blocking mode)
        
        Args:
            stream_id: Stream identifier
            frames: Number of frames to read
            
        Returns:
            Audio data or None if error
        """
        pass
    
    @abstractmethod
    def write_stream(self, stream_id: int, data: np.ndarray) -> bool:
        """
        Write audio data to output stream (blocking mode)
        
        Args:
            stream_id: Stream identifier
            data: Audio data to write
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def get_stream_latency(self, stream_id: int) -> tuple[float, float]:
        """
        Get stream latency
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Tuple of (input_latency_ms, output_latency_ms)
        """
        pass
    
    @abstractmethod
    def get_stream_cpu_load(self, stream_id: int) -> float:
        """
        Get CPU load for stream
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            CPU load as percentage (0.0 to 1.0)
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if this driver is available on the current system
        
        Returns:
            True if driver is available
        """
        return False
    
    def get_driver_info(self) -> Dict[str, Any]:
        """
        Get driver information
        
        Returns:
            Dictionary with driver details
        """
        return {
            'type': self.driver_type.value,
            'initialized': self.is_initialized,
            'active_streams': len(self.active_streams),
            'devices_count': len(self.devices_cache)
        }
    
    def refresh_devices(self):
        """Refresh device cache"""
        self.devices_cache = self.enumerate_devices()
    
    def get_default_input_device(self) -> Optional[AudioDeviceInfo]:
        """Get default input device"""
        for device in self.devices_cache:
            if device.is_default_input:
                return device
        return None
    
    def get_default_output_device(self) -> Optional[AudioDeviceInfo]:
        """Get default output device"""
        for device in self.devices_cache:
            if device.is_default_output:
                return device
        return None
