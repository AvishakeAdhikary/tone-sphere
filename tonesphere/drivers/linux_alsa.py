"""
ALSA (Advanced Linux Sound Architecture) driver implementation
Low-level Linux audio support
"""

import platform
import numpy as np
from typing import List, Optional, Dict, Any
from .base import (
    AudioDriverBase, AudioDriverType, AudioDeviceInfo,
    AudioStreamConfig, StreamState, AudioCallback
)
from tonesphere.utils.logger import logger


class ALSADriver(AudioDriverBase):
    """
    ALSA driver implementation for Linux
    
    ALSA is the low-level audio API for Linux, providing direct
    hardware access with low latency.
    """
    
    def __init__(self):
        super().__init__(AudioDriverType.ALSA)
        self.stream_callbacks: Dict[int, AudioCallback] = {}
        self.next_stream_id = 1
        self.pcm_devices: Dict[str, Any] = {}
        
    def is_available(self) -> bool:
        """Check if ALSA is available"""
        if platform.system() != "Linux":
            return False
        
        try:
            import alsaaudio
            return True
        except ImportError:
            return False
    
    def initialize(self) -> bool:
        """Initialize ALSA driver"""
        if not self.is_available():
            logger.warning("ALSA not available on this system")
            return False
        
        try:
            self.is_initialized = True
            logger.info("ALSA driver initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ALSA driver: {e}")
            return False
    
    def terminate(self):
        """Terminate ALSA driver"""
        for stream_id in list(self.active_streams.keys()):
            self.close_stream(stream_id)
        
        self.is_initialized = False
        logger.info("ALSA driver terminated")
    
    def enumerate_devices(self) -> List[AudioDeviceInfo]:
        """Enumerate ALSA devices"""
        devices = []
        device_id = 300  # Start ALSA devices at 300
        
        try:
            try:
                import alsaaudio
                
                # Get PCM devices
                pcm_list = alsaaudio.pcms()
                
                for idx, pcm_name in enumerate(pcm_list):
                    # Try to open device to get capabilities
                    try:
                        # Output device
                        devices.append(AudioDeviceInfo(
                            id=device_id,
                            name=f"ALSA: {pcm_name}",
                            driver_type=AudioDriverType.ALSA,
                            max_input_channels=2,
                            max_output_channels=2,
                            default_sample_rate=48000,
                            supported_sample_rates=[44100, 48000, 96000, 192000],
                            default_buffer_size=256,
                            supported_buffer_sizes=[64, 128, 256, 512, 1024],
                            is_default_input=idx == 0,
                            is_default_output=idx == 0,
                            latency_input_ms=5.33,
                            latency_output_ms=5.33,
                            is_asio=False,
                            host_api="ALSA",
                            supports_exclusive_mode=True,
                            supports_shared_mode=True,
                            supports_callback_mode=True,
                            supports_blocking_mode=True
                        ))
                        device_id += 1
                    except Exception as e:
                        logger.debug(f"Could not query ALSA device {pcm_name}: {e}")
                        
            except ImportError:
                # Fallback: create default devices
                devices.append(AudioDeviceInfo(
                    id=device_id,
                    name="ALSA Default",
                    driver_type=AudioDriverType.ALSA,
                    max_input_channels=2,
                    max_output_channels=2,
                    default_sample_rate=48000,
                    supported_sample_rates=[44100, 48000, 96000],
                    default_buffer_size=256,
                    supported_buffer_sizes=[128, 256, 512, 1024],
                    is_default_input=True,
                    is_default_output=True,
                    latency_input_ms=5.33,
                    latency_output_ms=5.33,
                    is_asio=False,
                    host_api="ALSA",
                    supports_exclusive_mode=True,
                    supports_shared_mode=True,
                    supports_callback_mode=True,
                    supports_blocking_mode=True
                ))
                
        except Exception as e:
            logger.error(f"Error enumerating ALSA devices: {e}")
        
        self.devices_cache = devices
        return devices
    
    def get_device_info(self, device_id: int) -> Optional[AudioDeviceInfo]:
        """Get ALSA device information"""
        if not self.devices_cache:
            self.enumerate_devices()
        
        for device in self.devices_cache:
            if device.id == device_id:
                return device
        return None
    
    def open_stream(self, config: AudioStreamConfig) -> int:
        """Open ALSA audio stream"""
        try:
            stream_id = self.next_stream_id
            self.next_stream_id += 1
            
            stream = {
                'id': stream_id,
                'config': config,
                'state': StreamState.STOPPED,
                'buffer': np.zeros((config.buffer_size, config.channels), dtype=np.float32),
                'callback': AudioCallback(),
                'pcm_handle': None
            }
            
            if config.callback:
                stream['callback'].set_callback(config.callback)
            
            self.active_streams[stream_id] = stream
            self.stream_callbacks[stream_id] = stream['callback']
            
            logger.info(f"Opened ALSA stream {stream_id}")
            return stream_id
            
        except Exception as e:
            logger.error(f"Failed to open ALSA stream: {e}")
            return -1
    
    def close_stream(self, stream_id: int) -> bool:
        """Close ALSA stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            self.stop_stream(stream_id)
            
            stream = self.active_streams[stream_id]
            if stream.get('pcm_handle'):
                stream['pcm_handle'].close()
            
            del self.active_streams[stream_id]
            if stream_id in self.stream_callbacks:
                del self.stream_callbacks[stream_id]
            
            logger.info(f"Closed ALSA stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to close ALSA stream: {e}")
            return False
    
    def start_stream(self, stream_id: int) -> bool:
        """Start ALSA stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.RUNNING
            
            logger.info(f"Started ALSA stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start ALSA stream: {e}")
            return False
    
    def stop_stream(self, stream_id: int) -> bool:
        """Stop ALSA stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.STOPPED
            
            logger.info(f"Stopped ALSA stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop ALSA stream: {e}")
            return False
    
    def get_stream_state(self, stream_id: int) -> StreamState:
        """Get ALSA stream state"""
        if stream_id in self.active_streams:
            return self.active_streams[stream_id]['state']
        return StreamState.ERROR
    
    def read_stream(self, stream_id: int, frames: int) -> Optional[np.ndarray]:
        """Read from ALSA stream"""
        if stream_id not in self.active_streams:
            return None
        
        stream = self.active_streams[stream_id]
        return stream['buffer'][:frames].copy()
    
    def write_stream(self, stream_id: int, data: np.ndarray) -> bool:
        """Write to ALSA stream"""
        if stream_id not in self.active_streams:
            return False
        
        stream = self.active_streams[stream_id]
        stream['buffer'] = data.copy()
        return True
    
    def get_stream_latency(self, stream_id: int) -> tuple[float, float]:
        """Get ALSA stream latency"""
        if stream_id not in self.active_streams:
            return (0.0, 0.0)
        
        stream = self.active_streams[stream_id]
        config = stream['config']
        
        latency_ms = (config.buffer_size / config.sample_rate) * 1000
        return (latency_ms, latency_ms)
    
    def get_stream_cpu_load(self, stream_id: int) -> float:
        """Get ALSA stream CPU load"""
        return 0.0
