"""
DirectSound driver implementation for Windows
Legacy Windows audio API with broader compatibility
"""

import platform
import numpy as np
from typing import List, Optional, Dict, Any
from .base import (
    AudioDriverBase, AudioDriverType, AudioDeviceInfo,
    AudioStreamConfig, StreamState, AudioCallback
)
from tonesphere.utils.logger import logger


class DirectSoundDriver(AudioDriverBase):
    """
    DirectSound driver implementation for Windows
    
    DirectSound is the legacy Windows audio API. While it has higher
    latency than WASAPI or ASIO, it's widely compatible and works on
    all Windows versions from XP onwards.
    """
    
    def __init__(self):
        super().__init__(AudioDriverType.DIRECTSOUND)
        self.stream_callbacks: Dict[int, AudioCallback] = {}
        self.next_stream_id = 1
        
    def is_available(self) -> bool:
        """Check if DirectSound is available"""
        return platform.system() == "Windows"
    
    def initialize(self) -> bool:
        """Initialize DirectSound driver"""
        if not self.is_available():
            logger.warning("DirectSound not available on this system")
            return False
        
        try:
            self.is_initialized = True
            logger.info("DirectSound driver initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize DirectSound driver: {e}")
            return False
    
    def terminate(self):
        """Terminate DirectSound driver"""
        for stream_id in list(self.active_streams.keys()):
            self.close_stream(stream_id)
        
        self.is_initialized = False
        logger.info("DirectSound driver terminated")
    
    def enumerate_devices(self) -> List[AudioDeviceInfo]:
        """Enumerate DirectSound devices"""
        devices = []
        device_id = 200  # Start DirectSound devices at 200
        
        try:
            # DirectSound typically has default playback and capture devices
            devices.append(AudioDeviceInfo(
                id=device_id,
                name="DirectSound Default Output",
                driver_type=AudioDriverType.DIRECTSOUND,
                max_input_channels=0,
                max_output_channels=2,
                default_sample_rate=44100,
                supported_sample_rates=[44100, 48000],
                default_buffer_size=1024,
                supported_buffer_sizes=[512, 1024, 2048, 4096],
                is_default_input=False,
                is_default_output=True,
                latency_input_ms=0.0,
                latency_output_ms=23.2,  # ~1024 samples @ 44.1kHz
                is_asio=False,
                host_api="DirectSound",
                supports_exclusive_mode=False,
                supports_shared_mode=True,
                supports_callback_mode=True,
                supports_blocking_mode=True
            ))
            device_id += 1
            
            devices.append(AudioDeviceInfo(
                id=device_id,
                name="DirectSound Default Input",
                driver_type=AudioDriverType.DIRECTSOUND,
                max_input_channels=2,
                max_output_channels=0,
                default_sample_rate=44100,
                supported_sample_rates=[44100, 48000],
                default_buffer_size=1024,
                supported_buffer_sizes=[512, 1024, 2048, 4096],
                is_default_input=True,
                is_default_output=False,
                latency_input_ms=23.2,
                latency_output_ms=0.0,
                is_asio=False,
                host_api="DirectSound",
                supports_exclusive_mode=False,
                supports_shared_mode=True,
                supports_callback_mode=True,
                supports_blocking_mode=True
            ))
            device_id += 1
            
            # Detect running audio applications using native detection
            try:
                from tonesphere.utils.app_detector import NativeAppDetector
                detector = NativeAppDetector()
                audio_apps = detector.get_audio_applications()
                
                for app in audio_apps:
                    # Create input device
                    devices.append(AudioDeviceInfo(
                        id=device_id,
                        name=f"{app.name} (DirectSound Input)",
                        driver_type=AudioDriverType.DIRECTSOUND,
                        max_input_channels=2,
                        max_output_channels=0,
                        default_sample_rate=44100,
                        supported_sample_rates=[44100, 48000],
                        default_buffer_size=1024,
                        supported_buffer_sizes=[512, 1024, 2048],
                        is_default_input=False,
                        is_default_output=False,
                        latency_input_ms=23.2,
                        latency_output_ms=0.0,
                        is_asio=False,
                        host_api="DirectSound Application",
                        supports_exclusive_mode=False,
                        supports_shared_mode=True,
                        supports_callback_mode=True,
                        supports_blocking_mode=True
                    ))
                    device_id += 1
                    
                    # Create output device
                    devices.append(AudioDeviceInfo(
                        id=device_id,
                        name=f"{app.name} (DirectSound Output)",
                        driver_type=AudioDriverType.DIRECTSOUND,
                        max_input_channels=0,
                        max_output_channels=2,
                        default_sample_rate=44100,
                        supported_sample_rates=[44100, 48000],
                        default_buffer_size=1024,
                        supported_buffer_sizes=[512, 1024, 2048],
                        is_default_input=False,
                        is_default_output=False,
                        latency_input_ms=0.0,
                        latency_output_ms=23.2,
                        is_asio=False,
                        host_api="DirectSound Application",
                        supports_exclusive_mode=False,
                        supports_shared_mode=True,
                        supports_callback_mode=True,
                        supports_blocking_mode=True
                    ))
                    device_id += 1
                
                if audio_apps:
                    logger.info(f"Detected {len(audio_apps)} audio applications via DirectSound")
                    
            except Exception as e:
                logger.debug(f"Could not detect audio applications: {e}")
            
        except Exception as e:
            logger.error(f"Error enumerating DirectSound devices: {e}")
        
        self.devices_cache = devices
        return devices
    
    def get_device_info(self, device_id: int) -> Optional[AudioDeviceInfo]:
        """Get DirectSound device information"""
        if not self.devices_cache:
            self.enumerate_devices()
        
        for device in self.devices_cache:
            if device.id == device_id:
                return device
        return None
    
    def open_stream(self, config: AudioStreamConfig) -> int:
        """Open DirectSound audio stream"""
        try:
            stream_id = self.next_stream_id
            self.next_stream_id += 1
            
            stream = {
                'id': stream_id,
                'config': config,
                'state': StreamState.STOPPED,
                'buffer': np.zeros((config.buffer_size, config.channels), dtype=np.float32),
                'callback': AudioCallback()
            }
            
            if config.callback:
                stream['callback'].set_callback(config.callback)
            
            self.active_streams[stream_id] = stream
            self.stream_callbacks[stream_id] = stream['callback']
            
            logger.info(f"Opened DirectSound stream {stream_id}")
            return stream_id
            
        except Exception as e:
            logger.error(f"Failed to open DirectSound stream: {e}")
            return -1
    
    def close_stream(self, stream_id: int) -> bool:
        """Close DirectSound stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            self.stop_stream(stream_id)
            del self.active_streams[stream_id]
            if stream_id in self.stream_callbacks:
                del self.stream_callbacks[stream_id]
            
            logger.info(f"Closed DirectSound stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to close DirectSound stream: {e}")
            return False
    
    def start_stream(self, stream_id: int) -> bool:
        """Start DirectSound stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.RUNNING
            
            logger.info(f"Started DirectSound stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start DirectSound stream: {e}")
            return False
    
    def stop_stream(self, stream_id: int) -> bool:
        """Stop DirectSound stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.STOPPED
            
            logger.info(f"Stopped DirectSound stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop DirectSound stream: {e}")
            return False
    
    def get_stream_state(self, stream_id: int) -> StreamState:
        """Get DirectSound stream state"""
        if stream_id in self.active_streams:
            return self.active_streams[stream_id]['state']
        return StreamState.ERROR
    
    def read_stream(self, stream_id: int, frames: int) -> Optional[np.ndarray]:
        """Read from DirectSound stream"""
        if stream_id not in self.active_streams:
            return None
        
        stream = self.active_streams[stream_id]
        return stream['buffer'][:frames].copy()
    
    def write_stream(self, stream_id: int, data: np.ndarray) -> bool:
        """Write to DirectSound stream"""
        if stream_id not in self.active_streams:
            return False
        
        stream = self.active_streams[stream_id]
        stream['buffer'] = data.copy()
        return True
    
    def get_stream_latency(self, stream_id: int) -> tuple[float, float]:
        """Get DirectSound stream latency"""
        if stream_id not in self.active_streams:
            return (0.0, 0.0)
        
        stream = self.active_streams[stream_id]
        config = stream['config']
        
        latency_ms = (config.buffer_size / config.sample_rate) * 1000
        return (latency_ms, latency_ms)
    
    def get_stream_cpu_load(self, stream_id: int) -> float:
        """Get DirectSound stream CPU load"""
        return 0.0
