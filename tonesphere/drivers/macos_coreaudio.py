"""
CoreAudio driver implementation for macOS
Native macOS audio framework
"""

import platform
import numpy as np
from typing import List, Optional, Dict, Any
from .base import (
    AudioDriverBase, AudioDriverType, AudioDeviceInfo,
    AudioStreamConfig, StreamState, AudioCallback
)
from tonesphere.utils.logger import logger


class CoreAudioDriver(AudioDriverBase):
    """
    CoreAudio driver implementation for macOS
    
    CoreAudio is the native audio framework for macOS, providing
    low-latency audio with professional features.
    """
    
    def __init__(self):
        super().__init__(AudioDriverType.COREAUDIO)
        self.stream_callbacks: Dict[int, AudioCallback] = {}
        self.next_stream_id = 1
        self.audio_units: Dict[int, Any] = {}
        
    def is_available(self) -> bool:
        """Check if CoreAudio is available"""
        return platform.system() == "Darwin"
    
    def initialize(self) -> bool:
        """Initialize CoreAudio driver"""
        if not self.is_available():
            logger.warning("CoreAudio not available on this system")
            return False
        
        try:
            self.is_initialized = True
            logger.info("CoreAudio driver initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize CoreAudio driver: {e}")
            return False
    
    def terminate(self):
        """Terminate CoreAudio driver"""
        for stream_id in list(self.active_streams.keys()):
            self.close_stream(stream_id)
        
        self.audio_units.clear()
        self.is_initialized = False
        logger.info("CoreAudio driver terminated")
    
    def enumerate_devices(self) -> List[AudioDeviceInfo]:
        """Enumerate CoreAudio devices"""
        devices = []
        device_id = 700  # Start CoreAudio devices at 700
        
        try:
            # CoreAudio device enumeration would use AudioObjectGetPropertyData
            # For now, create default devices
            
            devices.append(AudioDeviceInfo(
                id=device_id,
                name="CoreAudio Default Output",
                driver_type=AudioDriverType.COREAUDIO,
                max_input_channels=0,
                max_output_channels=2,
                default_sample_rate=48000,
                supported_sample_rates=[44100, 48000, 96000, 192000],
                default_buffer_size=256,
                supported_buffer_sizes=[64, 128, 256, 512, 1024],
                is_default_input=False,
                is_default_output=True,
                latency_input_ms=0.0,
                latency_output_ms=5.33,
                is_asio=False,
                host_api="CoreAudio",
                supports_exclusive_mode=True,
                supports_shared_mode=True,
                supports_callback_mode=True,
                supports_blocking_mode=True
            ))
            device_id += 1
            
            devices.append(AudioDeviceInfo(
                id=device_id,
                name="CoreAudio Default Input",
                driver_type=AudioDriverType.COREAUDIO,
                max_input_channels=2,
                max_output_channels=0,
                default_sample_rate=48000,
                supported_sample_rates=[44100, 48000, 96000, 192000],
                default_buffer_size=256,
                supported_buffer_sizes=[64, 128, 256, 512, 1024],
                is_default_input=True,
                is_default_output=False,
                latency_input_ms=5.33,
                latency_output_ms=0.0,
                is_asio=False,
                host_api="CoreAudio",
                supports_exclusive_mode=True,
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
                        name=f"{app.name} (CoreAudio Input)",
                        driver_type=AudioDriverType.COREAUDIO,
                        max_input_channels=2,
                        max_output_channels=0,
                        default_sample_rate=48000,
                        supported_sample_rates=[44100, 48000, 96000],
                        default_buffer_size=256,
                        supported_buffer_sizes=[128, 256, 512],
                        is_default_input=False,
                        is_default_output=False,
                        latency_input_ms=5.33,
                        latency_output_ms=0.0,
                        is_asio=False,
                        host_api="CoreAudio Application",
                        supports_exclusive_mode=True,
                        supports_shared_mode=True,
                        supports_callback_mode=True,
                        supports_blocking_mode=True
                    ))
                    device_id += 1
                    
                    # Create output device
                    devices.append(AudioDeviceInfo(
                        id=device_id,
                        name=f"{app.name} (CoreAudio Output)",
                        driver_type=AudioDriverType.COREAUDIO,
                        max_input_channels=0,
                        max_output_channels=2,
                        default_sample_rate=48000,
                        supported_sample_rates=[44100, 48000, 96000],
                        default_buffer_size=256,
                        supported_buffer_sizes=[128, 256, 512],
                        is_default_input=False,
                        is_default_output=False,
                        latency_input_ms=0.0,
                        latency_output_ms=5.33,
                        is_asio=False,
                        host_api="CoreAudio Application",
                        supports_exclusive_mode=True,
                        supports_shared_mode=True,
                        supports_callback_mode=True,
                        supports_blocking_mode=True
                    ))
                    device_id += 1
                
                if audio_apps:
                    logger.info(f"Detected {len(audio_apps)} audio applications via CoreAudio")
                    
            except Exception as e:
                logger.debug(f"Could not detect audio applications: {e}")
            
        except Exception as e:
            logger.error(f"Error enumerating CoreAudio devices: {e}")
        
        self.devices_cache = devices
        return devices
    
    def get_device_info(self, device_id: int) -> Optional[AudioDeviceInfo]:
        """Get CoreAudio device information"""
        if not self.devices_cache:
            self.enumerate_devices()
        
        for device in self.devices_cache:
            if device.id == device_id:
                return device
        return None
    
    def open_stream(self, config: AudioStreamConfig) -> int:
        """Open CoreAudio audio stream"""
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
            
            logger.info(f"Opened CoreAudio stream {stream_id}")
            return stream_id
            
        except Exception as e:
            logger.error(f"Failed to open CoreAudio stream: {e}")
            return -1
    
    def close_stream(self, stream_id: int) -> bool:
        """Close CoreAudio stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            self.stop_stream(stream_id)
            del self.active_streams[stream_id]
            if stream_id in self.stream_callbacks:
                del self.stream_callbacks[stream_id]
            
            logger.info(f"Closed CoreAudio stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to close CoreAudio stream: {e}")
            return False
    
    def start_stream(self, stream_id: int) -> bool:
        """Start CoreAudio stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.RUNNING
            
            logger.info(f"Started CoreAudio stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start CoreAudio stream: {e}")
            return False
    
    def stop_stream(self, stream_id: int) -> bool:
        """Stop CoreAudio stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.STOPPED
            
            logger.info(f"Stopped CoreAudio stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop CoreAudio stream: {e}")
            return False
    
    def get_stream_state(self, stream_id: int) -> StreamState:
        """Get CoreAudio stream state"""
        if stream_id in self.active_streams:
            return self.active_streams[stream_id]['state']
        return StreamState.ERROR
    
    def read_stream(self, stream_id: int, frames: int) -> Optional[np.ndarray]:
        """Read from CoreAudio stream"""
        if stream_id not in self.active_streams:
            return None
        
        stream = self.active_streams[stream_id]
        return stream['buffer'][:frames].copy()
    
    def write_stream(self, stream_id: int, data: np.ndarray) -> bool:
        """Write to CoreAudio stream"""
        if stream_id not in self.active_streams:
            return False
        
        stream = self.active_streams[stream_id]
        stream['buffer'] = data.copy()
        return True
    
    def get_stream_latency(self, stream_id: int) -> tuple[float, float]:
        """Get CoreAudio stream latency"""
        if stream_id not in self.active_streams:
            return (0.0, 0.0)
        
        stream = self.active_streams[stream_id]
        config = stream['config']
        
        latency_ms = (config.buffer_size / config.sample_rate) * 1000
        return (latency_ms, latency_ms)
    
    def get_stream_cpu_load(self, stream_id: int) -> float:
        """Get CoreAudio stream CPU load"""
        return 0.0
