"""
WASAPI (Windows Audio Session API) driver implementation
Provides modern Windows audio support with low latency
"""

import platform
import ctypes
import numpy as np
from typing import List, Optional, Dict, Any
from .base import (
    AudioDriverBase, AudioDriverType, AudioDeviceInfo,
    AudioStreamConfig, StreamState, AudioCallback
)
from tonesphere.utils.logger import logger


class WASAPIDriver(AudioDriverBase):
    """
    WASAPI driver implementation for Windows Vista and later
    
    WASAPI provides both shared and exclusive mode audio with lower
    latency than DirectSound. Exclusive mode provides ASIO-like latency.
    """
    
    def __init__(self):
        super().__init__(AudioDriverType.WASAPI)
        self.com_initialized = False
        self.audio_client = None
        self.stream_callbacks: Dict[int, AudioCallback] = {}
        self.next_stream_id = 1
        
    def is_available(self) -> bool:
        """Check if WASAPI is available"""
        if platform.system() != "Windows":
            return False
        
        try:
            # Check Windows version (Vista+)
            version = platform.version()
            major_version = int(version.split('.')[0])
            return major_version >= 6  # Vista is 6.0
        except:
            return False
    
    def initialize(self) -> bool:
        """Initialize WASAPI driver"""
        if not self.is_available():
            logger.warning("WASAPI not available on this system")
            return False
        
        try:
            # Initialize COM
            if platform.system() == "Windows":
                try:
                    import comtypes
                    comtypes.CoInitialize()
                    self.com_initialized = True
                except:
                    # Fallback to basic initialization
                    pass
            
            self.is_initialized = True
            logger.info("WASAPI driver initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WASAPI driver: {e}")
            return False
    
    def terminate(self):
        """Terminate WASAPI driver"""
        # Close all active streams
        for stream_id in list(self.active_streams.keys()):
            self.close_stream(stream_id)
        
        # Uninitialize COM
        if self.com_initialized:
            try:
                import comtypes
                comtypes.CoUninitialize()
            except:
                pass
        
        self.is_initialized = False
        logger.info("WASAPI driver terminated")
    
    def enumerate_devices(self) -> List[AudioDeviceInfo]:
        """Enumerate WASAPI devices"""
        devices = []
        
        try:
            # Use Windows MMDevice API to enumerate devices
            if platform.system() == "Windows":
                devices = self._enumerate_windows_devices()
        except Exception as e:
            logger.error(f"Error enumerating WASAPI devices: {e}")
        
        self.devices_cache = devices
        return devices
    
    def _enumerate_windows_devices(self) -> List[AudioDeviceInfo]:
        """Enumerate Windows audio devices using MMDevice API and detect audio applications"""
        devices = []
        device_id = 100  # Start WASAPI devices at 100
        
        try:
            import comtypes
            from comtypes import GUID, POINTER
            
            # This is a simplified version - full implementation would use
            # IMMDeviceEnumerator and IMMDevice interfaces
            
            # For now, create placeholder devices
            # In production, this would enumerate actual WASAPI devices
            
            # Default output device
            devices.append(AudioDeviceInfo(
                id=device_id,
                name="WASAPI Default Output",
                driver_type=AudioDriverType.WASAPI,
                max_input_channels=0,
                max_output_channels=2,
                default_sample_rate=48000,
                supported_sample_rates=[44100, 48000, 96000, 192000],
                default_buffer_size=480,
                supported_buffer_sizes=[240, 480, 960, 1920],
                is_default_input=False,
                is_default_output=True,
                latency_input_ms=0.0,
                latency_output_ms=10.0,
                is_asio=False,
                host_api="WASAPI",
                supports_exclusive_mode=True,
                supports_shared_mode=True,
                supports_callback_mode=True,
                supports_blocking_mode=True
            ))
            device_id += 1
            
            # Default input device
            devices.append(AudioDeviceInfo(
                id=device_id,
                name="WASAPI Default Input",
                driver_type=AudioDriverType.WASAPI,
                max_input_channels=2,
                max_output_channels=0,
                default_sample_rate=48000,
                supported_sample_rates=[44100, 48000, 96000, 192000],
                default_buffer_size=480,
                supported_buffer_sizes=[240, 480, 960, 1920],
                is_default_input=True,
                is_default_output=False,
                latency_input_ms=10.0,
                latency_output_ms=0.0,
                is_asio=False,
                host_api="WASAPI",
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
                    # Create input device (receive audio from app)
                    devices.append(AudioDeviceInfo(
                        id=device_id,
                        name=f"{app.name} (WASAPI Input)",
                        driver_type=AudioDriverType.WASAPI,
                        max_input_channels=2,
                        max_output_channels=0,
                        default_sample_rate=48000,
                        supported_sample_rates=[44100, 48000, 96000],
                        default_buffer_size=480,
                        supported_buffer_sizes=[240, 480, 960],
                        is_default_input=False,
                        is_default_output=False,
                        latency_input_ms=10.0,
                        latency_output_ms=0.0,
                        is_asio=False,
                        host_api="WASAPI Application",
                        supports_exclusive_mode=False,
                        supports_shared_mode=True,
                        supports_callback_mode=True,
                        supports_blocking_mode=True
                    ))
                    device_id += 1
                    
                    # Create output device (send audio to app)
                    devices.append(AudioDeviceInfo(
                        id=device_id,
                        name=f"{app.name} (WASAPI Output)",
                        driver_type=AudioDriverType.WASAPI,
                        max_input_channels=0,
                        max_output_channels=2,
                        default_sample_rate=48000,
                        supported_sample_rates=[44100, 48000, 96000],
                        default_buffer_size=480,
                        supported_buffer_sizes=[240, 480, 960],
                        is_default_input=False,
                        is_default_output=False,
                        latency_input_ms=0.0,
                        latency_output_ms=10.0,
                        is_asio=False,
                        host_api="WASAPI Application",
                        supports_exclusive_mode=False,
                        supports_shared_mode=True,
                        supports_callback_mode=True,
                        supports_blocking_mode=True
                    ))
                    device_id += 1
                
                if audio_apps:
                    logger.info(f"Detected {len(audio_apps)} audio applications via WASAPI")
                    
            except Exception as e:
                logger.debug(f"Could not detect audio applications: {e}")
            
        except Exception as e:
            logger.error(f"Error in _enumerate_windows_devices: {e}")
        
        return devices
    
    def get_device_info(self, device_id: int) -> Optional[AudioDeviceInfo]:
        """Get WASAPI device information"""
        if not self.devices_cache:
            self.enumerate_devices()
        
        for device in self.devices_cache:
            if device.id == device_id:
                return device
        return None
    
    def open_stream(self, config: AudioStreamConfig) -> int:
        """Open WASAPI audio stream"""
        try:
            stream_id = self.next_stream_id
            self.next_stream_id += 1
            
            # Create stream object
            stream = {
                'id': stream_id,
                'config': config,
                'state': StreamState.STOPPED,
                'buffer': np.zeros((config.buffer_size, config.channels), dtype=np.float32),
                'callback': AudioCallback(),
                'exclusive_mode': config.exclusive_mode
            }
            
            if config.callback:
                stream['callback'].set_callback(config.callback)
            
            self.active_streams[stream_id] = stream
            self.stream_callbacks[stream_id] = stream['callback']
            
            logger.info(f"Opened WASAPI stream {stream_id} (exclusive={config.exclusive_mode})")
            return stream_id
            
        except Exception as e:
            logger.error(f"Failed to open WASAPI stream: {e}")
            return -1
    
    def close_stream(self, stream_id: int) -> bool:
        """Close WASAPI stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            self.stop_stream(stream_id)
            del self.active_streams[stream_id]
            if stream_id in self.stream_callbacks:
                del self.stream_callbacks[stream_id]
            
            logger.info(f"Closed WASAPI stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to close WASAPI stream: {e}")
            return False
    
    def start_stream(self, stream_id: int) -> bool:
        """Start WASAPI stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.RUNNING
            
            logger.info(f"Started WASAPI stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start WASAPI stream: {e}")
            return False
    
    def stop_stream(self, stream_id: int) -> bool:
        """Stop WASAPI stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.STOPPED
            
            logger.info(f"Stopped WASAPI stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop WASAPI stream: {e}")
            return False
    
    def get_stream_state(self, stream_id: int) -> StreamState:
        """Get WASAPI stream state"""
        if stream_id in self.active_streams:
            return self.active_streams[stream_id]['state']
        return StreamState.ERROR
    
    def read_stream(self, stream_id: int, frames: int) -> Optional[np.ndarray]:
        """Read from WASAPI stream"""
        if stream_id not in self.active_streams:
            return None
        
        stream = self.active_streams[stream_id]
        return stream['buffer'][:frames].copy()
    
    def write_stream(self, stream_id: int, data: np.ndarray) -> bool:
        """Write to WASAPI stream"""
        if stream_id not in self.active_streams:
            return False
        
        stream = self.active_streams[stream_id]
        stream['buffer'] = data.copy()
        return True
    
    def get_stream_latency(self, stream_id: int) -> tuple[float, float]:
        """Get WASAPI stream latency"""
        if stream_id not in self.active_streams:
            return (0.0, 0.0)
        
        stream = self.active_streams[stream_id]
        config = stream['config']
        
        # Exclusive mode has lower latency
        if stream['exclusive_mode']:
            latency_ms = (config.buffer_size / config.sample_rate) * 1000
        else:
            # Shared mode has higher latency
            latency_ms = (config.buffer_size / config.sample_rate) * 1000 * 2
        
        return (latency_ms, latency_ms)
    
    def get_stream_cpu_load(self, stream_id: int) -> float:
        """Get WASAPI stream CPU load"""
        return 0.0
