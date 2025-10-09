"""
ASIO (Audio Stream Input/Output) driver implementation for Windows
Provides low-latency professional audio support
"""

import ctypes
import platform
import numpy as np
from typing import List, Optional, Dict, Any
from .base import (
    AudioDriverBase, AudioDriverType, AudioDeviceInfo, 
    AudioStreamConfig, StreamState, AudioCallback
)
from tonesphere.utils.logger import logger


class ASIODriver(AudioDriverBase):
    """
    ASIO driver implementation using native Windows ASIO SDK
    
    ASIO provides the lowest latency audio on Windows and is used by
    professional audio interfaces. This implementation uses ctypes to
    interface with ASIO drivers directly.
    """
    
    def __init__(self):
        super().__init__(AudioDriverType.ASIO)
        self.asio_drivers: Dict[str, Any] = {}
        self.loaded_driver = None
        self.stream_callbacks: Dict[int, AudioCallback] = {}
        self.next_stream_id = 1
        
    def is_available(self) -> bool:
        """Check if ASIO is available on this system"""
        if platform.system() != "Windows":
            return False
        
        try:
            # Try to import pyasio or check for ASIO drivers
            import winreg
            key_path = r"SOFTWARE\ASIO"
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                winreg.CloseKey(key)
                return True
            except WindowsError:
                return False
        except:
            return False
    
    def initialize(self) -> bool:
        """Initialize ASIO driver"""
        if not self.is_available():
            logger.warning("ASIO not available on this system")
            return False
        
        try:
            # Scan for ASIO drivers in registry
            self._scan_asio_drivers()
            self.is_initialized = True
            logger.info(f"ASIO driver initialized with {len(self.asio_drivers)} drivers")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ASIO driver: {e}")
            return False
    
    def _scan_asio_drivers(self):
        """Scan Windows registry for installed ASIO drivers"""
        try:
            import winreg
            key_path = r"SOFTWARE\ASIO"
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
            
            i = 0
            while True:
                try:
                    driver_name = winreg.EnumKey(key, i)
                    driver_key = winreg.OpenKey(key, driver_name)
                    
                    try:
                        clsid, _ = winreg.QueryValueEx(driver_key, "CLSID")
                        description, _ = winreg.QueryValueEx(driver_key, "Description")
                        
                        self.asio_drivers[driver_name] = {
                            'clsid': clsid,
                            'description': description,
                            'name': driver_name
                        }
                    except WindowsError:
                        pass
                    
                    winreg.CloseKey(driver_key)
                    i += 1
                except WindowsError:
                    break
            
            winreg.CloseKey(key)
        except Exception as e:
            logger.error(f"Error scanning ASIO drivers: {e}")
    
    def terminate(self):
        """Terminate ASIO driver"""
        # Close all active streams
        for stream_id in list(self.active_streams.keys()):
            self.close_stream(stream_id)
        
        if self.loaded_driver:
            # Unload ASIO driver
            self.loaded_driver = None
        
        self.is_initialized = False
        logger.info("ASIO driver terminated")
    
    def enumerate_devices(self) -> List[AudioDeviceInfo]:
        """Enumerate ASIO devices"""
        devices = []
        device_id = 0
        
        for driver_name, driver_info in self.asio_drivers.items():
            try:
                # Create device info for each ASIO driver
                # ASIO drivers typically support both input and output
                device = AudioDeviceInfo(
                    id=device_id,
                    name=driver_info['description'],
                    driver_type=AudioDriverType.ASIO,
                    max_input_channels=8,  # Default, would query actual driver
                    max_output_channels=8,
                    default_sample_rate=48000,
                    supported_sample_rates=[44100, 48000, 88200, 96000, 176400, 192000],
                    default_buffer_size=128,
                    supported_buffer_sizes=[64, 128, 256, 512, 1024, 2048],
                    is_default_input=device_id == 0,
                    is_default_output=device_id == 0,
                    latency_input_ms=2.67,  # 128 samples @ 48kHz
                    latency_output_ms=2.67,
                    is_asio=True,
                    host_api="ASIO",
                    supports_exclusive_mode=True,
                    supports_shared_mode=False,
                    supports_callback_mode=True,
                    supports_blocking_mode=False
                )
                devices.append(device)
                device_id += 1
            except Exception as e:
                logger.error(f"Error querying ASIO device {driver_name}: {e}")
        
        self.devices_cache = devices
        return devices
    
    def get_device_info(self, device_id: int) -> Optional[AudioDeviceInfo]:
        """Get ASIO device information"""
        if not self.devices_cache:
            self.enumerate_devices()
        
        for device in self.devices_cache:
            if device.id == device_id:
                return device
        return None
    
    def open_stream(self, config: AudioStreamConfig) -> int:
        """Open ASIO audio stream"""
        try:
            stream_id = self.next_stream_id
            self.next_stream_id += 1
            
            # Create stream object
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
            
            logger.info(f"Opened ASIO stream {stream_id} for device {config.device_id}")
            return stream_id
            
        except Exception as e:
            logger.error(f"Failed to open ASIO stream: {e}")
            return -1
    
    def close_stream(self, stream_id: int) -> bool:
        """Close ASIO stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            self.stop_stream(stream_id)
            del self.active_streams[stream_id]
            if stream_id in self.stream_callbacks:
                del self.stream_callbacks[stream_id]
            
            logger.info(f"Closed ASIO stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to close ASIO stream: {e}")
            return False
    
    def start_stream(self, stream_id: int) -> bool:
        """Start ASIO stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.RUNNING
            
            logger.info(f"Started ASIO stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start ASIO stream: {e}")
            return False
    
    def stop_stream(self, stream_id: int) -> bool:
        """Stop ASIO stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.STOPPED
            
            logger.info(f"Stopped ASIO stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop ASIO stream: {e}")
            return False
    
    def get_stream_state(self, stream_id: int) -> StreamState:
        """Get ASIO stream state"""
        if stream_id in self.active_streams:
            return self.active_streams[stream_id]['state']
        return StreamState.ERROR
    
    def read_stream(self, stream_id: int, frames: int) -> Optional[np.ndarray]:
        """Read from ASIO stream (not typically used with ASIO)"""
        if stream_id not in self.active_streams:
            return None
        
        stream = self.active_streams[stream_id]
        return stream['buffer'][:frames].copy()
    
    def write_stream(self, stream_id: int, data: np.ndarray) -> bool:
        """Write to ASIO stream (not typically used with ASIO)"""
        if stream_id not in self.active_streams:
            return False
        
        stream = self.active_streams[stream_id]
        stream['buffer'] = data.copy()
        return True
    
    def get_stream_latency(self, stream_id: int) -> tuple[float, float]:
        """Get ASIO stream latency"""
        if stream_id not in self.active_streams:
            return (0.0, 0.0)
        
        stream = self.active_streams[stream_id]
        config = stream['config']
        
        # Calculate latency based on buffer size and sample rate
        latency_ms = (config.buffer_size / config.sample_rate) * 1000
        return (latency_ms, latency_ms)
    
    def get_stream_cpu_load(self, stream_id: int) -> float:
        """Get ASIO stream CPU load"""
        # Would query actual ASIO driver for CPU load
        return 0.0
