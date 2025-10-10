"""
Wireless Audio Driver (Bluetooth & WiFi Audio)
Native OS-level support for wireless audio devices
"""

import platform
import subprocess
from typing import List, Optional, Dict, Any
from .base import (
    AudioDriverBase, AudioDriverType, AudioDeviceInfo,
    AudioStreamConfig, StreamState, AudioCallback
)
from tonesphere.utils.logger import logger
import numpy as np


class WirelessAudioDriver(AudioDriverBase):
    """
    Wireless audio driver for Bluetooth and WiFi audio devices
    Integrates with OS-level Bluetooth and network audio services
    """
    
    def __init__(self):
        super().__init__(AudioDriverType.AUTO)  # Will be set based on platform
        self.system = platform.system()
        self.bluetooth_devices = {}
        self.wifi_devices = {}
        self.stream_callbacks: Dict[int, AudioCallback] = {}
        self.next_stream_id = 1
        
    def is_available(self) -> bool:
        """Check if wireless audio is available"""
        if self.system == "Windows":
            return self._check_windows_bluetooth()
        elif self.system == "Linux":
            return self._check_linux_bluetooth()
        elif self.system == "Darwin":
            return self._check_macos_bluetooth()
        return False
    
    def _check_windows_bluetooth(self) -> bool:
        """Check Windows Bluetooth availability"""
        try:
            import ctypes
            # Check if Bluetooth radio is available
            result = subprocess.run(
                ['powershell', '-Command', 'Get-PnpDevice -Class Bluetooth'],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False
    
    def _check_linux_bluetooth(self) -> bool:
        """Check Linux Bluetooth availability"""
        try:
            # Check if bluetoothctl is available
            result = subprocess.run(['which', 'bluetoothctl'], capture_output=True)
            if result.returncode == 0:
                return True
            
            # Check if bluez is running
            result = subprocess.run(['systemctl', 'is-active', 'bluetooth'], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    def _check_macos_bluetooth(self) -> bool:
        """Check macOS Bluetooth availability"""
        try:
            result = subprocess.run(
                ['system_profiler', 'SPBluetoothDataType'],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False
    
    def initialize(self) -> bool:
        """Initialize wireless audio driver"""
        if not self.is_available():
            logger.warning("Wireless audio not available on this system")
            return False
        
        try:
            self.is_initialized = True
            logger.info("Wireless audio driver initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize wireless audio driver: {e}")
            return False
    
    def terminate(self):
        """Terminate wireless audio driver"""
        for stream_id in list(self.active_streams.keys()):
            self.close_stream(stream_id)
        
        self.bluetooth_devices.clear()
        self.wifi_devices.clear()
        self.is_initialized = False
        logger.info("Wireless audio driver terminated")
    
    def enumerate_devices(self) -> List[AudioDeviceInfo]:
        """Enumerate wireless audio devices"""
        devices = []
        device_id = 1000  # Start wireless devices at 1000
        
        try:
            if self.system == "Windows":
                devices.extend(self._enumerate_windows_wireless(device_id))
            elif self.system == "Linux":
                devices.extend(self._enumerate_linux_wireless(device_id))
            elif self.system == "Darwin":
                devices.extend(self._enumerate_macos_wireless(device_id))
        except Exception as e:
            logger.error(f"Error enumerating wireless devices: {e}")
        
        self.devices_cache = devices
        return devices
    
    def _enumerate_windows_wireless(self, start_id: int) -> List[AudioDeviceInfo]:
        """Enumerate Windows Bluetooth and WiFi audio devices"""
        devices = []
        device_id = start_id
        
        try:
            # Get Bluetooth audio devices using PowerShell
            ps_script = """
            Get-PnpDevice -Class AudioEndpoint | Where-Object {
                $_.FriendlyName -match 'Bluetooth|Wireless|A2DP|Hands-Free'
            } | Select-Object FriendlyName, Status | ConvertTo-Json
            """
            
            result = subprocess.run(
                ['powershell', '-NoProfile', '-Command', ps_script],
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if result.returncode == 0 and result.stdout.strip():
                import json
                try:
                    bt_devices = json.loads(result.stdout)
                    if not isinstance(bt_devices, list):
                        bt_devices = [bt_devices]
                    
                    for bt_dev in bt_devices:
                        name = bt_dev.get('FriendlyName', 'Unknown Bluetooth Device')
                        status = bt_dev.get('Status', 'Unknown')
                        
                        if status == 'OK':
                            # Create input device
                            devices.append(AudioDeviceInfo(
                                id=device_id,
                                name=f"{name} (Bluetooth Input)",
                                driver_type=AudioDriverType.WASAPI,
                                max_input_channels=2,
                                max_output_channels=0,
                                default_sample_rate=48000,
                                supported_sample_rates=[44100, 48000],
                                default_buffer_size=960,
                                supported_buffer_sizes=[480, 960, 1920],
                                is_default_input=False,
                                is_default_output=False,
                                latency_input_ms=20.0,
                                latency_output_ms=0.0,
                                is_asio=False,
                                host_api="Bluetooth",
                                supports_exclusive_mode=False,
                                supports_shared_mode=True,
                                supports_callback_mode=True,
                                supports_blocking_mode=True
                            ))
                            device_id += 1
                            
                            # Create output device
                            devices.append(AudioDeviceInfo(
                                id=device_id,
                                name=f"{name} (Bluetooth Output)",
                                driver_type=AudioDriverType.WASAPI,
                                max_input_channels=0,
                                max_output_channels=2,
                                default_sample_rate=48000,
                                supported_sample_rates=[44100, 48000],
                                default_buffer_size=960,
                                supported_buffer_sizes=[480, 960, 1920],
                                is_default_input=False,
                                is_default_output=False,
                                latency_input_ms=0.0,
                                latency_output_ms=20.0,
                                is_asio=False,
                                host_api="Bluetooth",
                                supports_exclusive_mode=False,
                                supports_shared_mode=True,
                                supports_callback_mode=True,
                                supports_blocking_mode=True
                            ))
                            device_id += 1
                except:
                    pass
            
            # Check for WiFi audio devices (AirPlay, DLNA, etc.)
            self._add_wifi_audio_devices_windows(devices, device_id)
            
        except Exception as e:
            logger.debug(f"Error enumerating Windows wireless devices: {e}")
        
        return devices
    
    def _enumerate_linux_wireless(self, start_id: int) -> List[AudioDeviceInfo]:
        """Enumerate Linux Bluetooth and WiFi audio devices"""
        devices = []
        device_id = start_id
        
        try:
            # Use bluetoothctl to list paired devices
            result = subprocess.run(
                ['bluetoothctl', 'devices'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('Device'):
                        parts = line.split(maxsplit=2)
                        if len(parts) >= 3:
                            mac_addr = parts[1]
                            name = parts[2]
                            
                            # Check if device is connected
                            info_result = subprocess.run(
                                ['bluetoothctl', 'info', mac_addr],
                                capture_output=True,
                                text=True,
                                timeout=1
                            )
                            
                            if 'Connected: yes' in info_result.stdout:
                                # Create input device
                                devices.append(AudioDeviceInfo(
                                    id=device_id,
                                    name=f"{name} (Bluetooth Input)",
                                    driver_type=AudioDriverType.PULSEAUDIO,
                                    max_input_channels=2,
                                    max_output_channels=0,
                                    default_sample_rate=48000,
                                    supported_sample_rates=[44100, 48000],
                                    default_buffer_size=1024,
                                    supported_buffer_sizes=[512, 1024, 2048],
                                    is_default_input=False,
                                    is_default_output=False,
                                    latency_input_ms=20.0,
                                    latency_output_ms=0.0,
                                    is_asio=False,
                                    host_api="Bluetooth",
                                    supports_exclusive_mode=False,
                                    supports_shared_mode=True,
                                    supports_callback_mode=True,
                                    supports_blocking_mode=True
                                ))
                                device_id += 1
                                
                                # Create output device
                                devices.append(AudioDeviceInfo(
                                    id=device_id,
                                    name=f"{name} (Bluetooth Output)",
                                    driver_type=AudioDriverType.PULSEAUDIO,
                                    max_input_channels=0,
                                    max_output_channels=2,
                                    default_sample_rate=48000,
                                    supported_sample_rates=[44100, 48000],
                                    default_buffer_size=1024,
                                    supported_buffer_sizes=[512, 1024, 2048],
                                    is_default_input=False,
                                    is_default_output=False,
                                    latency_input_ms=0.0,
                                    latency_output_ms=20.0,
                                    is_asio=False,
                                    host_api="Bluetooth",
                                    supports_exclusive_mode=False,
                                    supports_shared_mode=True,
                                    supports_callback_mode=True,
                                    supports_blocking_mode=True
                                ))
                                device_id += 1
            
            # Check for WiFi audio (PulseAudio network sinks/sources)
            self._add_wifi_audio_devices_linux(devices, device_id)
            
        except Exception as e:
            logger.debug(f"Error enumerating Linux wireless devices: {e}")
        
        return devices
    
    def _enumerate_macos_wireless(self, start_id: int) -> List[AudioDeviceInfo]:
        """Enumerate macOS Bluetooth and WiFi audio devices (AirPlay, etc.)"""
        devices = []
        device_id = start_id
        
        try:
            # Get Bluetooth audio devices
            result = subprocess.run(
                ['system_profiler', 'SPBluetoothDataType', '-json'],
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if result.returncode == 0:
                import json
                try:
                    data = json.loads(result.stdout)
                    bt_data = data.get('SPBluetoothDataType', [])
                    
                    for item in bt_data:
                        devices_list = item.get('device_connected', [])
                        for device in devices_list:
                            name = device.get('device_name', 'Unknown Bluetooth Device')
                            
                            # Create input device
                            devices.append(AudioDeviceInfo(
                                id=device_id,
                                name=f"{name} (Bluetooth Input)",
                                driver_type=AudioDriverType.COREAUDIO,
                                max_input_channels=2,
                                max_output_channels=0,
                                default_sample_rate=48000,
                                supported_sample_rates=[44100, 48000],
                                default_buffer_size=512,
                                supported_buffer_sizes=[256, 512, 1024],
                                is_default_input=False,
                                is_default_output=False,
                                latency_input_ms=15.0,
                                latency_output_ms=0.0,
                                is_asio=False,
                                host_api="Bluetooth",
                                supports_exclusive_mode=False,
                                supports_shared_mode=True,
                                supports_callback_mode=True,
                                supports_blocking_mode=True
                            ))
                            device_id += 1
                            
                            # Create output device
                            devices.append(AudioDeviceInfo(
                                id=device_id,
                                name=f"{name} (Bluetooth Output)",
                                driver_type=AudioDriverType.COREAUDIO,
                                max_input_channels=0,
                                max_output_channels=2,
                                default_sample_rate=48000,
                                supported_sample_rates=[44100, 48000],
                                default_buffer_size=512,
                                supported_buffer_sizes=[256, 512, 1024],
                                is_default_input=False,
                                is_default_output=False,
                                latency_input_ms=0.0,
                                latency_output_ms=15.0,
                                is_asio=False,
                                host_api="Bluetooth",
                                supports_exclusive_mode=False,
                                supports_shared_mode=True,
                                supports_callback_mode=True,
                                supports_blocking_mode=True
                            ))
                            device_id += 1
                except:
                    pass
            
            # Check for AirPlay devices
            self._add_airplay_devices(devices, device_id)
            
        except Exception as e:
            logger.debug(f"Error enumerating macOS wireless devices: {e}")
        
        return devices
    
    def _add_wifi_audio_devices_windows(self, devices: List[AudioDeviceInfo], start_id: int):
        """Add Windows WiFi audio devices (DLNA, etc.)"""
        # Implementation for Windows network audio discovery
        pass
    
    def _add_wifi_audio_devices_linux(self, devices: List[AudioDeviceInfo], start_id: int):
        """Add Linux WiFi audio devices (PulseAudio network, etc.)"""
        try:
            # Check for PulseAudio network sinks
            result = subprocess.run(
                ['pactl', 'list', 'sinks', 'short'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if 'network' in line.lower() or 'tunnel' in line.lower():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            name = parts[1]
                            devices.append(AudioDeviceInfo(
                                id=start_id,
                                name=f"{name} (Network Audio)",
                                driver_type=AudioDriverType.PULSEAUDIO,
                                max_input_channels=0,
                                max_output_channels=2,
                                default_sample_rate=48000,
                                supported_sample_rates=[44100, 48000],
                                default_buffer_size=2048,
                                supported_buffer_sizes=[1024, 2048, 4096],
                                is_default_input=False,
                                is_default_output=False,
                                latency_input_ms=0.0,
                                latency_output_ms=50.0,
                                is_asio=False,
                                host_api="Network Audio",
                                supports_exclusive_mode=False,
                                supports_shared_mode=True,
                                supports_callback_mode=True,
                                supports_blocking_mode=True
                            ))
                            start_id += 1
        except:
            pass
    
    def _add_airplay_devices(self, devices: List[AudioDeviceInfo], start_id: int):
        """Add macOS AirPlay devices"""
        # Implementation for AirPlay device discovery
        pass
    
    def get_device_info(self, device_id: int) -> Optional[AudioDeviceInfo]:
        """Get wireless device information"""
        if not self.devices_cache:
            self.enumerate_devices()
        
        for device in self.devices_cache:
            if device.id == device_id:
                return device
        return None
    
    def open_stream(self, config: AudioStreamConfig) -> int:
        """Open wireless audio stream"""
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
            
            logger.info(f"Opened wireless audio stream {stream_id}")
            return stream_id
            
        except Exception as e:
            logger.error(f"Failed to open wireless audio stream: {e}")
            return -1
    
    def close_stream(self, stream_id: int) -> bool:
        """Close wireless audio stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            self.stop_stream(stream_id)
            del self.active_streams[stream_id]
            if stream_id in self.stream_callbacks:
                del self.stream_callbacks[stream_id]
            
            logger.info(f"Closed wireless audio stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to close wireless audio stream: {e}")
            return False
    
    def start_stream(self, stream_id: int) -> bool:
        """Start wireless audio stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.RUNNING
            
            logger.info(f"Started wireless audio stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start wireless audio stream: {e}")
            return False
    
    def stop_stream(self, stream_id: int) -> bool:
        """Stop wireless audio stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.STOPPED
            
            logger.info(f"Stopped wireless audio stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop wireless audio stream: {e}")
            return False
    
    def get_stream_state(self, stream_id: int) -> StreamState:
        """Get wireless audio stream state"""
        if stream_id in self.active_streams:
            return self.active_streams[stream_id]['state']
        return StreamState.ERROR
    
    def read_stream(self, stream_id: int, frames: int) -> Optional[np.ndarray]:
        """Read from wireless audio stream"""
        if stream_id not in self.active_streams:
            return None
        
        stream = self.active_streams[stream_id]
        return stream['buffer'][:frames].copy()
    
    def write_stream(self, stream_id: int, data: np.ndarray) -> bool:
        """Write to wireless audio stream"""
        if stream_id not in self.active_streams:
            return False
        
        stream = self.active_streams[stream_id]
        stream['buffer'] = data.copy()
        return True
    
    def get_stream_latency(self, stream_id: int) -> tuple[float, float]:
        """Get wireless audio stream latency"""
        if stream_id not in self.active_streams:
            return (0.0, 0.0)
        
        stream = self.active_streams[stream_id]
        config = stream['config']
        
        # Wireless audio typically has higher latency
        latency_ms = (config.buffer_size / config.sample_rate) * 1000 + 20.0  # Add 20ms for wireless
        return (latency_ms, latency_ms)
    
    def get_stream_cpu_load(self, stream_id: int) -> float:
        """Get wireless audio stream CPU load"""
        return 0.0
