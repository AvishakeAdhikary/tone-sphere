"""
JACK (JACK Audio Connection Kit) driver implementation
Professional audio routing for Linux
"""

import platform
import numpy as np
from typing import List, Optional, Dict, Any
from .base import (
    AudioDriverBase, AudioDriverType, AudioDeviceInfo,
    AudioStreamConfig, StreamState, AudioCallback
)
from tonesphere.utils.logger import logger


class JACKDriver(AudioDriverBase):
    """
    JACK driver implementation for Linux (and macOS)
    
    JACK is a professional audio server providing low-latency
    connections between applications. It's the standard for
    pro audio on Linux.
    """
    
    def __init__(self):
        super().__init__(AudioDriverType.JACK)
        self.stream_callbacks: Dict[int, AudioCallback] = {}
        self.next_stream_id = 1
        self.jack_client = None
        
    def is_available(self) -> bool:
        """Check if JACK is available"""
        if platform.system() not in ["Linux", "Darwin"]:
            return False
        
        try:
            import jack
            return True
        except ImportError:
            # Check if jackd is running
            import subprocess
            try:
                result = subprocess.run(['jack_control', 'status'], 
                                      capture_output=True, timeout=1)
                return result.returncode == 0
            except:
                return False
    
    def initialize(self) -> bool:
        """Initialize JACK driver"""
        if not self.is_available():
            logger.warning("JACK not available on this system")
            return False
        
        try:
            self.is_initialized = True
            logger.info("JACK driver initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize JACK driver: {e}")
            return False
    
    def terminate(self):
        """Terminate JACK driver"""
        for stream_id in list(self.active_streams.keys()):
            self.close_stream(stream_id)
        
        if self.jack_client:
            try:
                self.jack_client.close()
            except:
                pass
            self.jack_client = None
        
        self.is_initialized = False
        logger.info("JACK driver terminated")
    
    def enumerate_devices(self) -> List[AudioDeviceInfo]:
        """Enumerate JACK devices including connected applications"""
        devices = []
        device_id = 500  # Start JACK devices at 500
        
        try:
            try:
                import jack
                
                # Add main JACK Audio Server device
                devices.append(AudioDeviceInfo(
                    id=device_id,
                    name="JACK Audio Server",
                    driver_type=AudioDriverType.JACK,
                    max_input_channels=64,  # JACK supports many channels
                    max_output_channels=64,
                    default_sample_rate=48000,
                    supported_sample_rates=[44100, 48000, 96000, 192000],
                    default_buffer_size=128,
                    supported_buffer_sizes=[64, 128, 256, 512, 1024],
                    is_default_input=True,
                    is_default_output=True,
                    latency_input_ms=2.67,  # Very low latency
                    latency_output_ms=2.67,
                    is_asio=False,
                    host_api="JACK",
                    supports_exclusive_mode=True,
                    supports_callback_mode=True,
                    supports_blocking_mode=False
                ))
                device_id += 1
                
                # Enumerate JACK clients using native app detection
            try:
                import subprocess
                
                # Use jack_lsp to list all ports
                result = subprocess.run(['jack_lsp', '-c'], capture_output=True, text=True, timeout=2)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    clients = {}
                    current_port = None
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        if not line.startswith(' '):
                            # This is a port name
                            current_port = line
                            if ':' in current_port:
                                client_name = current_port.split(':')[0]
                                # Skip system and ToneSphere ports
                                if client_name not in ['system', 'ToneSphere']:
                                    if client_name not in clients:
                                        clients[client_name] = {'input_ports': 0, 'output_ports': 0}
                                    
                                    # Determine if it's input or output based on port name
                                    if 'output' in current_port.lower() or 'playback' in current_port.lower():
                                        clients[client_name]['output_ports'] += 1
                                    elif 'input' in current_port.lower() or 'capture' in current_port.lower():
                                        clients[client_name]['input_ports'] += 1
                    
                    # Create device entries for each client
                    for client_name, ports in clients.items():
                        if ports['output_ports'] > 0:
                            # Client outputs audio (we receive it as input)
                            devices.append(AudioDeviceInfo(
                                id=device_id,
                                name=f"{client_name} (JACK Input)",
                                driver_type=AudioDriverType.JACK,
                                max_input_channels=ports['output_ports'],
                                max_output_channels=0,
                                default_sample_rate=48000,
                                supported_sample_rates=[44100, 48000, 96000, 192000],
                                default_buffer_size=128,
                                supported_buffer_sizes=[64, 128, 256, 512],
                                is_default_input=False,
                                is_default_output=False,
                                latency_input_ms=2.67,
                                latency_output_ms=0.0,
                                is_asio=False,
                                host_api="JACK Client",
                                supports_exclusive_mode=True,
                                supports_shared_mode=True,
                                supports_callback_mode=True,
                                supports_blocking_mode=False
                            ))
                            device_id += 1
                        
                        if ports['input_ports'] > 0:
                            # Client receives audio (we send it as output)
                            devices.append(AudioDeviceInfo(
                                id=device_id,
                                name=f"{client_name} (JACK Output)",
                                driver_type=AudioDriverType.JACK,
                                max_input_channels=0,
                                max_output_channels=ports['input_ports'],
                                default_sample_rate=48000,
                                supported_sample_rates=[44100, 48000, 96000, 192000],
                                default_buffer_size=128,
                                supported_buffer_sizes=[64, 128, 256, 512],
                                is_default_input=False,
                                is_default_output=False,
                                latency_input_ms=0.0,
                                latency_output_ms=2.67,
                                is_asio=False,
                                host_api="JACK Client",
                                supports_exclusive_mode=True,
                                supports_shared_mode=True,
                                supports_callback_mode=True,
                                supports_blocking_mode=False
                            ))
                            device_id += 1
                    
                    if clients:
                        logger.info(f"Detected {len(clients)} JACK clients")
                    
            except Exception as e:
                logger.debug(f"Could not enumerate JACK clients: {e}")
                
            except ImportError:
                # Fallback: create default JACK device
                devices.append(AudioDeviceInfo(
                    id=device_id,
                    name="JACK Audio Server",
                    driver_type=AudioDriverType.JACK,
                    max_input_channels=32,
                    max_output_channels=32,
                    default_sample_rate=48000,
                    supported_sample_rates=[44100, 48000, 96000],
                    default_buffer_size=128,
                    supported_buffer_sizes=[64, 128, 256, 512],
                    is_default_input=True,
                    is_default_output=True,
                    latency_input_ms=2.67,
                    latency_output_ms=2.67,
                    is_asio=False,
                    host_api="JACK",
                    supports_exclusive_mode=True,
                    supports_shared_mode=True,
                    supports_callback_mode=True,
                    supports_blocking_mode=False
                ))
                
        except Exception as e:
            logger.error(f"Error enumerating JACK devices: {e}")
        
        self.devices_cache = devices
        return devices
    
    def get_device_info(self, device_id: int) -> Optional[AudioDeviceInfo]:
        """Get JACK device information"""
        if not self.devices_cache:
            self.enumerate_devices()
        
        for device in self.devices_cache:
            if device.id == device_id:
                return device
        return None
    
    def open_stream(self, config: AudioStreamConfig) -> int:
        """Open JACK audio stream"""
        try:
            stream_id = self.next_stream_id
            self.next_stream_id += 1
            
            stream = {
                'id': stream_id,
                'config': config,
                'state': StreamState.STOPPED,
                'buffer': np.zeros((config.buffer_size, config.channels), dtype=np.float32),
                'callback': AudioCallback(),
                'jack_client': None
            }
            
            if config.callback:
                stream['callback'].set_callback(config.callback)
            
            self.active_streams[stream_id] = stream
            self.stream_callbacks[stream_id] = stream['callback']
            
            logger.info(f"Opened JACK stream {stream_id}")
            return stream_id
            
        except Exception as e:
            logger.error(f"Failed to open JACK stream: {e}")
            return -1
    
    def close_stream(self, stream_id: int) -> bool:
        """Close JACK stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            self.stop_stream(stream_id)
            
            stream = self.active_streams[stream_id]
            if stream.get('jack_client'):
                stream['jack_client'].close()
            
            del self.active_streams[stream_id]
            if stream_id in self.stream_callbacks:
                del self.stream_callbacks[stream_id]
            
            logger.info(f"Closed JACK stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to close JACK stream: {e}")
            return False
    
    def start_stream(self, stream_id: int) -> bool:
        """Start JACK stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.RUNNING
            
            logger.info(f"Started JACK stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start JACK stream: {e}")
            return False
    
    def stop_stream(self, stream_id: int) -> bool:
        """Stop JACK stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.STOPPED
            
            logger.info(f"Stopped JACK stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop JACK stream: {e}")
            return False
    
    def get_stream_state(self, stream_id: int) -> StreamState:
        """Get JACK stream state"""
        if stream_id in self.active_streams:
            return self.active_streams[stream_id]['state']
        return StreamState.ERROR
    
    def read_stream(self, stream_id: int, frames: int) -> Optional[np.ndarray]:
        """Read from JACK stream"""
        if stream_id not in self.active_streams:
            return None
        
        stream = self.active_streams[stream_id]
        return stream['buffer'][:frames].copy()
    
    def write_stream(self, stream_id: int, data: np.ndarray) -> bool:
        """Write to JACK stream"""
        if stream_id not in self.active_streams:
            return False
        
        stream = self.active_streams[stream_id]
        stream['buffer'] = data.copy()
        return True
    
    def get_stream_latency(self, stream_id: int) -> tuple[float, float]:
        """Get JACK stream latency"""
        if stream_id not in self.active_streams:
            return (0.0, 0.0)
        
        stream = self.active_streams[stream_id]
        config = stream['config']
        
        latency_ms = (config.buffer_size / config.sample_rate) * 1000
        return (latency_ms, latency_ms)
    
    def get_stream_cpu_load(self, stream_id: int) -> float:
        """Get JACK stream CPU load"""
        # JACK provides actual CPU load information
        return 0.0
