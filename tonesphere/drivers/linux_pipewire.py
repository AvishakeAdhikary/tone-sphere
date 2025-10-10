"""
PipeWire driver implementation for Linux
Modern multimedia framework replacing PulseAudio and JACK
"""

import platform
import numpy as np
from typing import List, Optional, Dict, Any
from .base import (
    AudioDriverBase, AudioDriverType, AudioDeviceInfo,
    AudioStreamConfig, StreamState, AudioCallback
)
from tonesphere.utils.logger import logger


class PipeWireDriver(AudioDriverBase):
    """
    PipeWire driver implementation for Linux
    
    PipeWire is the next-generation multimedia framework for Linux,
    designed to handle both audio and video with low latency.
    It's compatible with both PulseAudio and JACK applications.
    """
    
    def __init__(self):
        super().__init__(AudioDriverType.PIPEWIRE)
        self.stream_callbacks: Dict[int, AudioCallback] = {}
        self.next_stream_id = 1
        self.pw_context = None
        
    def is_available(self) -> bool:
        """Check if PipeWire is available"""
        if platform.system() != "Linux":
            return False
        
        try:
            # Check if PipeWire is running
            import subprocess
            result = subprocess.run(['pw-cli', 'info', '0'], 
                                  capture_output=True, timeout=1)
            return result.returncode == 0
        except:
            return False
    
    def initialize(self) -> bool:
        """Initialize PipeWire driver"""
        if not self.is_available():
            logger.warning("PipeWire not available on this system")
            return False
        
        try:
            self.is_initialized = True
            logger.info("PipeWire driver initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PipeWire driver: {e}")
            return False
    
    def terminate(self):
        """Terminate PipeWire driver"""
        for stream_id in list(self.active_streams.keys()):
            self.close_stream(stream_id)
        
        if self.pw_context:
            self.pw_context = None
        
        self.is_initialized = False
        logger.info("PipeWire driver terminated")
    
    def enumerate_devices(self) -> List[AudioDeviceInfo]:
        """Enumerate PipeWire devices including connected applications"""
        devices = []
        device_id = 600  # Start PipeWire devices at 600
        
        try:
            # Add default PipeWire devices
            devices.append(AudioDeviceInfo(
                id=device_id,
                name="PipeWire Default Output",
                driver_type=AudioDriverType.PIPEWIRE,
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
                host_api="PipeWire",
                supports_exclusive_mode=True,
                supports_shared_mode=True,
                supports_callback_mode=True,
                supports_blocking_mode=True
            ))
            device_id += 1
            
            devices.append(AudioDeviceInfo(
                id=device_id,
                name="PipeWire Default Input",
                driver_type=AudioDriverType.PIPEWIRE,
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
                host_api="PipeWire",
                supports_exclusive_mode=True,
                supports_shared_mode=True,
                supports_callback_mode=True,
                supports_blocking_mode=True
            ))
            device_id += 1
            
            # Enumerate PipeWire nodes using native detection
            try:
                import subprocess
                import json
                
                # Use pw-dump to list all nodes
                result = subprocess.run(['pw-dump'], capture_output=True, text=True, timeout=2)
                
                if result.returncode == 0:
                    pw_data = json.loads(result.stdout)
                    
                    # Filter for audio stream nodes (applications)
                    app_nodes = {}
                    for node in pw_data:
                        if node.get('type') != 'PipeWire:Interface:Node':
                            continue
                        
                        info = node.get('info', {})
                        props = info.get('props', {})
                        
                        # Look for application nodes
                        media_class = props.get('media.class', '')
                        app_name = props.get('application.name', props.get('node.name', ''))
                        node_name = props.get('node.name', '')
                        
                        # Skip system nodes and our own
                        if not app_name or 'ToneSphere' in app_name:
                            continue
                        
                        # Filter for audio stream nodes
                        if 'Stream/' in media_class or 'Audio/' in media_class:
                            # Determine if it's a source (output from app) or sink (input to app)
                            is_source = 'Source' in media_class or 'Output' in media_class
                            is_sink = 'Sink' in media_class or 'Input' in media_class
                            
                            if app_name not in app_nodes:
                                app_nodes[app_name] = {
                                    'sources': 0,
                                    'sinks': 0,
                                    'sample_rate': 48000
                                }
                            
                            # Get channel count
                            params = info.get('params', {})
                            format_info = params.get('Format', [{}])[0] if params.get('Format') else {}
                            channels = format_info.get('channels', 2)
                            
                            if is_source:
                                app_nodes[app_name]['sources'] = max(app_nodes[app_name]['sources'], channels)
                            if is_sink:
                                app_nodes[app_name]['sinks'] = max(app_nodes[app_name]['sinks'], channels)
                    
                    # Create device entries for each application
                    for app_name, node_info in app_nodes.items():
                        if node_info['sources'] > 0:
                            # Application outputs audio (we receive it as input)
                            devices.append(AudioDeviceInfo(
                                id=device_id,
                                name=f"{app_name} (PipeWire Input)",
                                driver_type=AudioDriverType.PIPEWIRE,
                                max_input_channels=node_info['sources'],
                                max_output_channels=0,
                                default_sample_rate=node_info['sample_rate'],
                                supported_sample_rates=[44100, 48000, 96000],
                                default_buffer_size=256,
                                supported_buffer_sizes=[64, 128, 256, 512],
                                is_default_input=False,
                                is_default_output=False,
                                latency_input_ms=5.33,
                                latency_output_ms=0.0,
                                is_asio=False,
                                host_api="PipeWire Node",
                                supports_exclusive_mode=True,
                                supports_shared_mode=True,
                                supports_callback_mode=True,
                                supports_blocking_mode=True
                            ))
                            device_id += 1
                        
                        if node_info['sinks'] > 0:
                            # Application receives audio (we send it as output)
                            devices.append(AudioDeviceInfo(
                                id=device_id,
                                name=f"{app_name} (PipeWire Output)",
                                driver_type=AudioDriverType.PIPEWIRE,
                                max_input_channels=0,
                                max_output_channels=node_info['sinks'],
                                default_sample_rate=node_info['sample_rate'],
                                supported_sample_rates=[44100, 48000, 96000],
                                default_buffer_size=256,
                                supported_buffer_sizes=[64, 128, 256, 512],
                                is_default_input=False,
                                is_default_output=False,
                                latency_input_ms=0.0,
                                latency_output_ms=5.33,
                                is_asio=False,
                                host_api="PipeWire Node",
                                supports_exclusive_mode=True,
                                supports_shared_mode=True,
                                supports_callback_mode=True,
                                supports_blocking_mode=True
                            ))
                            device_id += 1
                    
                    if app_nodes:
                        logger.info(f"Detected {len(app_nodes)} PipeWire application nodes")
                    
            except Exception as e:
                logger.debug(f"Could not enumerate PipeWire nodes: {e}")
            
        except Exception as e:
            logger.error(f"Error enumerating PipeWire devices: {e}")
        
        self.devices_cache = devices
        return devices
    
    def get_device_info(self, device_id: int) -> Optional[AudioDeviceInfo]:
        """Get PipeWire device information"""
        if not self.devices_cache:
            self.enumerate_devices()
        
        for device in self.devices_cache:
            if device.id == device_id:
                return device
        return None
    
    def open_stream(self, config: AudioStreamConfig) -> int:
        """Open PipeWire audio stream"""
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
            
            logger.info(f"Opened PipeWire stream {stream_id}")
            return stream_id
            
        except Exception as e:
            logger.error(f"Failed to open PipeWire stream: {e}")
            return -1
    
    def close_stream(self, stream_id: int) -> bool:
        """Close PipeWire stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            self.stop_stream(stream_id)
            del self.active_streams[stream_id]
            if stream_id in self.stream_callbacks:
                del self.stream_callbacks[stream_id]
            
            logger.info(f"Closed PipeWire stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to close PipeWire stream: {e}")
            return False
    
    def start_stream(self, stream_id: int) -> bool:
        """Start PipeWire stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.RUNNING
            
            logger.info(f"Started PipeWire stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start PipeWire stream: {e}")
            return False
    
    def stop_stream(self, stream_id: int) -> bool:
        """Stop PipeWire stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.STOPPED
            
            logger.info(f"Stopped PipeWire stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop PipeWire stream: {e}")
            return False
    
    def get_stream_state(self, stream_id: int) -> StreamState:
        """Get PipeWire stream state"""
        if stream_id in self.active_streams:
            return self.active_streams[stream_id]['state']
        return StreamState.ERROR
    
    def read_stream(self, stream_id: int, frames: int) -> Optional[np.ndarray]:
        """Read from PipeWire stream"""
        if stream_id not in self.active_streams:
            return None
        
        stream = self.active_streams[stream_id]
        return stream['buffer'][:frames].copy()
    
    def write_stream(self, stream_id: int, data: np.ndarray) -> bool:
        """Write to PipeWire stream"""
        if stream_id not in self.active_streams:
            return False
        
        stream = self.active_streams[stream_id]
        stream['buffer'] = data.copy()
        return True
    
    def get_stream_latency(self, stream_id: int) -> tuple[float, float]:
        """Get PipeWire stream latency"""
        if stream_id not in self.active_streams:
            return (0.0, 0.0)
        
        stream = self.active_streams[stream_id]
        config = stream['config']
        
        latency_ms = (config.buffer_size / config.sample_rate) * 1000
        return (latency_ms, latency_ms)
    
    def get_stream_cpu_load(self, stream_id: int) -> float:
        """Get PipeWire stream CPU load"""
        return 0.0
