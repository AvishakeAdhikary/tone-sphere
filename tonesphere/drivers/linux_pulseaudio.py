"""
PulseAudio driver implementation for Linux
High-level audio server with network support
"""

import platform
import numpy as np
from typing import List, Optional, Dict, Any
from .base import (
    AudioDriverBase, AudioDriverType, AudioDeviceInfo,
    AudioStreamConfig, StreamState, AudioCallback
)
from tonesphere.utils.logger import logger


class PulseAudioDriver(AudioDriverBase):
    """
    PulseAudio driver implementation for Linux
    
    PulseAudio is a sound server that sits on top of ALSA,
    providing network audio, per-application volume control,
    and easier device management.
    """
    
    def __init__(self):
        super().__init__(AudioDriverType.PULSEAUDIO)
        self.stream_callbacks: Dict[int, AudioCallback] = {}
        self.next_stream_id = 1
        self.pa_context = None
        
    def is_available(self) -> bool:
        """Check if PulseAudio is available"""
        if platform.system() != "Linux":
            return False
        
        try:
            import pulsectl
            return True
        except ImportError:
            # Check if pulseaudio is running
            import subprocess
            try:
                result = subprocess.run(['pulseaudio', '--check'], 
                                      capture_output=True, timeout=1)
                return result.returncode == 0
            except:
                return False
    
    def initialize(self) -> bool:
        """Initialize PulseAudio driver"""
        if not self.is_available():
            logger.warning("PulseAudio not available on this system")
            return False
        
        try:
            self.is_initialized = True
            logger.info("PulseAudio driver initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PulseAudio driver: {e}")
            return False
    
    def terminate(self):
        """Terminate PulseAudio driver"""
        for stream_id in list(self.active_streams.keys()):
            self.close_stream(stream_id)
        
        if self.pa_context:
            self.pa_context = None
        
        self.is_initialized = False
        logger.info("PulseAudio driver terminated")
    
    def enumerate_devices(self) -> List[AudioDeviceInfo]:
        """Enumerate PulseAudio devices and detect running audio applications"""
        devices = []
        device_id = 400  # Start PulseAudio devices at 400
        
        try:
            try:
                import pulsectl
                
                with pulsectl.Pulse('tonesphere-enum') as pulse:
                    # Get sinks (output devices)
                    for sink in pulse.sink_list():
                        devices.append(AudioDeviceInfo(
                            id=device_id,
                            name=f"PulseAudio: {sink.description}",
                            driver_type=AudioDriverType.PULSEAUDIO,
                            max_input_channels=0,
                            max_output_channels=sink.channel_count,
                            default_sample_rate=sink.sample_spec.rate,
                            supported_sample_rates=[44100, 48000, 96000],
                            default_buffer_size=512,
                            supported_buffer_sizes=[256, 512, 1024, 2048],
                            is_default_input=False,
                            is_default_output=(sink.name == pulse.server_info().default_sink_name),
                            latency_input_ms=0.0,
                            latency_output_ms=10.67,
                            is_asio=False,
                            host_api="PulseAudio",
                            supports_exclusive_mode=False,
                            supports_shared_mode=True,
                            supports_callback_mode=True,
                            supports_blocking_mode=True
                        ))
                        device_id += 1
                    
                    # Get sources (input devices)
                    for source in pulse.source_list():
                        if not source.name.endswith('.monitor'):  # Skip monitor sources
                            devices.append(AudioDeviceInfo(
                                id=device_id,
                                name=f"PulseAudio: {source.description}",
                                driver_type=AudioDriverType.PULSEAUDIO,
                                max_input_channels=source.channel_count,
                                max_output_channels=0,
                                default_sample_rate=source.sample_spec.rate,
                                supported_sample_rates=[44100, 48000, 96000],
                                default_buffer_size=512,
                                supported_buffer_sizes=[256, 512, 1024, 2048],
                                is_default_input=(source.name == pulse.server_info().default_source_name),
                                is_default_output=False,
                                latency_input_ms=10.67,
                                latency_output_ms=0.0,
                                is_asio=False,
                                host_api="PulseAudio",
                                supports_exclusive_mode=False,
                                supports_shared_mode=True,
                                supports_callback_mode=True,
                                supports_blocking_mode=True
                            ))
                            device_id += 1
                    
                    # Detect running audio applications via sink inputs and source outputs
                    app_streams = {}
                    
                    # Check sink inputs (applications playing audio)
                    for sink_input in pulse.sink_input_list():
                        app_name = sink_input.proplist.get('application.name', 'Unknown')
                        if app_name and app_name not in ['ToneSphere', 'tonesphere']:
                            if app_name not in app_streams:
                                app_streams[app_name] = {'has_output': True, 'has_input': False}
                            else:
                                app_streams[app_name]['has_output'] = True
                    
                    # Check source outputs (applications recording audio)
                    for source_output in pulse.source_output_list():
                        app_name = source_output.proplist.get('application.name', 'Unknown')
                        if app_name and app_name not in ['ToneSphere', 'tonesphere']:
                            if app_name not in app_streams:
                                app_streams[app_name] = {'has_output': False, 'has_input': True}
                            else:
                                app_streams[app_name]['has_input'] = True
                    
                    # Create device entries for detected applications
                    for app_name, stream_info in app_streams.items():
                        if stream_info['has_output']:
                            # Application is playing audio (we can receive it as input)
                            devices.append(AudioDeviceInfo(
                                id=device_id,
                                name=f"{app_name} (PulseAudio Input)",
                                driver_type=AudioDriverType.PULSEAUDIO,
                                max_input_channels=2,
                                max_output_channels=0,
                                default_sample_rate=48000,
                                supported_sample_rates=[44100, 48000, 96000],
                                default_buffer_size=512,
                                supported_buffer_sizes=[256, 512, 1024],
                                is_default_input=False,
                                is_default_output=False,
                                latency_input_ms=10.67,
                                latency_output_ms=0.0,
                                is_asio=False,
                                host_api="PulseAudio Stream",
                                supports_exclusive_mode=False,
                                supports_shared_mode=True,
                                supports_callback_mode=True,
                                supports_blocking_mode=True
                            ))
                            device_id += 1
                        
                        if stream_info['has_input']:
                            # Application is recording audio (we can send it as output)
                            devices.append(AudioDeviceInfo(
                                id=device_id,
                                name=f"{app_name} (PulseAudio Output)",
                                driver_type=AudioDriverType.PULSEAUDIO,
                                max_input_channels=0,
                                max_output_channels=2,
                                default_sample_rate=48000,
                                supported_sample_rates=[44100, 48000, 96000],
                                default_buffer_size=512,
                                supported_buffer_sizes=[256, 512, 1024],
                                is_default_input=False,
                                is_default_output=False,
                                latency_input_ms=0.0,
                                latency_output_ms=10.67,
                                is_asio=False,
                                host_api="PulseAudio Stream",
                                supports_exclusive_mode=False,
                                supports_shared_mode=True,
                                supports_callback_mode=True,
                                supports_blocking_mode=True
                            ))
                            device_id += 1
                    
                    if app_streams:
                        logger.info(f"Detected {len(app_streams)} PulseAudio application streams")
                            
            except ImportError:
                # Fallback: create default devices
                devices.append(AudioDeviceInfo(
                    id=device_id,
                    name="PulseAudio Default Output",
                    driver_type=AudioDriverType.PULSEAUDIO,
                    max_input_channels=0,
                    max_output_channels=2,
                    default_sample_rate=48000,
                    supported_sample_rates=[44100, 48000],
                    default_buffer_size=512,
                    supported_buffer_sizes=[256, 512, 1024],
                    is_default_input=False,
                    is_default_output=True,
                    latency_input_ms=0.0,
                    latency_output_ms=10.67,
                    is_asio=False,
                    host_api="PulseAudio",
                    supports_exclusive_mode=False,
                    supports_shared_mode=True,
                    supports_callback_mode=True,
                    supports_blocking_mode=True
                ))
                device_id += 1
                
                devices.append(AudioDeviceInfo(
                    id=device_id,
                    name="PulseAudio Default Input",
                    driver_type=AudioDriverType.PULSEAUDIO,
                    max_input_channels=2,
                    max_output_channels=0,
                    default_sample_rate=48000,
                    supported_sample_rates=[44100, 48000],
                    default_buffer_size=512,
                    supported_buffer_sizes=[256, 512, 1024],
                    is_default_input=True,
                    is_default_output=False,
                    latency_input_ms=10.67,
                    latency_output_ms=0.0,
                    is_asio=False,
                    host_api="PulseAudio",
                    supports_exclusive_mode=False,
                    supports_shared_mode=True,
                    supports_callback_mode=True,
                    supports_blocking_mode=True
                ))
                
        except Exception as e:
            logger.error(f"Error enumerating PulseAudio devices: {e}")
        
        self.devices_cache = devices
        return devices
    
    def get_device_info(self, device_id: int) -> Optional[AudioDeviceInfo]:
        """Get PulseAudio device information"""
        if not self.devices_cache:
            self.enumerate_devices()
        
        for device in self.devices_cache:
            if device.id == device_id:
                return device
        return None
    
    def open_stream(self, config: AudioStreamConfig) -> int:
        """Open PulseAudio audio stream"""
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
            
            logger.info(f"Opened PulseAudio stream {stream_id}")
            return stream_id
            
        except Exception as e:
            logger.error(f"Failed to open PulseAudio stream: {e}")
            return -1
    
    def close_stream(self, stream_id: int) -> bool:
        """Close PulseAudio stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            self.stop_stream(stream_id)
            del self.active_streams[stream_id]
            if stream_id in self.stream_callbacks:
                del self.stream_callbacks[stream_id]
            
            logger.info(f"Closed PulseAudio stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to close PulseAudio stream: {e}")
            return False
    
    def start_stream(self, stream_id: int) -> bool:
        """Start PulseAudio stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.RUNNING
            
            logger.info(f"Started PulseAudio stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start PulseAudio stream: {e}")
            return False
    
    def stop_stream(self, stream_id: int) -> bool:
        """Stop PulseAudio stream"""
        if stream_id not in self.active_streams:
            return False
        
        try:
            stream = self.active_streams[stream_id]
            stream['state'] = StreamState.STOPPED
            
            logger.info(f"Stopped PulseAudio stream {stream_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop PulseAudio stream: {e}")
            return False
    
    def get_stream_state(self, stream_id: int) -> StreamState:
        """Get PulseAudio stream state"""
        if stream_id in self.active_streams:
            return self.active_streams[stream_id]['state']
        return StreamState.ERROR
    
    def read_stream(self, stream_id: int, frames: int) -> Optional[np.ndarray]:
        """Read from PulseAudio stream"""
        if stream_id not in self.active_streams:
            return None
        
        stream = self.active_streams[stream_id]
        return stream['buffer'][:frames].copy()
    
    def write_stream(self, stream_id: int, data: np.ndarray) -> bool:
        """Write to PulseAudio stream"""
        if stream_id not in self.active_streams:
            return False
        
        stream = self.active_streams[stream_id]
        stream['buffer'] = data.copy()
        return True
    
    def get_stream_latency(self, stream_id: int) -> tuple[float, float]:
        """Get PulseAudio stream latency"""
        if stream_id not in self.active_streams:
            return (0.0, 0.0)
        
        stream = self.active_streams[stream_id]
        config = stream['config']
        
        latency_ms = (config.buffer_size / config.sample_rate) * 1000
        return (latency_ms, latency_ms)
    
    def get_stream_cpu_load(self, stream_id: int) -> float:
        """Get PulseAudio stream CPU load"""
        return 0.0
