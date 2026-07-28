"""
Audio Stream Manager
Manages audio streams with proper callback handling and routing
"""

import numpy as np
import threading
import time
from typing import Dict, Optional, Callable, List, Any
from tonesphere.utils.logger import logger
from tonesphere.drivers.base import AudioStreamConfig, StreamState


class AudioStream:
    """Represents an active audio stream"""
    
    def __init__(self, stream_id: int, config: AudioStreamConfig, 
                 driver_stream_id: int, device_id: int):
        self.stream_id = stream_id
        self.config = config
        self.driver_stream_id = driver_stream_id
        self.device_id = device_id
        self.state = StreamState.STOPPED
        
        # Audio buffers
        self.input_buffer = np.zeros((config.buffer_size, config.channels), dtype=np.float32)
        self.output_buffer = np.zeros((config.buffer_size, config.channels), dtype=np.float32)
        
        # Callback
        self.callback: Optional[Callable] = config.callback
        
        # Statistics
        self.frames_processed = 0
        self.xruns = 0  # Buffer under/overruns
        self.last_callback_time = 0.0
        self.cpu_load = 0.0
        
    def process_callback(self, input_data: Optional[np.ndarray], 
                        output_data: np.ndarray, 
                        frames: int) -> np.ndarray:
        """Process audio callback"""
        start_time = time.time()
        
        try:
            if self.callback:
                result = self.callback(input_data, output_data, frames, 
                                     {'time': start_time}, 0)
                self.frames_processed += frames
                
                # Calculate CPU load
                elapsed = time.time() - start_time
                buffer_duration = frames / self.config.sample_rate
                self.cpu_load = (elapsed / buffer_duration) if buffer_duration > 0 else 0.0
                
                return result
            else:
                return output_data
                
        except Exception as e:
            logger.error(f"Error in stream callback: {e}")
            self.xruns += 1
            return output_data


class AudioStreamManager:
    """
    Manages all audio streams with proper threading and callbacks
    """
    
    def __init__(self, driver_manager):
        self.driver_manager = driver_manager
        self.streams: Dict[int, AudioStream] = {}
        self.next_stream_id = 1
        self.stream_lock = threading.Lock()
        self.running = False

    def create_stream(self, device_id: int, config: AudioStreamConfig) -> int:
        """
        Create a new audio stream
        
        Args:
            device_id: Device to open stream on
            config: Stream configuration
            
        Returns:
            Stream ID or -1 on error
        """
        with self.stream_lock:
            try:
                # Open stream with driver
                driver_stream_id = self.driver_manager.open_stream(config)
                if driver_stream_id < 0:
                    logger.error(f"Failed to open driver stream for device {device_id}")
                    return -1
                
                # Create stream object
                stream_id = self.next_stream_id
                self.next_stream_id += 1
                
                stream = AudioStream(stream_id, config, driver_stream_id, device_id)
                self.streams[stream_id] = stream
                
                logger.info(f"Created stream {stream_id} for device {device_id}")
                return stream_id
                
            except Exception as e:
                logger.error(f"Error creating stream: {e}")
                return -1
    
    def destroy_stream(self, stream_id: int) -> bool:
        """Destroy an audio stream"""
        with self.stream_lock:
            if stream_id not in self.streams:
                return False
            
            try:
                stream = self.streams[stream_id]
                
                # Stop stream if running
                if stream.state == StreamState.RUNNING:
                    self.stop_stream(stream_id)
                
                # Close driver stream
                self.driver_manager.close_stream(stream.driver_stream_id)
                
                # Remove from streams
                del self.streams[stream_id]
                
                logger.info(f"Destroyed stream {stream_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error destroying stream: {e}")
                return False
    
    def start_stream(self, stream_id: int) -> bool:
        """Start an audio stream"""
        with self.stream_lock:
            if stream_id not in self.streams:
                return False
            
            try:
                stream = self.streams[stream_id]
                
                if stream.state == StreamState.RUNNING:
                    return True
                
                # Start driver stream
                if not self.driver_manager.start_stream(stream.driver_stream_id):
                    logger.error(f"Failed to start driver stream {stream.driver_stream_id}")
                    return False
                
                stream.state = StreamState.RUNNING
                logger.info(f"Started stream {stream_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error starting stream: {e}")
                return False
    
    def stop_stream(self, stream_id: int) -> bool:
        """Stop an audio stream"""
        with self.stream_lock:
            if stream_id not in self.streams:
                return False
            
            try:
                stream = self.streams[stream_id]
                
                if stream.state == StreamState.STOPPED:
                    return True
                
                # Stop driver stream
                if not self.driver_manager.stop_stream(stream.driver_stream_id):
                    logger.error(f"Failed to stop driver stream {stream.driver_stream_id}")
                    return False
                
                stream.state = StreamState.STOPPED
                logger.info(f"Stopped stream {stream_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
                return False
    
    def get_stream(self, stream_id: int) -> Optional[AudioStream]:
        """Get stream by ID"""
        return self.streams.get(stream_id)
    
    def get_all_streams(self) -> List[AudioStream]:
        """Get all streams"""
        with self.stream_lock:
            return list(self.streams.values())
    
    def start_processing(self):
        """
        Mark the manager as processing.

        There is deliberately no thread here. This used to spawn a loop that woke
        1000 times a second and whose entire body was `stream.last_callback_time =
        time.time()` — it moved no audio and only burned CPU. Audio will be pumped
        by the driver's own callback thread, not by us polling.
        """
        if self.running:
            return

        self.running = True
        logger.info("Stream manager active (no audio path yet — see README Roadmap)")

    def stop_processing(self):
        """Mark the manager as stopped."""
        if not self.running:
            return

        self.running = False
        logger.info("Stream manager stopped")


    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all streams"""
        with self.stream_lock:
            stats = {
                'total_streams': len(self.streams),
                'running_streams': sum(1 for s in self.streams.values() 
                                      if s.state == StreamState.RUNNING),
                'streams': {}
            }
            
            for stream_id, stream in self.streams.items():
                stats['streams'][stream_id] = {
                    'device_id': stream.device_id,
                    'state': stream.state.name,
                    'frames_processed': stream.frames_processed,
                    'xruns': stream.xruns,
                    'cpu_load': stream.cpu_load
                }
            
            return stats
    
    def cleanup(self):
        """Cleanup all streams"""
        self.stop_processing()
        
        with self.stream_lock:
            for stream_id in list(self.streams.keys()):
                self.destroy_stream(stream_id)
        
        logger.info("Stream manager cleanup complete")
