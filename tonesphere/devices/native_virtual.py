"""
In-process virtual audio endpoints.

IMPORTANT: these are NOT operating-system audio devices. They exist only inside
the ToneSphere process and cannot be selected from other applications' sound
settings. A system-visible virtual device requires a signed kernel-mode driver
(WDM/AVStream on Windows) — see the Roadmap in README.md.

They are useful today as internal mix buses that routing can target.
"""

import numpy as np
import threading
import queue
from typing import Dict, Optional, Callable, Any
from tonesphere.utils.logger import logger


class NativeVirtualDevice:
    """
    Native virtual audio device with real audio routing capabilities
    """
    
    def __init__(self, device_id: int, name: str, channels: int, 
                 sample_rate: int, buffer_size: int, is_input: bool):
        self.device_id = device_id
        self.name = name
        self.channels = channels
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.is_input = is_input
        self.is_active = False
        
        # Single ingress queue. There used to be separate input_buffer and
        # output_buffer queues: write_audio() always fed input_buffer while the
        # output-device loop only ever read output_buffer, so audio written to an
        # output device was silently discarded.
        self.frame_queue: queue.Queue = queue.Queue(maxsize=10)

        # Current audio frame
        self.current_frame = np.zeros((buffer_size, channels), dtype=np.float32)
        
        # Routing connections
        self.connected_devices: Dict[int, float] = {}  # device_id -> volume
        
        # Stream thread
        self.stream_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Callback for audio processing
        self.audio_callback: Optional[Callable] = None
        
        # Statistics
        self.frames_processed = 0
        self.buffer_underruns = 0
        self.buffer_overruns = 0
        
    @property
    def is_running(self) -> bool:
        """Whether the device's processing thread is active."""
        return self.running

    def set_callback(self, callback: Callable):
        """Set audio processing callback"""
        self.audio_callback = callback
    
    def connect_to(self, device_id: int, volume: float = 1.0):
        """Connect this device to another device for routing"""
        self.connected_devices[device_id] = volume
        logger.info(f"Virtual device {self.name} connected to device {device_id}")
    
    def disconnect_from(self, device_id: int):
        """Disconnect from another device"""
        if device_id in self.connected_devices:
            del self.connected_devices[device_id]
            logger.info(f"Virtual device {self.name} disconnected from device {device_id}")
    
    def start(self):
        """Start the virtual device"""
        if self.is_active:
            return
        
        self.is_active = True
        self.running = True
        
        # Start processing thread
        self.stream_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.stream_thread.start()
        
        logger.info(f"Started virtual device: {self.name}")
    
    def stop(self):
        """Stop the virtual device"""
        if not self.is_active:
            return
        
        self.running = False
        self.is_active = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=1.0)

        self.drain_buffers()

        logger.info(f"Stopped virtual device: {self.name}")

    def drain_buffers(self):
        """Discard any queued frames."""
        while True:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                return

    def _process_loop(self):
        """
        Consume queued frames into `current_frame`.

        Inputs and outputs behave identically here — the direction only describes
        which side of the graph the endpoint sits on, not how frames are carried.
        """
        while self.running:
            try:
                self._process_frame()
                self.frames_processed += 1
            except Exception as e:
                logger.error(f"Error in virtual device {self.name} processing: {e}")

    def _process_frame(self):
        try:
            audio_data = self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            # Starved: hold silence rather than repeating the last frame, which
            # would sound like a stutter once this feeds real hardware.
            audio_data = np.zeros((self.buffer_size, self.channels), dtype=np.float32)
            self.buffer_underruns += 1

        if self.audio_callback:
            audio_data = self.audio_callback(audio_data)

        self.current_frame = audio_data

    def write_audio(self, audio_data: np.ndarray):
        """Push a frame into this device. Drops the oldest frame when backed up."""
        frame = np.ascontiguousarray(audio_data, dtype=np.float32)
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            self.buffer_overruns += 1
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except (queue.Empty, queue.Full):
                pass

    def read_audio(self) -> np.ndarray:
        """Read the most recently processed frame."""
        return self.current_frame.copy()


    def get_statistics(self) -> Dict[str, Any]:
        """Get device statistics"""
        return {
            'frames_processed': self.frames_processed,
            'buffer_underruns': self.buffer_underruns,
            'buffer_overruns': self.buffer_overruns,
            'queued_frames': self.frame_queue.qsize(),
            'is_active': self.is_active
        }


# NOTE: `NativeVirtualDeviceManager` used to live here as a second, parallel
# device registry. The engine created its devices in `VirtualDeviceManager` but
# routed them through this one, so every route was recorded against an empty
# registry: `create_routing()` reported success and no audio ever moved. There is
# now exactly one registry — `devices.virtual_device_manager.VirtualDeviceManager`.
