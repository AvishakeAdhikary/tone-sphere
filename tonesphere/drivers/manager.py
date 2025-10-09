"""
Audio Driver Manager
Handles driver selection, initialization, and management
"""

import platform
from typing import List, Optional, Dict, Any
from .base import AudioDriverBase, AudioDriverType, AudioDeviceInfo, AudioStreamConfig
from tonesphere.utils.logger import logger


class AudioDriverManager:
    """
    Manages all audio drivers and provides unified interface
    
    Automatically detects available drivers and selects the best one
    for the current platform.
    """
    
    def __init__(self, preferred_driver: AudioDriverType = AudioDriverType.AUTO):
        self.preferred_driver = preferred_driver
        self.active_driver: Optional[AudioDriverBase] = None
        self.available_drivers: Dict[AudioDriverType, AudioDriverBase] = {}
        self._initialize_drivers()
        
    def _initialize_drivers(self):
        """Initialize all available drivers"""
        system = platform.system()
        
        # Import and register drivers based on platform
        if system == "Windows":
            self._register_windows_drivers()
        elif system == "Linux":
            self._register_linux_drivers()
        elif system == "Darwin":
            self._register_macos_drivers()
        
        # Select and activate best driver
        self._select_driver()
    
    def _register_windows_drivers(self):
        """Register Windows audio drivers"""
        try:
            from .windows_asio import ASIODriver
            asio = ASIODriver()
            if asio.is_available():
                self.available_drivers[AudioDriverType.ASIO] = asio
                logger.info("ASIO driver available")
        except Exception as e:
            logger.debug(f"ASIO driver not available: {e}")
        
        try:
            from .windows_wasapi import WASAPIDriver
            wasapi = WASAPIDriver()
            if wasapi.is_available():
                self.available_drivers[AudioDriverType.WASAPI] = wasapi
                logger.info("WASAPI driver available")
        except Exception as e:
            logger.debug(f"WASAPI driver not available: {e}")
        
        try:
            from .windows_directsound import DirectSoundDriver
            ds = DirectSoundDriver()
            if ds.is_available():
                self.available_drivers[AudioDriverType.DIRECTSOUND] = ds
                logger.info("DirectSound driver available")
        except Exception as e:
            logger.debug(f"DirectSound driver not available: {e}")
    
    def _register_linux_drivers(self):
        """Register Linux audio drivers"""
        try:
            from .linux_pipewire import PipeWireDriver
            pw = PipeWireDriver()
            if pw.is_available():
                self.available_drivers[AudioDriverType.PIPEWIRE] = pw
                logger.info("PipeWire driver available")
        except Exception as e:
            logger.debug(f"PipeWire driver not available: {e}")
        
        try:
            from .linux_jack import JACKDriver
            jack = JACKDriver()
            if jack.is_available():
                self.available_drivers[AudioDriverType.JACK] = jack
                logger.info("JACK driver available")
        except Exception as e:
            logger.debug(f"JACK driver not available: {e}")
        
        try:
            from .linux_pulseaudio import PulseAudioDriver
            pulse = PulseAudioDriver()
            if pulse.is_available():
                self.available_drivers[AudioDriverType.PULSEAUDIO] = pulse
                logger.info("PulseAudio driver available")
        except Exception as e:
            logger.debug(f"PulseAudio driver not available: {e}")
        
        try:
            from .linux_alsa import ALSADriver
            alsa = ALSADriver()
            if alsa.is_available():
                self.available_drivers[AudioDriverType.ALSA] = alsa
                logger.info("ALSA driver available")
        except Exception as e:
            logger.debug(f"ALSA driver not available: {e}")
    
    def _register_macos_drivers(self):
        """Register macOS audio drivers"""
        try:
            from .macos_coreaudio import CoreAudioDriver
            ca = CoreAudioDriver()
            if ca.is_available():
                self.available_drivers[AudioDriverType.COREAUDIO] = ca
                logger.info("CoreAudio driver available")
        except Exception as e:
            logger.debug(f"CoreAudio driver not available: {e}")
        
        # JACK is also available on macOS
        try:
            from .linux_jack import JACKDriver
            jack = JACKDriver()
            if jack.is_available():
                self.available_drivers[AudioDriverType.JACK] = jack
                logger.info("JACK driver available on macOS")
        except Exception as e:
            logger.debug(f"JACK driver not available on macOS: {e}")
    
    def _select_driver(self):
        """Select the best available driver"""
        if not self.available_drivers:
            logger.error("No audio drivers available!")
            return
        
        # If specific driver requested, use it
        if self.preferred_driver != AudioDriverType.AUTO:
            if self.preferred_driver in self.available_drivers:
                self.active_driver = self.available_drivers[self.preferred_driver]
                logger.info(f"Using preferred driver: {self.preferred_driver.value}")
                return
            else:
                logger.warning(f"Preferred driver {self.preferred_driver.value} not available")
        
        # Auto-select best driver based on platform
        system = platform.system()
        
        if system == "Windows":
            # Priority: ASIO > WASAPI > DirectSound
            for driver_type in [AudioDriverType.ASIO, AudioDriverType.WASAPI, AudioDriverType.DIRECTSOUND]:
                if driver_type in self.available_drivers:
                    self.active_driver = self.available_drivers[driver_type]
                    logger.info(f"Auto-selected driver: {driver_type.value}")
                    return
        
        elif system == "Linux":
            # Priority: JACK > PipeWire > PulseAudio > ALSA
            for driver_type in [AudioDriverType.JACK, AudioDriverType.PIPEWIRE, 
                              AudioDriverType.PULSEAUDIO, AudioDriverType.ALSA]:
                if driver_type in self.available_drivers:
                    self.active_driver = self.available_drivers[driver_type]
                    logger.info(f"Auto-selected driver: {driver_type.value}")
                    return
        
        elif system == "Darwin":
            # Priority: CoreAudio > JACK
            for driver_type in [AudioDriverType.COREAUDIO, AudioDriverType.JACK]:
                if driver_type in self.available_drivers:
                    self.active_driver = self.available_drivers[driver_type]
                    logger.info(f"Auto-selected driver: {driver_type.value}")
                    return
        
        # Fallback: use first available driver
        self.active_driver = list(self.available_drivers.values())[0]
        logger.info(f"Using fallback driver: {self.active_driver.driver_type.value}")
    
    def initialize(self) -> bool:
        """Initialize the active driver"""
        if not self.active_driver:
            logger.error("No active driver to initialize")
            return False
        
        return self.active_driver.initialize()
    
    def terminate(self):
        """Terminate all drivers"""
        for driver in self.available_drivers.values():
            try:
                if driver.is_initialized:
                    driver.terminate()
            except Exception as e:
                logger.error(f"Error terminating driver: {e}")
    
    def get_active_driver(self) -> Optional[AudioDriverBase]:
        """Get the currently active driver"""
        return self.active_driver
    
    def get_available_drivers(self) -> List[AudioDriverType]:
        """Get list of available driver types"""
        return list(self.available_drivers.keys())
    
    def switch_driver(self, driver_type: AudioDriverType) -> bool:
        """
        Switch to a different driver
        
        Args:
            driver_type: Driver type to switch to
            
        Returns:
            True if successful
        """
        if driver_type not in self.available_drivers:
            logger.error(f"Driver {driver_type.value} not available")
            return False
        
        # Terminate current driver
        if self.active_driver and self.active_driver.is_initialized:
            self.active_driver.terminate()
        
        # Switch to new driver
        self.active_driver = self.available_drivers[driver_type]
        success = self.active_driver.initialize()
        
        if success:
            logger.info(f"Switched to driver: {driver_type.value}")
        else:
            logger.error(f"Failed to switch to driver: {driver_type.value}")
        
        return success
    
    def enumerate_devices(self) -> List[AudioDeviceInfo]:
        """Enumerate devices from active driver"""
        if not self.active_driver:
            return []
        return self.active_driver.enumerate_devices()
    
    def get_device_info(self, device_id: int) -> Optional[AudioDeviceInfo]:
        """Get device information"""
        if not self.active_driver:
            return None
        return self.active_driver.get_device_info(device_id)
    
    def open_stream(self, config: AudioStreamConfig) -> int:
        """Open audio stream"""
        if not self.active_driver:
            return -1
        return self.active_driver.open_stream(config)
    
    def close_stream(self, stream_id: int) -> bool:
        """Close audio stream"""
        if not self.active_driver:
            return False
        return self.active_driver.close_stream(stream_id)
    
    def start_stream(self, stream_id: int) -> bool:
        """Start audio stream"""
        if not self.active_driver:
            return False
        return self.active_driver.start_stream(stream_id)
    
    def stop_stream(self, stream_id: int) -> bool:
        """Stop audio stream"""
        if not self.active_driver:
            return False
        return self.active_driver.stop_stream(stream_id)
    
    def get_driver_info(self) -> Dict[str, Any]:
        """Get information about all drivers"""
        info = {
            'active_driver': self.active_driver.driver_type.value if self.active_driver else None,
            'available_drivers': [dt.value for dt in self.available_drivers.keys()],
            'platform': platform.system()
        }
        
        if self.active_driver:
            info['active_driver_details'] = self.active_driver.get_driver_info()
        
        return info
