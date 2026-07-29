"""
Channel Control System
Provides per-channel volume, muting, and routing controls
"""

from dataclasses import dataclass

import numpy as np

from tonesphere.utils.logger import logger


@dataclass
class ChannelConfig:
    """Configuration for a single audio channel"""
    channel_index: int
    volume: float = 1.0  # 0.0 to 2.0
    muted: bool = False
    solo: bool = False
    pan: float = 0.0  # -1.0 (left) to 1.0 (right)
    inverted: bool = False  # Phase inversion


class DeviceChannelControl:
    """
    Channel control for a single device
    Manages individual channel settings
    """

    def __init__(self, device_id: int, num_channels: int):
        self.device_id = device_id
        self.num_channels = num_channels
        self.channels: dict[int, ChannelConfig] = {}

        # Initialize channels
        for i in range(num_channels):
            self.channels[i] = ChannelConfig(channel_index=i)

        # Device-level controls
        self.master_volume = 1.0
        self.master_muted = False
        self.channels_swapped = False

    def set_channel_volume(self, channel: int, volume: float):
        """Set volume for specific channel"""
        if channel in self.channels:
            self.channels[channel].volume = max(0.0, min(2.0, volume))
            logger.debug(f"Device {self.device_id} channel {channel} volume: {volume}")

    def set_channel_mute(self, channel: int, muted: bool):
        """Mute/unmute specific channel"""
        if channel in self.channels:
            self.channels[channel].muted = muted
            logger.debug(f"Device {self.device_id} channel {channel} muted: {muted}")

    def set_channel_solo(self, channel: int, solo: bool):
        """Solo specific channel"""
        if channel in self.channels:
            self.channels[channel].solo = solo
            logger.debug(f"Device {self.device_id} channel {channel} solo: {solo}")

    def set_channel_pan(self, channel: int, pan: float):
        """Set pan for specific channel (-1.0 left to 1.0 right)"""
        if channel in self.channels:
            self.channels[channel].pan = max(-1.0, min(1.0, pan))
            logger.debug(f"Device {self.device_id} channel {channel} pan: {pan}")

    def set_channel_inverted(self, channel: int, inverted: bool):
        """Invert phase of specific channel"""
        if channel in self.channels:
            self.channels[channel].inverted = inverted
            logger.debug(f"Device {self.device_id} channel {channel} inverted: {inverted}")

    def swap_channels(self):
        """Swap left and right channels (for stereo)"""
        self.channels_swapped = not self.channels_swapped
        logger.info(f"Device {self.device_id} channels swapped: {self.channels_swapped}")

    def set_master_volume(self, volume: float):
        """Set master volume for device"""
        self.master_volume = max(0.0, min(2.0, volume))
        logger.debug(f"Device {self.device_id} master volume: {volume}")

    def set_master_mute(self, muted: bool):
        """Mute/unmute entire device"""
        self.master_muted = muted
        logger.debug(f"Device {self.device_id} master muted: {muted}")

    def apply_to_strip(self, strip):
        """
        Push this configuration into a realtime `engine.dsp.ChannelStrip`.

        This class is the control-side model: it is what the GUI and REST API edit, and it
        is convenient but not realtime-safe. The strip is the callback-side counterpart,
        with preallocated buffers and smoothed gains. Keeping them separate is what lets
        settings be edited freely on the UI thread without ever blocking or allocating on
        the audio thread.

        Until this method existed, none of these settings affected audio at all.
        """
        any_solo = any(channel.solo for channel in self.channels.values())

        for index in range(self.num_channels):
            config = self.channels.get(index)
            if config is None:
                continue

            # Solo is exclusive: anything not soloed goes silent while any solo is active.
            silenced = config.muted or (any_solo and not config.solo)

            strip.set_channel_mute(index, silenced)
            if not silenced:
                strip.set_channel_gain(index, config.volume)
            strip.set_channel_pan(index, config.pan)
            strip.set_channel_inverted(index, config.inverted)

        strip.set_swapped(self.channels_swapped)
        strip.set_master_gain(0.0 if self.master_muted else self.master_volume)

    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Offline processing, for tests and file rendering.

        NOT for the audio callback: it allocates on every call and steps gain rather than
        ramping it, which would click. `apply_to_strip` feeds the realtime path.

        Args:
            audio_data: Input audio (frames, channels)

        Returns:
            Processed audio
        """
        if audio_data.shape[1] != self.num_channels:
            logger.warning(f"Channel count mismatch: expected {self.num_channels}, got {audio_data.shape[1]}")
            return audio_data

        # Master mute
        if self.master_muted:
            return np.zeros_like(audio_data)

        processed = audio_data.copy()

        # Swap channels if enabled (stereo only)
        if self.channels_swapped and self.num_channels == 2:
            processed = processed[:, [1, 0]]

        # Check if any channel is soloed
        any_solo = any(ch.solo for ch in self.channels.values())

        # Process each channel
        for ch_idx in range(self.num_channels):
            if ch_idx not in self.channels:
                continue

            ch_config = self.channels[ch_idx]

            # Solo logic
            if any_solo and not ch_config.solo:
                processed[:, ch_idx] = 0.0
                continue

            # Mute
            if ch_config.muted:
                processed[:, ch_idx] = 0.0
                continue

            # Phase inversion
            if ch_config.inverted:
                processed[:, ch_idx] *= -1.0

            # Volume
            processed[:, ch_idx] *= ch_config.volume

            # Pan, using the same constant-power law as the realtime path so offline and
            # live rendering agree. The previous version only attenuated the opposite
            # channel, so a centred source was 3 dB louder than a panned one.
            if self.num_channels == 2 and ch_config.pan != 0.0:
                from tonesphere.engine.dsp import pan_gains

                left, right = pan_gains(ch_config.pan)
                processed[:, ch_idx] *= left if ch_idx == 0 else right

        # Master volume
        processed *= self.master_volume

        return processed

    def get_channel_info(self, channel: int) -> dict | None:
        """Get information about a specific channel"""
        if channel not in self.channels:
            return None

        ch = self.channels[channel]
        return {
            'channel_index': ch.channel_index,
            'volume': ch.volume,
            'muted': ch.muted,
            'solo': ch.solo,
            'pan': ch.pan,
            'inverted': ch.inverted
        }

    def get_all_channels_info(self) -> dict:
        """Get information about all channels"""
        return {
            'device_id': self.device_id,
            'num_channels': self.num_channels,
            'master_volume': self.master_volume,
            'master_muted': self.master_muted,
            'channels_swapped': self.channels_swapped,
            'channels': {
                ch_idx: self.get_channel_info(ch_idx)
                for ch_idx in self.channels.keys()
            }
        }


class ChannelControlManager:
    """
    Manages channel controls for all devices
    """

    def __init__(self):
        self.device_controls: dict[int, DeviceChannelControl] = {}

    def add_device(self, device_id: int, num_channels: int):
        """Add a device to channel control"""
        if device_id not in self.device_controls:
            self.device_controls[device_id] = DeviceChannelControl(device_id, num_channels)
            logger.info(f"Added channel control for device {device_id} with {num_channels} channels")

    def remove_device(self, device_id: int):
        """Remove a device from channel control"""
        if device_id in self.device_controls:
            del self.device_controls[device_id]
            logger.info(f"Removed channel control for device {device_id}")

    def get_device_control(self, device_id: int) -> DeviceChannelControl | None:
        """Get channel control for a device"""
        return self.device_controls.get(device_id)

    def process_device_audio(self, device_id: int, audio_data: np.ndarray) -> np.ndarray:
        """Process audio for a device through its channel controls"""
        control = self.device_controls.get(device_id)
        if control:
            return control.process_audio(audio_data)
        return audio_data

    def set_device_channel_volume(self, device_id: int, channel: int, volume: float):
        """Set volume for device channel"""
        control = self.device_controls.get(device_id)
        if control:
            control.set_channel_volume(channel, volume)

    def set_device_channel_mute(self, device_id: int, channel: int, muted: bool):
        """Mute device channel"""
        control = self.device_controls.get(device_id)
        if control:
            control.set_channel_mute(channel, muted)

    def set_device_channel_solo(self, device_id: int, channel: int, solo: bool):
        """Solo device channel"""
        control = self.device_controls.get(device_id)
        if control:
            control.set_channel_solo(channel, solo)

    def set_device_channel_pan(self, device_id: int, channel: int, pan: float):
        """Set pan for device channel"""
        control = self.device_controls.get(device_id)
        if control:
            control.set_channel_pan(channel, pan)

    def set_device_channel_inverted(self, device_id: int, channel: int, inverted: bool):
        """Invert phase of device channel"""
        control = self.device_controls.get(device_id)
        if control:
            control.set_channel_inverted(channel, inverted)

    def swap_device_channels(self, device_id: int):
        """Swap channels for device"""
        control = self.device_controls.get(device_id)
        if control:
            control.swap_channels()

    def set_device_master_volume(self, device_id: int, volume: float):
        """Set master volume for device"""
        control = self.device_controls.get(device_id)
        if control:
            control.set_master_volume(volume)

    def set_device_master_mute(self, device_id: int, muted: bool):
        """Mute entire device"""
        control = self.device_controls.get(device_id)
        if control:
            control.set_master_mute(muted)

    def get_device_info(self, device_id: int) -> dict | None:
        """Get channel information for device"""
        control = self.device_controls.get(device_id)
        if control:
            return control.get_all_channels_info()
        return None

    def get_all_devices_info(self) -> dict[int, dict]:
        """Get channel information for all devices"""
        return {
            device_id: control.get_all_channels_info()
            for device_id, control in self.device_controls.items()
        }
