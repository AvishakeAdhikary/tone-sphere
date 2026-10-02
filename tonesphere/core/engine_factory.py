"""
Engine construction and the compatibility wrapper.

`UnifiedAudioEngine` exists because the GUI, CLI and REST API were written against it.
It forwards to `AudioEngine` and adds nothing, so new code should use `AudioEngine`
directly.
"""

from typing import Any

import numpy as np

from tonesphere.core.engine import AudioEngine
from tonesphere.engine import HostApi
from tonesphere.utils.config import ConfigManager
from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)

# Legacy config values map onto host APIs. The old config offered 'asio', 'wasapi',
# 'directsound', 'alsa', 'pulseaudio', 'jack', 'pipewire' and 'coreaudio' as if each were
# an implemented backend; several never existed even as stubs.
_DRIVER_ALIASES = {
    'auto': None,
    'asio': HostApi.ASIO,
    'wasapi': HostApi.WASAPI,
    'wdmks': HostApi.WDMKS,
    'wdm-ks': HostApi.WDMKS,
    'directsound': HostApi.DIRECTSOUND,
    'mme': HostApi.MME,
    'alsa': HostApi.ALSA,
    'jack': HostApi.JACK,
    'coreaudio': HostApi.COREAUDIO,
    # PipeWire and PulseAudio are reached through ALSA or JACK compatibility layers;
    # PortAudio has no separate backend for either.
    'pipewire': HostApi.JACK,
    'pulseaudio': HostApi.ALSA,
}


def resolve_host_api(name: str) -> HostApi | None:
    """Map a config string to a host API. Unknown names fall back to auto-selection."""
    key = (name or 'auto').strip().lower()

    if key in _DRIVER_ALIASES:
        return _DRIVER_ALIASES[key]

    resolved = HostApi.from_name(name)
    if resolved != HostApi.UNKNOWN:
        return resolved

    logger.warning(f"Unknown driver '{name}' in config; auto-selecting")
    return None


def create_audio_engine(config_manager: ConfigManager | None = None) -> AudioEngine:
    """Build an engine from configuration."""
    if config_manager is None:
        config_manager = ConfigManager()

    config = config_manager.load_config()
    engine_config = config.get('engine', {})
    vdev_config = config.get('virtual_devices', {})

    sample_rate = engine_config.get('sample_rate', 48000)
    buffer_size = engine_config.get('buffer_size', 256)
    host_api = resolve_host_api(engine_config.get('preferred_driver', 'auto'))
    exclusive = engine_config.get('exclusive_mode', True)

    logger.info(
        f"Creating engine: {sample_rate} Hz, {buffer_size} frames, "
        f"{host_api.value if host_api else 'auto'}, exclusive={exclusive}"
    )

    return AudioEngine(
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        preferred_driver=host_api,
        max_virtual_inputs=vdev_config.get('max_inputs', 10),
        max_virtual_outputs=vdev_config.get('max_outputs', 10),
        exclusive=exclusive,
    )


class UnifiedAudioEngine:
    """Thin forwarder to `AudioEngine`, kept for the existing GUI/CLI/API callers."""

    def __init__(self, config_manager: ConfigManager | None = None):
        self.engine = create_audio_engine(config_manager)

    # --- Lifecycle ---

    def initialize(self):
        return self.engine.initialize()

    def start_engine(self):
        return self.engine.start_engine()

    def stop_engine(self):
        return self.engine.stop_engine()

    def cleanup(self):
        return self.engine.cleanup()

    # --- Devices ---

    def get_devices(self, include_all_backends: bool = False) -> list[dict]:
        return self.engine.get_devices(include_all_backends)

    def default_output_id(self) -> int | None:
        return self.engine.default_output_id()

    def default_input_id(self) -> int | None:
        return self.engine.default_input_id()

    def refresh_devices(self) -> bool:
        return self.engine.refresh_devices()

    def create_virtual_input(self, name: str, channels: int = 2) -> int | None:
        return self.engine.create_virtual_input(name, channels)

    def create_virtual_output(self, name: str, channels: int = 2) -> int | None:
        return self.engine.create_virtual_output(name, channels)

    def list_virtual_devices(self) -> list[dict]:
        return self.engine.list_virtual_devices()

    def get_virtual_device_counts(self) -> dict:
        return self.engine.get_virtual_device_counts()

    def delete_virtual_device(self, device_id: int) -> bool:
        return self.engine.delete_virtual_device(device_id)

    def update_virtual_device_sample_rate(self, device_id: int, sample_rate: int) -> bool:
        return self.engine.update_virtual_device_sample_rate(device_id, sample_rate)

    def update_virtual_device_channels(self, device_id: int, channels: int) -> bool:
        return self.engine.update_virtual_device_channels(device_id, channels)

    def create_linux_system_sink(self, name: str, channels: int = 2) -> int | None:
        return self.engine.create_linux_system_sink(name, channels)

    def remove_linux_system_sink(self, device_id: int) -> bool:
        return self.engine.remove_linux_system_sink(device_id)

    def create_macos_system_device(self) -> int | None:
        return self.engine.create_macos_system_device()

    def remove_macos_system_device(self, device_id: int) -> bool:
        return self.engine.remove_macos_system_device(device_id)

    def write_to_bus(self, device_id: int, audio: np.ndarray) -> int:
        return self.engine.write_to_bus(device_id, audio)

    # --- Routing ---

    def create_routing(self, source_id: int, destination_id: int, volume: float = 1.0):
        return self.engine.create_routing(source_id, destination_id, volume)

    def remove_routing(self, source_id: int, destination_id: int) -> bool:
        return self.engine.remove_routing(source_id, destination_id)

    def set_routing_volume(self, source_id: int, destination_id: int, volume: float):
        return self.engine.set_routing_volume(source_id, destination_id, volume)

    def set_routing_volume_db(self, source_id: int, destination_id: int, gain_db: float):
        return self.engine.set_routing_volume_db(source_id, destination_id, gain_db)

    def set_routing_mute(self, source_id: int, destination_id: int, muted: bool):
        return self.engine.set_routing_mute(source_id, destination_id, muted)

    def clear_all_routing(self):
        return self.engine.clear_all_routing()

    def create_monitor_patch(self, muted: bool = True):
        return self.engine.create_monitor_patch(muted)

    @property
    def has_routes(self) -> bool:
        return self.engine.has_routes

    @property
    def state(self) -> str:
        return self.engine.state

    def get_routing_matrix(self) -> dict:
        return self.engine.get_routing_matrix()

    # --- Measurements ---

    def get_performance_stats(self) -> dict:
        return self.engine.get_performance_stats()

    def get_meters(self) -> dict[int, dict[str, float]]:
        return self.engine.get_meters()

    def clear_clip_indicators(self):
        return self.engine.clear_clip_indicators()

    def get_ring_statistics(self) -> dict[str, dict]:
        return self.engine.get_ring_statistics()

    # --- Backend ---

    def get_driver_info(self) -> dict[str, Any]:
        return self.engine.get_driver_info()

    def get_available_drivers(self) -> list[str]:
        return self.engine.get_available_drivers()

    def switch_driver(self, driver_type: str) -> bool:
        return self.engine.switch_driver(driver_type)

    def set_exclusive_mode(self, exclusive: bool) -> bool:
        return self.engine.set_exclusive_mode(exclusive)

    def set_sample_rate(self, sample_rate: int) -> bool:
        return self.engine.set_sample_rate(sample_rate)

    def set_buffer_size(self, buffer_size: int) -> bool:
        return self.engine.set_buffer_size(buffer_size)

    # --- Network ---

    def start_network_streaming(self):
        return self.engine.start_network_streaming()

    def stop_network_streaming(self):
        return self.engine.stop_network_streaming()

    def get_network_clients(self):
        return self.engine.get_network_clients()

    def connect_to_network(self, host: str, port: int) -> bool:
        return self.engine.connect_to_network(host, port)

    def disconnect_from_network(self, conn_id: str):
        return self.engine.disconnect_from_network(conn_id)

    def get_network_connections(self):
        return self.engine.get_network_connections()

    def get_network_statistics(self):
        return self.engine.get_network_statistics()

    def register_network_receive(
        self,
        device_id: int,
        transport: str = 'tcp',
        target_latency_ms: float = 40.0,
        conceal: str = 'silence',
    ):
        return self.engine.register_network_receive(
            device_id, transport, target_latency_ms, conceal
        )

    def unregister_network_receive(self, device_id: int) -> bool:
        return self.engine.unregister_network_receive(device_id)

    def send_device_to_network(self, device_id: int, target=None, transport: str = 'udp'):
        return self.engine.send_device_audio_to_network(device_id, target, transport)

    def disable_network_send(self, device_id: int) -> bool:
        return self.engine.disable_network_send(device_id)

    def list_network_sends(self) -> list[dict]:
        return self.engine.list_network_sends()

    def set_network_quality(self, quality: str):
        return self.engine.set_network_quality(quality)

    # --- Network: UDP (realtime) ---

    def start_udp_transport(self, bind_host: str = '127.0.0.1', bind_port: int = 9002):
        return self.engine.start_udp_transport(bind_host, bind_port)

    def stop_udp_transport(self):
        return self.engine.stop_udp_transport()

    def add_udp_peer(self, name: str, host: str, port: int) -> str:
        return self.engine.add_udp_peer(name, host, port)

    def remove_udp_peer(self, name: str) -> bool:
        return self.engine.remove_udp_peer(name)

    def get_udp_peers(self) -> dict[str, str]:
        return self.engine.get_udp_peers()

    # --- Channel controls ---

    # These forward to AudioEngine rather than reaching into the control manager, so that
    # every change is pushed into the running audio path. Calling the manager directly
    # updates state that no sample ever sees.

    def get_device_channels(self, device_id: int):
        return self.engine.channel_control_manager.get_device_info(device_id)

    def set_channel_volume(self, device_id: int, channel: int, volume: float):
        self.engine.set_channel_volume(device_id, channel, volume)

    def set_channel_mute(self, device_id: int, channel: int, muted: bool):
        self.engine.set_channel_mute(device_id, channel, muted)

    def set_channel_solo(self, device_id: int, channel: int, solo: bool):
        self.engine.set_channel_solo(device_id, channel, solo)

    def set_channel_pan(self, device_id: int, channel: int, pan: float):
        self.engine.set_channel_pan(device_id, channel, pan)

    def set_channel_inverted(self, device_id: int, channel: int, inverted: bool):
        self.engine.set_channel_inverted(device_id, channel, inverted)

    def swap_channels(self, device_id: int):
        self.engine.swap_channels(device_id)

    def set_device_master_volume(self, device_id: int, volume: float):
        self.engine.set_device_master_volume(device_id, volume)

    def set_device_master_mute(self, device_id: int, muted: bool):
        self.engine.set_device_master_mute(device_id, muted)

    def set_routing_pan(self, source_id: int, destination_id: int, pan: float):
        self.engine.set_routing_pan(source_id, destination_id, pan)

    def set_routing_invert(self, source_id: int, destination_id: int, invert: bool):
        self.engine.set_routing_invert(source_id, destination_id, invert)

    def set_routing_source_channel(self, source_id: int, destination_id: int, channel: int | None):
        return self.engine.set_routing_source_channel(source_id, destination_id, channel)

    def monitor(self, input_id: int, output_id: int, source_channel: int | None = None):
        return self.engine.monitor(input_id, output_id, source_channel)

    def save_session(self):
        self.engine.save_session()

    def save_preset(self, path):
        return self.engine.save_preset(path)

    def load_preset(self, path) -> str:
        return self.engine.load_preset(path)

    def restore_session(self):
        return self.engine.restore_session()

    def set_master_volume(self, volume: float):
        return self.engine.set_master_volume(volume)

    def failed_device_ids(self) -> dict[int, str]:
        return self.engine.failed_device_ids()

    def virtual_device_status(self) -> dict[str, Any]:
        return self.engine.virtual_device_status()

    def manage_virtual_cable(self, operation: str, *args: str) -> tuple[bool, str]:
        return self.engine.manage_virtual_cable(operation, *args)

    def handle_device_change(self) -> dict[str, Any]:
        return self.engine.handle_device_change()

    def last_device_change(self) -> dict[str, Any] | None:
        return self.engine.last_device_change()

    # --- Plugins ---

    def hosts_plugins(self) -> bool:
        return self.engine.hosts_plugins()

    def scan_plugins(self, paths=None, retry_failed: bool = False) -> list:
        return self.engine.scan_plugins(paths, retry_failed)

    def add_plugin(self, device_id: int, info, is_input: bool = True) -> tuple[bool, str]:
        return self.engine.add_plugin(device_id, info, is_input)

    def remove_plugin(self, device_id: int, index: int, is_input: bool = True) -> bool:
        return self.engine.remove_plugin(device_id, index, is_input)

    def plugin_instances(self, device_id: int, is_input: bool = True) -> list:
        return self.engine.plugin_instances(device_id, is_input)

    def list_plugins(self, device_id: int, is_input: bool = True) -> list[dict[str, Any]]:
        return self.engine.list_plugins(device_id, is_input)

    def set_plugin_bypassed(self, device_id: int, index: int, bypassed: bool, is_input: bool = True) -> bool:
        return self.engine.set_plugin_bypassed(device_id, index, bypassed, is_input)

    def set_plugin_parameter(self, device_id: int, index: int, param_id: int, normalized: float,
                             is_input: bool = True):
        return self.engine.set_plugin_parameter(device_id, index, param_id, normalized, is_input)

    def find_plugin(self, path: str, uid: str | None = None):
        return self.engine.find_plugin(path, uid)

    def create_instrument(self, info, name: str | None = None, channels: int = 2):
        return self.engine.create_instrument(info, name, channels)

    def instruments(self) -> list[dict[str, Any]]:
        return self.engine.instruments()

    def send_midi(self, device_id: int, status: int, data1: int, data2: int = 0) -> tuple[bool, str]:
        return self.engine.send_midi(device_id, status, data1, data2)

    def midi_inputs(self) -> list[str]:
        return self.engine.midi_inputs()

    def connect_midi_input(self, port: int, device_id: int) -> tuple[bool, str]:
        return self.engine.connect_midi_input(port, device_id)

    def note_on(self, device_id: int, note: int, velocity: int = 100, channel: int = 0) -> tuple[bool, str]:
        return self.engine.note_on(device_id, note, velocity, channel)

    def note_off(self, device_id: int, note: int, channel: int = 0) -> tuple[bool, str]:
        return self.engine.note_off(device_id, note, channel)

    def add_builtin(self, device_id: int, kind: str, is_input: bool = True, values=None) -> tuple[bool, str]:
        return self.engine.add_builtin(device_id, kind, is_input, values)

    def list_inserts(self, device_id: int, is_input: bool = True) -> list[dict[str, Any]]:
        return self.engine.list_inserts(device_id, is_input)

    def insert_instance(self, device_id: int, index: int, is_input: bool = True):
        return self.engine.insert_instance(device_id, index, is_input)

    def remove_insert(self, device_id: int, index: int, is_input: bool = True) -> bool:
        return self.engine.remove_insert(device_id, index, is_input)

    def move_insert(self, device_id: int, index: int, to: int, is_input: bool = True) -> bool:
        return self.engine.move_insert(device_id, index, to, is_input)

    def set_insert_bypassed(self, device_id: int, index: int, bypassed: bool, is_input: bool = True) -> bool:
        return self.engine.set_insert_bypassed(device_id, index, bypassed, is_input)

    def set_insert_parameter(self, device_id: int, index: int, param_id: int, normalized: float,
                             is_input: bool = True) -> bool:
        return self.engine.set_insert_parameter(device_id, index, param_id, normalized, is_input)

    def set_builtin_value(self, device_id: int, index: int, param: int, value: float, is_input: bool = True):
        return self.engine.set_builtin_value(device_id, index, param, value, is_input)

    # --- Properties ---

    @property
    def is_running(self) -> bool:
        return self.engine.is_running

    @property
    def sample_rate(self) -> int:
        return self.engine.sample_rate

    @property
    def buffer_size(self) -> int:
        return self.engine.buffer_size

    @property
    def master_volume(self) -> float:
        return self.engine.master_volume

    @master_volume.setter
    def master_volume(self, value: float):
        self.engine.set_master_volume(value)
