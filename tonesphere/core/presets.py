"""
Saving and recalling a setup.

A preset stores the patch and the mixer state, not the hardware. Devices are referenced by
their stable key rather than by PortAudio index, because indices shift whenever anything is
plugged in — recalling a preset by index would silently point your guitar at whatever
happened to land at slot 3 today.

Recall is deliberately partial: a preset saved with an interface attached must still load
usefully when it is not. Missing devices are reported, not fatal, so you get the rest of
your setup back and a list of what is absent.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)

PRESET_VERSION = 1


@dataclass
class RecallResult:
    """What happened when a preset was applied."""
    applied: bool
    missing_devices: List[str] = field(default_factory=list)
    restored_routes: int = 0
    skipped_routes: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.applied and not self.missing_devices and not self.skipped_routes

    def summary(self) -> str:
        if not self.applied:
            return "Preset could not be applied"

        parts = [f"{self.restored_routes} route(s) restored"]
        if self.skipped_routes:
            parts.append(f"{self.skipped_routes} skipped")
        if self.missing_devices:
            parts.append(f"missing: {', '.join(self.missing_devices[:3])}")
        return ", ".join(parts)


class PresetManager:
    """Reads and writes presets for an `AudioEngine`."""

    def __init__(self, engine, directory: Optional[Path] = None):
        self.engine = engine
        self.directory = Path(directory) if directory else Path("presets")

    # --- Capture ---

    def capture(self, name: str) -> Dict[str, Any]:
        """
        Snapshot the current setup.

        Engine settings (rate, buffer, backend) are recorded but applied separately on
        recall, because changing them restarts the audio device and the caller may not
        want that as a side effect of loading a mix.
        """
        engine = self.engine

        buses = [
            {
                'name': meta['name'],
                'direction': meta['direction'],
                'channels': meta['channels'],
                'id': bus_id,
            }
            for bus_id, meta in sorted(engine._bus_meta.items())
        ]

        routes = []
        for (source_id, dest_id), route in engine.routing_matrix.connections.items():
            source = engine._node_for(source_id)
            dest = engine._node_for(dest_id)
            if source is None or dest is None:
                continue

            routes.append({
                'source': self._reference(source_id),
                'dest': self._reference(dest_id),
                'gain': round(route.volume, 6),
                'muted': route.muted,
                'pan': round(route.pan, 4),
                'inverted': route.inverted,
            })

        channels = {}
        for device_id, control in engine.channel_control_manager.device_controls.items():
            reference = self._reference(device_id)
            if reference is None:
                continue
            channels[reference] = {
                'master_volume': round(control.master_volume, 6),
                'master_muted': control.master_muted,
                'swapped': control.channels_swapped,
                'channels': [
                    {
                        'index': index,
                        'volume': round(config.volume, 6),
                        'muted': config.muted,
                        'solo': config.solo,
                        'pan': round(config.pan, 4),
                        'inverted': config.inverted,
                    }
                    for index, config in sorted(control.channels.items())
                ],
            }

        return {
            'version': PRESET_VERSION,
            'name': name,
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'engine': {
                'sample_rate': engine.sample_rate,
                'buffer_size': engine.buffer_size,
                'host_api': engine.host.host_api.value if engine.host.host_api else None,
                'exclusive': engine.host.exclusive,
                'master_volume': round(engine.master_volume, 6),
            },
            'buses': buses,
            'routes': routes,
            'channels': channels,
        }

    def _reference(self, device_id: int) -> Optional[str]:
        """
        A durable reference for a device or bus.

        Devices use their host-API-qualified name, buses their assigned name. Never the
        PortAudio index: those are positional and change when hardware appears.
        """
        node = self.engine._node_for(device_id)
        if node is None:
            return None

        if node.kind == 'bus':
            meta = self.engine._bus_meta.get(device_id)
            return f"bus:{meta['name']}" if meta else None

        return f"device:{node.ref}"

    # --- Recall ---

    def apply(self, preset: Dict[str, Any], apply_engine_settings: bool = False) -> RecallResult:
        """
        Apply a preset to the engine.

        Partial recall is a feature: a preset saved with an interface attached should still
        restore the rest of your setup when that interface is absent, and tell you what it
        could not do.
        """
        engine = self.engine
        result = RecallResult(applied=True)

        version = preset.get('version', 0)
        if version > PRESET_VERSION:
            result.warnings.append(
                f"Preset was written by a newer version ({version}); "
                f"unknown settings will be ignored"
            )

        if apply_engine_settings:
            self._apply_engine_settings(preset.get('engine', {}), result)

        engine.clear_all_routing()

        reference_to_id = self._recreate_buses(preset.get('buses', []))
        reference_to_id.update(self._map_devices())

        self._apply_routes(preset.get('routes', []), reference_to_id, result)
        self._apply_channels(preset.get('channels', {}), reference_to_id, result)

        master = preset.get('engine', {}).get('master_volume')
        if master is not None:
            engine.master_volume = float(master)

        logger.info(f"Preset '{preset.get('name', 'unnamed')}': {result.summary()}")
        return result

    def _apply_engine_settings(self, settings: Dict[str, Any], result: RecallResult):
        engine = self.engine

        host_api = settings.get('host_api')
        if host_api and host_api != (engine.host.host_api.value if engine.host.host_api else None):
            if not engine.switch_driver(host_api):
                result.warnings.append(f"Backend '{host_api}' is not available here")

        rate = settings.get('sample_rate')
        if rate and rate != engine.sample_rate:
            engine.set_sample_rate(int(rate))

        buffer_size = settings.get('buffer_size')
        if buffer_size and buffer_size != engine.buffer_size:
            engine.set_buffer_size(int(buffer_size))

        exclusive = settings.get('exclusive')
        if exclusive is not None:
            engine.set_exclusive_mode(bool(exclusive))

    def _recreate_buses(self, buses: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Recreate the preset's buses, reusing any that already exist under the same name.

        Reuse rather than recreate so applying a preset twice does not accumulate
        duplicate buses.
        """
        engine = self.engine
        mapping: Dict[str, int] = {}

        existing = {meta['name']: bus_id for bus_id, meta in engine._bus_meta.items()}

        for bus in buses:
            name = bus.get('name')
            if not name:
                continue

            if name in existing:
                mapping[f"bus:{name}"] = existing[name]
                continue

            channels = int(bus.get('channels', 2))
            if bus.get('direction') == 'output':
                bus_id = engine.create_virtual_output(name, channels)
            else:
                bus_id = engine.create_virtual_input(name, channels)

            if bus_id is not None:
                mapping[f"bus:{name}"] = bus_id

        return mapping

    def _map_devices(self) -> Dict[str, int]:
        return {
            f"device:{device.key}": device_id
            for device_id, device in self.engine._device_by_id.items()
        }

    def _apply_routes(self, routes: List[Dict[str, Any]],
                      mapping: Dict[str, int], result: RecallResult):
        engine = self.engine

        for route in routes:
            source_ref = route.get('source')
            dest_ref = route.get('dest')

            source_id = mapping.get(source_ref)
            dest_id = mapping.get(dest_ref)

            if source_id is None or dest_id is None:
                result.skipped_routes += 1
                for reference in (source_ref, dest_ref):
                    if reference and reference not in mapping:
                        label = reference.split(':', 1)[-1]
                        if label not in result.missing_devices:
                            result.missing_devices.append(label)
                continue

            success, message = engine.create_routing(
                source_id, dest_id, float(route.get('gain', 1.0))
            )

            if not success:
                result.skipped_routes += 1
                result.warnings.append(message)
                continue

            if route.get('muted'):
                engine.set_routing_mute(source_id, dest_id, True)
            if route.get('pan'):
                engine.set_routing_pan(source_id, dest_id, float(route['pan']))
            if route.get('inverted'):
                engine.set_routing_invert(source_id, dest_id, True)

            result.restored_routes += 1

    def _apply_channels(self, channels: Dict[str, Any],
                        mapping: Dict[str, int], result: RecallResult):
        engine = self.engine

        for reference, settings in channels.items():
            device_id = mapping.get(reference)
            if device_id is None:
                continue

            control = engine.channel_control_manager.device_controls.get(device_id)
            if control is None:
                continue

            control.master_volume = float(settings.get('master_volume', 1.0))
            control.master_muted = bool(settings.get('master_muted', False))
            control.channels_swapped = bool(settings.get('swapped', False))

            for entry in settings.get('channels', []):
                index = int(entry.get('index', 0))
                if index not in control.channels:
                    continue
                config = control.channels[index]
                config.volume = float(entry.get('volume', 1.0))
                config.muted = bool(entry.get('muted', False))
                config.solo = bool(entry.get('solo', False))
                config.pan = float(entry.get('pan', 0.0))
                config.inverted = bool(entry.get('inverted', False))

            engine.apply_channel_controls(device_id)

    # --- Files ---

    def save(self, name: str, path: Optional[Path] = None) -> Path:
        """Write a preset. Returns where it went."""
        preset = self.capture(name)

        target = Path(path) if path else self.directory / f"{self._safe_name(name)}.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)

        with open(target, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(preset, handle, sort_keys=False, allow_unicode=True)

        logger.info(f"Saved preset to {target}")
        return target

    def load(self, path: Path, apply_engine_settings: bool = False) -> RecallResult:
        source = Path(path)
        if not source.is_file():
            return RecallResult(applied=False, warnings=[f"No such preset: {source}"])

        try:
            with open(source, 'r', encoding='utf-8') as handle:
                preset = yaml.safe_load(handle)
        except Exception as e:
            return RecallResult(applied=False, warnings=[f"Could not read preset: {e}"])

        if not isinstance(preset, dict):
            return RecallResult(applied=False, warnings=["Preset file is not valid"])

        return self.apply(preset, apply_engine_settings)

    def list_presets(self) -> List[Dict[str, Any]]:
        """Presets in the preset directory, newest first."""
        if not self.directory.is_dir():
            return []

        found = []
        for path in self.directory.glob("*.yaml"):
            try:
                with open(path, 'r', encoding='utf-8') as handle:
                    data = yaml.safe_load(handle) or {}
                found.append({
                    'path': str(path),
                    'name': data.get('name', path.stem),
                    'created': data.get('created', ''),
                    'routes': len(data.get('routes', [])),
                })
            except Exception as e:
                logger.debug(f"Skipping unreadable preset {path}: {e}")

        return sorted(found, key=lambda item: item['created'], reverse=True)

    @staticmethod
    def _safe_name(name: str) -> str:
        keep = "-_ ()"
        cleaned = "".join(c for c in name if c.isalnum() or c in keep).strip()
        return (cleaned or "preset").replace(" ", "_")
