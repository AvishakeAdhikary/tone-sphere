"""
ToneSphere audio engine.

A facade over `tonesphere.engine.AudioHost` that keeps the integer-device-id surface the
GUI, CLI and REST API were written against, while the audio itself runs on real
PortAudio streams.

What changed from the version this replaces: it had eight "driver" classes that returned
arrays of zeros, two disjoint device registries, and a polling thread that moved nothing.
Audio now runs in the driver's own callback, routing is an immutable graph swapped by
reference, and every statistic is either measured or reported as None.

Ids
---
Callers use ints. The engine owns the mapping from int to graph node, because PortAudio's
device indices shift when hardware is plugged or unplugged, and a UI holding a stale int
must not silently start controlling a different device. Physical ids are derived from the
device key, buses are allocated from a separate range.
"""

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tonesphere.core.channel_control import ChannelControlManager
from tonesphere.core.models import AudioDevice, DeviceType
from tonesphere.core.processor import AudioProcessor
from tonesphere.core.routing import AudioRoutingMatrix
from tonesphere.engine import (
    AudioBackendUnavailable, AudioHost, Connection, DeviceInfo, HostApi, RoutingGraph,
    bus_node, db_to_linear, device_node, enumerate_devices, linear_to_db, preferred_host_api,
)
from tonesphere.network.audio_router import NetworkAudioRouter, NetworkQuality
from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)

# Buses get ids from here up, so a bus id can never be mistaken for a device id.
BUS_ID_BASE = 10000


class AudioEngine:
    """Routing engine over real audio hardware."""

    def __init__(
        self,
        sample_rate: int = 48000,
        buffer_size: int = 256,
        preferred_driver: Optional[HostApi] = None,
        max_virtual_inputs: int = 10,
        max_virtual_outputs: int = 10,
        exclusive: bool = True,
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.max_virtual_inputs = max_virtual_inputs
        self.max_virtual_outputs = max_virtual_outputs

        self.host = AudioHost(
            samplerate=sample_rate,
            blocksize=buffer_size,
            host_api=preferred_driver,
            exclusive=exclusive,
        )

        # Kept from the previous implementation: these state models were always sound,
        # they were simply never connected to any audio.
        self.routing_matrix = AudioRoutingMatrix()
        self.channel_control_manager = ChannelControlManager()
        self.processor = AudioProcessor(sample_rate, buffer_size)
        self.network_router = NetworkAudioRouter(quality=NetworkQuality.HIGH)

        # Id bookkeeping.
        self._devices: List[DeviceInfo] = []
        self._id_to_node: Dict[int, Any] = {}
        self._node_to_id: Dict[str, int] = {}
        self._device_by_id: Dict[int, DeviceInfo] = {}
        self._bus_meta: Dict[int, Dict[str, Any]] = {}
        self._next_bus_id = BUS_ID_BASE

        self._lock = threading.RLock()
        self._initialized = False
        self._started = False
        self._backend_error: Optional[str] = None
        self._problems: List[str] = []

        self.master_volume = 1.0

    # --- Lifecycle ---

    def initialize(self):
        """Enumerate hardware and pick a backend. Does not open any stream."""
        with self._lock:
            try:
                self._devices = enumerate_devices()
                self._backend_error = None
            except AudioBackendUnavailable as e:
                # A missing PortAudio is a real, reportable condition, not something to
                # paper over with an empty device list that looks like "no hardware".
                self._backend_error = str(e)
                self._devices = []
                logger.error(f"Audio backend unavailable: {e}")
                self._initialized = True
                return

            if self.host.host_api is None:
                self.host.host_api = preferred_host_api()

            self.host._devices = self._devices
            self._reindex()

            for device_id, device in self._device_by_id.items():
                channels = max(device.max_input_channels, device.max_output_channels)
                self.channel_control_manager.add_device(device_id, channels)

            self._initialized = True
            api = self.host.host_api.value if self.host.host_api else 'none'
            logger.info(f"Engine initialized: {len(self._devices)} devices, {api}")

    def start_engine(self):
        """
        Open and start the streams the current routing needs.

        With no routes there is nothing to open, and that is a legitimate idle state rather
        than a failure — `has_routes` distinguishes it so the UI can say "running, nothing
        patched" instead of the contradiction of a stopped engine behind a STOP button.
        """
        with self._lock:
            if not self._initialized:
                self.initialize()

            if self._backend_error:
                raise RuntimeError(f"Cannot start: {self._backend_error}")

            if self.host.is_running:
                return

            self._started = True

            if not self.routing_matrix.connections:
                self._problems = []
                logger.info("Engine started with no routes — patch something to hear audio")
                return

            self._problems = self.host.configure(self._build_graph())
            self._problems += self.host.start()

            for problem in self._problems:
                logger.warning(f"Engine start: {problem}")

    @property
    def has_routes(self) -> bool:
        return bool(self.routing_matrix.connections)

    @property
    def state(self) -> str:
        """
        One word for the UI. Distinguishes the three real states so 'Stopped' is never
        shown while the start button reads STOP.
        """
        if self.host.is_running:
            return 'degraded' if self.host.failed_streams() else 'running'
        if self._started:
            return 'idle'
        return 'stopped'

    def create_monitor_patch(self, muted: bool = True) -> Tuple[bool, str]:
        """
        Patch the default input straight to the default output.

        This is the guitarist's path: plug into the interface, hear yourself through the
        headphones. Created muted by default on purpose — on a laptop the default input is
        the built-in microphone and the default output is the built-in speakers, and
        unmuting that combination is an acoustic feedback loop at whatever volume the
        machine happens to be set to. The caller unmutes once it knows the devices are
        safe, or once the user asks.
        """
        with self._lock:
            source_id = self.default_input_id()
            dest_id = self.default_output_id()

            if source_id is None:
                return False, "No input device available"
            if dest_id is None:
                return False, "No output device available"

            success, message = self.create_routing(source_id, dest_id, volume=1.0)
            if not success:
                return False, message

            if muted:
                self.set_routing_mute(source_id, dest_id, True)

            source = self._device_by_id[source_id]
            dest = self._device_by_id[dest_id]
            suffix = " (muted — unmute when you know it will not feed back)" if muted else ""

            return True, f"{source.name} -> {dest.name}{suffix}"

    def stop_engine(self):
        with self._lock:
            self._started = False
            self.host.stop()

    @property
    def is_running(self) -> bool:
        return self.host.is_running

    def cleanup(self):
        with self._lock:
            self.host.cleanup()
            logger.info("Engine cleanup complete")

    # --- Id mapping ---

    def _reindex(self):
        """
        Rebuild the int-id mapping, preserving ids for devices that are still present.

        Preserving matters: a UI holding id 7 must keep pointing at the same speakers
        after an unrelated USB interface is unplugged.
        """
        preserved = {
            node_key: device_id
            for node_key, device_id in self._node_to_id.items()
            if node_key.startswith('device:')
        }

        self._device_by_id.clear()
        next_id = 0

        for device in self._devices:
            node = device_node(device.key)
            node_key = str(node)

            device_id = preserved.get(node_key)
            if device_id is None:
                while next_id in self._device_by_id or next_id >= BUS_ID_BASE:
                    next_id += 1
                device_id = next_id
                next_id += 1

            self._id_to_node[device_id] = node
            self._node_to_id[node_key] = device_id
            self._device_by_id[device_id] = device

    def _node_for(self, device_id: int):
        return self._id_to_node.get(device_id)

    def _id_for(self, node) -> Optional[int]:
        return self._node_to_id.get(str(node))

    # --- Devices ---

    def refresh_devices(self) -> bool:
        """
        Re-enumerate hardware, dropping routes to anything that disappeared.

        Leaving routes pointing at an unplugged device would make the engine fail to
        configure on the next start with a confusing error, so they are pruned here.
        """
        with self._lock:
            try:
                self._devices = enumerate_devices()
            except AudioBackendUnavailable as e:
                self._backend_error = str(e)
                return False

            self.host._devices = self._devices
            live_keys = {device_node(d.key) for d in self._devices}

            self._reindex()

            graph = self.host.graph_holder.current()
            for node in graph.nodes():
                if node.kind == 'device' and node not in live_keys:
                    graph = graph.without_node(node)
                    logger.info(f"Dropped routes to removed device: {node.ref}")

            self.host.apply_graph(graph)

            for device_id, device in self._device_by_id.items():
                if device_id not in self.channel_control_manager.device_controls:
                    channels = max(device.max_input_channels, device.max_output_channels)
                    self.channel_control_manager.add_device(device_id, channels)

            logger.info(f"Refreshed: {len(self._devices)} devices")
            return True

    def get_devices(self, include_all_backends: bool = False) -> List[Dict]:
        """
        Routable endpoints on the active backend.

        Filtered by host API by default, because the same speakers are enumerated once per
        backend — this machine reports 24 devices that are really 3 pieces of hardware
        seen through MME, DirectSound, WASAPI and WDM-KS. Showing all of them invites the
        user to pick the 120 ms DirectSound copy of the device they wanted. Every serious
        audio application picks a driver first, then its devices; `include_all_backends`
        is there for a settings screen that wants to offer the choice.

        `latency_ms` is what the driver reports for the device. It is not
        `measured_latency_ms` from the engine statistics — that is what the open stream
        actually achieved, and the two differ substantially.
        """
        with self._lock:
            devices: List[Dict] = []
            active_api = self.host.host_api

            for device_id, device in sorted(self._device_by_id.items()):
                if (not include_all_backends and active_api is not None
                        and device.host_api != active_api):
                    continue

                if device.can_input:
                    devices.append({
                        'id': device_id,
                        'name': f"{device.name} (In)" if device.is_duplex else device.name,
                        'type': DeviceType.PHYSICAL_INPUT.value,
                        'channels': device.max_input_channels,
                        'sample_rate': device.default_samplerate,
                        'is_asio': device.host_api == HostApi.ASIO,
                        'is_active': self.host.is_running,
                        'latency_ms': device.default_low_input_latency_ms,
                        'host_api': device.host_api_name,
                        'supports_exclusive': device.supports_exclusive,
                        'direction': 'input',
                    })

                if device.can_output:
                    devices.append({
                        'id': device_id,
                        'name': f"{device.name} (Out)" if device.is_duplex else device.name,
                        'type': DeviceType.PHYSICAL_OUTPUT.value,
                        'channels': device.max_output_channels,
                        'sample_rate': device.default_samplerate,
                        'is_asio': device.host_api == HostApi.ASIO,
                        'is_active': self.host.is_running,
                        'latency_ms': device.default_low_output_latency_ms,
                        'host_api': device.host_api_name,
                        'supports_exclusive': device.supports_exclusive,
                        'direction': 'output',
                    })

            for bus_id, meta in sorted(self._bus_meta.items()):
                devices.append({
                    'id': bus_id,
                    'name': meta['name'],
                    'type': meta['type'],
                    'channels': meta['channels'],
                    'sample_rate': self.sample_rate,
                    'is_asio': False,
                    'is_active': self.host.is_running,
                    'latency_ms': 0.0,
                    'host_api': 'ToneSphere bus (in-process)',
                    'supports_exclusive': False,
                    'direction': meta['direction'],
                })

            return devices

    def get_device_info(self, device_id: int) -> Optional[DeviceInfo]:
        return self._device_by_id.get(device_id)

    def default_output_id(self) -> Optional[int]:
        """
        The device a user would expect audio to come out of.

        Prefers the OS default on the active backend, then any output on it. Avoids
        picking, say, MME's "Microsoft Sound Mapper", which is a routing shim rather than
        a real endpoint.
        """
        return self._default_id(want_output=True)

    def default_input_id(self) -> Optional[int]:
        return self._default_id(want_output=False)

    def _default_id(self, want_output: bool) -> Optional[int]:
        active_api = self.host.host_api

        candidates = [
            (device_id, device)
            for device_id, device in sorted(self._device_by_id.items())
            if (device.can_output if want_output else device.can_input)
            and (active_api is None or device.host_api == active_api)
        ]

        if not candidates:
            return None

        for device_id, device in candidates:
            if device.is_default_output if want_output else device.is_default_input:
                return device_id

        # "Sound Mapper" and "Primary Sound Driver" are host-API shims, not hardware.
        for device_id, device in candidates:
            lowered = device.name.lower()
            if 'sound mapper' not in lowered and 'primary sound' not in lowered:
                return device_id

        return candidates[0][0]

    # --- Buses (previously called "virtual devices") ---

    def create_virtual_input(self, name: str, channels: int = 2) -> Optional[int]:
        """
        Create an input bus.

        Note this is an in-process summing point, not an operating-system device: other
        applications cannot select it. The name is kept for API compatibility.
        """
        return self._create_bus(name, channels, direction='input')

    def create_virtual_output(self, name: str, channels: int = 2) -> Optional[int]:
        return self._create_bus(name, channels, direction='output')

    def _create_bus(self, name: str, channels: int, direction: str) -> Optional[int]:
        with self._lock:
            existing = sum(1 for m in self._bus_meta.values() if m['direction'] == direction)
            limit = self.max_virtual_inputs if direction == 'input' else self.max_virtual_outputs

            if existing >= limit:
                logger.warning(f"Bus limit reached ({limit} {direction}s)")
                return None

            bus_id = self._next_bus_id
            self._next_bus_id += 1

            bus_name = name or f"ToneSphere {direction.title()} {existing + 1}"
            internal = f"{direction}_{bus_id}"

            self.host.create_bus(internal, channels)

            node = bus_node(internal)
            self._id_to_node[bus_id] = node
            self._node_to_id[str(node)] = bus_id
            self._bus_meta[bus_id] = {
                'name': bus_name,
                'internal': internal,
                'channels': channels,
                'direction': direction,
                'type': (DeviceType.VIRTUAL_INPUT if direction == 'input'
                         else DeviceType.VIRTUAL_OUTPUT).value,
            }

            self.channel_control_manager.add_device(bus_id, channels)
            logger.info(f"Created {direction} bus '{bus_name}' (id {bus_id})")
            return bus_id

    def remove_virtual_device(self, device_id: int) -> bool:
        with self._lock:
            meta = self._bus_meta.pop(device_id, None)
            if meta is None:
                return False

            node = self._id_to_node.pop(device_id, None)
            if node is not None:
                self._node_to_id.pop(str(node), None)
                self.host.apply_graph(self.host.graph_holder.current().without_node(node))

            self.host.remove_bus(meta['internal'])
            logger.info(f"Removed bus '{meta['name']}'")
            return True

    delete_virtual_device = remove_virtual_device

    def list_virtual_devices(self) -> List[Dict]:
        return [
            {
                'id': bus_id,
                'name': meta['name'],
                'type': meta['direction'],
                'channels': meta['channels'],
                'sample_rate': self.sample_rate,
                'is_running': self.host.is_running,
            }
            for bus_id, meta in sorted(self._bus_meta.items())
        ]

    def get_virtual_device_counts(self) -> Dict:
        inputs = sum(1 for m in self._bus_meta.values() if m['direction'] == 'input')
        outputs = sum(1 for m in self._bus_meta.values() if m['direction'] == 'output')
        return {
            'input_count': inputs,
            'output_count': outputs,
            'total_count': inputs + outputs,
            'max_inputs': self.max_virtual_inputs,
            'max_outputs': self.max_virtual_outputs,
            'inputs_available': self.max_virtual_inputs - inputs,
            'outputs_available': self.max_virtual_outputs - outputs,
        }

    def update_virtual_device_channels(self, device_id: int, channels: int) -> bool:
        with self._lock:
            meta = self._bus_meta.get(device_id)
            if meta is None or channels < 1:
                return False

            was_running = self.host.is_running
            if was_running:
                self.host.stop()

            self.host.remove_bus(meta['internal'])
            meta['channels'] = channels
            self.host.create_bus(meta['internal'], channels)
            self.channel_control_manager.add_device(device_id, channels)

            if was_running:
                self.start_engine()
            return True

    def update_virtual_device_sample_rate(self, device_id: int, sample_rate: int) -> bool:
        """
        Buses run at the engine's rate.

        A bus is a summing point inside one graph; giving it its own rate would mean
        resampling on every route into and out of it. Changing the engine rate is the
        honest operation, so say so rather than silently doing nothing.
        """
        if device_id not in self._bus_meta:
            return False

        logger.warning(
            "Buses run at the engine sample rate; use set_sample_rate() to change it"
        )
        return False

    def write_to_bus(self, device_id: int, audio: np.ndarray) -> int:
        """Feed audio into a bus from outside the audio callback."""
        meta = self._bus_meta.get(device_id)
        if meta is None:
            return 0
        return self.host.write_bus(meta['internal'], audio)

    # --- Routing ---

    def _build_graph(self) -> RoutingGraph:
        """Translate the int-keyed routing matrix into a graph the host can run."""
        connections = []

        for (source_id, dest_id), route in self.routing_matrix.connections.items():
            source = self._node_for(source_id)
            dest = self._node_for(dest_id)
            if source is None or dest is None:
                continue

            connections.append(Connection(
                source=source,
                dest=dest,
                gain=route.volume,
                muted=route.muted,
            ))

        soloed = frozenset(
            node for node in (
                self._node_for(cid)
                for cid, channel in self.routing_matrix.channels.items()
                if channel.solo
            ) if node is not None
        )

        return RoutingGraph(
            connections=tuple(connections),
            soloed=soloed,
            master_gain=self.master_volume,
        )

    def create_routing(self, source_id: int, destination_id: int,
                       volume: float = 1.0) -> Tuple[bool, str]:
        """
        Route source to destination.

        Rejects cycles: a feedback loop in an audio graph is not a subtle bug, it is a
        runaway howl at whatever volume the user's headphones were set to.
        """
        with self._lock:
            source = self._node_for(source_id)
            dest = self._node_for(destination_id)

            if source is None:
                return False, f"Unknown source device {source_id}"
            if dest is None:
                return False, f"Unknown destination device {destination_id}"

            if self.host.graph_holder.current().would_feedback(source, dest):
                return False, "Refused: this would create a feedback loop"

            success, message = self.routing_matrix.create_routing(source_id, destination_id, volume)
            if not success:
                return success, message

            problems = self._publish_graph()

            # A route whose stream would not open carries no audio, so it is not a
            # success. Roll it back and say what went wrong, rather than returning True
            # with the error tucked into the message where a caller will ignore it.
            if self._route_is_dead(source, dest):
                self.routing_matrix.remove_routing(source_id, destination_id)
                self._publish_graph()
                detail = problems[0] if problems else "device could not be opened"
                return False, f"Could not open the audio path: {detail}"

            if problems:
                return True, f"{message} ({problems[0]})"

            return True, message

    def _route_is_dead(self, source, dest) -> bool:
        """Whether either end of a route failed to open a stream."""
        if not self.host.is_running:
            return False
        dead = set(self.host.dead_nodes())
        return source in dead or dest in dead

    def remove_routing(self, source_id: int, destination_id: int) -> bool:
        with self._lock:
            if not self.routing_matrix.remove_routing(source_id, destination_id):
                return False
            self._publish_graph()
            return True

    def set_routing_volume(self, source_id: int, destination_id: int, volume: float):
        """Change a route's gain. Takes effect on the next block, ramped, no restart."""
        with self._lock:
            self.routing_matrix.set_routing_volume(source_id, destination_id, volume)
            self._publish_graph()

    def set_routing_volume_db(self, source_id: int, destination_id: int, gain_db: float):
        self.set_routing_volume(source_id, destination_id, db_to_linear(gain_db))

    def set_routing_mute(self, source_id: int, destination_id: int, muted: bool):
        with self._lock:
            route = self.routing_matrix.connections.get((source_id, destination_id))
            if route is None:
                return
            route.muted = muted
            self._publish_graph()

    def clear_all_routing(self):
        with self._lock:
            self.routing_matrix.connections.clear()
            self._publish_graph()

    def _publish_graph(self) -> List[str]:
        """
        Hand the current routing to the host.

        Gain, mute and solo changes apply on the next block with no interruption. A route
        to a device with no open stream needs that stream opening, which means a
        reconfigure — handled here so callers never have to know which kind of change they
        just made.
        """
        graph = self._build_graph()
        problems = self.host.apply_graph(graph)

        # No routes means nothing to keep open. Leaving streams running would hold devices
        # exclusively for no reason and keep reporting 'degraded' from a config that is
        # no longer in effect.
        if not graph.connections:
            if self.host.is_running:
                self.host.stop()
            self._problems = []
            return []

        needs_streams = self._started and graph.connections
        must_reconfigure = bool(problems) and self.host.is_running

        if needs_streams and (must_reconfigure or not self.host.is_running):
            if self.host.is_running:
                logger.info("Reconfiguring streams for new routing")
                self.host.stop()

            problems = self.host.configure(graph)
            problems += self.host.start()

            for problem in problems:
                logger.warning(f"Routing change: {problem}")

        self._problems = problems
        return problems

    def get_routing_matrix(self) -> Dict:
        connections = {}
        for (source, dest), route in self.routing_matrix.connections.items():
            connections[f"{source}_{dest}"] = {
                'source_id': route.source_id,
                'destination_id': route.destination_id,
                'state': route.state.value,
                'volume': route.volume,
                'volume_db': round(linear_to_db(route.volume), 1),
                'muted': route.muted,
                'solo': route.solo,
            }
        return connections

    # --- Metering ---

    def get_meters(self) -> Dict[int, Dict[str, float]]:
        """
        Current levels per device id, in dBFS.

        Returns nothing when stopped rather than zeros: a meter reading of 0.0 while no
        audio is running would suggest silence was measured, when nothing was.
        """
        if not self.host.is_running:
            return {}

        meters: Dict[int, Dict[str, float]] = {}

        for key, reading in self.host.meters.read_summaries().items():
            device_id = self._meter_key_to_id(key)
            if device_id is None:
                continue

            meters[device_id] = {
                'peak_db': reading.peak_db,
                'rms_db': reading.rms_db,
                'peak_hold_db': reading.peak_hold_db,
                'clipped': reading.clipped,
            }

        return meters

    def _meter_key_to_id(self, key: str) -> Optional[int]:
        if key.startswith('bus::'):
            return self._node_to_id.get(f"bus:{key[5:]}")

        device_key = key.rsplit('::', 1)[0]
        return self._node_to_id.get(f"device:{device_key}")

    def clear_clip_indicators(self):
        self.host.meters.clear_clips()

    # --- Statistics ---

    def get_performance_stats(self) -> Dict:
        """
        Measured engine statistics.

        Unmeasured values are None, never 0.0 — see `utils.formatting`. `xruns` is
        PortAudio's own count of dropouts, and `drift_corrections` counts how often two
        device clocks had to be resynchronised.
        """
        stats = self.host.statistics().as_dict()

        stats['total_devices'] = len(self._device_by_id)
        stats['virtual_devices'] = len(self._bus_meta)
        stats['active_streams'] = stats.pop('stream_count', 0)
        stats['buffer_underruns'] = stats['xruns']
        stats['active_driver'] = stats.get('host_api')
        stats['backend_error'] = self._backend_error
        stats['problems'] = list(self._problems)

        return stats

    def get_ring_statistics(self) -> Dict[str, dict]:
        return self.host.ring_statistics()

    # --- Backend info ---

    def get_driver_info(self) -> Dict[str, Any]:
        from tonesphere.engine.devices import describe_backend

        info = describe_backend()
        info['active_driver'] = self.host.host_api.value if self.host.host_api else None
        info['exclusive_mode'] = self.host.exclusive
        info['platform'] = __import__('platform').system()
        if self._backend_error:
            info['error'] = self._backend_error
        return info

    def get_available_drivers(self) -> List[str]:
        from tonesphere.engine.devices import available_host_apis

        try:
            return [api.value for api in available_host_apis()]
        except AudioBackendUnavailable:
            return []

    def switch_driver(self, driver_type: str) -> bool:
        """
        Change backend. Requires a restart, because streams belong to a backend.

        Existing routes are dropped: a device key includes its host API, so the same
        speakers are a different node under WASAPI than under WDM-KS.
        """
        with self._lock:
            try:
                api = HostApi.from_name(driver_type)
                if api == HostApi.UNKNOWN:
                    api = HostApi(driver_type)
            except ValueError:
                logger.error(f"Unknown host API: {driver_type}")
                return False

            if api not in self.get_available_drivers_enum():
                logger.error(f"{api.value} is not available on this system")
                return False

            was_running = self.host.is_running
            self.host.stop()

            self.host.host_api = api
            self.routing_matrix.connections.clear()
            self.host.apply_graph(RoutingGraph())

            self.refresh_devices()

            if was_running:
                self.start_engine()

            logger.info(f"Switched to {api.value}")
            return True

    def get_available_drivers_enum(self) -> List[HostApi]:
        from tonesphere.engine.devices import available_host_apis

        try:
            return available_host_apis()
        except AudioBackendUnavailable:
            return []

    def set_exclusive_mode(self, exclusive: bool) -> bool:
        """
        Exclusive mode bypasses the system mixer. Measured 8.3 ms versus 22 ms shared on
        WASAPI here, at the cost of no other application being able to use the device.
        """
        with self._lock:
            if self.host.exclusive == exclusive:
                return True

            was_running = self.host.is_running
            self.host.stop()
            self.host.exclusive = exclusive

            if was_running:
                self.start_engine()
            return True

    def set_sample_rate(self, sample_rate: int) -> bool:
        with self._lock:
            was_running = self.host.is_running
            self.host.stop()

            self.sample_rate = sample_rate
            self.host.samplerate = sample_rate
            self.processor.sample_rate = sample_rate

            if was_running:
                self.start_engine()
            return True

    def set_buffer_size(self, buffer_size: int) -> bool:
        """
        Smaller is lower latency and higher risk of dropouts. 256 frames is a reasonable
        default; 64 is achievable on exclusive-mode hardware.
        """
        with self._lock:
            was_running = self.host.is_running
            self.host.stop()

            self.buffer_size = buffer_size
            self.host.blocksize = buffer_size
            self.processor.buffer_size = buffer_size

            if was_running:
                self.start_engine()
            return True

    # --- Network (unchanged; see README for its current state) ---

    def start_network_streaming(self):
        self.network_router.start_server()

    def stop_network_streaming(self):
        self.network_router.stop_server()

    def get_network_clients(self) -> List[str]:
        return self.network_router.get_connected_clients()

    def connect_to_network(self, host: str, port: int) -> bool:
        return self.network_router.connect_to(host, port)

    def disconnect_from_network(self, conn_id: str):
        self.network_router.disconnect_from(conn_id)

    def get_network_connections(self) -> List[str]:
        return self.network_router.get_connections()

    def get_network_statistics(self) -> Dict:
        return self.network_router.get_statistics()

    def register_network_receive(self, device_id: int):
        """Feed audio arriving from the network into a bus."""
        def receive_callback(audio_data: np.ndarray, packet):
            self.write_to_bus(device_id, audio_data)

        self.network_router.register_receive_callback(device_id, receive_callback)

    def send_device_audio_to_network(self, device_id: int, target: Optional[str] = None):
        logger.warning(
            "Network send is not wired to the audio path; see the Roadmap in README.md"
        )
