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
from typing import Any

import numpy as np

from tonesphere.core.channel_control import ChannelControlManager
from tonesphere.core.models import DeviceType
from tonesphere.core.routing import AudioRoutingMatrix
from tonesphere.engine import (
    AudioBackendUnavailable,
    AudioHost,
    Connection,
    DeviceInfo,
    HostApi,
    RoutingGraph,
    bus_node,
    db_to_linear,
    device_node,
    enumerate_devices,
    linear_to_db,
    network_node,
    preferred_host_api,
)
from tonesphere.network.audio_router import NetworkAudioRouter, NetworkQuality
from tonesphere.network.jitter_buffer import JitterBuffer
from tonesphere.network.send_worker import NetworkSendWorker
from tonesphere.network.udp_transport import MalformedPacket, UdpAudioTransport
from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)

# Buses get ids from here up, so a bus id can never be mistaken for a device id.
BUS_ID_BASE = 10000

# Network send sinks get ids from here up — above the bus range, so the three id spaces
# (hardware, bus, network) stay disjoint and a stale int held by a UI can never end up
# pointing at a different kind of thing than it did.
NETWORK_ID_BASE = 20000

# Default port for the UDP transport. One above the TCP router's 9001, so the two can run
# side by side on one machine without either having to be reconfigured.
DEFAULT_UDP_PORT = 9002


class AudioEngine:
    """Routing engine over real audio hardware."""

    def __init__(
        self,
        sample_rate: int = 48000,
        buffer_size: int = 256,
        preferred_driver: HostApi | None = None,
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
        self.network_router = NetworkAudioRouter(quality=NetworkQuality.HIGH)

        # Both transports live at once rather than one replacing the other: TCP is right
        # for a bulk feed that must not lose a sample and can afford buffering, UDP for
        # monitoring that would rather drop a block than wait. Neither opens a socket
        # until asked.
        self.udp_transport = UdpAudioTransport(
            quality=NetworkQuality.HIGH, sample_rate=sample_rate
        )

        # Id bookkeeping.
        self._devices: list[DeviceInfo] = []
        self._id_to_node: dict[int, Any] = {}
        self._node_to_id: dict[str, int] = {}
        self._device_by_id: dict[int, DeviceInfo] = {}
        self._bus_meta: dict[int, dict[str, Any]] = {}
        self._next_bus_id = BUS_ID_BASE
        # Keyed by the bus each capture feeds, so stopping one is the same operation as
        # removing its bus.
        self._process_captures: dict[int, dict[str, Any]] = {}

        self._network_sinks: dict[int, dict[str, Any]] = {}
        self._next_network_id = NETWORK_ID_BASE
        self._send_worker: NetworkSendWorker | None = None
        self._udp_receives: dict[int, dict[str, Any]] = {}

        # OS-level virtual endpoints: Linux null-sinks (Track 2) and the macOS CoreAudio
        # HAL device (Track 3). Both are, once they exist, ordinary PortAudio devices —
        # PulseAudio/PipeWire and the ALSA bridge in `engine/linux_virtual.py` are the
        # transport on Linux, and `coreaudiod` is on macOS — so `refresh_devices()`
        # enumerates one exactly like a sound card and `_reindex()` assigns it an id from
        # the normal hardware range (below `BUS_ID_BASE`). This dict is keyed by that same
        # id and never allocates one of its own, so there is no new id range to keep
        # disjoint from `BUS_ID_BASE`/`NETWORK_ID_BASE` — it exists only to remember which
        # already-real device ids are ours, which backend put them there (`'backend'`), and
        # to drive the `origin` label in `get_devices()`. One registry, not one per
        # platform: the two backends differ in how an endpoint appears, not in what it is
        # once it has.
        self._system_virtual_devices: dict[int, dict[str, Any]] = {}

        self._lock = threading.RLock()
        self._initialized = False
        self._started = False
        self._backend_error: str | None = None
        self._problems: list[str] = []

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

        A routing that touches no hardware at all — bus to bus, or bus to a network sink —
        is the same legitimate idle state. It is a complete, working configuration that
        moves audio through ring buffers and a socket without any PortAudio stream, so
        reporting "route something to a device first" about it would be wrong twice over:
        the route exists, and it is already carrying audio.
        """
        with self._lock:
            if not self._initialized:
                self.initialize()

            if self._backend_error:
                raise RuntimeError(f"Cannot start: {self._backend_error}")

            if self.host.is_running:
                return

            self._started = True

            graph = self._build_graph()

            if not any(node.kind == 'device' for node in graph.nodes()):
                self._problems = []
                logger.info("Engine started with no hardware routes — nothing to open")
                return

            self._problems = self.host.configure(graph)
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

    def create_monitor_patch(self, muted: bool = True) -> tuple[bool, str]:
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
        # Outside the lock: stopping a capture joins its thread, and no other caller
        # should be blocked behind that.
        self._stop_all_process_captures()

        with self._lock:
            self._started = False
            self.host.stop()

    @property
    def is_running(self) -> bool:
        return self.host.is_running

    def cleanup(self):
        self._stop_all_process_captures()

        with self._lock:
            # Network threads are torn down here rather than in `stop_engine()`, because
            # `stop_engine()` is a pause the engine restarts from and a peer should not
            # have its stream dropped by one. `cleanup()` is the terminal call, so nothing
            # is left running past it.
            self.stop_udp_transport()
            self.network_router.stop_server()

            # Same reasoning as the network transports: a Linux virtual sink is an
            # OS-level resource, not a thread, so pausing the engine must not unload it —
            # only the terminal `cleanup()` does, matching the "ephemeral, torn down on
            # exit" lifecycle Track 2 is designed around.
            self._teardown_all_system_virtual_devices()

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

    def _id_for(self, node) -> int | None:
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

    def get_devices(self, include_all_backends: bool = False) -> list[dict]:
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

        `origin` says what actually put this endpoint here — `'hardware'`, an in-process
        `'in_process_bus'`, or an OS-level `'os_virtual_endpoint'` (a Linux sink from
        Track 2, or the macOS HAL device from Track 3) — so a caller can tell a real card
        from a bus from an endpoint ToneSphere published to the OS, without guessing from
        `host_api`. `host_api` stays PortAudio's literal truth either way (e.g. `"ALSA"`
        for a Linux sink and `"Core Audio"` for the HAL device, since that is genuinely
        what enumerated them) rather than being overloaded to carry this.
        """
        with self._lock:
            devices: list[dict] = []
            active_api = self.host.host_api

            for device_id, device in sorted(self._device_by_id.items()):
                if (not include_all_backends and active_api is not None
                        and device.host_api != active_api):
                    continue

                origin = (
                    'os_virtual_endpoint' if device_id in self._system_virtual_devices
                    else 'hardware'
                )

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
                        'origin': origin,
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
                        'origin': origin,
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
                    'origin': 'in_process_bus',
                })

            return devices

    def get_device_info(self, device_id: int) -> DeviceInfo | None:
        return self._device_by_id.get(device_id)

    def default_output_id(self) -> int | None:
        """
        The device a user would expect audio to come out of.

        Prefers the OS default on the active backend, then any output on it. Avoids
        picking, say, MME's "Microsoft Sound Mapper", which is a routing shim rather than
        a real endpoint.
        """
        return self._default_id(want_output=True)

    def default_input_id(self) -> int | None:
        return self._default_id(want_output=False)

    def _default_id(self, want_output: bool) -> int | None:
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

    def create_virtual_input(self, name: str, channels: int = 2) -> int | None:
        """
        Create an input bus.

        Note this is an in-process summing point, not an operating-system device: other
        applications cannot select it. The name is kept for API compatibility.
        """
        return self._create_bus(name, channels, direction='input')

    def create_virtual_output(self, name: str, channels: int = 2) -> int | None:
        return self._create_bus(name, channels, direction='output')

    def _create_bus(self, name: str, channels: int, direction: str) -> int | None:
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

            # Anything network-attached to this bus goes with it. Left behind, the send
            # worker would keep reading a route whose source no longer exists and the
            # playout thread would keep writing into a bus that is gone — both reporting
            # healthy counts for audio going nowhere.
            self.disable_network_send(device_id)
            self.unregister_network_receive(device_id)

            node = self._id_to_node.pop(device_id, None)
            if node is not None:
                self._node_to_id.pop(str(node), None)
                self.host.apply_graph(self.host.graph_holder.current().without_node(node))

            self.host.remove_bus(meta['internal'])
            logger.info(f"Removed bus '{meta['name']}'")
            return True

    delete_virtual_device = remove_virtual_device

    def list_virtual_devices(self) -> list[dict]:
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

    def get_virtual_device_counts(self) -> dict:
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

    # --- Linux virtual sink (Track 2) ---

    def create_linux_system_sink(self, name: str, channels: int = 2) -> int | None:
        """
        Create an OS-visible sink other Linux applications can select, and return the
        PortAudio device id that reaches it.

        Confirmed on real CI, the hard way (see `engine/linux_virtual.py`'s module
        docstring): PortAudio's ALSA backend does not enumerate a custom-named PCM even
        with a proper ALSA `hint` block and even after retrying past any startup race —
        `aplay -L` sees it, `sd.query_devices()` never does. What PortAudio *does* always
        expose is the generic `pulse` device, which is simply "whatever PulseAudio's
        current default sink/source is" — so `linux_virtual.create_virtual_sink` points
        those defaults at the sink it just made, and this method hands back `pulse`'s own
        device id. `_system_virtual_devices` remembers which id that is, for `origin`
        labelling and so `remove_linux_system_sink` knows what to restore — it does not
        allocate ids of its own. None, never a placeholder id, if the OS never actually
        made the sink reachable this way: unavailable on this platform, no Pulse/PipeWire
        server running, or no `pulse` PortAudio device at all are real, distinct failures,
        but none of them get a device id that would carry no audio.

        Because this retargets PulseAudio's one system-wide default, only one Linux system
        sink can be meaningfully active at a time — a second call while one is already
        active just retargets the same default again, which `linux_virtual`'s own module
        docstring states plainly rather than pretending both could be independently live.
        """
        from tonesphere.engine.linux_virtual import (
            LinuxVirtualSinkError,
            create_virtual_sink,
            remove_virtual_sink,
        )

        with self._lock:
            try:
                handle = create_virtual_sink(name, channels=channels, samplerate=self.sample_rate)
            except LinuxVirtualSinkError as e:
                logger.error(f"Could not create Linux virtual sink '{name}': {e}")
                return None

            self.refresh_devices()
            device_id = self._find_device_id_by_name('pulse')

            if device_id is None:
                known_names = [d.name for d in self._device_by_id.values()]
                logger.error(
                    f"'{name}' was loaded via pactl (module {handle.module_id}) and made "
                    f"PulseAudio's default, but this machine's PortAudio has no 'pulse' "
                    f"device at all to reach it through — tearing it down rather than "
                    f"reporting a device id backed by nothing. PortAudio's names: {known_names}."
                )
                try:
                    remove_virtual_sink(handle)
                except LinuxVirtualSinkError as cleanup_error:
                    logger.error(f"Could not clean up after a failed sink creation: {cleanup_error}")
                return None

            self._system_virtual_devices[device_id] = {
                'name': name, 'handle': handle, 'backend': 'linux-null-sink',
            }
            logger.info(f"Created Linux virtual sink '{name}' -> device {device_id} (via 'pulse')")
            return device_id

    def _find_device_id_by_name(self, name: str) -> int | None:
        """Substring match, because both backends' endpoints arrive under a decorated
        name: ALSA prefixes the pcm name, CoreAudio can suffix the device name."""
        for device_id, device in self._device_by_id.items():
            if name in device.name:
                return device_id
        return None

    def remove_linux_system_sink(self, device_id: int) -> bool:
        """
        Tear down the OS-level sink, then this engine's record of it.

        The OS teardown happens first: if `pactl unload-module` were to fail after the
        bookkeeping was already cleared, `get_devices()` would start reporting a sink
        that still exists as ordinary `'hardware'`, which is exactly the mislabelling
        `origin` exists to prevent.
        """
        from tonesphere.engine.linux_virtual import LinuxVirtualSinkError, remove_virtual_sink
        from tonesphere.engine.macos_virtual import MACOS_BACKEND

        with self._lock:
            entry = self._system_virtual_devices.get(device_id)
            if entry is None:
                return False

            # Both backends share this registry, and both expose a remove endpoint that
            # takes a device id, so an id from the other one can genuinely arrive here.
            if entry.get('backend') == MACOS_BACKEND:
                return False

            try:
                remove_virtual_sink(entry['handle'])
            except LinuxVirtualSinkError as e:
                logger.error(f"Could not remove Linux virtual sink '{entry['name']}': {e}")
                return False

            self._system_virtual_devices.pop(device_id, None)

            node = self._id_to_node.pop(device_id, None)
            if node is not None:
                self._node_to_id.pop(str(node), None)
                self.host.apply_graph(self.host.graph_holder.current().without_node(node))

            self.refresh_devices()
            logger.info(f"Removed Linux virtual sink '{entry['name']}'")
            return True

    # --- macOS CoreAudio HAL device (Track 3) ---

    def create_macos_system_device(self) -> int | None:
        """
        Attach to the ToneSphere CoreAudio HAL device and return the PortAudio device id
        it enumerates as, or None with a logged reason if it is not there.

        Note "attach", not "create": unlike the Linux backend, which loads a null-sink on
        demand, nothing here can bring a device into existence. Installing a HAL plug-in
        means copying a bundle into `/Library/Audio/Plug-Ins/HAL` with `sudo` and
        restarting `coreaudiod`, which interrupts audio for every application on the
        machine — not something to do behind a REST call. A human runs
        `native/coreaudio-plugin`'s `make install` once; this method finds the resulting
        device by the exact name the plug-in publishes and records it in the same
        `_system_virtual_devices` registry the Linux sinks use, so `get_devices()` labels
        it `os_virtual_endpoint` rather than plain hardware.

        Matching is on that exact name on purpose: a user who has BlackHole or Soundflower
        installed has an OS-visible loopback device that ToneSphere did not put there, and
        claiming it as ours would be a straightforward lie about what this project did.
        """
        from tonesphere.engine.macos_virtual import (
            MACOS_BACKEND,
            MACOS_DEVICE_NAME,
            hal_plugin_path,
            unavailable_reason,
        )

        with self._lock:
            reason = unavailable_reason()
            if reason is not None:
                logger.error(f"Cannot attach the macOS HAL device: {reason}")
                return None

            self.refresh_devices()
            device_id = self._find_device_id_by_name(MACOS_DEVICE_NAME)

            if device_id is None:
                logger.error(
                    f"The plug-in bundle is installed at {hal_plugin_path()} but "
                    f"'{MACOS_DEVICE_NAME}' is not in PortAudio's device list — "
                    f"coreaudiod has most likely not been restarted since it was "
                    f"installed (`sudo launchctl kickstart -k "
                    f"system/com.apple.audio.coreaudiod`)"
                )
                return None

            self._system_virtual_devices[device_id] = {
                'name': MACOS_DEVICE_NAME, 'handle': None, 'backend': MACOS_BACKEND,
            }
            logger.info(f"Attached the macOS HAL device '{MACOS_DEVICE_NAME}' -> device {device_id}")
            return device_id

    def remove_macos_system_device(self, device_id: int) -> bool:
        """
        Forget the registration. The device itself keeps existing.

        This is the honest asymmetry with `remove_linux_system_sink`, which really does
        unload the endpoint: uninstalling a HAL plug-in needs `sudo` and another
        machine-wide `coreaudiod` restart, so all this can do is stop claiming the device
        as ToneSphere's — after which `get_devices()` reports it as ordinary hardware,
        which is exactly what an installed-but-unclaimed HAL device is from ToneSphere's
        point of view. Existing routes to it are deliberately left alone: unlike a removed
        null-sink, the device is still there and still carrying whatever was patched to it.
        """
        from tonesphere.engine.macos_virtual import MACOS_BACKEND

        with self._lock:
            entry = self._system_virtual_devices.get(device_id)
            if entry is None or entry.get('backend') != MACOS_BACKEND:
                return False

            self._system_virtual_devices.pop(device_id, None)
            logger.info(
                f"Released the macOS HAL device at {device_id} "
                f"(the plug-in stays installed; see native/coreaudio-plugin/README.md)"
            )
            return True

    def _teardown_all_system_virtual_devices(self):
        """
        No OS-level endpoint outlives the engine that created it.

        The macOS branch releases a registration rather than an OS resource, because that
        is all it ever held — the plug-in is installed by a human and stays installed.
        """
        from tonesphere.engine.macos_virtual import MACOS_BACKEND

        for device_id, entry in list(self._system_virtual_devices.items()):
            if entry.get('backend') == MACOS_BACKEND:
                self.remove_macos_system_device(device_id)
            else:
                self.remove_linux_system_sink(device_id)

    # --- Per-process capture (Windows) ---

    def start_process_capture(self, pid: int, name: str | None = None,
                              include_process_tree: bool = True) -> int:
        """
        Capture one application's audio output into a new bus, and return the bus id.

        The bus is created inside the capture's `on_format` callback, once Windows has
        said what it is actually delivering, so its channel count cannot disagree with
        the audio arriving in it. Raises `ProcessCaptureError` on any real failure — it
        never hands back a bus id backed by silence, which on Windows is what a naive
        implementation gets, since activation succeeds for process ids that do not exist.
        """
        from tonesphere.engine.process_capture import ProcessCapture, ProcessCaptureError

        with self._lock:
            if any(c['pid'] == pid for c in self._process_captures.values()):
                raise ProcessCaptureError(f"Process {pid} is already being captured")

            capture = ProcessCapture(
                pid, include_process_tree=include_process_tree,
                sample_rate=self.sample_rate, channels=2)

            created: dict[str, Any] = {}

            def on_format(fmt):
                bus_id = self.create_virtual_input(
                    name or f"Capture: process {pid}", channels=fmt.channels)
                if bus_id is None:
                    raise ProcessCaptureError(
                        "No input bus available for the capture (limit reached)")

                created['bus_id'] = bus_id
                resampler = self._capture_resampler(fmt)
                created['resampler'] = resampler

                if resampler is None:
                    return lambda block: self.write_to_bus(bus_id, block)
                return lambda block: self.write_to_bus(bus_id, resampler(block))

            try:
                fmt = capture.start(on_format)
            except ProcessCaptureError:
                bus_id = created.get('bus_id')
                if bus_id is not None:
                    self.remove_virtual_device(bus_id)
                raise

            bus_id = created['bus_id']
            self._process_captures[bus_id] = {
                'capture': capture,
                'pid': pid,
                'name': name or f"Capture: process {pid}",
                'format': fmt,
                'resampled': created['resampler'] is not None,
            }

            logger.info(f"Process {pid} capture -> bus {bus_id} ({fmt.describe()})")
            return bus_id

    def _capture_resampler(self, fmt):
        """
        Convert a capture's rate to the engine's, or None when they already agree.

        Measured on Windows 11: process loopback delivers whatever rate it is asked for,
        so this is normally None. It exists for the case where Windows delivers something
        else, because the alternative to converting is playing the capture at the wrong
        pitch and calling it working. `DriftResampler`'s ±0.2% clamp is for clock drift,
        so the clamp is lifted here — this is outright rate conversion, not drift.
        """
        if fmt.sample_rate == self.sample_rate:
            return None

        from tonesphere.engine.dsp import DriftResampler

        logger.warning(
            f"Capture delivers {fmt.sample_rate} Hz into a {self.sample_rate} Hz engine; "
            f"resampling by {fmt.sample_rate / self.sample_rate:.4f}"
        )

        resampler = DriftResampler(fmt.channels, self.buffer_size,
                                   max_ratio_deviation=1.0)
        resampler.set_ratio(fmt.sample_rate / self.sample_rate)

        def convert(block: np.ndarray) -> np.ndarray:
            frames = int(block.shape[0] / resampler.ratio)
            if frames < 1:
                return block[:0]
            out = np.zeros((frames, fmt.channels), dtype=np.float32)
            resampler.process(block, out, frames)
            return out

        return convert

    def stop_process_capture(self, bus_id: int) -> bool:
        """Stop a capture and remove the bus it fed. False if there is no such capture."""
        with self._lock:
            entry = self._process_captures.pop(bus_id, None)

        if entry is None:
            return False

        entry['capture'].stop()

        with self._lock:
            self.remove_virtual_device(bus_id)

        logger.info(f"Stopped capture of process {entry['pid']}")
        return True

    def process_capture_status(self, bus_id: int | None = None) -> list[dict]:
        """
        Measured state of every running capture, or just one.

        `running` comes from the capture thread being alive, so a capture that died takes
        its claim to be running with it instead of leaving a stale flag behind.
        """
        with self._lock:
            entries = (
                [(bus_id, self._process_captures[bus_id])]
                if bus_id is not None and bus_id in self._process_captures
                else sorted(self._process_captures.items())
            )

            return [
                {
                    'bus_id': captured_id,
                    'name': entry['name'],
                    'resampled': entry['resampled'],
                    **entry['capture'].statistics(),
                }
                for captured_id, entry in entries
            ]

    def _stop_all_process_captures(self):
        """No capture thread outlives the engine that started it."""
        with self._lock:
            bus_ids = list(self._process_captures)

        for bus_id in bus_ids:
            self.stop_process_capture(bus_id)

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
                pan=route.pan,
                invert=route.inverted,
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
                       volume: float = 1.0) -> tuple[bool, str]:
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
            # A network send is a routing-matrix route like any other, so clearing the
            # patchbay clears it too — and its registration has to go with it, or the send
            # worker would keep reporting a route the graph no longer contains.
            for device_id in [meta['device_id'] for meta in self._network_sinks.values()]:
                self.disable_network_send(device_id)

            self.routing_matrix.connections.clear()
            self._publish_graph()

    def _publish_graph(self) -> list[str]:
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

        # Only a route touching real hardware needs a stream. A graph made only of buses
        # and network sinks is a complete, working configuration — it moves audio through
        # ring buffers and a socket — and trying to open streams for it would report
        # "route something to a device first" about a route that already exists.
        needs_streams = self._started and any(
            node.kind == 'device' for node in graph.nodes()
        )
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

    def get_routing_matrix(self) -> dict:
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
                'pan': route.pan,
                'inverted': route.inverted,
            }
        return connections

    # --- Channel controls ---

    def apply_channel_controls(self, device_id: int | None = None):
        """
        Push channel-control state into the running audio path.

        Call after any set_channel_* change. Before this existed, `ChannelControlManager`
        stored volume, mute, solo, pan and polarity for every device and none of it
        reached a single sample — the UI's faders moved and nothing happened.

        A device with no open stream is skipped rather than treated as an error: the
        setting is kept on the control side and applied when its stream next opens.
        """
        with self._lock:
            targets = [device_id] if device_id is not None else list(self._id_to_node)

            for target in targets:
                control = self.channel_control_manager.device_controls.get(target)
                if control is None:
                    continue

                node = self._node_for(target)
                if node is None:
                    continue

                if node.kind == 'bus':
                    # Buses have no device strip; their level is the route gain.
                    continue

                input_strip, output_strip = self.host.strips_for(node.ref)
                for strip in (input_strip, output_strip):
                    if strip is not None:
                        control.apply_to_strip(strip)

    def set_channel_volume(self, device_id: int, channel: int, volume: float):
        self.channel_control_manager.set_device_channel_volume(device_id, channel, volume)
        self.apply_channel_controls(device_id)

    def set_channel_mute(self, device_id: int, channel: int, muted: bool):
        self.channel_control_manager.set_device_channel_mute(device_id, channel, muted)
        self.apply_channel_controls(device_id)

    def set_channel_solo(self, device_id: int, channel: int, solo: bool):
        self.channel_control_manager.set_device_channel_solo(device_id, channel, solo)
        self.apply_channel_controls(device_id)

    def set_channel_pan(self, device_id: int, channel: int, pan: float):
        self.channel_control_manager.set_device_channel_pan(device_id, channel, pan)
        self.apply_channel_controls(device_id)

    def set_channel_inverted(self, device_id: int, channel: int, inverted: bool):
        control = self.channel_control_manager.device_controls.get(device_id)
        if control is not None:
            control.set_channel_inverted(channel, inverted)
            self.apply_channel_controls(device_id)

    def swap_channels(self, device_id: int):
        self.channel_control_manager.swap_device_channels(device_id)
        self.apply_channel_controls(device_id)

    def set_device_master_volume(self, device_id: int, volume: float):
        self.channel_control_manager.set_device_master_volume(device_id, volume)
        self.apply_channel_controls(device_id)

    def set_device_master_mute(self, device_id: int, muted: bool):
        self.channel_control_manager.set_device_master_mute(device_id, muted)
        self.apply_channel_controls(device_id)

    # --- Inserts (EQ, dynamics, plugins) ---

    def get_inserts(self, device_id: int, is_input: bool = True):
        """
        The insert chain for a device, or None if it has no open stream.

        None is normal while stopped. Inserts belong to a stream, so they exist only once
        that stream is open; configuration should be reapplied after a restart.
        """
        node = self._node_for(device_id)
        if node is None or node.kind != 'device':
            return None
        return self.host.inserts_for(node.ref, is_input)

    def load_plugin(self, device_id: int, path: str, is_input: bool = True) -> tuple[bool, str]:
        """
        Load a VST3/AU plugin onto a device's insert chain.

        This is the feature the project exists for: rather than paying for a separate
        router to get a guitar into Guitar Rig, load Guitar Rig here and monitor through it
        on the same low-latency path.
        """
        inserts = self.get_inserts(device_id, is_input)
        if inserts is None:
            return False, "Device has no open stream — start the engine and patch it first"

        try:
            if inserts.plugins is None:
                inserts.enable_plugins()
            index = inserts.plugins.load(path)
        except Exception as e:
            return False, str(e)

        latency = self.host.total_plugin_latency_ms()
        note = f" (+{latency:.1f} ms plugin latency)" if latency > 0.05 else ""
        return True, f"Loaded {inserts.plugins.names[index]}{note}"

    def list_plugins(self, device_id: int, is_input: bool = True) -> list[dict[str, Any]]:
        inserts = self.get_inserts(device_id, is_input)
        if inserts is None or inserts.plugins is None:
            return []
        return inserts.plugins.describe()

    def remove_plugin(self, device_id: int, index: int, is_input: bool = True) -> bool:
        inserts = self.get_inserts(device_id, is_input)
        if inserts is None or inserts.plugins is None:
            return False
        inserts.plugins.remove(index)
        return True

    @staticmethod
    def discover_plugins(paths: list[str] | None = None) -> list[str]:
        from tonesphere.engine.effects import PluginChain

        return PluginChain.discover(paths)

    @staticmethod
    def plugin_hosting_available() -> bool:
        from tonesphere.engine.effects import PluginChain

        return PluginChain.is_available()

    # --- Route parameters ---

    def set_routing_pan(self, source_id: int, destination_id: int, pan: float):
        """
        Pan one route. Per route, not per source: the same guitar can sit centre in the
        headphone mix and hard left in a recording feed.
        """
        with self._lock:
            route = self.routing_matrix.connections.get((source_id, destination_id))
            if route is None:
                return
            route.pan = min(max(pan, -1.0), 1.0)
            self._publish_graph()

    def set_routing_invert(self, source_id: int, destination_id: int, invert: bool):
        with self._lock:
            route = self.routing_matrix.connections.get((source_id, destination_id))
            if route is None:
                return
            route.inverted = invert
            self._publish_graph()

    # --- Metering ---

    def get_meters(self) -> dict[int, dict[str, float]]:
        """
        Current levels per device id, in dBFS.

        Returns nothing when stopped rather than zeros: a meter reading of 0.0 while no
        audio is running would suggest silence was measured, when nothing was.
        """
        if not self.host.is_running:
            return {}

        meters: dict[int, dict[str, float]] = {}

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

    def _meter_key_to_id(self, key: str) -> int | None:
        if key.startswith('bus::'):
            return self._node_to_id.get(f"bus:{key[5:]}")

        device_key = key.rsplit('::', 1)[0]
        return self._node_to_id.get(f"device:{device_key}")

    def clear_clip_indicators(self):
        self.host.meters.clear_clips()

    # --- Statistics ---

    def get_performance_stats(self) -> dict:
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

    def get_ring_statistics(self) -> dict[str, dict]:
        return self.host.ring_statistics()

    # --- Backend info ---

    def get_driver_info(self) -> dict[str, Any]:
        from tonesphere.engine.devices import describe_backend

        info = describe_backend()
        info['active_driver'] = self.host.host_api.value if self.host.host_api else None
        info['exclusive_mode'] = self.host.exclusive
        info['platform'] = __import__('platform').system()
        if self._backend_error:
            info['error'] = self._backend_error
        return info

    def get_available_drivers(self) -> list[str]:
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
            self.clear_all_routing()
            self.host.apply_graph(RoutingGraph())

            self.refresh_devices()

            if was_running:
                self.start_engine()

            logger.info(f"Switched to {api.value}")
            return True

    def get_available_drivers_enum(self) -> list[HostApi]:
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

            # The network side paces itself from the rate, so leaving it on the old one
            # would send at the wrong speed and slowly under- or overrun the peer.
            self.udp_transport.sample_rate = sample_rate
            if self._send_worker is not None:
                self._send_worker.sample_rate = sample_rate

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

            if self._send_worker is not None:
                self._send_worker.frames_per_read = buffer_size

            if was_running:
                self.start_engine()
            return True

    # --- Network: TCP (bulk) ---

    def start_network_streaming(self):
        self.network_router.start_server()

    def stop_network_streaming(self):
        self.network_router.stop_server()

    def get_network_clients(self) -> list[str]:
        return self.network_router.get_connected_clients()

    def connect_to_network(self, host: str, port: int) -> bool:
        return self.network_router.connect_to(host, port)

    def disconnect_from_network(self, conn_id: str):
        self.network_router.disconnect_from(conn_id)

    def get_network_connections(self) -> list[str]:
        return self.network_router.get_connections()

    # --- Network: UDP (realtime) ---

    def start_udp_transport(
        self, bind_host: str = '127.0.0.1', bind_port: int = DEFAULT_UDP_PORT
    ) -> tuple[bool, str]:
        """
        Bind the UDP socket and start receiving.

        Defaults to `127.0.0.1` rather than `0.0.0.0`: binding every interface raises a
        Windows firewall prompt, which is not a thing a local monitoring setup or a test
        run should do unasked. A caller that wants to be reachable from another machine
        passes that machine-visible address in deliberately.
        """
        with self._lock:
            try:
                host, port = self.udp_transport.start(bind_host, bind_port)
            except OSError as e:
                return False, f"Could not bind {bind_host}:{bind_port} — {e}"
            return True, f"UDP transport listening on {host}:{port}"

    def stop_udp_transport(self):
        """Stop the socket, the send worker and every playout thread."""
        with self._lock:
            if self._send_worker is not None:
                self._send_worker.stop()

            for state in list(self._udp_receives.values()):
                self._stop_playout(state)
            self._udp_receives.clear()

            self.udp_transport.stop()

    def add_udp_peer(self, name: str, host: str, port: int) -> str:
        return self.udp_transport.add_peer(name, host, port)

    def remove_udp_peer(self, name: str) -> bool:
        return self.udp_transport.remove_peer(name)

    def get_udp_peers(self) -> dict[str, str]:
        return {
            name: f"{host}:{port}"
            for name, (host, port) in self.udp_transport.peers().items()
        }

    def set_network_quality(self, quality: str) -> tuple[bool, str]:
        """Set the quality preset on both transports, so one scale means one thing."""
        try:
            preset = NetworkQuality(quality)
        except ValueError:
            options = ', '.join(q.value for q in NetworkQuality)
            return False, f"Unknown quality '{quality}' — choose one of: {options}"

        with self._lock:
            self.network_router.quality = preset
            self.udp_transport.quality = preset
        return True, f"Network quality set to {preset.value}"

    # --- Network send (the direction that was never wired) ---

    def send_device_audio_to_network(
        self, device_id: int, target: str | None = None, transport: str = 'udp'
    ) -> tuple[bool, str]:
        """
        Stream a device's or bus's audio out over the network.

        The send target is registered as a real routing-matrix connection, not as a tap on
        the side of the graph. That is not a stylistic choice: `_build_graph` reconstructs
        the entire host graph from `routing_matrix.connections` and every routing mutation
        republishes it, so a connection pushed straight into the host would be erased the
        next time anything else in the patchbay changed. Going through
        `routing_matrix.create_routing` is the difference between "works" and "works until
        the user drags a cable".

        Once it is a graph node, the host allocates it a ring like any other route and the
        send worker is an ordinary consumer of that ring.
        """
        if transport != 'udp':
            # The TCP path has a working receive side and no send side, and this is not
            # the change that gives it one. Refused rather than logged-and-ignored, which
            # is what this method did before.
            return False, (
                "TCP send is still not wired to the audio path — only its receive side is. "
                "Use transport='udp' for the realtime path."
            )

        with self._lock:
            source = self._node_for(device_id)
            if source is None:
                return False, f"Unknown source device {device_id}"

            existing = self._sink_id_for_source(device_id)
            if existing is not None:
                return False, (
                    f"Device {device_id} is already sending to the network "
                    f"(sink {existing}); disable it first to change the target"
                )

            if not self.udp_transport.is_running:
                # Port 0: a send-only peer does not need a predictable port, and asking
                # for one it does not need is one more thing that can already be in use.
                started, message = self.start_udp_transport(bind_port=0)
                if not started:
                    return False, message

            sink_id = self._next_network_id
            self._next_network_id += 1

            node = network_node(f"send_{sink_id}")
            self._id_to_node[sink_id] = node
            self._node_to_id[str(node)] = sink_id

            success, message = self.create_routing(device_id, sink_id)
            if not success:
                self._id_to_node.pop(sink_id, None)
                self._node_to_id.pop(str(node), None)
                return False, f"Could not route {device_id} to the network: {message}"

            self._network_sinks[sink_id] = {
                'device_id': device_id,
                'source': str(source),
                'dest': str(node),
                'target': target,
                'transport': 'udp',
            }

            worker = self._ensure_send_worker()
            worker.add_route(
                sink_id=sink_id, device_id=device_id,
                source_key=str(source), dest_key=str(node), target=target,
            )
            worker.start()

            peers = len(self.udp_transport.peers())
            where = f"peer '{target}'" if target else f"{peers} peer(s)"
            note = "" if peers else " — no peers registered yet, so nothing is leaving this machine"
            return True, f"Device {device_id} is sending to {where}{note}"

    def disable_network_send(self, device_id: int) -> bool:
        """Tear a network send route back down, matrix entry included."""
        with self._lock:
            sink_id = self._sink_id_for_source(device_id)
            if sink_id is None:
                return False

            self.remove_routing(device_id, sink_id)

            if self._send_worker is not None:
                self._send_worker.remove_route(sink_id)
                if self._send_worker.route_count == 0:
                    self._send_worker.stop()

            node = self._id_to_node.pop(sink_id, None)
            if node is not None:
                self._node_to_id.pop(str(node), None)

            self._network_sinks.pop(sink_id, None)
            logger.info(f"Network send disabled for device {device_id}")
            return True

    def list_network_sends(self) -> list[dict]:
        return [
            {'sink_id': sink_id, **meta}
            for sink_id, meta in sorted(self._network_sinks.items())
        ]

    def _sink_id_for_source(self, device_id: int) -> int | None:
        for sink_id, meta in self._network_sinks.items():
            if meta['device_id'] == device_id:
                return sink_id
        return None

    def _ensure_send_worker(self) -> NetworkSendWorker:
        if self._send_worker is None:
            self._send_worker = NetworkSendWorker(
                transport=self.udp_transport,
                # `read_available`, not `read_route`: the sender is paced by a timer, and
                # a zero-filled short read would put silence on the wire every time that
                # timer ran late. See the note in `send_worker`'s docstring.
                reader=self.host.read_available,
                sample_rate=self.sample_rate,
                frames_per_read=self.buffer_size,
            )
        return self._send_worker

    # --- Network receive ---

    def register_network_receive(
        self,
        device_id: int,
        transport: str = 'tcp',
        target_latency_ms: float = 40.0,
        conceal: str = 'silence',
    ) -> tuple[bool, str]:
        """
        Feed audio arriving from the network into a bus.

        TCP keeps its original behaviour: packets are written to the bus as they arrive,
        which is acceptable for a path that is already buffered end to end.

        UDP does not. Writing datagrams on arrival hands the audio path the network's
        timing instead of the clock's, which stutters even on a link losing nothing, so
        packets go into a `JitterBuffer` and a paced thread drains it. The buffer's
        geometry is taken from the first packet that actually arrives rather than assumed,
        because the sender's channel count and packet size are its choice, not ours.
        """
        if transport == 'tcp':
            def receive_callback(audio_data: np.ndarray, packet):
                self.write_to_bus(device_id, audio_data)

            self.network_router.register_receive_callback(device_id, receive_callback)
            return True, f"Device {device_id} will receive TCP audio"

        if transport != 'udp':
            return False, f"Unknown transport '{transport}' — use 'tcp' or 'udp'"

        with self._lock:
            if device_id not in self._bus_meta:
                return False, (
                    f"Device {device_id} is not a bus — network audio can only be "
                    f"written into a bus"
                )

            if device_id in self._udp_receives:
                return False, f"Device {device_id} is already receiving UDP audio"

            if not self.udp_transport.is_running:
                started, message = self.start_udp_transport()
                if not started:
                    return False, message

            state: dict[str, Any] = {
                'device_id': device_id,
                'buffer': None,
                'target_latency_ms': target_latency_ms,
                'conceal': conceal,
                'packets_received': 0,
                'packets_rejected': 0,
                'frames_written': 0,
                'writes_refused': 0,
                'late_wakes': 0,
                'error': None,
                'stop': threading.Event(),
                'thread': None,
            }

            self.udp_transport.register_handler(
                device_id, self._make_udp_receive_handler(state)
            )

            thread = threading.Thread(
                target=self._playout_loop, args=(state,),
                name=f"udp-playout-{device_id}", daemon=True,
            )
            state['thread'] = thread
            self._udp_receives[device_id] = state
            thread.start()

            bound = self.udp_transport.bound_address
            return True, (
                f"Device {device_id} will receive UDP audio on "
                f"{bound[0]}:{bound[1]} ({target_latency_ms:.0f} ms jitter buffer)"
            )

    def unregister_network_receive(self, device_id: int) -> bool:
        with self._lock:
            self.network_router.unregister_receive_callback(device_id)
            self.udp_transport.unregister_handler(device_id)

            state = self._udp_receives.pop(device_id, None)
            if state is None:
                return False

            self._stop_playout(state)
            return True

    def _make_udp_receive_handler(self, state: dict[str, Any]):
        """
        The receive-thread callback for one device.

        Runs on the transport's receive thread, so it does no more than decode and insert.
        Everything paced happens in `_playout_loop`.
        """
        def handler(packet):
            try:
                audio = packet.decode()
            except MalformedPacket as e:
                state['packets_rejected'] += 1
                state['error'] = f"undecodable packet: {e}"
                return

            buffer = state['buffer']

            if buffer is None or buffer.channels != packet.channels \
                    or buffer.frames_per_packet != packet.frame_count:
                # First packet, or a sender that changed its geometry mid-stream. Either
                # way the buffer's existing contents describe a different signal, so it is
                # rebuilt and re-primed rather than fed blocks it cannot align.
                buffer = JitterBuffer(
                    frames_per_packet=packet.frame_count,
                    channels=packet.channels,
                    sample_rate=self.sample_rate,
                    target_latency_ms=state['target_latency_ms'],
                    conceal=state['conceal'],
                )
                state['buffer'] = buffer

            state['packets_received'] += 1
            buffer.push(packet.sequence, audio)

        return handler

    def _playout_loop(self, state: dict[str, Any]):
        """
        Drain one jitter buffer into its bus on a paced clock.

        Pulls one packet per *elapsed* slot rather than one per wake-up, which is the
        difference between working and not. An MTU-sized packet is about 3 ms of audio,
        and `threading.Event.wait` resolves to the system tick — measured here at roughly
        8-15 ms on Windows — so one pull per wake would drain the buffer at a third of the
        rate packets arrive at. Measured before this was fixed: the buffer climbed to its
        64-packet cap and started evicting, reporting a healthy link as packet loss.

        Pulling by elapsed time keeps the average rate exact and only coarsens the write
        granularity, which costs nothing: the destination is a ring buffer drained by the
        audio callback, not a converter that needs samples handed to it on the tick.

        While no buffer exists — nothing has arrived yet — it polls slowly rather than
        spinning, and inserts nothing at all: a buffer that has not started is not a
        buffer producing silence.
        """
        stop: threading.Event = state['stop']
        device_id = state['device_id']
        next_slot = time.monotonic()

        # A ceiling on catch-up, so a long stall re-bases the clock instead of dumping a
        # second of audio into the bus in one write.
        max_slots_per_wake = 32

        while not stop.is_set():
            buffer = state['buffer']

            if buffer is None:
                if stop.wait(0.01):
                    break
                next_slot = time.monotonic()
                continue

            slot = buffer.frames_per_packet / self.sample_rate
            now = time.monotonic()

            if now < next_slot:
                if stop.wait(next_slot - now):
                    break
                now = time.monotonic()

            due = 0
            while next_slot <= now and due < max_slots_per_wake:
                next_slot += slot
                due += 1

            if due >= max_slots_per_wake:
                state['late_wakes'] += 1
                next_slot = now + slot

            for _ in range(due):
                try:
                    block = buffer.pull()
                except Exception as e:
                    # A playout thread that dies must not leave the caller reading healthy
                    # statistics off a dead thread; record why and stop.
                    state['error'] = f"playout failed: {e}"
                    logger.error(f"UDP playout for device {device_id} stopped: {e}")
                    return

                if block is None:
                    # Still priming. Re-base the clock so the slots spent waiting are not
                    # owed later as a burst the moment priming completes.
                    next_slot = time.monotonic() + slot
                    break

                written = self.write_to_bus(device_id, block)
                state['frames_written'] += written
                if written == 0:
                    # Nothing is routed out of this bus, so the audio has nowhere to go.
                    # Counted separately from a network problem, because it is not one.
                    state['writes_refused'] += 1

    def _stop_playout(self, state: dict[str, Any]):
        state['stop'].set()
        thread = state.get('thread')
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    # --- Network statistics ---

    def get_network_statistics(self) -> dict:
        """
        Both transports, with TCP's keys left flat where they have always been.

        Existing callers read `packets_sent`/`bytes_sent`/`quality` off the top level, so
        moving them under a `'tcp'` key would break the CLI and the REST response for no
        gain. UDP is additive, under `'udp'`.
        """
        stats = dict(self.network_router.get_statistics())

        receives = {}
        for device_id, state in self._udp_receives.items():
            buffer = state['buffer']
            thread = state.get('thread')
            receives[str(device_id)] = {
                'device_id': device_id,
                'packets_received': state['packets_received'],
                'packets_rejected': state['packets_rejected'],
                'frames_written': state['frames_written'],
                'writes_refused': state['writes_refused'],
                'late_wakes': state['late_wakes'],
                'playout_alive': bool(thread is not None and thread.is_alive()),
                'error': state['error'],
                # None, not an empty stat block: before the first packet arrives there is
                # no buffer and therefore nothing measured about one.
                'jitter_buffer': buffer.statistics() if buffer is not None else None,
            }

        send = (
            self._send_worker.statistics() if self._send_worker is not None
            else {'running': False, 'routes': {}}
        )

        for route in send.get('routes', {}).values():
            route['ring'] = self.host.route_statistics(route['source'], route['dest'])

        stats['udp'] = {
            'transport': self.udp_transport.statistics(),
            'send': send,
            'receive': receives,
        }

        return stats
