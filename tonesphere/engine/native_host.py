"""
The audio host on Windows: `AudioHost`'s interface, backed by the native engine.

`AudioEngine` (and through it the UI, the API and the CLI) talks to a host through a small
surface — configure, start, stop, apply_graph, buses, strips, meters, statistics. This
class provides that surface over `tonesphere_native.dll`, so none of those callers change.
What changes is underneath: no Python runs on the audio thread, routing is compiled into a
plan the engine swaps in without stopping, devices are MMDevice endpoints (or an ASIO
driver) addressed by stable IDs, and every statistic is timed on the audio thread itself.

Buses. A native bus is a summing node, which nothing outside the engine can write to. So
each bus also gets a companion ring source ("feed") routed into it at unity: whatever
Python writes to the bus — network receive, per-process capture, the diagnostic tone —
goes through that ring. Device -> bus -> device needs no feed and simply works, which the
PortAudio host never did.

Clocks. A plan that routes a device runs from that device's clock (the first output, else
the first input; the rest cross clock boundaries through drift-corrected rings). A plan
that routes no device at all — a bus feeding a network stream — runs from the engine's own
timer, because otherwise nothing would ever run a block.
"""

import threading
import time

import numpy as np

from tonesphere.engine.devices import DeviceInfo, HostApi
from tonesphere.engine.dsp import balance_gains
from tonesphere.engine.graph import GraphHolder, NodeId, RoutingGraph, bus_node, loopback_node
from tonesphere.engine.host import HostStatistics
from tonesphere.engine.meters import MeterReading
from tonesphere.native import NativeEngine, NativeError, NativeUnavailable
from tonesphere.native.plan import BUS, INPUT, OUTPUT, Endpoint, PlanCompiler
from tonesphere.utils.logger import get_logger
from tonesphere.utils.threads import synchronized

logger = get_logger(__name__)

WASAPI_NAME = "Windows WASAPI"
ASIO_NAME = "ASIO"
FEED = 'feed'
# The engine's block capacity. Each device callback runs the engine once, however many
# frames the device asks for, so this bounds the largest device buffer — it is not a
# latency setting. The configured buffer size is what each stream asks the device for.
CAPACITY_FRAMES = 4096
FEED_RING_BLOCKS = 32
NETWORK_RING_BLOCKS = 32
PEAK_HOLD_S = 1.5


def native_devices(host_api: HostApi | None) -> list[DeviceInfo]:
    """Endpoints as `DeviceInfo`, from MMDevice — or, when ASIO is chosen, one per ASIO driver."""
    if host_api == HostApi.ASIO:
        return _asio_devices()
    from tonesphere.native import wasapi

    devices = []
    for i, e in enumerate(wasapi.endpoints()):
        render = e.flow == 'render'
        period = e.min_period_ms or 0.0
        devices.append(DeviceInfo(
            index=i, name=e.name, host_api=HostApi.WASAPI, host_api_name=WASAPI_NAME,
            max_input_channels=0 if render else e.mix_channels,
            max_output_channels=e.mix_channels if render else 0,
            default_samplerate=e.mix_sample_rate,
            default_low_input_latency_ms=0.0 if render else period,
            default_low_output_latency_ms=period if render else 0.0,
            default_high_input_latency_ms=0.0 if render else (e.default_period_ms or 0.0),
            default_high_output_latency_ms=(e.default_period_ms or 0.0) if render else 0.0,
            is_default_input=not render and e.is_default,
            is_default_output=render and e.is_default,
            endpoint_id=e.id,
        ))
    return devices


def _asio_devices() -> list[DeviceInfo]:
    from tonesphere.native import asio

    devices = []
    for i, d in enumerate(asio.drivers()):
        if not d.dll_present:
            continue
        try:
            info = asio.query(d.name)
        except NativeError as e:
            logger.warning(f"ASIO driver {d.name} could not be queried: {e}")
            continue
        rate = int(info.sample_rate) if info.sample_rate else 48000
        latency_in = info.input_latency_frames / rate * 1000 if rate else 0.0
        latency_out = info.output_latency_frames / rate * 1000 if rate else 0.0
        devices.append(DeviceInfo(
            index=i, name=d.name, host_api=HostApi.ASIO, host_api_name=ASIO_NAME,
            max_input_channels=len(info.inputs), max_output_channels=len(info.outputs),
            default_samplerate=rate,
            default_low_input_latency_ms=latency_in, default_low_output_latency_ms=latency_out,
            default_high_input_latency_ms=latency_in, default_high_output_latency_ms=latency_out,
            supported_samplerates=tuple(info.sample_rates), endpoint_id=d.name,
        ))
    return devices


class _Strip:
    """
    A device side's channel strip, as `DeviceChannelControl.apply_to_strip` drives it,
    mapped onto the native node's strip. Mute is a trim of zero because the native strip
    has per-channel trim and polarity but no per-channel mute. A stereo strip's balance
    (channel 0's pan) is folded into the trims, by the engine's own balance law; pan on a
    non-stereo strip belongs on routes.
    """

    def __init__(self, engine: NativeEngine, node: int, channels: int):
        self._engine = engine
        self._node = node
        self.channels = channels
        self._gain = [1.0] * channels
        self._muted = [False] * channels
        self._inverted = [False] * channels
        self._balance = [1.0] * channels
        self._swapped = False
        self._master = 1.0

    def rebind(self, engine: NativeEngine, node: int):
        """Point at a rebuilt engine (a rate or block-size change) and push every setting again."""
        self._engine = engine
        self._node = node
        for c in range(self.channels):
            self._push(c)
            engine.set_channel_inverted(node, c, self._inverted[c])
        if self.channels >= 2:
            engine.set_node_swapped(node, self._swapped)
        engine.set_node_gain(node, self._master)

    def _push(self, channel: int):
        trim = 0.0 if self._muted[channel] else self._gain[channel] * self._balance[channel]
        self._engine.set_channel_trim(self._node, channel, trim)

    def set_channel_gain(self, channel: int, gain: float):
        if channel < self.channels:
            self._gain[channel] = max(0.0, float(gain))
            self._push(channel)

    def set_channel_mute(self, channel: int, muted: bool):
        if channel < self.channels:
            self._muted[channel] = bool(muted)
            self._push(channel)

    def set_channel_inverted(self, channel: int, inverted: bool):
        if channel < self.channels:
            self._inverted[channel] = bool(inverted)
            self._engine.set_channel_inverted(self._node, channel, inverted)

    def set_channel_pan(self, channel: int, pan: float):
        if self.channels == 2 and channel == 0:
            self._balance = list(balance_gains(pan))
            self._push(0)
            self._push(1)

    def set_swapped(self, swapped: bool):
        if self.channels >= 2:
            self._swapped = bool(swapped)
            self._engine.set_node_swapped(self._node, self._swapped)

    def set_master_gain(self, gain: float):
        self._master = max(0.0, float(gain))
        self._engine.set_node_gain(self._node, self._master)


class _Meters:
    """
    `MeterRegistry`'s read side over native meters. The engine reports the peak since the
    last read and the latest block's RMS; peak hold and the clip latch live here, on the
    control side, so the audio thread does nothing but measure.
    """

    def __init__(self, host: "NativeHost"):
        self._host = host
        self._held: dict[str, list[tuple[float, float]]] = {}
        self._clipped: set[str] = set()

    def read_summaries(self) -> dict[str, MeterReading]:
        with self._host._lock:
            return self._read_summaries()

    def _read_summaries(self) -> dict[str, MeterReading]:
        out = {}
        now = time.monotonic()
        engine = self._host._engine
        for key, node in self._host._meter_nodes().items():
            try:
                m = engine.meter(node)
            except NativeError:
                continue
            peaks = [m['peak'], *m['channel_peak']]
            holds = self._held.get(key)
            if holds is None or len(holds) != len(peaks):
                holds = [(0.0, 0.0)] * len(peaks)
            for i, peak in enumerate(peaks):
                held, until = holds[i]
                if peak >= held or now > until:
                    holds[i] = (peak, now + PEAK_HOLD_S)
            self._held[key] = holds
            if m['clipped']:
                self._clipped.add(key)
            out[key] = MeterReading(peak=m['peak'], rms=m['rms'], peak_hold=holds[0][0], clipped=key in self._clipped,
                                    channel_peaks=tuple(m['channel_peak']), channel_rms=tuple(m['channel_rms']),
                                    channel_holds=tuple(h for h, _ in holds[1:]))
        engine.reset_meters()
        return out

    def clear_clips(self):
        self._clipped.clear()


@synchronized()
class NativeHost:
    """
    Thread-safe: every public method takes the host's lock, the data paths
    (`write_bus`, `read_available`) included, so a rate or block-size change can swap and
    close the native engine while network and capture threads are writing into it.
    """

    backend = 'native'

    def __init__(self, samplerate: int = 48000, blocksize: int = 256, host_api: HostApi | None = None,
                 exclusive: bool = False):
        self._samplerate = samplerate
        self._blocksize = blocksize
        self.host_api = host_api or HostApi.WASAPI
        self.exclusive = exclusive
        self._devices: list[DeviceInfo] = []
        self.graph_holder = GraphHolder()
        self._lock = threading.RLock()
        self._engine = NativeEngine(samplerate, max(blocksize, CAPACITY_FRAMES))
        self._compiler = PlanCompiler()
        self._buses: dict[str, int] = {}
        self._plan = None
        self._device_sides: dict[tuple[str, str], int] = {}   # (device key, role) -> native node
        self._loopback_sides: dict[str, int] = {}              # output device key -> native source
        self._loopback_clock: tuple[int, str] | None = None     # (native sink, device key)
        self._stream_set: frozenset = frozenset()
        self._running = False
        self._clock = False
        self._strips: dict[tuple[str, str], _Strip] = {}
        # Each device side's (and bus's) processing chain, in order: `BuiltinInsert`s and
        # VST3 `PluginInstance`s. Its slot in the engine is its position here.
        self._chains: dict[tuple[NodeId, str], list] = {}
        self._last_error: str | None = None
        self.meters = _Meters(self)

    # The engine's rate and block size are fixed when it is created, so changing either
    # rebuilds it — keeping strip settings, and reopening every plugin at the new rate
    # with the state it had.

    @property
    def samplerate(self) -> int:
        return self._samplerate

    @samplerate.setter
    def samplerate(self, value: int):
        if value != self._samplerate:
            self._samplerate = value
            self._rebuild()

    @property
    def blocksize(self) -> int:
        return self._blocksize

    @blocksize.setter
    def blocksize(self, value: int):
        if value != self._blocksize:
            self._blocksize = value
            self._rebuild()

    def _rebuild(self):
        with self._lock:
            self.stop()
            reopened = {side: [entry if getattr(entry, 'is_builtin', False) else self._reopen(entry)
                               for entry in chain]
                        for side, chain in self._chains.items()}
            old = self._engine
            self._engine = NativeEngine(self._samplerate, max(self._blocksize, CAPACITY_FRAMES))
            old.close()
            self._chains = reopened
            self._republish()
            for side, strip in self._strips.items():
                if side in self._device_sides:
                    strip.rebind(self._engine, self._device_sides[side])

    def _reopen(self, instance):
        from tonesphere.plugins import PluginInstance

        state = instance.state()
        instance.close()
        fresh = PluginInstance(instance.info, self._samplerate, self._engine.max_block, instance.channels)
        fresh.restore(state)
        fresh.bypassed = instance.bypassed
        return fresh

    # --- Insert chains: built-in processors and VST3 plugins, per device side or bus ---

    def _side(self, node: NodeId, is_input: bool) -> tuple[NodeId, str]:
        # A bus is one native node; its chain runs on what it has summed.
        return (node, OUTPUT) if node.kind == 'bus' else (node, INPUT if is_input else OUTPUT)

    def _side_channels(self, node: NodeId, is_input: bool) -> int:
        if node.kind == 'bus':
            channels = self._buses.get(node.ref)
            if channels is None:
                raise ValueError(f"bus {node.ref} does not exist")
            return channels
        if node.kind != 'device':
            raise ValueError(f"{node} cannot host effects")
        device = self._device(node.ref)
        if device is None:
            raise ValueError(f"{node.ref}: device not present")
        channels = device.max_input_channels if is_input else device.max_output_channels
        if channels < 1:
            raise ValueError(f"{device.name} has no {'input' if is_input else 'output'} channels")
        return channels

    def chain_for(self, node: NodeId, is_input: bool) -> list:
        return list(self._chains.get(self._side(node, is_input), []))

    def add_plugin(self, node: NodeId, is_input: bool, info) -> int:
        """Open `info` for this side's width and append it to the chain; its index there."""
        from tonesphere.plugins import PluginInstance

        channels = self._side_channels(node, is_input)
        instance = PluginInstance(info, self._samplerate, self._engine.max_block, channels)
        return self._append(node, is_input, instance)

    def add_builtin(self, node: NodeId, is_input: bool, kind: str, values: list[float] | None = None) -> int:
        from tonesphere.engine.builtins import BuiltinInsert

        self._side_channels(node, is_input)
        return self._append(node, is_input, BuiltinInsert(kind, values))

    def _append(self, node: NodeId, is_input: bool, entry) -> int:
        from tonesphere.native._abi import MAX_INSERTS

        chain = self._chains.setdefault(self._side(node, is_input), [])
        if len(chain) >= MAX_INSERTS:
            if not getattr(entry, 'is_builtin', False):
                entry.close()
            raise ValueError(f"a chain holds at most {MAX_INSERTS} effects")
        chain.append(entry)
        self._republish()
        return len(chain) - 1

    def remove_insert(self, node: NodeId, is_input: bool, index: int) -> bool:
        chain = self._chains.get(self._side(node, is_input), [])
        if not 0 <= index < len(chain):
            return False
        entry = chain.pop(index)
        self._republish()
        if not getattr(entry, 'is_builtin', False):
            entry.close()  # released once the plan that used it is retired
        return True

    def move_insert(self, node: NodeId, is_input: bool, index: int, to: int) -> bool:
        chain = self._chains.get(self._side(node, is_input), [])
        if not (0 <= index < len(chain) and 0 <= to < len(chain)):
            return False
        chain.insert(to, chain.pop(index))
        self._republish()
        return True

    def set_insert_bypassed(self, node: NodeId, is_input: bool, index: int, bypassed: bool) -> bool:
        """The host's bypass: the processor is not run at all, and a plugin's latency no longer counts."""
        chain = self._chains.get(self._side(node, is_input), [])
        if not 0 <= index < len(chain):
            return False
        chain[index].bypassed = bool(bypassed)
        self._republish()
        return True

    def set_builtin_value(self, node: NodeId, is_input: bool, index: int, param: int, value: float) -> float:
        """Set a built-in's parameter in its own unit; it reaches the audio thread on the next block."""
        side = self._side(node, is_input)
        entry = self._chains.get(side, [])[index]
        if not getattr(entry, 'is_builtin', False):
            raise ValueError("not a built-in effect")
        applied = entry.set_value(param, value)
        native = self._plan.native_ids.get(side) if self._plan is not None else None
        if native is not None:
            self._engine.set_insert_param(native, index, param, applied)
        return applied

    def _plan_inserts(self) -> dict[tuple[NodeId, str], list[tuple]]:
        from tonesphere.native import VST3

        out = {}
        for side, chain in self._chains.items():
            out[side] = [(slot, entry.type, entry.bypassed) if getattr(entry, 'is_builtin', False)
                         else (slot, VST3, entry.bypassed, entry.handle)
                         for slot, entry in enumerate(chain)]
        return out

    def _push_builtin_values(self):
        """After a plan swap: a moved or rebuilt processor would otherwise run its defaults."""
        for side, chain in self._chains.items():
            native = self._plan.native_ids.get(side)
            if native is None:
                continue
            for slot, entry in enumerate(chain):
                if getattr(entry, 'is_builtin', False):
                    for param, value in enumerate(entry.values):
                        self._engine.set_insert_param(native, slot, param, value)

    def insert_readout(self, node: NodeId, is_input: bool, index: int) -> float | None:
        """A built-in's gain reduction (compressor, limiter) as the audio thread last applied it."""
        native = self._plan.native_ids.get(self._side(node, is_input)) if self._plan is not None else None
        if native is None or not self._running:
            return None
        try:
            return self._engine.insert_readout(native, index)
        except NativeError:
            return None

    def plugin_latency_samples(self) -> int:
        """Reported by the plugins on output paths plus the worst input path: what a player hears."""
        per_side = {side: sum(e.latency_samples for e in chain
                              if not getattr(e, 'is_builtin', False) and not e.bypassed)
                    for side, chain in self._chains.items()}
        worst_in = max((n for (_node, role), n in per_side.items() if role == INPUT), default=0)
        worst_out = max((n for (_node, role), n in per_side.items() if role == OUTPUT), default=0)
        return worst_in + worst_out

    # --- Backend description, for AudioEngine.get_driver_info ---

    def available_host_apis(self) -> list[HostApi]:
        apis = [HostApi.WASAPI]
        try:
            from tonesphere.native import asio

            if asio.available() and any(d.dll_present for d in asio.drivers()):
                apis.append(HostApi.ASIO)
        except Exception as e:
            logger.debug(f"ASIO not offered: {e}")
        return apis

    def describe(self) -> dict:
        from tonesphere.native import build_info

        apis = self.available_host_apis()
        devices = self._devices
        return {
            'engine': build_info(),
            'backend': 'native',
            'host_apis': [api.value for api in apis],
            'preferred_host_api': HostApi.WASAPI.value,
            'device_count': len(devices),
            'input_count': sum(1 for d in devices if d.can_input),
            'output_count': sum(1 for d in devices if d.can_output),
            'asio_available': HostApi.ASIO in apis,
        }

    # --- Devices ---

    def enumerate(self) -> list[DeviceInfo]:
        return native_devices(self.host_api)

    def _device(self, key: str) -> DeviceInfo | None:
        return next((d for d in self._devices if d.key == key), None)

    @property
    def supports_loopback(self) -> bool:
        """Whole-system loopback of an output is a WASAPI shared-mode feature; ASIO has none."""
        return self.host_api == HostApi.WASAPI

    # --- Buses ---

    def create_bus(self, name: str, channels: int = 2):
        with self._lock:
            self._buses[name] = channels
            self._republish()

    def remove_bus(self, name: str):
        with self._lock:
            self._buses.pop(name, None)
            self._republish()

    def write_bus(self, name: str, block: np.ndarray) -> int:
        """Queue audio into a bus from any control-side producer; frames accepted (short when full)."""
        channels = self._buses.get(name)
        if channels is None or self._plan is None:
            return 0
        # A bus with nowhere to send audio takes none, as with the PortAudio host: callers
        # count that as an unrouted write, which means something different from a fault.
        source = bus_node(name)
        if not any(c.source == source for c in self.graph_holder.current().connections):
            return 0
        feed = self._compiler.native_id(source, FEED)
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        if block.shape[1] != channels:
            if block.shape[1] == 1:
                block = np.repeat(block, channels, axis=1)
            else:
                fitted = np.zeros((block.shape[0], channels), np.float32)
                width = min(channels, block.shape[1])
                fitted[:, :width] = block[:, :width]
                block = fitted
        try:
            return self._engine.port_write(feed, block)
        except NativeError:
            return 0

    def read_available(self, source: str, dest: str, max_frames: int) -> np.ndarray | None:
        """Everything a network sink has queued, up to `max_frames`; None if no such route."""
        graph = self.graph_holder.current()
        if not any(str(c.source) == source and str(c.dest) == dest for c in graph.connections):
            return None
        node = self._compiler.native_id(_parse(dest), OUTPUT)
        try:
            frames = min(self._engine.port_available(node), max_frames)
            return self._engine.port_read(node, frames) if frames > 0 else np.zeros((0, 2), np.float32)
        except NativeError:
            return None

    def node_for_side(self, device_key: str, is_input: bool) -> int | None:
        return self._device_sides.get((device_key, INPUT if is_input else OUTPUT))

    # --- Graph ---

    def _endpoints(self, graph: RoutingGraph) -> tuple[dict[NodeId, Endpoint], list[str]]:
        endpoints: dict[NodeId, Endpoint] = {}
        problems = []
        for node in graph.nodes():
            if node.kind == 'device':
                device = self._device(node.ref)
                if device is None:
                    problems.append(f"{node.ref}: device not present")
                    continue
                endpoints[node] = Endpoint(device.max_input_channels, device.max_output_channels)
            elif node.kind == 'loopback':
                device = self._device(node.ref)
                if device is None or not device.can_output or not self.supports_loopback:
                    problems.append(f"loopback of {node.ref}: not available")
                    continue
                endpoints[node] = Endpoint(device.max_output_channels, 0, limiter=False)
            elif node.kind == 'bus':
                channels = self._buses.get(node.ref)
                if channels is None:
                    problems.append(f"bus {node.ref} does not exist")
                    continue
                endpoints[node] = Endpoint(channels, channels, limiter=False)
        for name, channels in self._buses.items():
            endpoints.setdefault(bus_node(name), Endpoint(channels, channels, limiter=False))
        for c in graph.connections:
            if c.dest.kind == 'network' and c.source in endpoints:
                source = endpoints[c.source]
                channels = source.input_channels or source.output_channels
                endpoints[c.dest] = Endpoint(0, channels, ring_frames=self.blocksize * NETWORK_RING_BLOCKS,
                                             limiter=False)
        return endpoints, problems

    def _compile(self, graph: RoutingGraph):
        endpoints, problems = self._endpoints(graph)
        usable = RoutingGraph(
            connections=tuple(c for c in graph.connections if c.source in endpoints and c.dest in endpoints),
            soloed=graph.soloed, master_gain=graph.master_gain,
        )
        try:
            plan = self._compiler.compile(usable, endpoints, self._plan_inserts(),
                                          always=tuple(bus_node(name) for name in self._buses))
        except (KeyError, ValueError) as e:
            return None, problems + [str(e)]

        # Each bus's feed: a ring source Python writes to, routed into the bus at unity.
        from tonesphere.native import Node, Route
        for name, channels in self._buses.items():
            feed = self._compiler.native_id(bus_node(name), FEED)
            bus = self._compiler.native_id(bus_node(name), BUS)
            plan.nodes.append(Node.source(feed, channels, ring_frames=self.blocksize * FEED_RING_BLOCKS))
            plan.routes.append(Route(feed, bus))

        # A loopback has no clock of its own (Windows sends it nothing during silence), so a
        # plan whose only device is a loopback also renders silence into that output, in
        # shared mode, to be its clock — which also keeps its loopback delivering packets.
        self._loopback_clock = None
        loopbacks = [node for (node, _role) in plan.native_ids if node.kind == 'loopback']
        if loopbacks and not any(node.kind == 'device' for (node, _role) in plan.native_ids):
            ref = loopbacks[0].ref
            clock = self._compiler.native_id(loopback_node(ref), 'clock')
            plan.nodes.append(Node.sink(clock, endpoints[loopbacks[0]].input_channels, limiter=False))
            self._loopback_clock = (clock, ref)
        return plan, problems

    def _apply(self, plan):
        self._engine.apply_plan(plan.nodes, plan.routes, plan.inserts)
        self._engine.set_master_gain(plan.master_gain)
        self._plan = plan
        self._push_builtin_values()
        self._device_sides = {(node.ref, role): native for (node, role), native in plan.native_ids.items()
                              if node.kind == 'device'}
        self._loopback_sides = {node.ref: native for (node, _role), native in plan.native_ids.items()
                                if node.kind == 'loopback'}
        for side in list(self._strips):
            if side not in self._device_sides:
                del self._strips[side]

    def _wanted_streams(self) -> frozenset:
        return frozenset((key, role, native) for (key, role), native in self._device_sides.items()) | \
            frozenset((key, 'loopback', native) for key, native in self._loopback_sides.items())

    def _republish(self):
        """Recompile and swap in the current graph; used when buses or inserts change."""
        plan, problems = self._compile(self.graph_holder.current())
        if plan is not None:
            try:
                self._apply(plan)
            except NativeError as e:
                logger.warning(f"Plan refused: {e}")

    def apply_graph(self, graph: RoutingGraph) -> list[str]:
        """
        Swap in a new routing. Gain, mute, solo and bus changes take effect on the next
        block without stopping anything. If the set of devices in use changed, a problem is
        returned so the caller reconfigures, exactly as with the PortAudio host.
        """
        with self._lock:
            self.graph_holder.commit(graph)
            plan, problems = self._compile(graph)
            if plan is None:
                return problems
            try:
                self._apply(plan)
            except NativeError as e:
                return problems + [str(e)]
            if self._running and self._wanted_streams() != self._stream_set:
                problems.append("the devices in use changed")
            return problems

    def configure(self, graph: RoutingGraph) -> list[str]:
        with self._lock:
            self.graph_holder.commit(graph)
            plan, problems = self._compile(graph)
            if plan is None:
                return problems
            try:
                self._apply(plan)
            except NativeError as e:
                problems.append(str(e))
            return problems

    # --- Lifecycle ---

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> list[str]:
        with self._lock:
            if self._running:
                return []
            if self._plan is None:
                self.configure(self.graph_holder.current())
            sides = sorted(self._device_sides.items(), key=lambda item: (item[0][1] != OUTPUT, item[0][0]))
            problems = []
            try:
                if self._loopback_sides and self.host_api != HostApi.WASAPI:
                    problems.append("whole-system loopback needs WASAPI; ASIO has none")
                if not sides and not self._loopback_sides:
                    self._engine.start_clock(self._blocksize)
                    self._clock = True
                elif self.host_api == HostApi.ASIO:
                    problems += self._start_asio(sides)
                else:
                    problems += self._start_wasapi(sides)
            except NativeError as e:
                self._last_error = str(e)
                return problems + [f"could not start: {e}"]
            self._running = True
            self._stream_set = self._wanted_streams()
            for status in self._engine.stream_status():
                if status['state'] == 'failed':
                    problems.append(f"{self._key_for_node(status['node_id'])}: {status['message']}")
            return problems

    def _start_wasapi(self, sides) -> list[str]:
        from tonesphere.native.wasapi import StreamSpec

        specs = []
        for (key, role), native in sides:
            device = self._device(key)
            if device is None or not device.endpoint_id:
                continue
            channels = device.max_output_channels if role == OUTPUT else device.max_input_channels
            specs.append(StreamSpec(native, 'render' if role == OUTPUT else 'capture', channels, device.endpoint_id,
                                    exclusive=self.exclusive, period_frames=self._blocksize))
        if not specs and self._loopback_clock is not None:
            clock, key = self._loopback_clock
            device = self._device(key)
            specs.append(StreamSpec(clock, 'render', device.max_output_channels, device.endpoint_id))
        for key, native in sorted(self._loopback_sides.items()):
            device = self._device(key)
            if device is not None and device.endpoint_id:
                # Loopback is shared mode only: it hears the Windows mixer, which an
                # exclusive stream on the same device bypasses.
                specs.append(StreamSpec(native, 'loopback', device.max_output_channels, device.endpoint_id))
        self._engine.start_wasapi(specs, master=0)
        return []

    def _start_asio(self, sides) -> list[str]:
        from tonesphere.native import asio

        keys = {key for (key, _), _ in sides}
        if len(keys) > 1:
            return ["ASIO drives one device per process; only one ASIO device can be routed at a time"]
        key = next(iter(keys))
        device = self._device(key)
        in_node = self._device_sides.get((key, INPUT), 0)
        out_node = self._device_sides.get((key, OUTPUT), 0)
        asio.start(self._engine, device.endpoint_id,
                   input_node=in_node, inputs=tuple(range(device.max_input_channels)) if in_node else (),
                   output_node=out_node, outputs=tuple(range(device.max_output_channels)) if out_node else (),
                   buffer_frames=self._asio_buffer(device))
        return []

    def _asio_buffer(self, device: DeviceInfo) -> int:
        """The configured buffer if the driver allows it, else its preferred size (0)."""
        from tonesphere.native import asio

        try:
            info = asio.query(device.endpoint_id)
        except NativeError:
            return 0
        size = self._blocksize
        allowed = info.min_buffer <= size <= info.max_buffer
        if allowed and info.granularity == -1:
            allowed = size & (size - 1) == 0
        elif allowed and info.granularity > 0:
            allowed = (size - info.min_buffer) % info.granularity == 0
        return size if allowed else 0

    def stop(self):
        with self._lock:
            if not self._running:
                return
            try:
                self._engine.stop_backend()
            except NativeError:
                pass
            self._running = False
            self._clock = False
            self._stream_set = frozenset()

    def cleanup(self):
        self.stop()
        for chain in self._chains.values():
            for entry in chain:
                if not getattr(entry, 'is_builtin', False):
                    entry.close()
        self._chains.clear()
        self._engine.close()

    # --- Strips ---

    def strips_for(self, device_key: str):
        """(input strip, output strip) for a device side that is in the plan, else None for that side."""
        def strip(role):
            native = self._device_sides.get((device_key, role))
            if native is None:
                return None
            device = self._device(device_key)
            channels = (device.max_input_channels if role == INPUT else device.max_output_channels) if device else 2
            if (device_key, role) not in self._strips:
                self._strips[(device_key, role)] = _Strip(self._engine, native, channels)
            return self._strips[(device_key, role)]
        return strip(INPUT), strip(OUTPUT)

    # --- Health and statistics ---

    def _key_for_node(self, native: int) -> str:
        for (key, _), node in self._device_sides.items():
            if node == native:
                return key
        for key, node in self._loopback_sides.items():
            if node == native:
                return f"loopback::{key}"
        return str(native)

    def failed_streams(self) -> dict[str, str]:
        if not self._running:
            return {}
        return {self._key_for_node(s['node_id']): s['message'] or 'stream failed'
                for s in self._engine.stream_status() if s['state'] == 'failed'}

    def dead_nodes(self) -> list[NodeId]:
        return [loopback_node(key[len('loopback::'):]) if key.startswith('loopback::') else NodeId('device', key)
                for key in self.failed_streams()]

    def _meter_nodes(self) -> dict[str, int]:
        nodes = {f"{key}::{'in' if role == INPUT else 'out'}": native
                 for (key, role), native in self._device_sides.items()}
        for key, native in self._loopback_sides.items():
            nodes[f"loopback::{key}"] = native
        for name in self._buses:
            nodes[f"bus::{name}"] = self._compiler.native_id(bus_node(name), BUS)
        return nodes

    def statistics(self) -> HostStatistics:
        stats = HostStatistics(samplerate=self.samplerate, blocksize=self.blocksize,
                               nominal_latency_ms=self.blocksize / self.samplerate * 1000, backend='native')
        stats.host_api = WASAPI_NAME if self.host_api == HostApi.WASAPI else ASIO_NAME
        if not self._running:
            return stats
        stats.running = True
        s = self._engine.stats()
        streams = self._engine.stream_status()
        devices = [st for st in streams if st['kind'] != 5]
        stats.callback_count = s['blocks']
        stats.xruns = s['xruns']
        stats.audio_thread_allocations = s['rt_allocations']
        if s['blocks']:
            stats.callback_min_ms = s['callback_ns_min'] / 1e6
            stats.callback_mean_ms = s['callback_ns_mean'] / 1e6
            stats.callback_max_ms = s['callback_ns_max'] / 1e6
            stats.callback_p99_ms = s['callback_ns_p99'] / 1e6
            stats.processing_load = s['processing_load']
            stats.cpu_load = s['mean_load'] * 100 if s['mean_load'] is not None else None
        stats.stream_count = len(devices)
        stats.live_stream_count = sum(1 for st in devices if st['state'] == 'running')
        stats.exclusive = any(st['exclusive'] for st in devices)
        stats.failed_streams = self.failed_streams()
        ins = [st['reported_latency_ms'] for st in devices if st['kind'] != 1 and st['reported_latency_ms']]
        outs = [st['reported_latency_ms'] for st in devices if st['kind'] == 1 and st['reported_latency_ms']]
        stats.input_latency_ms = max(ins) if ins else None
        stats.output_latency_ms = max(outs) if outs else None
        if ins or outs:
            plugins_ms = self.plugin_latency_samples() / self.samplerate * 1000
            stats.reported_latency_ms = (stats.input_latency_ms or 0.0) + (stats.output_latency_ms or 0.0) + plugins_ms
        return stats

    def route_statistics(self, source: str, dest: str) -> dict | None:
        try:
            node = self._compiler.native_id(_parse(dest), OUTPUT)
            return {'available': self._engine.port_available(node)}
        except (NativeError, ValueError):
            return None

    def ring_statistics(self) -> dict[str, dict]:
        s = self._engine.stats()
        return {'native': {'overruns': s['ring_overruns'], 'underruns': s['ring_underruns']}}

    @property
    def last_callback_error(self) -> str | None:
        for event in self._engine.events():
            self._last_error = f"native event {event['code']} on node {event['arg0']}"
        return self._last_error

    def stream_status(self) -> list[dict]:
        return self._engine.stream_status() if self._running else []


def _parse(node: str) -> NodeId:
    kind, _, ref = node.partition(':')
    if not ref:
        raise ValueError(f"not a node: {node}")
    return NodeId(kind, ref)


def make_host(samplerate: int, blocksize: int, host_api: HostApi | None, exclusive: bool,
              backend: str | None = None):
    """
    The native host on Windows when its DLL is present; otherwise the PortAudio host. The
    choice is reported through `HostStatistics.backend`, never hidden, and a Windows build
    that lacks its native engine says so in the log rather than quietly running on the
    Python callback path. `backend='portaudio'` (or TONESPHERE_HOST=portaudio) asks for the
    PortAudio host explicitly.
    """
    import os
    import sys

    from tonesphere.engine.host import AudioHost

    backend = backend or os.environ.get('TONESPHERE_HOST')
    if sys.platform == 'win32' and backend != 'portaudio':
        try:
            wanted = host_api if host_api in (HostApi.WASAPI, HostApi.ASIO) else None
            return NativeHost(samplerate, blocksize, wanted, exclusive)
        except NativeUnavailable as e:
            logger.warning(f"Native engine unavailable, using the PortAudio host (not real-time safe): {e}")
    return AudioHost(samplerate=samplerate, blocksize=blocksize, host_api=host_api, exclusive=exclusive)
