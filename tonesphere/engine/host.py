"""
The audio host: real PortAudio streams, and the callback that actually moves samples.

This replaces eight "driver" classes that each allocated a NumPy array of zeros and
called it a stream. There is one backend, PortAudio, because PortAudio is already the
abstraction over ASIO/WASAPI/WDM-KS/CoreAudio/ALSA/JACK and reimplementing that in
Python was the original mistake.

Topology
--------
One stream per device. A device used in both directions gets a single duplex stream,
which matters more than it sounds: a duplex stream has one clock, so input and output
cannot drift, and the block we just captured can be mixed into the block we are about to
play with no buffering in between. That is the guitar path — interface in, headphones
out — and it is the lowest latency configuration available.

Cross-device routes cannot share a clock, so they pass through a ring buffer and are
subject to drift correction.

Realtime discipline
-------------------
Inside a callback we do not allocate, lock, log, raise, or touch anything that might
take the GIL for an unbounded time. Every buffer is preallocated in `_rebuild_scratch`.
Errors are counted, not raised — a traceback out of a callback silently kills the stream
in PortAudio and the user just hears silence with no explanation.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from tonesphere.engine.devices import (
    AudioBackendUnavailable, DeviceInfo, HostApi, enumerate_devices, find_device,
)
from tonesphere.engine.dsp import ChannelStrip, DriftResampler, Limiter, Panner
from tonesphere.engine.effects import InsertChain
from tonesphere.engine.graph import (
    Connection, GraphHolder, NodeId, RoutingGraph, bus_node, device_node,
)
from tonesphere.engine.meters import MeterRegistry
from tonesphere.engine.ringbuffer import AudioRingBuffer
from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)

# Ring depth as a multiple of the block size. Four blocks absorbs normal scheduling
# jitter between two device clocks while adding only a few milliseconds.
RING_BLOCKS = 4

# Drift correction, in blocks of buffered audio.
#
# The ring is steered towards TARGET. Above HIGH we give up on resampling and discard,
# which only happens on a gross desync such as a device stalling and resuming.
DRIFT_TARGET_BLOCKS = 1.5
DRIFT_HIGH_BLOCKS = 3.0

# How aggressively fill error is turned into a resampling ratio. Low on purpose: the
# correction must stay well inside the resampler's clamp so it can never become audible
# pitch movement, and drift is measured in parts per million, not percent.
DRIFT_CORRECTION_STRENGTH = 0.0005

# Gain ramp length as a fraction of a block. A gain step applied instantly produces a
# click; ramping across the block makes it inaudible.
GAIN_RAMP_FRACTION = 1.0


class _RouteTable:
    """
    One ring buffer per route, published to the callbacks by reference swap.

    Why per route and not per source: `read_into` consumes. If a single ring were shared
    by two destinations, whichever output callback ran first would take the audio and the
    second would get silence — fan-out would be broken, intermittently and differently on
    every run. Giving each route its own ring means the producer writes its block once per
    consumer and every consumer reads a complete copy.

    The table is immutable, like the graph, and rebuilt on a routing change. Existing
    rings are carried across so unchanged routes are not interrupted. Swapping one
    reference means a callback either sees the old table or the new one, never a dict
    mid-resize.
    """

    __slots__ = ('rings', 'by_source', 'panners', 'resamplers')

    def __init__(
        self,
        rings: Dict[Tuple[str, str], AudioRingBuffer],
        panners: Optional[Dict[Tuple[str, str], Panner]] = None,
        resamplers: Optional[Dict[Tuple[str, str], DriftResampler]] = None,
    ):
        self.rings = rings

        # Per-route processing state lives here alongside the ring, rather than on the
        # destination stream. Pan and drift are properties of a route, and keying them by
        # route means any renderer finds them — including a bus destination, which has no
        # device stream to hang them off.
        self.panners = panners if panners is not None else {}
        self.resamplers = resamplers if resamplers is not None else {}

        # Producer side: every ring a given source must fan its block out to.
        by_source: Dict[str, List[AudioRingBuffer]] = {}
        for (source_key, _dest_key), ring in rings.items():
            by_source.setdefault(source_key, []).append(ring)

        self.by_source: Dict[str, Tuple[AudioRingBuffer, ...]] = {
            key: tuple(items) for key, items in by_source.items()
        }

    def ring_for(self, source_key: str, dest_key: str) -> Optional[AudioRingBuffer]:
        return self.rings.get((source_key, dest_key))

    def sinks_for(self, source_key: str) -> Tuple[AudioRingBuffer, ...]:
        return self.by_source.get(source_key, ())

    def panner_for(self, key: Tuple[str, str]) -> Optional[Panner]:
        return self.panners.get(key)

    def resampler_for(self, key: Tuple[str, str]) -> Optional[DriftResampler]:
        return self.resamplers.get(key)

    def statistics(self) -> Dict[str, dict]:
        return {
            f"{source} -> {dest}": ring.statistics()
            for (source, dest), ring in self.rings.items()
        }


@dataclass
class StreamConfig:
    """How to open one device."""
    device: DeviceInfo
    samplerate: int
    blocksize: int
    input_channels: int = 0
    output_channels: int = 0
    exclusive: bool = False

    @property
    def is_duplex(self) -> bool:
        return self.input_channels > 0 and self.output_channels > 0


@dataclass
class HostStatistics:
    """
    What the engine actually measured. Unmeasured values stay None.

    `measured_latency_ms` comes from PortAudio's reported stream latency, which accounts
    for the driver's own buffering — unlike blocksize/samplerate arithmetic, which only
    describes our contribution and reads far lower than the truth.
    """
    running: bool = False
    samplerate: Optional[int] = None
    blocksize: Optional[int] = None
    host_api: Optional[str] = None
    exclusive: bool = False

    input_latency_ms: Optional[float] = None
    output_latency_ms: Optional[float] = None
    measured_latency_ms: Optional[float] = None
    nominal_latency_ms: Optional[float] = None

    cpu_load: Optional[float] = None
    xruns: int = 0
    callback_count: int = 0
    callback_errors: int = 0
    drift_corrections: int = 0
    stream_count: int = 0
    live_stream_count: int = 0
    failed_streams: Dict[str, str] = field(default_factory=dict)

    @property
    def fully_healthy(self) -> bool:
        """Running with every configured stream open. Partial success is not health."""
        return self.running and not self.failed_streams

    def as_dict(self) -> dict:
        return {
            'running': self.running,
            'samplerate': self.samplerate,
            'blocksize': self.blocksize,
            'host_api': self.host_api,
            'exclusive': self.exclusive,
            'input_latency_ms': self.input_latency_ms,
            'output_latency_ms': self.output_latency_ms,
            'measured_latency_ms': self.measured_latency_ms,
            'nominal_latency_ms': self.nominal_latency_ms,
            'cpu_usage': self.cpu_load,
            'xruns': self.xruns,
            'callback_count': self.callback_count,
            'callback_errors': self.callback_errors,
            'drift_corrections': self.drift_corrections,
            'stream_count': self.stream_count,
            'live_stream_count': self.live_stream_count,
            'failed_streams': dict(self.failed_streams),
            'fully_healthy': self.fully_healthy,
            'audio_path_active': self.running,
        }


class _DeviceStream:
    """
    One PortAudio stream plus the preallocated buffers its callback needs.

    Everything the callback touches is created here, on the control thread, before the
    stream starts.
    """

    __slots__ = (
        'config', 'node', 'stream', 'output_meter_key', 'input_meter_key',
        '_mix', '_source_scratch', '_channel_scratch', '_gain_ramp', '_prev_state',
        '_limiter', '_resample_in',
        'input_strip', 'output_strip', '_capture_scratch',
        'input_inserts', 'output_inserts',
        'xruns', 'callback_count', 'callback_errors', 'drift_corrections',
        'error', 'used_exclusive',
    )

    def __init__(self, config: StreamConfig):
        self.config = config
        self.node = device_node(config.device.key)
        self.stream = None

        # Why this stream is not carrying audio, if it is not. None means healthy.
        self.error: Optional[str] = None

        # Which mode actually opened, which is not always the one requested.
        self.used_exclusive = False

        self.input_meter_key = f"{config.device.key}::in"
        self.output_meter_key = f"{config.device.key}::out"

        block = config.blocksize

        # The output accumulator. Summing into a preallocated buffer avoids one
        # allocation per block per stream.
        self._mix = np.zeros((block, max(config.output_channels, 1)), dtype=np.float32)

        # Staging for one source's contribution before channel mapping.
        self._source_scratch = np.zeros((block, 8), dtype=np.float32)
        self._channel_scratch = np.zeros((block, max(config.output_channels, 1)), dtype=np.float32)

        # A 0..1 ramp used to interpolate gain changes across a block.
        self._gain_ramp = np.linspace(0.0, 1.0, block, dtype=np.float32).reshape(-1, 1)

        # Last block's gain per route, plus the source it came from. The source is kept
        # so a route that has just been muted or unrouted can still be rendered for one
        # final block, ramping down to zero instead of cutting off with a click.
        self._prev_state: Dict[Tuple[str, str], Tuple[NodeId, float]] = {}

        # Catches summing overs before they reach the device's integer converter, where
        # they would clip hard rather than gracefully.
        self._limiter: Optional[Limiter] = None
        if config.output_channels:
            self._limiter = Limiter(config.samplerate, block)

        # Oversized so a ratio above 1.0, which needs more input than output, still fits.
        self._resample_in = np.zeros((block * 2, 8), dtype=np.float32)

        # Per-device trim, mute, pan and polarity. Fed from `core.channel_control`, which
        # held exactly this state and was never connected to any audio.
        self.input_strip: Optional[ChannelStrip] = None
        self.output_strip: Optional[ChannelStrip] = None
        self._capture_scratch: Optional[np.ndarray] = None

        if config.input_channels:
            self.input_strip = ChannelStrip(config.input_channels, block)
            # PortAudio's input buffer must not be written to, so processing happens in
            # our own memory.
            self._capture_scratch = np.zeros((block, config.input_channels), dtype=np.float32)

        if config.output_channels:
            self.output_strip = ChannelStrip(config.output_channels, block)

        # Insert chains: high-pass, EQ, dynamics, VST3 plugins, delay. Created empty and
        # skipped entirely until something is enabled, so an untouched channel costs
        # nothing beyond a boolean check.
        self.input_inserts: Optional[InsertChain] = None
        self.output_inserts: Optional[InsertChain] = None

        if config.input_channels:
            self.input_inserts = InsertChain(
                config.samplerate, block, config.input_channels
            )
        if config.output_channels:
            self.output_inserts = InsertChain(
                config.samplerate, block, config.output_channels
            )

        self.xruns = 0
        self.callback_count = 0
        self.callback_errors = 0
        self.drift_corrections = 0

    @property
    def is_live(self) -> bool:
        """Whether this stream is genuinely open and carrying audio."""
        return self.stream is not None and self.error is None

    def source_buffer(self, channels: int) -> np.ndarray:
        """A view of the staging buffer wide enough for `channels`. Never allocates."""
        if channels <= self._source_scratch.shape[1]:
            return self._source_scratch[:, :channels]
        # Wider than expected (a >8 channel interface). Grow once, then reuse.
        self._source_scratch = np.zeros(
            (self.config.blocksize, channels), dtype=np.float32
        )
        return self._source_scratch


class AudioHost:
    """
    Owns the streams and runs the mix.

    Lifecycle: `configure()` decides the topology from the graph, `start()` opens and
    starts streams, `apply_graph()` publishes routing changes without restarting
    anything. Only `configure`/`start`/`stop` touch PortAudio.
    """

    def __init__(
        self,
        samplerate: int = 48000,
        blocksize: int = 256,
        host_api: Optional[HostApi] = None,
        exclusive: bool = False,
    ):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.host_api = host_api
        self.exclusive = exclusive

        self.graph_holder = GraphHolder()
        self.meters = MeterRegistry()

        self._streams: Dict[str, _DeviceStream] = {}
        self._devices: List[DeviceInfo] = []

        # Buses are declared channel counts; their audio lives in the route table like
        # everything else, so a bus feeding two destinations fans out correctly.
        self._bus_channels: Dict[str, int] = {}
        self._bus_pending: Dict[str, AudioRingBuffer] = {}

        # Swapped by reference on a routing change. Read once per callback.
        self._routes = _RouteTable({})

        self._running = False
        self._lock = threading.RLock()   # guards configure/start/stop only, never a callback
        self._sd = None

        # Set when a callback hits something it cannot handle, so the control thread can
        # report it. The callback only ever writes; it never logs.
        self._last_callback_error: Optional[str] = None

    # --- Backend ---

    def _sounddevice(self):
        if self._sd is None:
            try:
                import sounddevice as sd
            except (ImportError, OSError) as e:
                raise AudioBackendUnavailable(f"PortAudio unavailable: {e}") from e
            self._sd = sd
        return self._sd

    def refresh_devices(self) -> List[DeviceInfo]:
        with self._lock:
            self._devices = enumerate_devices()
            return list(self._devices)

    @property
    def devices(self) -> List[DeviceInfo]:
        if not self._devices:
            self.refresh_devices()
        return list(self._devices)

    @property
    def is_running(self) -> bool:
        return self._running

    # --- Buses ---

    def create_bus(self, name: str, channels: int = 2) -> str:
        """
        Create an in-process mix bus.

        Not an operating-system device: other applications cannot select it. It is a
        summing point inside the graph.
        """
        with self._lock:
            if name not in self._bus_channels:
                self._bus_channels[name] = channels
                self.meters.ensure(f"bus::{name}", channels)
                self._rebuild_routes()
            return name

    def remove_bus(self, name: str):
        with self._lock:
            self._bus_channels.pop(name, None)
            self._bus_pending.pop(name, None)
            self.meters.remove(f"bus::{name}")
            self._rebuild_routes()

    def bus_channels(self, name: str) -> Optional[int]:
        return self._bus_channels.get(name)

    # --- Route table ---

    def _channels_of(self, node: NodeId) -> Optional[int]:
        """How wide a node's audio is, for sizing its rings."""
        if node.kind == 'bus':
            return self._bus_channels.get(node.ref)

        stream = self._streams.get(node.ref)
        if stream is not None and stream.config.input_channels:
            return stream.config.input_channels

        device = find_device(node.ref, self._devices) if self._devices else None
        if device is not None and device.can_input:
            return min(device.max_input_channels, 8)
        return None

    def _rebuild_routes(self):
        """
        Build the ring for every route in the current graph and publish the table.

        Rings for unchanged routes are carried over so a routing edit elsewhere in the
        graph does not interrupt audio already flowing.
        """
        graph = self.graph_holder.current()
        existing = self._routes

        rings: Dict[Tuple[str, str], AudioRingBuffer] = {}
        panners: Dict[Tuple[str, str], Panner] = {}
        resamplers: Dict[Tuple[str, str], DriftResampler] = {}

        for connection in graph.connections:
            source_key = str(connection.source)
            dest_key = str(connection.dest)
            channels = self._channels_of(connection.source)
            if channels is None:
                continue

            key = (source_key, dest_key)

            # Carry unchanged routes across so an edit elsewhere in the graph does not
            # interrupt audio already flowing, or reset a smoothed pan mid-move.
            previous = existing.rings.get(key)
            if previous is not None and previous.channels == channels:
                rings[key] = previous
            else:
                rings[key] = AudioRingBuffer(
                    capacity_frames=self.blocksize * RING_BLOCKS, channels=channels
                )

            panner = existing.panners.get(key) or Panner(self.blocksize)
            panner.set_pan(connection.pan)
            panners[key] = panner

            # Drift correction applies only between two *different* hardware devices,
            # which are the only things with independent clocks that can disagree.
            #
            # Not the same device in both directions: a duplex stream has one clock, so
            # there is nothing to correct. Not a bus at either end either: a bus has no
            # clock of its own, it is paced by whoever writes it. Resampling in those
            # cases adds interpolation error and a fractional offset to a signal that was
            # already perfectly aligned.
            crosses_clocks = (
                connection.source.kind == 'device'
                and connection.dest.kind == 'device'
                and connection.source.ref != connection.dest.ref
            )
            if crosses_clocks:
                resamplers[key] = (
                    existing.resamplers.get(key) or DriftResampler(channels, self.blocksize)
                )

        # Single reference swap: a callback sees either the old table or the new one,
        # never a dict mid-rebuild. Everything above happens on the control thread, so
        # the callback never allocates.
        self._routes = _RouteTable(rings, panners, resamplers)

    # --- Configuration ---

    def configure(self, graph: Optional[RoutingGraph] = None) -> List[str]:
        """
        Decide which streams the graph needs and prepare them.

        Returns any problems found, so the caller can surface them instead of us
        half-configuring and failing later inside a callback. Must be called stopped.
        """
        if self._running:
            raise RuntimeError("configure() while running; call stop() first")

        with self._lock:
            if graph is not None:
                self.graph_holder.commit(graph)
            graph = self.graph_holder.current()

            if not self._devices:
                self.refresh_devices()

            problems: List[str] = []
            needed: Dict[str, StreamConfig] = {}

            for node in graph.nodes():
                if node.kind != 'device':
                    continue

                device = find_device(node.ref, self._devices)
                if device is None:
                    problems.append(f"Device not found: {node.ref}")
                    continue

                # Does the graph read from it, write to it, or both?
                is_source = any(c.source == node for c in graph.connections)
                is_dest = any(c.dest == node for c in graph.connections)

                in_channels = min(device.max_input_channels, 8) if is_source else 0
                out_channels = min(device.max_output_channels, 8) if is_dest else 0

                if is_source and not device.can_input:
                    problems.append(f"{device.name} has no inputs but is routed from")
                    continue
                if is_dest and not device.can_output:
                    problems.append(f"{device.name} has no outputs but is routed to")
                    continue
                if in_channels == 0 and out_channels == 0:
                    continue

                needed[device.key] = StreamConfig(
                    device=device,
                    samplerate=self.samplerate,
                    blocksize=self.blocksize,
                    input_channels=in_channels,
                    output_channels=out_channels,
                    exclusive=self.exclusive and device.supports_exclusive,
                )

            # Verify each configuration with PortAudio before committing to it, so a bad
            # rate or channel count surfaces here rather than as silence later.
            validated: Dict[str, StreamConfig] = {}
            for key, config in needed.items():
                error = self._validate(config)
                if error:
                    problems.append(error)
                else:
                    validated[key] = config

            self._streams = {key: _DeviceStream(config) for key, config in validated.items()}

            for stream in self._streams.values():
                if stream.config.input_channels:
                    self.meters.ensure(stream.input_meter_key, stream.config.input_channels)
                if stream.config.output_channels:
                    self.meters.ensure(stream.output_meter_key, stream.config.output_channels)

            for bus_name, channels in self._bus_channels.items():
                self.meters.ensure(f"bus::{bus_name}", channels)

            self._rebuild_routes()

            if validated:
                duplex = sum(1 for c in validated.values() if c.is_duplex)
                logger.info(
                    f"Configured {len(validated)} stream(s), {duplex} duplex, "
                    f"{self.samplerate} Hz, {self.blocksize} frames"
                )

            return problems

    def _validate(self, config: StreamConfig) -> Optional[str]:
        """Ask PortAudio whether this configuration is actually openable."""
        sd = self._sounddevice()
        device = config.device

        try:
            if config.input_channels:
                sd.check_input_settings(
                    device=device.index,
                    channels=config.input_channels,
                    samplerate=config.samplerate,
                    extra_settings=self._extra_settings(config, is_input=True),
                )
            if config.output_channels:
                sd.check_output_settings(
                    device=device.index,
                    channels=config.output_channels,
                    samplerate=config.samplerate,
                    extra_settings=self._extra_settings(config, is_input=False),
                )
        except Exception as e:
            return f"{device.name} [{device.host_api_name}]: {e}"

        return None

    def _extra_settings(self, config: StreamConfig, is_input: bool):
        """
        Host-API-specific options.

        Exclusive mode is the whole reason WASAPI is competitive with ASIO: it hands the
        device to us and bypasses the system mixer and its resampling.
        """
        if not config.exclusive:
            return None

        sd = self._sounddevice()

        if config.device.host_api == HostApi.WASAPI:
            try:
                return sd.WasapiSettings(exclusive=True)
            except (AttributeError, TypeError):
                return None

        return None

    # --- Running ---

    def start(self) -> List[str]:
        """
        Open and start every configured stream.

        Returns one problem per stream that could not be opened. Partial success is real
        and must be reported: a capture device commonly refuses exclusive mode while the
        playback device accepts it, and the route between them then carries nothing. This
        used to log "Audio host running: 1 stream(s)" and report a healthy 8.3 ms latency
        while the input side had failed to open at all.
        """
        with self._lock:
            if self._running:
                return []

            if not self._streams:
                return ["No streams configured — route something to a device first"]

            sd = self._sounddevice()
            problems: List[str] = []
            live = 0

            for stream in self._streams.values():
                stream.error = None
                try:
                    self._open(sd, stream)
                    stream.stream.start()
                    live += 1
                except Exception as e:
                    stream.error = str(e)
                    stream.stream = None
                    problems.append(f"{stream.config.device.name}: {e}")
                    logger.error(f"Failed to start {stream.config.device.name}: {e}")

            if live == 0:
                return problems or ["No streams could be started"]

            self._running = True

            exclusive = sum(1 for s in self._streams.values() if s.is_live and s.used_exclusive)
            failed = len(self._streams) - live
            summary = f"Audio host running: {live} stream(s), {exclusive} exclusive"
            if failed:
                summary += f", {failed} FAILED"
            logger.info(summary)

            return problems

    def _open(self, sd, stream: _DeviceStream):
        """
        Open one stream, falling back from exclusive to shared mode if necessary.

        Exclusive mode is worth the try — 8.3 ms versus 22 ms measured on the same device —
        but many capture devices refuse it, and a shared stream that works beats an
        exclusive one that does not. Which mode won is recorded so the UI can say.
        """
        config = stream.config

        attempts = [config]
        if config.exclusive:
            from dataclasses import replace as _replace
            attempts.append(_replace(config, exclusive=False))

        last_error: Optional[Exception] = None

        for attempt in attempts:
            try:
                stream.stream = self._open_with(sd, stream, attempt)
                stream.used_exclusive = attempt.exclusive

                if config.exclusive and not attempt.exclusive:
                    logger.info(
                        f"{config.device.name}: exclusive mode refused, using shared "
                        f"(higher latency)"
                    )
                return
            except Exception as e:
                last_error = e
                stream.stream = None

        raise RuntimeError(self._explain_open_failure(config, last_error))

    def _explain_open_failure(self, config: StreamConfig, error: Optional[Exception]) -> str:
        """
        Turn a PortAudio error into something a user can act on.

        "Invalid device [PaErrorCode -9996]" on a device that enumerated fine almost always
        means the OS is refusing access rather than that anything is wrong with the
        request, and the usual cause on Windows is the microphone privacy setting. Saying
        so saves an hour of confusion.
        """
        text = str(error) if error else "unknown error"

        if '-9996' in text or 'Invalid device' in text:
            direction = "input" if config.input_channels else "output"
            hint = (f"{text} — the OS refused access to this {direction}. "
                    f"On Windows check Settings > Privacy & security > Microphone, "
                    f"and that the device is enabled in Sound settings.")
            return hint

        if '-9997' in text or 'Invalid sample rate' in text:
            return (f"{text} — {config.samplerate} Hz is not supported by this device. "
                    f"Try 44100 or the device's own default rate.")

        if '-9998' in text or 'Invalid number of channels' in text:
            requested = config.input_channels or config.output_channels
            return f"{text} — this device does not offer {requested} channels."

        if 'Device unavailable' in text or '-9985' in text:
            return (f"{text} — another application is using this device exclusively. "
                    f"Close it, or turn off exclusive mode.")

        return text

    def _open_with(self, sd, stream: _DeviceStream, config: StreamConfig):
        device = config.device

        common = dict(
            samplerate=config.samplerate,
            blocksize=config.blocksize,
            dtype='float32',
            latency='low',
            clip_off=True,       # we manage headroom; PortAudio clipping would hide it
            dither_off=True,     # dither belongs at the final integer conversion only
        )

        if config.is_duplex:
            settings = self._extra_settings(config, is_input=True)
            return sd.Stream(
                device=(device.index, device.index),
                channels=(config.input_channels, config.output_channels),
                callback=self._make_duplex_callback(stream),
                extra_settings=(settings, settings) if settings is not None else None,
                **common,
            )

        if config.input_channels:
            return sd.InputStream(
                device=device.index,
                channels=config.input_channels,
                callback=self._make_input_callback(stream),
                extra_settings=self._extra_settings(config, is_input=True),
                **common,
            )

        return sd.OutputStream(
            device=device.index,
            channels=config.output_channels,
            callback=self._make_output_callback(stream),
            extra_settings=self._extra_settings(config, is_input=False),
            **common,
        )

    def strips_for(self, device_key: str) -> Tuple[Optional[ChannelStrip], Optional[ChannelStrip]]:
        """
        The (input, output) strips for a device, or (None, None) if it has no stream.

        Callers push control-side state in through these. Both may be None while the host
        is stopped, which is normal — settings are stored on the control side and applied
        when streams are next opened.
        """
        stream = self._streams.get(device_key)
        if stream is None:
            return (None, None)
        return (stream.input_strip, stream.output_strip)

    def inserts_for(self, device_key: str, is_input: bool) -> Optional[InsertChain]:
        """The insert chain on one side of a device, or None if it has no stream."""
        stream = self._streams.get(device_key)
        if stream is None:
            return None
        return stream.input_inserts if is_input else stream.output_inserts

    def total_plugin_latency_ms(self) -> float:
        """
        Latency the loaded plugins add, on top of the driver's.

        Must be included in what we report: a look-ahead limiter adds several milliseconds
        by itself, and omitting it would make the latency figure wrong in exactly the
        direction that flatters us.
        """
        samples = 0
        for stream in self._streams.values():
            for chain in (stream.input_inserts, stream.output_inserts):
                if chain is not None:
                    samples += chain.latency_samples
        return samples / self.samplerate * 1000.0 if samples else 0.0

    def failed_streams(self) -> Dict[str, str]:
        """Device key -> why its stream is not carrying audio."""
        return {
            key: stream.error
            for key, stream in self._streams.items()
            if stream.error is not None
        }

    def dead_nodes(self) -> List[NodeId]:
        """
        Nodes whose stream failed, so callers can mark the routes that touch them.

        A route to a dead node is drawn as broken rather than active, because the audio is
        not arriving and the user needs to know that.
        """
        return [stream.node for stream in self._streams.values() if stream.error is not None]

    def stop(self):
        with self._lock:
            if not self._running:
                return

            self._running = False

            for stream in self._streams.values():
                self._close(stream)

            for ring in self._routes.rings.values():
                ring.clear()

            logger.info("Audio host stopped")

    def _close(self, stream: _DeviceStream):
        if stream.stream is None:
            return
        try:
            stream.stream.stop()
            stream.stream.close()
        except Exception as e:
            logger.warning(f"Error closing {stream.config.device.name}: {e}")
        finally:
            stream.stream = None
            stream._prev_state.clear()

    # --- Graph updates ---

    def apply_graph(self, graph: RoutingGraph) -> List[str]:
        """
        Publish a routing change.

        Gain, mute and solo changes take effect on the next block with no interruption,
        because the callback simply reads the new snapshot. A new route needs a ring, which
        is allocated here on the control thread and published by swapping the table — the
        callback never allocates. Only a route to a device with no open stream needs a
        restart, and we say so rather than silently dropping it.
        """
        current_devices = set(self._streams.keys())
        wanted_devices = {node.ref for node in graph.nodes() if node.kind == 'device'}

        self.graph_holder.commit(graph)
        self._rebuild_routes()

        missing = wanted_devices - current_devices
        if missing and self._running:
            return [f"Restart required to add: {', '.join(sorted(missing))}"]
        return []

    # --- Callbacks. Realtime context below this line. ---

    def _make_duplex_callback(self, stream: _DeviceStream) -> Callable:
        """
        Duplex: capture, then mix and play, in one callback.

        The capture is published to the ring *before* the mix runs, so a route from this
        device back out of this device sees the block we just captured. That is what
        makes in-to-out on one interface add no buffering beyond the block itself.
        """
        source_key = str(stream.node)

        def callback(indata, outdata, frames, time_info, status):
            try:
                stream.callback_count += 1
                if status:
                    stream.xruns += 1

                self._publish_capture(source_key, indata, stream)
                self._mix_into(stream, outdata, frames)

            except Exception as e:
                # Never propagate: PortAudio aborts the stream on an exception and the
                # user simply loses audio with no message.
                stream.callback_errors += 1
                self._last_callback_error = repr(e)
                outdata.fill(0.0)

        return callback

    def _make_input_callback(self, stream: _DeviceStream) -> Callable:
        source_key = str(stream.node)

        def callback(indata, frames, time_info, status):
            try:
                stream.callback_count += 1
                if status:
                    stream.xruns += 1

                self._publish_capture(source_key, indata, stream)
            except Exception as e:
                stream.callback_errors += 1
                self._last_callback_error = repr(e)

        return callback

    def _publish_capture(self, source_key: str, indata: np.ndarray, stream: _DeviceStream):
        """
        Apply the input strip, fan the block out to every route leaving this device, meter.

        Writing once per consumer is what makes fan-out correct: each destination reads a
        complete copy instead of racing the others for a shared one.

        Trim, mute and polarity are applied once here rather than per route, because they
        describe the device rather than any particular destination — and doing it once is
        also cheaper.
        """
        block = indata
        strip = stream.input_strip
        inserts = stream.input_inserts

        needs_work = (
            (strip is not None and not strip.is_transparent())
            or (inserts is not None and not inserts.is_transparent)
        )

        if needs_work:
            frames = indata.shape[0]
            scratch = stream._capture_scratch[:frames]
            scratch[:] = indata

            if strip is not None:
                strip.process(scratch, frames)
            if inserts is not None:
                # Channel inserts run before the split, so every destination receives the
                # same processed signal — which is what "insert on the channel" means.
                inserts.process(scratch, frames)

            block = scratch

        for ring in self._routes.sinks_for(source_key):
            ring.write(block)

        # Meter post-trim, so the meter shows what is actually being sent onward.
        bank = self.meters.get(stream.input_meter_key)
        if bank is not None:
            bank.measure(block)

    def _make_output_callback(self, stream: _DeviceStream) -> Callable:
        def callback(outdata, frames, time_info, status):
            try:
                stream.callback_count += 1
                if status:
                    stream.xruns += 1

                self._mix_into(stream, outdata, frames)
            except Exception as e:
                stream.callback_errors += 1
                self._last_callback_error = repr(e)
                outdata.fill(0.0)

        return callback

    def _mix_into(self, stream: _DeviceStream, outdata: np.ndarray, frames: int):
        """
        Sum every route feeding this device into `outdata`.

        Realtime: no allocation, no locks, no logging. The graph and route table are each
        read once, so a concurrent edit cannot change the routing halfway through a block.

        Note that inaudible routes are still rendered. A route that has just been muted or
        deleted gets one final block ramping to zero, because dropping it outright puts a
        step discontinuity in the waveform, and a step discontinuity is a click. Draining
        its ring also keeps drift correction from seeing a phantom backlog.
        """
        graph = self.graph_holder.current()
        routes = self._routes
        dest_key = str(stream.node)

        mix = stream._mix[:frames, :outdata.shape[1]]
        mix.fill(0.0)

        master = graph.master_gain
        out_channels = outdata.shape[1]

        # Everything the graph currently wants, audible or not: an explicitly muted route
        # needs rendering so it can ramp down.
        active_keys = set()

        for connection in graph.connections:
            if connection.dest != stream.node:
                continue

            source_key = str(connection.source)
            key = (source_key, dest_key)
            active_keys.add(key)

            ring = routes.ring_for(source_key, dest_key)
            if ring is None:
                continue

            gain = connection.effective_gain * master

            # Solo elsewhere silences this route; resolved via the graph's audible set so
            # the rule lives in one place.
            if graph.soloed and connection.source not in graph.soloed:
                gain = 0.0

            self._render_route(stream, ring, key, connection.source, gain,
                               mix, out_channels, frames, connection)

        # Routes the graph no longer has, but which were audible last block. Render one
        # more block ramping to silence, then forget them.
        if stream._prev_state:
            for key in list(stream._prev_state.keys()):
                if key in active_keys:
                    continue

                source_node, previous = stream._prev_state[key]
                if previous <= 0.0:
                    del stream._prev_state[key]
                    continue

                ring = routes.ring_for(key[0], key[1])
                if ring is None:
                    del stream._prev_state[key]
                    continue

                self._render_route(stream, ring, key, source_node, 0.0,
                                   mix, out_channels, frames, None)

        # Bus inserts, then output trim, then the limiter. The limiter is unconditionally
        # last so it protects the converter from whatever anything before it did.
        if stream.output_inserts is not None and not stream.output_inserts.is_transparent:
            stream.output_inserts.process(mix, frames)

        if stream.output_strip is not None and not stream.output_strip.is_transparent():
            stream.output_strip.process(mix, frames)

        if stream._limiter is not None:
            stream._limiter.process(mix, frames)

        outdata[:] = mix

        bank = self.meters.get(stream.output_meter_key)
        if bank is not None:
            bank.measure(outdata)

    def _render_route(
        self,
        stream: _DeviceStream,
        ring: AudioRingBuffer,
        key: Tuple[str, str],
        source_node: NodeId,
        gain: float,
        mix: np.ndarray,
        out_channels: int,
        frames: int,
        connection: Optional[Connection],
    ):
        """
        Add one route's contribution to the mix.

        Always consumes from the ring, even at zero gain, so a silent route cannot
        accumulate a backlog that later shows up as spurious drift.
        """
        self._drift_correct(ring, stream, key)
        source = self._read_source(stream, ring, key, frames)

        state = stream._prev_state.get(key)
        previous = state[1] if state is not None else gain

        if previous == 0.0 and gain == 0.0:
            stream._prev_state[key] = (source_node, 0.0)
            return

        if connection is not None and connection.invert:
            # Polarity flip. Applied before summing so it can cancel a correlated source,
            # which is the whole point of having it.
            source = source * -1.0

        mapped = self._map_channels(stream, source, out_channels, frames, connection, key)

        if previous == gain:
            if gain == 1.0:
                mix += mapped
            else:
                mix += mapped * gain
        else:
            # Linear ramp across the block: inaudible, and cheap enough to do on every
            # change rather than trying to work out which ones matter.
            ramp = stream._gain_ramp[:frames]
            mix += mapped * (previous + (gain - previous) * ramp)

        stream._prev_state[key] = (source_node, gain)

    def _drift_correct(
        self, ring: AudioRingBuffer, stream: _DeviceStream,
        key: Optional[Tuple[str, str]] = None,
    ):
        """
        Keep a cross-device ring from drifting away from its target fill level.

        Two devices at a nominal 48 kHz do not agree exactly, so a cross-device route's
        buffer creeps in one direction: latency grows until it overflows, or shrinks until
        it starves. Dropping or repeating a block fixes the level but costs an audible
        glitch every time, and at typical drift rates that is every few seconds.

        Instead the ring's fill level steers a resampling ratio. The correction is a
        fraction of a percent and clamped by `DriftResampler.MAX_RATIO_DEVIATION`, which is
        far below the threshold of audible pitch change. Discarding only remains as a
        backstop for a gross desync, such as a device that stalled and came back.
        """
        blocksize = stream.config.blocksize
        target = blocksize * DRIFT_TARGET_BLOCKS
        available = ring.available

        resampler = self._routes.resampler_for(key) if key is not None else None
        if resampler is not None:
            # Fill above target means the source is ahead: read slightly faster.
            error = (available - target) / max(target, 1.0)
            resampler.set_ratio(1.0 + error * DRIFT_CORRECTION_STRENGTH)

        hard_limit = int(blocksize * DRIFT_HIGH_BLOCKS)
        if available > hard_limit:
            # Beyond what resampling can pull back in reasonable time. Drop, and count it
            # so a persistently bad configuration is visible rather than merely audible.
            ring.discard(available - blocksize)
            stream.drift_corrections += 1

    def _map_channels(
        self,
        stream: _DeviceStream,
        source: np.ndarray,
        out_channels: int,
        frames: int,
        connection: Optional[Connection] = None,
        key: Optional[Tuple[str, str]] = None,
    ) -> np.ndarray:
        """
        Fit a source's channel count to the destination's, applying pan where it applies.

        Mono into stereo is a pan, not a duplication: with a constant-power law a centred
        mono source keeps the same perceived loudness as a hard-panned one. Stereo to mono
        averages rather than taking the left channel, so a hard-right guitar does not
        vanish. Equal counts with no pan pass through with no copy at all.
        """
        in_channels = source.shape[1]
        panner = self._panner_for(key, connection)

        if in_channels == 1 and out_channels == 2:
            # Always panned, even at centre. Placing a mono source in a stereo field is
            # what the pan law is *for*: centred means -3 dB per side, not unity per side.
            # Duplicating at unity would make a centred source 3 dB louder than a panned
            # one, so a pan sweep would audibly dip at the edges.
            if panner is not None:
                return panner.process_mono_to_stereo(source, frames)
            scratch = stream._channel_scratch[:frames, :2]
            scratch[:] = source[:, 0:1]
            return scratch

        if in_channels == out_channels:
            # Equal widths: centre really is a no-op, so skip the copy entirely.
            if panner is not None and out_channels == 2 and not panner.is_centred():
                scratch = stream._channel_scratch[:frames, :2]
                scratch[:] = source[:frames]
                return panner.process_stereo(scratch, frames)
            return source

        scratch = stream._channel_scratch[:frames, :out_channels]

        if in_channels == 1:
            scratch[:] = source[:, 0:1]
        elif out_channels == 1:
            np.mean(source, axis=1, out=scratch[:, 0])
        elif in_channels > out_channels:
            scratch[:] = source[:, :out_channels]
        else:
            scratch[:, :in_channels] = source
            scratch[:, in_channels:] = 0.0

        return scratch

    def _read_source(
        self, stream: _DeviceStream, ring: AudioRingBuffer,
        key: Tuple[str, str], frames: int,
    ) -> np.ndarray:
        """
        Pull one block from a route's ring, resampling it if the route crosses clocks.

        Without a resampler this is a straight read. With one, we read the slightly
        different number of input frames the current ratio calls for and interpolate,
        which is how drift is absorbed continuously instead of in audible jumps.
        """
        channels = ring.channels
        out = stream.source_buffer(channels)[:frames]

        resampler = self._routes.resampler_for(key)
        if resampler is None:
            ring.read_into(out)
            return out

        needed = min(resampler.input_frames_needed(frames), stream._resample_in.shape[0])
        if channels > stream._resample_in.shape[1]:
            stream._resample_in = np.zeros(
                (stream.config.blocksize * 2, channels), dtype=np.float32
            )

        source = stream._resample_in[:needed, :channels]
        ring.read_into(source)

        consumed = resampler.process(source, out, frames)

        # Hand back whatever the interpolator did not use, so no sample is lost or
        # duplicated across block boundaries.
        surplus = needed - consumed
        if surplus > 0:
            ring.unread(surplus)

        return out

    def _panner_for(
        self,
        key: Optional[Tuple[str, str]],
        connection: Optional[Connection],
    ) -> Optional[Panner]:
        """
        The panner for a route, with its position brought up to date.

        Panners are created on the control thread in `_rebuild_routes`; a lookup miss here
        means the route is brand new, and this block passes through unpanned rather than
        the callback allocating one. Whether a centred panner can be skipped depends on
        the channel mapping, so that decision belongs to the caller.
        """
        if key is None or connection is None:
            return None

        panner = self._routes.panner_for(key)
        if panner is None:
            return None

        if panner.pan != connection.pan:
            panner.set_pan(connection.pan)

        return panner

    # --- Bus input, for non-realtime producers ---

    def write_bus(self, name: str, frames: np.ndarray) -> int:
        """
        Push audio into a bus, fanning it out to every route leaving that bus.

        Used by tone generators, network receive and tests. Returns the number of frames
        accepted by the most backed-up consumer, so a caller can tell it is outrunning
        the audio clock.
        """
        if name not in self._bus_channels:
            return 0

        block = np.ascontiguousarray(frames, dtype=np.float32)
        sinks = self._routes.sinks_for(str(bus_node(name)))

        if not sinks:
            return 0

        written = min(ring.write(block) for ring in sinks)

        bank = self.meters.get(f"bus::{name}")
        if bank is not None:
            bank.measure(block)

        return written

    def read_bus(self, name: str, frames: int, dest: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Consume a bus's audio for one destination. Allocates, so not for callbacks.

        A bus has one ring per destination, so reading requires knowing which. With `dest`
        omitted the first is used, which is what a single-consumer test wants.
        """
        source_key = str(bus_node(name))

        if dest is not None:
            ring = self._routes.ring_for(source_key, dest)
        else:
            sinks = self._routes.sinks_for(source_key)
            ring = sinks[0] if sinks else None

        if ring is None:
            return None

        out = np.zeros((frames, ring.channels), dtype=np.float32)
        ring.read_into(out)
        return out

    # --- Statistics ---

    def statistics(self) -> HostStatistics:
        """
        What we measured. Anything we did not measure is None.

        `cpu_load` is PortAudio's own figure: the fraction of each block period spent
        inside our callback. Above ~0.8 dropouts are imminent.
        """
        stats = HostStatistics(
            running=self._running,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            exclusive=any(s.used_exclusive for s in self._streams.values() if s.is_live),
            stream_count=len(self._streams),
            live_stream_count=sum(1 for s in self._streams.values() if s.is_live),
            failed_streams=self.failed_streams(),
            nominal_latency_ms=self.blocksize / self.samplerate * 1000.0,
        )

        if self.host_api is not None:
            stats.host_api = self.host_api.value

        stats.xruns = sum(s.xruns for s in self._streams.values())
        stats.callback_count = sum(s.callback_count for s in self._streams.values())
        stats.callback_errors = sum(s.callback_errors for s in self._streams.values())
        stats.drift_corrections = sum(s.drift_corrections for s in self._streams.values())

        if not self._running:
            return stats

        input_latency = None
        output_latency = None
        cpu_loads = []

        for stream in self._streams.values():
            handle = stream.stream
            if handle is None or stream.error is not None:
                # A failed stream must not contribute a latency or CPU figure; averaging
                # it in would make a broken configuration look measured and healthy.
                continue

            try:
                cpu_loads.append(float(handle.cpu_load))
            except Exception:
                pass

            latency = getattr(handle, 'latency', None)
            if latency is None:
                continue

            if isinstance(latency, (tuple, list)):
                if stream.config.input_channels:
                    input_latency = max(input_latency or 0.0, float(latency[0]) * 1000.0)
                if stream.config.output_channels:
                    output_latency = max(output_latency or 0.0, float(latency[1]) * 1000.0)
            else:
                value = float(latency) * 1000.0
                if stream.config.input_channels:
                    input_latency = max(input_latency or 0.0, value)
                else:
                    output_latency = max(output_latency or 0.0, value)

            if stream.config.device.host_api and stats.host_api is None:
                stats.host_api = stream.config.device.host_api_name

        stats.input_latency_ms = input_latency
        stats.output_latency_ms = output_latency

        # Round trip is what a player feels: in, through us, and back out — including
        # whatever the loaded plugins add, which can be several milliseconds.
        if input_latency is not None or output_latency is not None:
            stats.measured_latency_ms = (
                (input_latency or 0.0)
                + (output_latency or 0.0)
                + self.total_plugin_latency_ms()
            )

        if cpu_loads:
            stats.cpu_load = max(cpu_loads) * 100.0

        return stats

    def ring_statistics(self) -> Dict[str, dict]:
        """Per-route buffer health. Overflow means a consumer is starving its producer."""
        return self._routes.statistics()

    @property
    def last_callback_error(self) -> Optional[str]:
        """Most recent exception swallowed by a callback, for diagnostics."""
        return self._last_callback_error

    def cleanup(self):
        self.stop()
        with self._lock:
            self._streams.clear()
            self._bus_channels.clear()
            self._routes = _RouteTable({})
            self.meters.clear()
