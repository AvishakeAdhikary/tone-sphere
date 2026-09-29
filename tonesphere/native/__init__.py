"""
Python side of the native real-time engine.

Everything here is control plane: it builds plans, sets targets, reads meters and
statistics, and moves audio in and out of ring ports from ordinary Python threads. None
of it runs on the audio thread, and the audio thread never calls back into Python.

The DLL is loaded lazily, so this package imports on every platform; `load()` raises
`NativeUnavailable` with the reason when there is no usable DLL. Nothing falls back to a
Python imitation of the engine — a missing native engine is reported, not papered over.
"""

import ctypes
import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tonesphere.native import _abi

DLL_NAME = "tonesphere_native.dll"


class NativeUnavailable(RuntimeError):
    """The native engine could not be loaded; the message says why."""


class NativeError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(f"{message} (ts_result {code})")
        self.code = code


def candidate_dirs() -> list[Path]:
    dirs = []
    if os.environ.get("TONESPHERE_NATIVE_DIR"):
        dirs.append(Path(os.environ["TONESPHERE_NATIVE_DIR"]))
    if getattr(sys, "_MEIPASS", None):
        dirs.append(Path(sys._MEIPASS) / "tonesphere" / "native" / "_bin")
    dirs.append(Path(__file__).resolve().parent / "_bin")
    return dirs


_lock = threading.Lock()
_dll: ctypes.CDLL | None = None


def load() -> ctypes.CDLL:
    global _dll
    with _lock:
        if _dll is not None:
            return _dll
        if sys.platform != "win32":
            raise NativeUnavailable("the native engine is built for Windows only")
        tried = []
        for directory in candidate_dirs():
            path = directory / DLL_NAME
            if not path.is_file():
                tried.append(f"{path} (missing)")
                continue
            try:
                dll = _abi.bind(ctypes.CDLL(str(path)))
            except OSError as e:
                tried.append(f"{path} ({e})")
                continue
            version = dll.ts_abi_version()
            if version != _abi.ABI_VERSION:
                raise NativeUnavailable(
                    f"{path} speaks ABI {version}, this Python expects {_abi.ABI_VERSION}; rebuild with "
                    f"`uv run python scripts/build_native.py`")
            _dll = dll
            return dll
        raise NativeUnavailable(
            "tonesphere_native.dll not found — build it with `uv run python scripts/build_native.py`. "
            "Tried: " + "; ".join(tried))


def available() -> bool:
    try:
        load()
        return True
    except NativeUnavailable:
        return False


def build_info() -> str:
    return load().ts_build_info().decode()


@dataclass(frozen=True)
class Node:
    id: int
    kind: int
    channels: int
    ring_frames: int = 0
    limiter: bool = False

    @classmethod
    def source(cls, id: int, channels: int, ring_frames: int = 0) -> "Node":
        return cls(id, _abi.NODE_SOURCE, channels, ring_frames)

    @classmethod
    def bus(cls, id: int, channels: int) -> "Node":
        return cls(id, _abi.NODE_BUS, channels)

    @classmethod
    def sink(cls, id: int, channels: int, ring_frames: int = 0, limiter: bool = False) -> "Node":
        return cls(id, _abi.NODE_SINK, channels, ring_frames, limiter)


@dataclass(frozen=True)
class Route:
    source: int
    dest: int
    gain: float = 1.0
    pan: float = 0.0
    muted: bool = False
    invert: bool = False


EQ = _abi.INSERT_EQ
COMPRESSOR = _abi.INSERT_COMPRESSOR
LIMITER = _abi.INSERT_LIMITER
DELAY = _abi.INSERT_DELAY
VST3 = _abi.INSERT_VST3


@dataclass(frozen=True)
class Insert:
    """
    A processor on a node. `slot` is both its address and its processing order. For a
    VST3 insert, `plugin` is the handle of an open `tonesphere.plugins.PluginInstance`.
    """
    node: int
    slot: int
    type: int
    bypassed: bool = False
    plugin: int = 0


def _float_ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class NativeEngine:
    """
    One engine instance. Control calls are serialised by an internal lock, which is the
    threading contract the C ABI asks for. `process()` is the audio-thread role and is
    deliberately *not* under that lock: the engine is designed for control calls to run
    concurrently with processing, and a test that did not exercise that would prove less.
    """

    def __init__(self, sample_rate: int = 48000, max_block: int = 256):
        self._dll = load()
        self._handle = self._dll.ts_engine_create(sample_rate, max_block)
        if not self._handle:
            raise NativeError(_abi.ERR_INVALID, f"could not create an engine at {sample_rate} Hz / {max_block} frames")
        self.sample_rate = sample_rate
        self.max_block = max_block
        self._control = threading.Lock()
        self._nodes: dict[int, Node] = {}

    def close(self):
        with self._control:
            if self._handle:
                self._dll.ts_engine_destroy(self._handle)
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _error(self) -> str:
        buffer = ctypes.create_string_buffer(1024)
        self._dll.ts_engine_last_error(self._handle, buffer, len(buffer))
        return buffer.value.decode(errors="replace")

    def _check(self, result: int) -> int:
        if result < 0:
            raise NativeError(result, self._error())
        return result

    def apply_plan(self, nodes: list[Node], routes: list[Route], inserts: list[Insert] = ()):
        node_array = (_abi.NodeDesc * max(1, len(nodes)))()
        for i, n in enumerate(nodes):
            flags = (_abi.NODE_FLAG_RING if n.ring_frames else 0) | (_abi.NODE_FLAG_LIMITER if n.limiter else 0)
            node_array[i] = _abi.NodeDesc(n.id, n.kind, n.channels, flags, n.ring_frames)
        route_array = (_abi.RouteDesc * max(1, len(routes)))()
        for i, r in enumerate(routes):
            flags = (_abi.ROUTE_FLAG_MUTED if r.muted else 0) | (_abi.ROUTE_FLAG_INVERT if r.invert else 0)
            route_array[i] = _abi.RouteDesc(r.source, r.dest, r.gain, r.pan, flags)
        insert_array = (_abi.InsertDesc * max(1, len(inserts)))()
        for i, x in enumerate(inserts):
            insert_array[i] = _abi.InsertDesc(x.node, x.slot, x.type, _abi.INSERT_FLAG_BYPASSED if x.bypassed else 0,
                                              x.plugin)
        plan = _abi.Plan(node_array, len(nodes), route_array, len(routes), insert_array, len(inserts))
        with self._control:
            self._check(self._dll.ts_engine_apply_plan(self._handle, ctypes.byref(plan)))
            self._nodes = {n.id: n for n in nodes}

    def set_route_gain(self, source: int, dest: int, gain: float):
        with self._control:
            self._check(self._dll.ts_engine_set_route_gain(self._handle, source, dest, gain))

    def set_route_muted(self, source: int, dest: int, muted: bool):
        with self._control:
            self._check(self._dll.ts_engine_set_route_muted(self._handle, source, dest, int(muted)))

    def set_route_pan(self, source: int, dest: int, pan: float):
        with self._control:
            self._check(self._dll.ts_engine_set_route_pan(self._handle, source, dest, pan))

    def set_master_gain(self, gain: float):
        with self._control:
            self._check(self._dll.ts_engine_set_master_gain(self._handle, gain))

    def set_node_gain(self, node: int, gain: float):
        with self._control:
            self._check(self._dll.ts_engine_set_node_gain(self._handle, node, gain))

    def set_node_muted(self, node: int, muted: bool):
        with self._control:
            self._check(self._dll.ts_engine_set_node_muted(self._handle, node, int(muted)))

    def set_channel_trim(self, node: int, channel: int, gain: float):
        with self._control:
            self._check(self._dll.ts_engine_set_channel_trim(self._handle, node, channel, gain))

    def set_channel_inverted(self, node: int, channel: int, inverted: bool):
        with self._control:
            self._check(self._dll.ts_engine_set_channel_inverted(self._handle, node, channel, int(inverted)))

    def set_insert_param(self, node: int, slot: int, param: int, value: float):
        with self._control:
            self._check(self._dll.ts_engine_set_insert_param(self._handle, node, slot, param, value))

    def insert_param(self, node: int, slot: int, param: int) -> float:
        value = ctypes.c_float()
        with self._control:
            self._check(self._dll.ts_engine_get_insert_param(self._handle, node, slot, param, ctypes.byref(value)))
        return value.value

    def set_insert_bypassed(self, node: int, slot: int, bypassed: bool):
        with self._control:
            self._check(self._dll.ts_engine_set_insert_bypassed(self._handle, node, slot, int(bypassed)))

    def insert_readout(self, node: int, slot: int) -> float | None:
        """Gain reduction in dB the audio thread last applied; None for a processor that has none."""
        value = ctypes.c_float()
        with self._control:
            self._check(self._dll.ts_engine_get_insert_readout(self._handle, node, slot, ctypes.byref(value)))
        return None if math.isnan(value.value) else value.value

    def process(self, inputs: dict[int, np.ndarray], outputs: dict[int, int]) -> dict[int, np.ndarray]:
        """
        Run one block with no device. `inputs` maps source node id to a (frames, channels)
        float32 block; `outputs` maps sink node id to the channel count wanted back.
        """
        frames = None
        keep = []
        in_ports = (_abi.PortBuffer * max(1, len(inputs)))()
        for i, (node_id, block) in enumerate(inputs.items()):
            block = np.ascontiguousarray(block, dtype=np.float32)
            if block.ndim == 1:
                block = block.reshape(-1, 1)
            if frames is None:
                frames = block.shape[0]
            elif block.shape[0] != frames:
                raise ValueError("every input block must have the same number of frames")
            keep.append(block)
            in_ports[i] = _abi.PortBuffer(node_id, block.shape[1], _float_ptr(block))
        if frames is None:
            raise ValueError("process() needs at least one input to know the block size; pass zeros for silence")

        results = {}
        out_ports = (_abi.PortBuffer * max(1, len(outputs)))()
        for i, (node_id, channels) in enumerate(outputs.items()):
            out = np.full((frames, channels), np.nan, dtype=np.float32)
            results[node_id] = out
            out_ports[i] = _abi.PortBuffer(node_id, channels, _float_ptr(out))

        self._check(self._dll.ts_engine_process(self._handle, in_ports, len(inputs), out_ports, len(outputs), frames))
        return results

    def port_write(self, node_id: int, block: np.ndarray) -> int:
        block = np.ascontiguousarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        with self._control:
            return self._check(self._dll.ts_port_write(self._handle, node_id, _float_ptr(block), block.shape[0]))

    def port_read(self, node_id: int, frames: int) -> np.ndarray:
        channels = self._nodes[node_id].channels
        out = np.zeros((frames, channels), dtype=np.float32)
        with self._control:
            got = self._check(self._dll.ts_port_read(self._handle, node_id, _float_ptr(out), frames))
        return out[:got]

    def port_available(self, node_id: int) -> int:
        with self._control:
            return self._check(self._dll.ts_port_available(self._handle, node_id))

    def stats(self) -> dict:
        """
        Callback timing as measured on the audio thread. Every duration is None until a
        block has run; `processing_load` is the worst block against its period, because a
        real-time budget is broken by its worst case, not its average.
        """
        raw = _abi.Stats()
        with self._control:
            self._check(self._dll.ts_engine_get_stats(self._handle, ctypes.byref(raw)))
        blocks = raw.blocks
        histogram = list(raw.histogram)
        measured = blocks > 0
        return {
            'blocks': blocks,
            'xruns': raw.xruns,
            'ring_overruns': raw.ring_overruns,
            'ring_underruns': raw.ring_underruns,
            'callback_ns_min': raw.callback_ns_min if measured else None,
            'callback_ns_max': raw.callback_ns_max if measured else None,
            'callback_ns_mean': raw.callback_ns_total / blocks if measured else None,
            'callback_ns_p99': min(_percentile(histogram, 0.99), raw.callback_ns_max) if measured else None,
            'period_ns': raw.period_ns if measured else None,
            'processing_load': raw.callback_ns_max / raw.period_ns if measured and raw.period_ns else None,
            'plan_generation': raw.plan_generation,
            'rt_allocations': raw.rt_allocations,
            'histogram': histogram,
        }

    def reset_stats(self):
        with self._control:
            self._check(self._dll.ts_engine_reset_stats(self._handle))

    def meter(self, node_id: int) -> dict:
        raw = _abi.Meter()
        with self._control:
            self._check(self._dll.ts_engine_get_meter(self._handle, node_id, ctypes.byref(raw)))
        return {'peak': raw.peak, 'rms': raw.rms, 'clipped': bool(raw.clipped), 'channels': raw.channels}

    def reset_meters(self):
        with self._control:
            self._check(self._dll.ts_engine_reset_meters(self._handle))

    def start_wasapi(self, streams, master: int = 0):
        """
        Open device streams and let the master stream's device clock run the engine. See
        `tonesphere.native.wasapi.StreamSpec`. From here until `stop_backend()` the engine
        has an audio thread of its own, and `process()` is refused.
        """
        array = (_abi.StreamDesc * len(streams))()
        for i, spec in enumerate(streams):
            array[i] = spec.to_desc()
        with self._control:
            self._check(self._dll.ts_engine_start_wasapi(self._handle, array, len(streams), master))

    def stop_backend(self):
        with self._control:
            self._check(self._dll.ts_engine_stop_backend(self._handle))

    def stream_status(self) -> list[dict]:
        buffer = (_abi.StreamStatus * 64)()
        with self._control:
            n = self._check(self._dll.ts_engine_stream_status(self._handle, buffer, len(buffer)))
        return [_stream_status(s) for s in buffer[:n]]

    def events(self) -> list[dict]:
        buffer = (_abi.Event * 256)()
        with self._control:
            n = self._check(self._dll.ts_engine_poll_events(self._handle, buffer, len(buffer)))
        return [{'code': e.code, 'arg0': e.arg0, 'arg1': e.arg1, 'block': e.block} for e in buffer[:n]]


_STREAM_STATES = {0: 'starting', 1: 'running', 2: 'failed', 3: 'stopped'}


def _stream_status(s) -> dict:
    """Latency here is what Windows reports for the stream; it is never a measurement."""
    return {
        'node_id': s.node_id,
        'kind': s.kind,
        'state': _STREAM_STATES.get(s.state, str(s.state)),
        'is_master': bool(s.is_master),
        'exclusive': s.share_mode == _abi.SHARE_EXCLUSIVE,
        'hresult': s.hresult,
        'sample_rate': s.sample_rate or None,
        'channels': s.channels,
        'bits': s.bits or None,
        'valid_bits': s.valid_bits or None,
        'is_float': bool(s.is_float),
        'buffer_frames': s.buffer_frames or None,
        'period_frames': s.period_frames or None,
        'reported_latency_ms': s.stream_latency_hns / 10_000 if s.stream_latency_hns > 0 else None,
        'frames': s.frames,
        'glitches': s.glitches,
        'underruns': s.underruns,
        'overruns': s.overruns,
        'drift_ratio': s.drift_ratio,
        'ring_fill': s.ring_fill,
        'raw': bool(s.raw),
        'message': s.error.decode(errors='replace'),
    }


def _percentile(histogram: list[int], q: float) -> float:
    """
    Upper edge of the bucket holding the q-th sample: conservative, never flattering. The
    caller caps it at the observed maximum, since a bucket edge can exceed every sample in it.
    """
    total = sum(histogram)
    threshold = q * total
    running = 0
    for i, count in enumerate(histogram):
        running += count
        if running >= threshold:
            return _abi.bucket_edge_ns(i)
    return _abi.bucket_edge_ns(len(histogram) - 1)


def convert(fmt: int, samples: np.ndarray) -> bytes:
    """float32 -> a device sample format, through the engine's own boundary conversion."""
    samples = np.ascontiguousarray(samples, dtype=np.float32).ravel()
    width = {_abi.FORMAT_FLOAT32: 4, _abi.FORMAT_FLOAT64: 8, _abi.FORMAT_INT16: 2,
             _abi.FORMAT_INT24: 3, _abi.FORMAT_INT32: 4}[fmt]
    out = ctypes.create_string_buffer(len(samples) * width)
    if load().ts_convert_from_float(fmt, _float_ptr(samples), out, len(samples)) != _abi.OK:
        raise NativeError(_abi.ERR_INVALID, "conversion refused")
    return out.raw


def unconvert(fmt: int, data: bytes, samples: int) -> np.ndarray:
    """A device sample format -> float32."""
    out = np.zeros(samples, dtype=np.float32)
    if load().ts_convert_to_float(fmt, data, _float_ptr(out), samples) != _abi.OK:
        raise NativeError(_abi.ERR_INVALID, "conversion refused")
    return out


class NativeResampler:
    """A FrameRing read through the engine's DriftResampler: the clock-boundary crossing on its own."""

    def __init__(self, channels: int, max_block: int, target_fill: int, ring_frames: int):
        self._dll = load()
        self._handle = self._dll.ts_resampler_create(channels, max_block, target_fill, ring_frames)
        if not self._handle:
            raise NativeError(_abi.ERR_INVALID, "could not create resampler")
        self.channels = channels

    def push(self, block: np.ndarray) -> int:
        block = np.ascontiguousarray(block, dtype=np.float32).reshape(-1, self.channels)
        return self._dll.ts_resampler_push(self._handle, _float_ptr(block), block.shape[0])

    def pull(self, frames: int) -> tuple[np.ndarray, int]:
        out = np.zeros((frames, self.channels), dtype=np.float32)
        missing = self._dll.ts_resampler_pull(self._handle, _float_ptr(out), frames)
        return out, missing

    @property
    def ratio(self) -> float:
        return self._dll.ts_resampler_ratio(self._handle)

    @property
    def fill(self) -> int:
        return self._dll.ts_resampler_fill(self._handle)

    def __del__(self):
        try:
            if self._handle:
                self._dll.ts_resampler_destroy(self._handle)
                self._handle = None
        except Exception:
            pass


class NativeRing:
    """The engine's SPSC ring on its own, so its guarantees can be tested directly."""

    def __init__(self, capacity_frames: int, channels: int):
        self._dll = load()
        self._handle = self._dll.ts_ring_create(capacity_frames, channels)
        if not self._handle:
            raise NativeError(_abi.ERR_INVALID, "could not create ring")
        self.channels = channels

    @property
    def capacity(self) -> int:
        return self._dll.ts_ring_capacity(self._handle)

    def write(self, block: np.ndarray) -> int:
        block = np.ascontiguousarray(block, dtype=np.float32).reshape(-1, self.channels)
        return self._dll.ts_ring_write(self._handle, _float_ptr(block), block.shape[0])

    def read(self, frames: int) -> np.ndarray:
        out = np.zeros((frames, self.channels), dtype=np.float32)
        got = self._dll.ts_ring_read(self._handle, _float_ptr(out), frames)
        return out[:got]

    def available(self) -> int:
        return self._dll.ts_ring_available(self._handle)

    def close(self):
        if self._handle:
            self._dll.ts_ring_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
