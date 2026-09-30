"""
The native ASIO host (`tonesphere_asio.dll`, GPLv3 — see `native/asio/LICENSE`).

A registry entry under HKLM\\SOFTWARE\\ASIO is not ASIO support. `drivers()` lists what is
registered; `query()` loads a driver and reads what it reports; only `NativeEngine` running
under `start()` with audio moving through the driver's buffer switch is ASIO working.
"""

import ctypes
from ctypes import POINTER, Structure, c_char, c_char_p, c_double, c_int32, c_uint32, c_void_p
from dataclasses import dataclass
from pathlib import Path

from tonesphere.native import NativeError, NativeUnavailable, _abi, candidate_dirs, load

DLL_NAME = "tonesphere_asio.dll"
ABI_VERSION = 1
MAX_CHANNELS = 64
RATES = (44100, 48000, 88200, 96000, 176400, 192000, 32000, 22050)

SAMPLE_TYPES = {
    16: 'int16', 17: 'int24', 18: 'int32', 19: 'float32', 20: 'float64',
    24: 'int32 (16-bit)', 25: 'int32 (18-bit)', 26: 'int32 (20-bit)', 27: 'int32 (24-bit)',
}


class _Driver(Structure):
    _fields_ = [("name", c_char * 128), ("description", c_char * 128), ("clsid", c_char * 64),
                ("dll_path", c_char * 260), ("dll_present", c_uint32), ("reserved", c_uint32)]


class _Channel(Structure):
    _fields_ = [("name", c_char * 32), ("sample_type", c_int32), ("group", c_int32),
                ("supported", c_uint32), ("reserved", c_uint32)]


class _Info(Structure):
    _fields_ = [
        ("driver_name", c_char * 128), ("driver_version", c_int32), ("inputs", c_int32), ("outputs", c_int32),
        ("min_buffer", c_int32), ("max_buffer", c_int32), ("preferred_buffer", c_int32), ("granularity", c_int32),
        ("current_sample_rate", c_double), ("rates_supported", c_uint32), ("input_latency", c_int32),
        ("output_latency", c_int32), ("post_output", c_uint32),
        ("input_channels", _Channel * MAX_CHANNELS), ("output_channels", _Channel * MAX_CHANNELS),
    ]


class _Config(Structure):
    _fields_ = [("driver", c_char * 128), ("buffer_frames", c_uint32), ("input_node", c_uint32),
                ("output_node", c_uint32), ("input_count", c_uint32), ("output_count", c_uint32),
                ("inputs", c_uint32 * MAX_CHANNELS), ("outputs", c_uint32 * MAX_CHANNELS)]


_dll = None


def load_asio() -> ctypes.CDLL:
    """The ASIO DLL, bound. Raises NativeUnavailable (with the reason) if it is not built."""
    global _dll
    if _dll is not None:
        return _dll
    load()  # the core DLL first: the ASIO DLL imports it
    for directory in candidate_dirs():
        path = Path(directory) / DLL_NAME
        if not path.is_file():
            continue
        dll = ctypes.CDLL(str(path))
        f_ptr = POINTER(ctypes.c_float)
        for name, restype, argtypes in [
            ("ts_asio_abi_version", c_int32, []),
            ("ts_asio_last_error", c_int32, [c_char_p, c_int32]),
            ("ts_asio_list", c_int32, [POINTER(_Driver), c_int32]),
            ("ts_asio_query", c_int32, [c_char_p, POINTER(_Info)]),
            ("ts_asio_start", c_int32, [c_void_p, POINTER(_Config)]),
            ("ts_asio_convert_in", c_int32, [c_int32, c_void_p, f_ptr, c_uint32]),
            ("ts_asio_convert_out", c_int32, [c_int32, f_ptr, c_void_p, c_uint32]),
        ]:
            fn = getattr(dll, name)
            fn.restype, fn.argtypes = restype, argtypes
        if dll.ts_asio_abi_version() != ABI_VERSION:
            raise NativeUnavailable(f"{path} speaks ASIO ABI {dll.ts_asio_abi_version()}, expected {ABI_VERSION}")
        _dll = dll
        return dll
    raise NativeUnavailable(
        f"{DLL_NAME} not found: it is built only when the ASIO SDK has been fetched "
        f"(`uv run python scripts/fetch_sdks.py asio`, then `scripts/build_native.py`)")


def available() -> bool:
    try:
        load_asio()
        return True
    except NativeUnavailable:
        return False


def _error() -> str:
    buffer = ctypes.create_string_buffer(1024)
    load_asio().ts_asio_last_error(buffer, len(buffer))
    return buffer.value.decode(errors='replace')


@dataclass(frozen=True)
class Driver:
    name: str
    description: str
    clsid: str
    dll_path: str
    dll_present: bool


def drivers() -> list[Driver]:
    """ASIO drivers registered on this machine. Registered is not the same as working."""
    dll = load_asio()
    count = dll.ts_asio_list(None, 0)
    buffer = (_Driver * max(1, count))()
    count = dll.ts_asio_list(buffer, len(buffer))
    return [Driver(d.name.decode(errors='replace'), d.description.decode(errors='replace'),
                   d.clsid.decode(), d.dll_path.decode(errors='replace'), bool(d.dll_present))
            for d in buffer[:count]]


@dataclass(frozen=True)
class Channel:
    index: int
    name: str
    sample_type: str
    supported: bool


@dataclass(frozen=True)
class DriverInfo:
    name: str
    version: int
    inputs: list[Channel]
    outputs: list[Channel]
    min_buffer: int
    max_buffer: int
    preferred_buffer: int
    granularity: int
    sample_rate: float
    sample_rates: list[int]
    input_latency_frames: int    # reported by the driver, not measured
    output_latency_frames: int
    post_output: bool


def query(name: str) -> DriverInfo:
    """Load the driver, read everything it reports, and unload it."""
    info = _Info()
    if load_asio().ts_asio_query(name.encode(), ctypes.byref(info)) != _abi.OK:
        raise NativeError(_abi.ERR_BACKEND, _error())

    def channels(array, count):
        return [Channel(i, c.name.decode(errors='replace'), SAMPLE_TYPES.get(c.sample_type, f"type {c.sample_type}"),
                        bool(c.supported)) for i, c in enumerate(array[:min(count, MAX_CHANNELS)])]

    return DriverInfo(
        name=info.driver_name.decode(errors='replace'), version=info.driver_version,
        inputs=channels(info.input_channels, info.inputs), outputs=channels(info.output_channels, info.outputs),
        min_buffer=info.min_buffer, max_buffer=info.max_buffer, preferred_buffer=info.preferred_buffer,
        granularity=info.granularity, sample_rate=info.current_sample_rate,
        sample_rates=[r for i, r in enumerate(RATES) if info.rates_supported & (1 << i)],
        input_latency_frames=info.input_latency, output_latency_frames=info.output_latency,
        post_output=bool(info.post_output),
    )


def start(engine, driver: str, *, input_node: int = 0, inputs: tuple[int, ...] = (),
          output_node: int = 0, outputs: tuple[int, ...] = (), buffer_frames: int = 0):
    """
    Run `engine` from the driver's buffer switch. `inputs`/`outputs` are the driver's
    channel indices, in the order they fill the node's channels. The engine's sample rate
    is the stream's: a driver that cannot run at it is refused, not resampled.
    """
    if len(inputs) > MAX_CHANNELS or len(outputs) > MAX_CHANNELS:
        raise ValueError("at most 64 channels each way")
    config = _Config()
    config.driver = driver.encode()
    config.buffer_frames = buffer_frames
    config.input_node, config.output_node = input_node, output_node
    config.input_count, config.output_count = len(inputs), len(outputs)
    for i, c in enumerate(inputs):
        config.inputs[i] = c
    for i, c in enumerate(outputs):
        config.outputs[i] = c
    with engine._control:
        if load_asio().ts_asio_start(engine._handle, ctypes.byref(config)) != _abi.OK:
            raise NativeError(_abi.ERR_BACKEND, _error())


def convert_in(sample_type: int, data: bytes, samples: int):
    import numpy as np
    out = np.zeros(samples, dtype=np.float32)
    if load_asio().ts_asio_convert_in(sample_type, data, out.ctypes.data_as(POINTER(ctypes.c_float)), samples) != 0:
        raise NativeError(_abi.ERR_INVALID, "unsupported sample type")
    return out


def convert_out(sample_type: int, samples, width: int) -> bytes:
    import numpy as np
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    out = ctypes.create_string_buffer(len(samples) * width)
    if load_asio().ts_asio_convert_out(sample_type, samples.ctypes.data_as(POINTER(ctypes.c_float)), out,
                                       len(samples)) != 0:
        raise NativeError(_abi.ERR_INVALID, "unsupported sample type")
    return out.raw
