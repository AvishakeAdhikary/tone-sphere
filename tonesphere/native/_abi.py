"""
ctypes mirror of `native/include/tonesphere_native.h`.

Every struct and prototype here must match the header exactly; `ABI_VERSION` is checked
against the DLL at load time so a stale DLL fails loudly instead of misreading memory.
"""

import ctypes
from ctypes import POINTER, Structure, c_char_p, c_float, c_int32, c_uint32, c_uint64, c_void_p

ABI_VERSION = 3

OK = 0
ERR_INVALID = -1
ERR_CYCLE = -2
ERR_NOMEM = -3
ERR_STATE = -4
ERR_UNSUPPORTED = -5
ERR_NOT_FOUND = -6
ERR_BACKEND = -7

NODE_SOURCE = 1
NODE_BUS = 2
NODE_SINK = 3
NODE_FLAG_RING = 0x1
NODE_FLAG_LIMITER = 0x2

ROUTE_FLAG_MUTED = 0x1
ROUTE_FLAG_INVERT = 0x2

INSERT_EQ = 1
INSERT_COMPRESSOR = 2
INSERT_LIMITER = 3
INSERT_DELAY = 4
INSERT_FLAG_BYPASSED = 0x1

EQ_OFF = 0
EQ_PEAKING = 1
EQ_LOW_SHELF = 2
EQ_HIGH_SHELF = 3
EQ_HIGHPASS = 4
EQ_LOWPASS = 5

MAX_CHANNELS = 64
MAX_INSERTS = 16
HISTOGRAM_BUCKETS = 64


def bucket_edge_ns(i: int) -> float:
    """Upper edge of histogram bucket i, as defined in the header."""
    return 1000.0 * 2.0 ** ((i + 1) / 4.0)

EVENT_RING_OVERRUN = 1
EVENT_RING_UNDERRUN = 2
EVENT_NONFINITE = 3
EVENT_XRUN = 4
EVENT_EVENTS_LOST = 5


class NodeDesc(Structure):
    _fields_ = [
        ("id", c_uint32),
        ("kind", c_uint32),
        ("channels", c_uint32),
        ("flags", c_uint32),
        ("ring_frames", c_uint32),
    ]


class RouteDesc(Structure):
    _fields_ = [
        ("source", c_uint32),
        ("dest", c_uint32),
        ("gain", c_float),
        ("pan", c_float),
        ("flags", c_uint32),
    ]


class InsertDesc(Structure):
    _fields_ = [
        ("node_id", c_uint32),
        ("slot", c_uint32),
        ("type", c_uint32),
        ("flags", c_uint32),
    ]


class Plan(Structure):
    _fields_ = [
        ("nodes", POINTER(NodeDesc)),
        ("node_count", c_uint32),
        ("routes", POINTER(RouteDesc)),
        ("route_count", c_uint32),
        ("inserts", POINTER(InsertDesc)),
        ("insert_count", c_uint32),
    ]


class Stats(Structure):
    _fields_ = [
        ("blocks", c_uint64),
        ("xruns", c_uint64),
        ("ring_overruns", c_uint64),
        ("ring_underruns", c_uint64),
        ("callback_ns_min", c_uint64),
        ("callback_ns_max", c_uint64),
        ("callback_ns_total", c_uint64),
        ("period_ns", c_uint64),
        ("plan_generation", c_uint64),
        ("rt_allocations", c_uint64),
        ("histogram", c_uint64 * HISTOGRAM_BUCKETS),
    ]


class Meter(Structure):
    _fields_ = [
        ("peak", c_float),
        ("rms", c_float),
        ("clipped", c_uint32),
        ("channels", c_uint32),
    ]


class Event(Structure):
    _fields_ = [
        ("code", c_uint32),
        ("arg0", c_uint32),
        ("arg1", c_uint64),
        ("block", c_uint64),
    ]


class PortBuffer(Structure):
    _fields_ = [
        ("node_id", c_uint32),
        ("channels", c_uint32),
        ("data", POINTER(c_float)),
    ]


def bind(dll: ctypes.CDLL) -> ctypes.CDLL:
    """Declare every prototype. Unbound ctypes calls default to int and silently truncate pointers."""
    engine = c_void_p
    ring = c_void_p
    f_ptr = POINTER(c_float)

    def proto(name, restype, *argtypes):
        fn = getattr(dll, name)
        fn.restype = restype
        fn.argtypes = list(argtypes)

    proto("ts_abi_version", c_int32)
    proto("ts_build_info", c_char_p)
    proto("ts_engine_create", engine, c_uint32, c_uint32)
    proto("ts_engine_destroy", None, engine)
    proto("ts_engine_last_error", c_int32, engine, ctypes.c_char_p, c_int32)
    proto("ts_engine_apply_plan", c_int32, engine, POINTER(Plan))
    proto("ts_engine_set_route_gain", c_int32, engine, c_uint32, c_uint32, c_float)
    proto("ts_engine_set_route_muted", c_int32, engine, c_uint32, c_uint32, c_int32)
    proto("ts_engine_set_route_pan", c_int32, engine, c_uint32, c_uint32, c_float)
    proto("ts_engine_set_master_gain", c_int32, engine, c_float)
    proto("ts_engine_set_node_gain", c_int32, engine, c_uint32, c_float)
    proto("ts_engine_set_node_muted", c_int32, engine, c_uint32, c_int32)
    proto("ts_engine_set_channel_trim", c_int32, engine, c_uint32, c_uint32, c_float)
    proto("ts_engine_set_channel_inverted", c_int32, engine, c_uint32, c_uint32, c_int32)
    proto("ts_engine_set_insert_param", c_int32, engine, c_uint32, c_uint32, c_uint32, c_float)
    proto("ts_engine_get_insert_param", c_int32, engine, c_uint32, c_uint32, c_uint32, f_ptr)
    proto("ts_engine_set_insert_bypassed", c_int32, engine, c_uint32, c_uint32, c_int32)
    proto("ts_engine_get_insert_readout", c_int32, engine, c_uint32, c_uint32, f_ptr)
    proto("ts_engine_process", c_int32, engine, POINTER(PortBuffer), c_uint32, POINTER(PortBuffer), c_uint32, c_uint32)
    proto("ts_port_write", c_int32, engine, c_uint32, f_ptr, c_uint32)
    proto("ts_port_read", c_int32, engine, c_uint32, f_ptr, c_uint32)
    proto("ts_port_available", c_int32, engine, c_uint32)
    proto("ts_engine_get_stats", c_int32, engine, POINTER(Stats))
    proto("ts_engine_reset_stats", c_int32, engine)
    proto("ts_engine_get_meter", c_int32, engine, c_uint32, POINTER(Meter))
    proto("ts_engine_reset_meters", c_int32, engine)
    proto("ts_engine_poll_events", c_int32, engine, POINTER(Event), c_int32)
    proto("ts_ring_create", ring, c_uint32, c_uint32)
    proto("ts_ring_destroy", None, ring)
    proto("ts_ring_capacity", c_uint32, ring)
    proto("ts_ring_write", c_uint32, ring, f_ptr, c_uint32)
    proto("ts_ring_read", c_uint32, ring, f_ptr, c_uint32)
    proto("ts_ring_available", c_uint32, ring)
    return dll
