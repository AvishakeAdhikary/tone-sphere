"""
Windows audio endpoints through the native WASAPI layer.

Endpoints are identified by their MMDevice endpoint ID — the string Windows itself uses,
stable across reboots, re-enumeration and a device being unplugged and plugged back in —
not by a PortAudio index, which changes whenever anything is added or removed.
"""

import ctypes
from dataclasses import dataclass

from tonesphere.native import NativeError, _abi, load


@dataclass(frozen=True)
class Endpoint:
    id: str
    name: str
    flow: str                  # 'render' | 'capture'
    state: int
    default_for: frozenset     # of 'console', 'multimedia', 'communications'
    mix_channels: int
    mix_sample_rate: int
    mix_bits: int
    mix_is_float: bool
    default_period_ms: float | None
    min_period_ms: float | None
    shared_min_period_frames: int | None  # IAudioClient3 low-latency shared mode; None if not offered
    raw_supported: bool                   # can bypass the driver's enhancement effects

    @property
    def is_default(self) -> bool:
        return 'console' in self.default_for


def _last_error() -> str:
    buffer = ctypes.create_string_buffer(512)
    load().ts_wasapi_last_error(buffer, len(buffer))
    return buffer.value.decode(errors='replace')


def _roles(bits: int) -> frozenset:
    names = {_abi.ROLE_CONSOLE: 'console', _abi.ROLE_MULTIMEDIA: 'multimedia',
             _abi.ROLE_COMMUNICATIONS: 'communications'}
    return frozenset(name for bit, name in names.items() if bits & bit)


def endpoints() -> list[Endpoint]:
    """Every active render and capture endpoint Windows reports."""
    dll = load()
    count = dll.ts_wasapi_enumerate(None, 0)
    if count < 0:
        raise NativeError(count, _last_error())
    buffer = (_abi.DeviceInfo * max(1, count + 8))()
    count = dll.ts_wasapi_enumerate(buffer, len(buffer))
    if count < 0:
        raise NativeError(count, _last_error())
    return [
        Endpoint(
            id=_abi.wide(d.id),
            name=_abi.wide(d.name),
            flow='render' if d.flow == _abi.FLOW_RENDER else 'capture',
            state=d.state,
            default_for=_roles(d.default_roles),
            mix_channels=d.mix_channels,
            mix_sample_rate=d.mix_sample_rate,
            mix_bits=d.mix_bits,
            mix_is_float=bool(d.mix_is_float),
            default_period_ms=d.default_period_hns / 10_000 if d.default_period_hns else None,
            min_period_ms=d.min_period_hns / 10_000 if d.min_period_hns else None,
            shared_min_period_frames=d.shared_min_period_frames or None,
            raw_supported=bool(d.raw_supported),
        )
        for d in buffer[:min(count, len(buffer))]
    ]


def default_endpoint(flow: str) -> Endpoint | None:
    return next((e for e in endpoints() if e.flow == flow and e.is_default), None)


def watch(enable: bool = True):
    """Start (or stop) receiving device arrival, removal, state and default-change events."""
    if load().ts_wasapi_watch(int(enable)) != _abi.OK:
        raise NativeError(_abi.ERR_BACKEND, _last_error())


_EVENT_KINDS = {
    _abi.DEVICE_EVENT_ADDED: 'added',
    _abi.DEVICE_EVENT_REMOVED: 'removed',
    _abi.DEVICE_EVENT_STATE_CHANGED: 'state_changed',
    _abi.DEVICE_EVENT_DEFAULT_CHANGED: 'default_changed',
    _abi.DEVICE_EVENT_LOST: 'events_lost',
}


def poll_events() -> list[dict]:
    buffer = (_abi.DeviceEvent * 256)()
    n = load().ts_wasapi_poll_events(buffer, len(buffer))
    if n < 0:
        raise NativeError(n, _last_error())
    return [
        {
            'kind': _EVENT_KINDS.get(e.kind, str(e.kind)),
            'id': _abi.wide(e.id),
            'flow': {_abi.FLOW_RENDER: 'render', _abi.FLOW_CAPTURE: 'capture'}.get(e.flow),
            'role': next(iter(_roles(e.role)), None),
            'state': e.state,
        }
        for e in buffer[:n]
    ]


@dataclass(frozen=True)
class StreamSpec:
    """
    One device stream bound to one engine node. `kind` is 'render', 'capture', 'loopback'
    (whatever a render endpoint plays) or 'process_loopback' (one process's audio).
    `exclusive` asks for exclusive mode; `allow_shared_fallback` lets a refusal fall back
    to shared mode, which the stream status then reports — it never happens silently.
    `raw` asks shared streams to bypass the driver's enhancement effects: a loudness
    equaliser or noise suppressor in a monitoring path changes what the player hears, so it
    is on by default, and the status says whether the endpoint honoured it.
    """
    node_id: int
    kind: str
    channels: int
    device_id: str = ''
    exclusive: bool = False
    allow_shared_fallback: bool = True
    process_id: int = 0
    include_process_tree: bool = True
    raw: bool = True
    period_frames: int = 0  # the device period to ask for; 0 = the engine's block

    def to_desc(self) -> _abi.StreamDesc:
        kinds = {'render': _abi.STREAM_RENDER, 'capture': _abi.STREAM_CAPTURE,
                 'loopback': _abi.STREAM_LOOPBACK, 'process_loopback': _abi.STREAM_PROCESS_LOOPBACK}
        flags = (_abi.STREAM_FLAG_ALLOW_SHARED_FALLBACK if self.allow_shared_fallback else 0) | \
                (_abi.STREAM_FLAG_INCLUDE_TREE if self.include_process_tree else 0) | \
                (_abi.STREAM_FLAG_RAW if self.raw else 0)
        return _abi.StreamDesc(
            _abi.to_wide(self.device_id, _abi.DEVICE_ID_CHARS),
            self.node_id,
            kinds[self.kind],
            _abi.SHARE_EXCLUSIVE if self.exclusive else _abi.SHARE_SHARED,
            flags,
            self.channels,
            self.process_id,
            self.period_frames,
        )
