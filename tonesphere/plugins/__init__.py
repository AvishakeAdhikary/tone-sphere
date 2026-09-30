"""
VST3 plugins, as the control plane sees them.

`PluginInstance` is an open plugin in the native host (`native/vst3/vst3_host.cpp`); insert
it into an engine plan with `Insert(node, slot, VST3, plugin=instance.handle)`. Python never
touches the plugin's VST3 objects: parameters, state and editors all go through the native
plugin thread, and processing happens on the engine's audio thread.

Finding a `.vst3` file is discovery, not hosting. A plugin is hosted when it has been
opened here and has processed audio — see tests/native/test_vst3.py for what that means
in practice.
"""

import base64
import ctypes
from dataclasses import dataclass
from pathlib import Path

from tonesphere.native import NativeError, _abi, load

# ParameterInfo::ParameterFlags
CAN_AUTOMATE = 1 << 0
IS_READ_ONLY = 1 << 1
IS_WRAP_AROUND = 1 << 2
IS_LIST = 1 << 3
IS_HIDDEN = 1 << 4
IS_PROGRAM_CHANGE = 1 << 15
IS_BYPASS = 1 << 16

# IComponentHandler::restartComponent flags a host must act on.
RELOAD_COMPONENT = 1 << 0
IO_CHANGED = 1 << 1
LATENCY_CHANGED = 1 << 3


class PluginError(RuntimeError):
    """A plugin could not be loaded, opened or driven; the message says why."""


def _error() -> str:
    buffer = ctypes.create_string_buffer(1024)
    load().ts_vst3_last_error(buffer, len(buffer))
    return buffer.value.decode(errors='replace')


def _text(raw: bytes) -> str:
    return raw.decode('utf-8', errors='replace')


@dataclass(frozen=True)
class PluginInfo:
    path: str
    uid: str
    name: str
    vendor: str
    version: str
    category: str
    subcategories: str
    sdk_version: str
    is_audio_effect: bool

    @property
    def is_instrument(self) -> bool:
        """An instrument needs note input, and nothing in ToneSphere sends MIDI yet."""
        return 'Instrument' in self.subcategories.split('|')

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "PluginInfo":
        return cls(**d)


def classes_in(path: str) -> list[PluginInfo]:
    """
    Load a module in *this* process and list its classes. A module whose load code
    crashes can take the process down: untrusted plugins go through
    `tonesphere.plugins.scan`, which does this in a subprocess.
    """
    dll = load()
    capacity = 64
    buffer = (_abi.Vst3Class * capacity)()
    # Absolute, always: the SDK's loader uses an altered DLL search path, which Windows
    # defines only for absolute paths.
    path = str(Path(path).resolve())
    count = dll.ts_vst3_scan(path.encode('utf-8'), buffer, capacity)
    if count < 0:
        raise PluginError(_error())
    return [
        PluginInfo(path=str(path), uid=_text(c.uid), name=_text(c.name), vendor=_text(c.vendor),
                   version=_text(c.version), category=_text(c.category), subcategories=_text(c.subcategories),
                   sdk_version=_text(c.sdk_version), is_audio_effect=bool(c.is_audio_effect))
        for c in buffer[:min(count, capacity)]
    ]


@dataclass(frozen=True)
class PluginParameter:
    id: int
    title: str
    short_title: str
    units: str
    step_count: int            # 0 continuous, 1 toggle, n discrete steps
    default_normalized: float
    normalized: float          # 0..1, the VST3 native form
    plain: float               # the controller's conversion to the plugin's own units
    display: str               # the plugin's own text for the current value
    flags: int

    @property
    def automatable(self) -> bool:
        return bool(self.flags & CAN_AUTOMATE)

    @property
    def read_only(self) -> bool:
        return bool(self.flags & IS_READ_ONLY)

    @property
    def is_bypass(self) -> bool:
        return bool(self.flags & IS_BYPASS)


@dataclass(frozen=True)
class PluginState:
    """
    A plugin's own serialised state, exactly as its getState wrote it: opaque bytes, never
    a Python object. Base64 in presets, because YAML and JSON carry text.
    """
    component: bytes
    controller: bytes

    def to_dict(self) -> dict:
        return {'component': base64.b64encode(self.component).decode(),
                'controller': base64.b64encode(self.controller).decode()}

    @classmethod
    def from_dict(cls, d: dict) -> "PluginState":
        return cls(base64.b64decode(d.get('component', '')), base64.b64decode(d.get('controller', '')))


@dataclass(frozen=True)
class PluginStatus:
    crashed: bool
    fault: str
    latency_samples: int       # reported by the plugin, not measured
    blocks: int
    restart_flags: int


class PluginInstance:
    """
    One open plugin, set up for `channels` in and out at `sample_rate` with blocks of at
    most `max_block`. It must match the node it is inserted on; the engine refuses it
    otherwise.
    """

    def __init__(self, info: PluginInfo, sample_rate: int, max_block: int, channels: int):
        self.info = info
        self.sample_rate = sample_rate
        self.max_block = max_block
        self.channels = channels
        self._dll = load()
        handle = ctypes.c_uint32(0)
        result = self._dll.ts_vst3_open(str(Path(info.path).resolve()).encode('utf-8'), info.uid.encode(),
                                        sample_rate, max_block,
                                        channels, ctypes.byref(handle))
        if result != _abi.OK:
            raise PluginError(f"{info.name}: {_error()}")
        self.handle = handle.value
        # The host's bypass, not the plugin's own bypass parameter; the native host applies it.
        self.bypassed = False

    def close(self):
        if self.handle:
            self._dll.ts_vst3_close(self.handle)
            self.handle = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _check(self, result: int) -> int:
        if result < 0:
            raise PluginError(f"{self.info.name}: {_error()}")
        return result

    def parameters(self) -> list[PluginParameter]:
        count = self._check(self._dll.ts_vst3_param_count(self.handle))
        out = []
        raw = _abi.Vst3Param()
        for i in range(count):
            self._check(self._dll.ts_vst3_param_info(self.handle, i, ctypes.byref(raw)))
            out.append(PluginParameter(
                id=raw.id, title=_text(raw.title), short_title=_text(raw.short_title), units=_text(raw.units),
                step_count=raw.step_count, default_normalized=raw.default_normalized, normalized=raw.normalized,
                plain=raw.plain, display=_text(raw.display), flags=raw.flags))
        return out

    def set_parameter(self, param_id: int, normalized: float):
        """Takes effect on the audio thread's next block."""
        self._check(self._dll.ts_vst3_set_param(self.handle, param_id, float(normalized)))

    def state(self) -> PluginState:
        parts = []
        for which in (0, 1):
            size = self._check(self._dll.ts_vst3_get_state(self.handle, which, None, 0))
            buffer = ctypes.create_string_buffer(max(1, size))
            self._check(self._dll.ts_vst3_get_state(self.handle, which, buffer, size))
            parts.append(buffer.raw[:size])
        return PluginState(*parts)

    def restore(self, state: PluginState):
        """
        Hand the plugin its own saved bytes. A plugin that rejects them (another plugin's
        state, or an incompatible version) raises PluginError and keeps its current state.
        """
        component = ctypes.create_string_buffer(state.component, len(state.component))
        controller = ctypes.create_string_buffer(state.controller, len(state.controller))
        self._check(self._dll.ts_vst3_set_state(self.handle, component, len(state.component),
                                                controller, len(state.controller)))

    def status(self) -> PluginStatus:
        raw = _abi.Vst3Status()
        self._check(self._dll.ts_vst3_get_status(self.handle, ctypes.byref(raw)))
        return PluginStatus(bool(raw.crashed), _text(raw.fault), raw.latency, raw.blocks, raw.restart_flags)

    @property
    def latency_samples(self) -> int:
        return self.status().latency_samples

    @property
    def event_inputs(self) -> int:
        """Event input buses: 1 for an instrument, which plays notes; usually 0 for an effect."""
        return max(0, int(self._dll.ts_vst3_event_inputs(self.handle)))

    def send_midi(self, status: int, data1: int, data2: int = 0, sample_offset: int = 0):
        """One MIDI message; see `ts_vst3_send_midi`. Any thread may call this."""
        self._check(self._dll.ts_vst3_send_midi(self.handle, status & 0xFF, data1 & 0x7F, data2 & 0x7F,
                                                max(0, int(sample_offset))))

    def note_on(self, note: int, velocity: int = 100, channel: int = 0):
        self.send_midi(0x90 | (channel & 0x0F), note, velocity)

    def note_off(self, note: int, velocity: int = 0, channel: int = 0):
        self.send_midi(0x80 | (channel & 0x0F), note, velocity)

    def has_editor(self) -> bool:
        return bool(self._dll.ts_vst3_has_editor(self.handle))

    def open_editor(self):
        self._check(self._dll.ts_vst3_open_editor(self.handle))

    def close_editor(self):
        self._check(self._dll.ts_vst3_close_editor(self.handle))


__all__ = [
    'PluginError', 'PluginInfo', 'PluginInstance', 'PluginParameter', 'PluginState', 'PluginStatus',
    'classes_in', 'NativeError',
]
