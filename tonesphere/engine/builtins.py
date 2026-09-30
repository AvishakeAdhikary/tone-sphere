"""
The native engine's own processors — EQ, compressor, limiter, delay — as the control plane
describes them: each parameter's range, scale and unit, and the values last set.

The engine keeps a processor's state by (node, slot, type) across plan swaps, but a
processor that moves slot, or an engine rebuilt for a new rate or block size, starts from
its defaults. So the values live here too, and the host pushes them to the engine after
every plan it applies: what the user set survives anything that rebuilds the chain.

Parameters are offered in the same shape as a VST3 plugin's (`plugins.Parameter`: id,
title, display, units, normalized, step count), so one parameter panel drives both.
"""

import math
from dataclasses import dataclass

from tonesphere.native import _abi

EQ, COMPRESSOR, LIMITER, DELAY = _abi.INSERT_EQ, _abi.INSERT_COMPRESSOR, _abi.INSERT_LIMITER, _abi.INSERT_DELAY
EQ_BANDS = 8
EQ_TYPES = ('off', 'peaking', 'low_shelf', 'high_shelf', 'highpass', 'lowpass')


@dataclass(frozen=True)
class ParamSpec:
    index: int
    key: str                      # i18n key under 'builtin.param.'
    minimum: float
    maximum: float
    default: float
    unit: str = ''                # 'Hz', 'dB', 'ms', 'ratio', 'dBFS' (a linear value shown in dB), '%'
    log: bool = False
    choices: tuple[str, ...] = ()
    band: int = 0                 # EQ band, 1-based; 0 for none

    def to_normalized(self, value: float) -> float:
        if self.choices:
            return value / (len(self.choices) - 1)
        if self.log:
            return math.log(value / self.minimum) / math.log(self.maximum / self.minimum)
        return (value - self.minimum) / (self.maximum - self.minimum)

    def from_normalized(self, normalized: float) -> float:
        n = min(max(float(normalized), 0.0), 1.0)
        if self.choices:
            return float(round(n * (len(self.choices) - 1)))
        if self.log:
            return self.minimum * (self.maximum / self.minimum) ** n
        return self.minimum + n * (self.maximum - self.minimum)

    def clamp(self, value: float) -> float:
        return min(max(float(value), self.minimum), self.maximum)


def _eq_specs() -> tuple[ParamSpec, ...]:
    specs = []
    for b in range(EQ_BANDS):
        specs += [ParamSpec(b * 4, 'eq_type', 0, len(EQ_TYPES) - 1, 0, choices=EQ_TYPES, band=b + 1),
                  ParamSpec(b * 4 + 1, 'eq_frequency', 20.0, 20000.0, 1000.0, 'Hz', log=True, band=b + 1),
                  ParamSpec(b * 4 + 2, 'eq_q', 0.1, 10.0, 0.707, log=True, band=b + 1),
                  ParamSpec(b * 4 + 3, 'eq_gain', -24.0, 24.0, 0.0, 'dB', band=b + 1)]
    return tuple(specs)


# The defaults are the native constructors' (native/engine/dsp.h), so a fresh insert and
# its description agree before anything is pushed.
KINDS: dict[str, tuple[int, tuple[ParamSpec, ...]]] = {
    'eq': (EQ, _eq_specs()),
    'compressor': (COMPRESSOR, (
        ParamSpec(0, 'threshold', -60.0, 0.0, -18.0, 'dB'),
        ParamSpec(1, 'ratio', 1.0, 20.0, 4.0, 'ratio', log=True),
        ParamSpec(2, 'attack', 0.1, 200.0, 10.0, 'ms', log=True),
        ParamSpec(3, 'release', 5.0, 2000.0, 100.0, 'ms', log=True),
        ParamSpec(4, 'knee', 0.0, 24.0, 6.0, 'dB'),
        ParamSpec(5, 'makeup', 0.0, 24.0, 0.0, 'dB'),
    )),
    'limiter': (LIMITER, (
        ParamSpec(0, 'ceiling', 0.1, 1.0, 0.99, 'dBFS', log=True),
        ParamSpec(1, 'release', 5.0, 1000.0, 80.0, 'ms', log=True),
    )),
    'delay': (DELAY, (
        ParamSpec(0, 'time', 1.0, 2000.0, 250.0, 'ms', log=True),
        ParamSpec(1, 'feedback', 0.0, 0.95, 0.35, '%'),
        ParamSpec(2, 'mix', 0.0, 1.0, 0.25, '%'),
    )),
}
KIND_OF_TYPE = {native_type: kind for kind, (native_type, _specs) in KINDS.items()}


def display(spec: ParamSpec, value: float) -> str:
    from tonesphere.i18n import tr

    if spec.choices:
        return tr(f'builtin.eq_type.{spec.choices[int(round(value))]}')
    if spec.unit == 'Hz':
        return f"{value / 1000:.2f} kHz" if value >= 1000 else f"{value:.0f} Hz"
    if spec.unit == 'dB':
        return f"{value:+.1f} dB"
    if spec.unit == 'dBFS':
        return f"{20 * math.log10(max(value, 1e-9)):.1f} dBFS"
    if spec.unit == 'ms':
        return f"{value:.1f} ms" if value < 100 else f"{value:.0f} ms"
    if spec.unit == 'ratio':
        return f"{value:.1f}:1"
    if spec.unit == '%':
        return f"{value * 100:.0f} %"
    return f"{value:.3f}"


@dataclass(frozen=True)
class BuiltinParameter:
    """One parameter, in the shape `plugins.Parameter` has, so one panel shows both."""
    id: int
    title: str
    display: str
    units: str
    normalized: float
    step_count: int
    read_only: bool = False
    value: float = 0.0


class BuiltinInsert:
    """A built-in processor in a chain: its kind, whether it is bypassed, and its values."""

    is_builtin = True

    def __init__(self, kind: str, values: list[float] | None = None, bypassed: bool = False):
        if kind not in KINDS:
            raise ValueError(f"unknown built-in effect '{kind}': one of {', '.join(KINDS)}")
        self.kind = kind
        self.type, self.specs = KINDS[kind]
        self.values = [s.default for s in self.specs]
        for i, v in enumerate(values or []):
            if i < len(self.specs):
                self.values[i] = self.specs[i].clamp(v)
        self.bypassed = bypassed

    @property
    def name(self) -> str:
        from tonesphere.i18n import tr

        return tr(f'builtin.{self.kind}')

    def set_value(self, index: int, value: float) -> float:
        spec = self.specs[index]
        self.values[index] = float(round(value)) if spec.choices else spec.clamp(value)
        return self.values[index]

    def set_normalized(self, index: int, normalized: float) -> float:
        return self.set_value(index, self.specs[index].from_normalized(normalized))

    def parameters(self) -> list[BuiltinParameter]:
        from tonesphere.i18n import tr

        out = []
        for spec, value in zip(self.specs, self.values, strict=True):
            title = tr(f'builtin.param.{spec.key}')
            if spec.band:
                title = tr('builtin.band', band=spec.band, parameter=title)
            out.append(BuiltinParameter(
                id=spec.index, title=title, display=display(spec, value),
                units='' if spec.unit in ('ratio', 'dBFS', '%') else spec.unit,
                normalized=spec.to_normalized(value), step_count=len(spec.choices) - 1 if spec.choices else 0,
                value=value))
        return out

    def to_dict(self) -> dict:
        return {'kind': 'builtin', 'type': self.kind, 'bypassed': self.bypassed, 'values': list(self.values)}
