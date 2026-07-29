from dataclasses import dataclass
from enum import Enum
from typing import Any


class DeviceType(Enum):
    PHYSICAL_INPUT = "physical_input"
    PHYSICAL_OUTPUT = "physical_output"
    VIRTUAL_INPUT = "virtual_input"
    VIRTUAL_OUTPUT = "virtual_output"

class RoutingState(Enum):
    DISABLED = 0
    ENABLED = 1
    MONITOR = 2  # Monitor only (listen but don't route)

class EffectType(Enum):
    EQ = "equalizer"
    COMPRESSOR = "compressor"
    REVERB = "reverb"
    DELAY = "delay"
    NOISE_GATE = "noise_gate"
    LIMITER = "limiter"

@dataclass
class AudioDevice:
    id: int
    name: str
    device_type: DeviceType
    channels: int
    sample_rate: int
    buffer_size: int
    is_asio: bool = False
    is_active: bool = False
    latency: float = 0.0

@dataclass
class RoutingConnection:
    source_id: int
    destination_id: int
    state: RoutingState
    volume: float = 1.0   # linear amplitude; 1.0 is unity (0 dB)
    muted: bool = False
    solo: bool = False
    # Pan and polarity belong to the route rather than the source, because one source can
    # legitimately sit centre in a headphone mix and hard left in a recording feed.
    pan: float = 0.0      # -1.0 hard left .. 1.0 hard right
    inverted: bool = False

@dataclass
class AudioEffect:
    effect_type: EffectType
    enabled: bool
    parameters: dict[str, Any]

@dataclass
class AudioChannel:
    id: int
    name: str
    device_id: int
    volume: float = 1.0
    muted: bool = False
    solo: bool = False
    effects: list[AudioEffect] = None

    def __post_init__(self):
        if self.effects is None:
            self.effects = []
