from typing import Optional

from pydantic import BaseModel

class DeviceInfo(BaseModel):
    id: int
    name: str
    type: str
    channels: int
    sample_rate: int
    is_asio: bool
    is_active: bool
    latency_ms: float

class CreateVirtualDeviceRequest(BaseModel):
    name: str
    channels: int = 2
    device_type: str  # "input" or "output"

class CreateRoutingRequest(BaseModel):
    source_id: int
    destination_id: int
    volume: float = 1.0

class SetVolumeRequest(BaseModel):
    source_id: int
    destination_id: int
    volume: float

class PerformanceStats(BaseModel):
    """
    Engine statistics. Fields that are not measured are None rather than 0.0, so
    clients can render "unknown" instead of showing an untaken measurement as a
    healthy-looking zero.
    """
    buffer_underruns: int
    nominal_latency_ms: float          # buffer_size / sample_rate — arithmetic, not measured
    cpu_usage: Optional[float] = None
    measured_latency_ms: Optional[float] = None
    audio_path_active: bool = False