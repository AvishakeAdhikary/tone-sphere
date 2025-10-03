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
    cpu_usage: float
    buffer_underruns: int
    latency_ms: float