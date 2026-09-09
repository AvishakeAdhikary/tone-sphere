
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
    # What actually put this endpoint here: 'hardware', an in-process 'in_process_bus',
    # or an OS-level 'os_virtual_endpoint' (a Linux sink from Track 2, or the macOS HAL
    # device from Track 3). Defaulted for any caller still constructing this model
    # without it, but `AudioEngine.get_devices()` always supplies it.
    origin: str = 'hardware'

class CreateVirtualDeviceRequest(BaseModel):
    name: str
    channels: int = 2
    device_type: str  # "input" or "output"

class CreateLinuxSinkRequest(BaseModel):
    """
    A PulseAudio/PipeWire `module-null-sink` ToneSphere creates and bridges into ALSA,
    so it enumerates as an ordinary PortAudio device. Linux-only; see
    `tonesphere/engine/linux_virtual.py`.
    """
    name: str
    channels: int = 2

class CreateRoutingRequest(BaseModel):
    source_id: int
    destination_id: int
    volume: float = 1.0

class SetVolumeRequest(BaseModel):
    source_id: int
    destination_id: int
    volume: float

class StartProcessCaptureRequest(BaseModel):
    """
    Capture one application's output by process id.

    `include_process_tree` is Windows' own distinction: a browser plays audio from child
    processes, so capturing only the pid you can see would get silence.
    """
    pid: int
    name: str | None = None
    include_process_tree: bool = True


class PerformanceStats(BaseModel):
    """
    Engine statistics. Fields that are not measured are None rather than 0.0, so
    clients can render "unknown" instead of showing an untaken measurement as a
    healthy-looking zero.
    """
    buffer_underruns: int
    nominal_latency_ms: float          # buffer_size / sample_rate — arithmetic, not measured
    cpu_usage: float | None = None
    measured_latency_ms: float | None = None
    audio_path_active: bool = False
