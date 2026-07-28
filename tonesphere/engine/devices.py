"""
Real audio device enumeration, via PortAudio.

Replaces the previous hand-written "drivers", each of which returned either a fixed
list of invented device names or nothing at all. Everything reported here comes from
PortAudio querying the operating system.

Host API notes (Windows), in the order we prefer them:

  WASAPI   Shared or exclusive. Exclusive bypasses the Windows mixer and reports
           2-3 ms on this hardware. The modern default.
  WDM-KS   Kernel streaming — talks to the driver below the audio engine, so latency
           is ASIO-class. Always exclusive; the device cannot be shared while open.
  ASIO     Absent from the PyPI PortAudio wheel: the ASIO SDK is Steinberg-licensed
           and cannot be redistributed, so wheels are built without it. Appears
           automatically if the user supplies an ASIO-enabled PortAudio.
  DirectSound / MME   Legacy, 90-120 ms. Compatibility fallback only.
"""

import platform
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)


class HostApi(str, Enum):
    """Audio backends, named as PortAudio reports them."""
    ASIO = "ASIO"
    WASAPI = "Windows WASAPI"
    WDMKS = "Windows WDM-KS"
    DIRECTSOUND = "Windows DirectSound"
    MME = "MME"
    ALSA = "ALSA"
    JACK = "JACK Audio Connection Kit"
    OSS = "OSS"
    COREAUDIO = "Core Audio"
    UNKNOWN = "Unknown"

    @classmethod
    def from_name(cls, name: str) -> "HostApi":
        for api in cls:
            if api.value.lower() == name.lower():
                return api
        return cls.UNKNOWN


# Preference order per platform. Latency first, compatibility last.
_PREFERENCE: Dict[str, Tuple[HostApi, ...]] = {
    "Windows": (HostApi.ASIO, HostApi.WASAPI, HostApi.WDMKS, HostApi.DIRECTSOUND, HostApi.MME),
    "Linux": (HostApi.JACK, HostApi.ALSA, HostApi.OSS),
    "Darwin": (HostApi.COREAUDIO, HostApi.JACK),
}

# Rates we probe when asking a device what it actually supports.
_CANDIDATE_RATES = (44100, 48000, 88200, 96000, 176400, 192000)


class AudioBackendUnavailable(RuntimeError):
    """PortAudio itself could not be loaded."""


@dataclass(frozen=True)
class DeviceInfo:
    """
    A real audio endpoint.

    `index` is PortAudio's device index and is the only identifier safe to hand back to
    PortAudio. It is not stable across a device being plugged or unplugged, so `key`
    exists for anything we persist.
    """
    index: int
    name: str
    host_api: HostApi
    host_api_name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: int
    default_low_input_latency_ms: float
    default_low_output_latency_ms: float
    default_high_input_latency_ms: float
    default_high_output_latency_ms: float
    is_default_input: bool = False
    is_default_output: bool = False
    supported_samplerates: Tuple[int, ...] = field(default_factory=tuple)

    @property
    def key(self) -> str:
        """Stable-ish identifier for configs: survives reindexing, not renaming."""
        return f"{self.host_api_name}::{self.name}"

    @property
    def can_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def can_output(self) -> bool:
        return self.max_output_channels > 0

    @property
    def is_duplex(self) -> bool:
        """Whether one stream can carry both directions — a single clock, so no drift."""
        return self.can_input and self.can_output

    @property
    def supports_exclusive(self) -> bool:
        """WDM-KS is always exclusive; WASAPI and ASIO can be."""
        return self.host_api in (HostApi.WASAPI, HostApi.WDMKS, HostApi.ASIO)

    def best_latency_ms(self, is_input: bool) -> float:
        return self.default_low_input_latency_ms if is_input else self.default_low_output_latency_ms

    def describe(self) -> str:
        directions = []
        if self.can_input:
            directions.append(f"{self.max_input_channels} in")
        if self.can_output:
            directions.append(f"{self.max_output_channels} out")
        return f"{self.name} [{self.host_api_name}] ({', '.join(directions)})"


def _import_sounddevice():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:
        # OSError happens when the PortAudio shared library is missing, which is
        # common on bare CI containers.
        raise AudioBackendUnavailable(f"PortAudio unavailable: {e}") from e
    return sd


def _clean_name(name: str) -> str:
    """
    Repair device names mangled by encoding.

    PortAudio returns Windows MME/DirectSound names in the system ANSI code page, but
    hands them over as if they were UTF-8, so "Intel® Smart Sound" arrives containing a
    replacement character. Salvage what we can rather than showing the user mojibake.
    """
    if '�' not in name:
        return name.strip()

    try:
        repaired = name.encode('utf-8', errors='replace').decode('utf-8')
        repaired = repaired.replace('�', '')
        # Collapse the double space left where the bad byte was.
        return ' '.join(repaired.split())
    except Exception:
        return name.strip()


def probe_samplerates(sd, index: int, is_input: bool, channels: int) -> Tuple[int, ...]:
    """
    Ask the device which rates it will actually accept.

    The old code published a hardcoded list including 192 kHz for every device. Here we
    ask, so the UI cannot offer a rate that fails the moment it is selected.
    """
    supported = []
    for rate in _CANDIDATE_RATES:
        try:
            if is_input:
                sd.check_input_settings(device=index, channels=channels, samplerate=rate)
            else:
                sd.check_output_settings(device=index, channels=channels, samplerate=rate)
            supported.append(rate)
        except Exception:
            continue
    return tuple(supported)


def enumerate_devices(probe_rates: bool = False) -> List[DeviceInfo]:
    """
    Every audio endpoint the OS reports.

    `probe_rates` opens and closes a trial stream per device per candidate rate, which
    takes a second or two in total — fine on a settings screen, too slow for a refresh
    on every UI tick, so it is off by default.
    """
    sd = _import_sounddevice()

    try:
        raw_devices = sd.query_devices()
        host_apis = sd.query_hostapis()
    except Exception as e:
        raise AudioBackendUnavailable(f"Could not query devices: {e}") from e

    default_input, default_output = _default_indices(sd)

    devices: List[DeviceInfo] = []
    for index, raw in enumerate(raw_devices):
        api_index = raw['hostapi']
        api_name = host_apis[api_index]['name'] if api_index < len(host_apis) else "Unknown"

        max_in = int(raw['max_input_channels'])
        max_out = int(raw['max_output_channels'])

        rates: Tuple[int, ...] = ()
        if probe_rates:
            rates = probe_samplerates(sd, index, max_in > 0, min(max_in or max_out, 2))

        devices.append(DeviceInfo(
            index=index,
            name=_clean_name(str(raw['name'])),
            host_api=HostApi.from_name(api_name),
            host_api_name=api_name,
            max_input_channels=max_in,
            max_output_channels=max_out,
            default_samplerate=int(raw['default_samplerate']),
            default_low_input_latency_ms=float(raw['default_low_input_latency']) * 1000.0,
            default_low_output_latency_ms=float(raw['default_low_output_latency']) * 1000.0,
            default_high_input_latency_ms=float(raw['default_high_input_latency']) * 1000.0,
            default_high_output_latency_ms=float(raw['default_high_output_latency']) * 1000.0,
            is_default_input=(index == default_input),
            is_default_output=(index == default_output),
            supported_samplerates=rates,
        ))

    return devices


def _default_indices(sd) -> Tuple[Optional[int], Optional[int]]:
    try:
        default = sd.default.device
        return (
            default[0] if default[0] != -1 else None,
            default[1] if default[1] != -1 else None,
        )
    except Exception:
        return (None, None)


def available_host_apis() -> List[HostApi]:
    """Backends PortAudio was compiled with and that have at least one device."""
    sd = _import_sounddevice()

    found = []
    for api in sd.query_hostapis():
        if api['devices']:
            resolved = HostApi.from_name(api['name'])
            if resolved not in found:
                found.append(resolved)
    return found


def preferred_host_api(available: Optional[List[HostApi]] = None) -> Optional[HostApi]:
    """
    The lowest-latency backend actually present.

    Contrast with the code this replaces, which selected ASIO whenever the registry key
    `HKLM\\SOFTWARE\\ASIO` existed — true on machines with zero ASIO drivers installed,
    after which enumeration returned nothing and no fallback was attempted. Here a
    backend has to have devices before it can be chosen.
    """
    if available is None:
        available = available_host_apis()

    for api in _PREFERENCE.get(platform.system(), ()):
        if api in available:
            return api

    return available[0] if available else None


def devices_for_host_api(api: HostApi, devices: Optional[List[DeviceInfo]] = None) -> List[DeviceInfo]:
    if devices is None:
        devices = enumerate_devices()
    return [d for d in devices if d.host_api == api]


def find_device(key_or_index, devices: Optional[List[DeviceInfo]] = None) -> Optional[DeviceInfo]:
    """Resolve a device by PortAudio index or by persisted key."""
    if devices is None:
        devices = enumerate_devices()

    if isinstance(key_or_index, int):
        for device in devices:
            if device.index == key_or_index:
                return device
        return None

    for device in devices:
        if device.key == key_or_index:
            return device

    # Fall back to a name match so a config survives the host API being renamed.
    for device in devices:
        if device.name == key_or_index:
            return device

    return None


def describe_backend() -> dict:
    """Backend summary for diagnostics and the UI's hardware bar."""
    sd = _import_sounddevice()

    try:
        devices = enumerate_devices()
        apis = available_host_apis()
        preferred = preferred_host_api(apis)

        return {
            'portaudio_version': sd.get_portaudio_version()[1],
            'host_apis': [api.value for api in apis],
            'preferred_host_api': preferred.value if preferred else None,
            'device_count': len(devices),
            'input_count': sum(1 for d in devices if d.can_input),
            'output_count': sum(1 for d in devices if d.can_output),
            'asio_available': HostApi.ASIO in apis,
        }
    except AudioBackendUnavailable as e:
        return {'error': str(e)}
