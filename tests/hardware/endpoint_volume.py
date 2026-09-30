"""
An output endpoint's master volume and mute, for the acoustic round-trip test only: it may
raise the laptop's speaker volume for the few seconds of a sweep, and always puts back
exactly what it found (`held_at`). IAudioEndpointVolume through ctypes, with the COM helpers
of `tonesphere/engine/wasapi_com.py`.
"""

import contextlib
import ctypes

from tonesphere.engine import wasapi_com as com

CLSID_MMDEVICE_ENUMERATOR = com.guid("BCDE0395-E52F-467C-8E3D-C4579291692E")
IID_IMMDEVICE_ENUMERATOR = com.guid("A95664D2-9614-4F35-A746-DE8DB63617E6")
IID_IAUDIO_ENDPOINT_VOLUME = com.guid("5CDF2C82-841E-4546-9722-0CF74078229A")
CLSCTX_ALL = 23


def _volume_interface(endpoint_id: str) -> int:
    ole32 = ctypes.WinDLL('ole32.dll')
    enumerator = ctypes.c_void_p()
    hr = ole32.CoCreateInstance(ctypes.byref(CLSID_MMDEVICE_ENUMERATOR), None, CLSCTX_ALL,
                                ctypes.byref(IID_IMMDEVICE_ENUMERATOR), ctypes.byref(enumerator))
    if com.failed(hr & 0xFFFFFFFF):
        raise OSError(f"CoCreateInstance(MMDeviceEnumerator): {com.describe_hresult(hr)}")
    device = ctypes.c_void_p()
    try:
        get_device = com.bind(enumerator, 5, com.HRESULT, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p))
        hr = get_device(enumerator, endpoint_id, ctypes.byref(device))
        if com.failed(hr):
            raise OSError(f"GetDevice: {com.describe_hresult(hr)}")
        volume = ctypes.c_void_p()
        activate = com.bind(device, 3, com.HRESULT, ctypes.POINTER(com.GUID), ctypes.c_ulong, ctypes.c_void_p,
                            ctypes.POINTER(ctypes.c_void_p))
        hr = activate(device, ctypes.byref(IID_IAUDIO_ENDPOINT_VOLUME), CLSCTX_ALL, None, ctypes.byref(volume))
        if com.failed(hr):
            raise OSError(f"Activate(IAudioEndpointVolume): {com.describe_hresult(hr)}")
        return volume.value
    finally:
        com.release(device)
        com.release(enumerator)


def get(endpoint_id: str) -> tuple[float, bool]:
    """(volume scalar 0..1, muted)."""
    owned = com.co_initialize()
    volume = _volume_interface(endpoint_id)
    try:
        level, mute = ctypes.c_float(), ctypes.c_int()
        com.bind(volume, 9, com.HRESULT, ctypes.POINTER(ctypes.c_float))(volume, ctypes.byref(level))
        com.bind(volume, 15, com.HRESULT, ctypes.POINTER(ctypes.c_int))(volume, ctypes.byref(mute))
        return level.value, bool(mute.value)
    finally:
        com.release(volume)
        if owned:
            com.co_uninitialize()


def set(endpoint_id: str, level: float, muted: bool):   # noqa: A001 - the counterpart of get
    owned = com.co_initialize()
    volume = _volume_interface(endpoint_id)
    try:
        com.bind(volume, 7, com.HRESULT, ctypes.c_float, ctypes.c_void_p)(volume, float(level), None)
        com.bind(volume, 14, com.HRESULT, ctypes.c_int, ctypes.c_void_p)(volume, int(muted), None)
    finally:
        com.release(volume)
        if owned:
            com.co_uninitialize()


@contextlib.contextmanager
def held_at(endpoint_id: str, level: float):
    """Unmuted at exactly `level` inside the block; as it was found, afterwards, whatever happens."""
    before = get(endpoint_id)
    set(endpoint_id, level, False)
    try:
        yield before
    finally:
        set(endpoint_id, *before)
