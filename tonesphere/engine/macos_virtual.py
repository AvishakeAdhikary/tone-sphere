"""
The macOS side of an OS-visible virtual device: a CoreAudio HAL plug-in.

Per `docs/VIRTUAL_AUDIO_DRIVER.md`, macOS needs an AudioServerPlugIn — a user-space
bundle in `/Library/Audio/Plug-Ins/HAL`, no kernel code and no paid certificate for a
local install. The bundle lives in `native/coreaudio-plugin/` and is built, signed,
installed and round-trip-tested by the `build-macos-plugin` CI job; this module is only
the Python half, and it deliberately does almost nothing:

Unlike the Linux backend (`linux_virtual.py`), which *creates* its endpoint on demand with
`pactl load-module`, nothing here can create anything. Installing a HAL plug-in needs
`sudo` and a `coreaudiod` restart that interrupts audio for the whole machine — not
something an audio application should do behind a REST call. So the device either exists
because a human ran `make install`, or it does not, and this module's whole job is to say
which, honestly, and to find the device by its exact name when it is there.

`MACOS_DEVICE_NAME` and `MACOS_DEVICE_UID` are the same strings the plug-in publishes
(`kDevice_Name` / `kDevice_UID` in `native/coreaudio-plugin/ToneSphereAudio.c`).
`tests/test_macos_virtual_device.py` reads that C file and fails if the two drift apart.
Matching on the exact name matters for a second reason: a user's separately installed
BlackHole must never be picked up and reported as ToneSphere's own device.

Everything here is macOS-only in effect but not in import: this module must import
cleanly on Windows/Linux (`tests/test_imports.py` walks every module and imports it), so
every OS check happens at call time, never at import time.
"""

import platform
from pathlib import Path

from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)

# Shared with native/coreaudio-plugin/ToneSphereAudio.c and Info.plist.
MACOS_DEVICE_NAME = "ToneSphere Audio"
MACOS_DEVICE_UID = "ToneSphereAudio_UID"
MACOS_BUNDLE_ID = "com.tonesphere.audio.driver"
MACOS_PLUGIN_BUNDLE_NAME = "ToneSphereAudio.driver"
MACOS_PLUGIN_INSTALL_DIR = "/Library/Audio/Plug-Ins/HAL"

# The value stored in `AudioEngine._system_virtual_devices[...]['backend']`, so one
# registry can hold both this and Track 2's Linux sinks and still know which is which.
MACOS_BACKEND = "macos-hal-plugin"

_INSTALL_HINT = (
    "build and install it with `make && make sign && make install` in "
    "native/coreaudio-plugin (see that directory's README.md)"
)


class MacOSVirtualDeviceError(RuntimeError):
    """
    A real failure attaching to the HAL device — never raised for a mere platform
    mismatch, which `hal_plugin_supported()` reports honestly instead.
    """


def hal_plugin_supported() -> bool:
    """
    Whether this OS can have a CoreAudio HAL plug-in at all.

    Only a platform check, deliberately: whether the bundle is actually *installed* is a
    separate, real condition reported by `hal_plugin_installed()`, and whether
    `coreaudiod` has actually published the device is a third one that only device
    enumeration can answer. The same "platform-only assertion" split
    `process_loopback_supported()` uses in `app_capture.py` and
    `virtual_sink_supported()` uses in `linux_virtual.py`.
    """
    return platform.system() == "Darwin"


def hal_plugin_path() -> Path:
    return Path(MACOS_PLUGIN_INSTALL_DIR) / MACOS_PLUGIN_BUNDLE_NAME


def hal_plugin_installed() -> bool:
    """
    Whether the bundle is on disk where `coreaudiod` looks for it.

    Not the same as "the device exists": `coreaudiod` only rescans that directory when it
    restarts, so a bundle can be installed and invisible until it does, and a device can
    still be live for a moment after the bundle is deleted. Both halves are reported
    separately by `plugin_status()` rather than collapsed into one optimistic flag.
    """
    if not hal_plugin_supported():
        return False
    return hal_plugin_path().is_dir()


def plugin_status() -> dict:
    """
    What is actually true about the plug-in on this machine, with nothing inferred.

    `device_visible` is None — not False — when this is not macOS or when PortAudio could
    not be asked, because "we did not look" and "we looked and it is not there" are
    different answers and this project reports a measurement it did not take as unknown.
    """
    supported = hal_plugin_supported()
    installed = hal_plugin_installed()

    visible: bool | None = None
    if supported:
        try:
            from tonesphere.engine.devices import enumerate_devices

            visible = any(MACOS_DEVICE_NAME in d.name for d in enumerate_devices())
        except Exception as e:
            logger.debug(f"Could not enumerate devices while checking the HAL plugin: {e}")

    return {
        "platform": platform.system(),
        "supported": supported,
        "installed": installed,
        "install_path": str(hal_plugin_path()),
        "device_name": MACOS_DEVICE_NAME,
        "device_uid": MACOS_DEVICE_UID,
        "device_visible": visible,
        "backend": MACOS_BACKEND,
    }


def unavailable_reason() -> str | None:
    """
    Why attaching would fail right now, in words a user can act on, or None if it should
    work. Kept in one place so the engine, the REST layer and the logs all say the same
    thing rather than each inventing their own wording.
    """
    if not hal_plugin_supported():
        return (
            f"The ToneSphere CoreAudio HAL plug-in is macOS-only; "
            f"unavailable on {platform.system()}"
        )
    if not hal_plugin_installed():
        return (
            f"The ToneSphere HAL plug-in is not installed at {hal_plugin_path()} — "
            f"{_INSTALL_HINT}"
        )
    return None
