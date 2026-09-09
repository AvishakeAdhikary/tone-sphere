"""
A Linux virtual sink, via PulseAudio/PipeWire's existing `module-null-sink` support.

Per `docs/VIRTUAL_AUDIO_DRIVER.md`, this is configuration, not code — no kernel work is
needed on Linux at all, unlike the Windows driver route that document explains is
deliberately not attempted. `pactl load-module module-null-sink` makes a sink other
Pulse/PipeWire clients can already select.

How ToneSphere itself reaches it, and why this is not what the original design intended
------------------------------------------------------------------------------------------
The first version of this module bridged the sink into ALSA as its own named PCM (a
`~/.asoundrc` block using ALSA's `pulse` plugin), on the theory that PortAudio's ALSA
enumeration walks every named PCM `snd_device_name_hint` reports, the way `aplay -L` does.
Verified wrong on real CI: `aplay -L` lists the custom PCM correctly (hint block and all),
but `sd.query_devices()` — the same PortAudio, asked directly — never includes it, only
ALSA's two built-in hinted entries, `pulse` and `default`. Five retries with a delay between
them ruled out a startup race; it is a permanent gap in what that PortAudio build enumerates,
not a timing one.

`pulse` itself, though, **is** a real, always-present PortAudio device — confirmed by its
own presence in that same device list — and it is simply "whatever PulseAudio's default
sink/source currently is". So instead of trying to make PortAudio see a new device, this
sets PulseAudio's default sink (and, for capture, default source) to point at the one we
created, and `create_linux_system_sink` in `core/engine.py` uses the existing `pulse` device
as-is. The `~/.asoundrc` bridge is kept alongside this, best-effort: it costs nothing, and
gives ALSA-native applications (anything that opens PCMs by name directly, bypassing
PortAudio) a way to select the sink by name even though PortAudio itself cannot enumerate it.

The real cost, stated plainly because `docs/VIRTUAL_AUDIO_DRIVER.md` said it would be if the
named-PCM route did not pan out: this changes what audio the *entire machine* uses by
default while active, not just what ToneSphere sees, and only one such sink can be "current"
this way — a second `create_linux_system_sink` call while one is already active would just
retarget the same single default. `remove_virtual_sink` restores whatever was default before,
so a crash mid-session is the only way this outlives ToneSphere's own process.

Everything here is Linux-only in effect but not in import: this module must still import
cleanly on Windows/macOS (`tests/test_imports.py` walks every module and imports it), so
every OS check happens at call time, never at import time.
"""

import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)

_PACTL_TIMEOUT_S = 5.0


class LinuxVirtualSinkError(RuntimeError):
    """
    A real failure creating, bridging or removing a sink — never raised for a mere
    platform mismatch, which `virtual_sink_supported()` reports honestly instead.
    """


@dataclass(frozen=True)
class VirtualSinkHandle:
    """Enough to find the sink again and tear down exactly what was created for it."""
    name: str
    sink_name: str
    module_id: int
    channels: int
    sample_rate: int
    monitor_name: str
    asoundrc_pcm: str
    # Whatever PulseAudio considered default before this sink took over, so teardown can
    # give the machine its previous audio routing back rather than leaving every app
    # pointed at a sink about to be unloaded. None means "no default was set" (unusual,
    # but not this module's job to invent one).
    previous_default_sink: str | None
    previous_default_source: str | None


def virtual_sink_supported() -> bool:
    """
    Whether this OS can even attempt a Pulse/PipeWire null-sink.

    Only a platform check, deliberately: whether a Pulse/PipeWire server is actually
    *running* is a separate, real failure surfaced by `create_virtual_sink` raising
    `LinuxVirtualSinkError`, not folded into this flag — the same "platform-only
    assertion" split `process_loopback_supported()` uses in `app_capture.py`.
    """
    return platform.system() == "Linux"


def _sanitize(name: str) -> str:
    """Pulse sink names and ALSA pcm names both reject anything but `[A-Za-z0-9_]`."""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return cleaned or "sink"


def _run_pactl(*args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["pactl", *args], capture_output=True, text=True, timeout=_PACTL_TIMEOUT_S,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as e:
        raise LinuxVirtualSinkError(f"pactl unavailable: {e}") from e


def _asoundrc_path() -> Path:
    return Path.home() / ".asoundrc"


def _markers(sink_name: str) -> tuple[str, str]:
    return f"# BEGIN TONESPHERE {sink_name}", f"# END TONESPHERE {sink_name}"


def _write_asoundrc_block(sink_name: str, pcm_name: str) -> None:
    """
    Append a clearly-delimited block bridging `pcm_name` (and `{pcm_name}_monitor`) to the
    null-sink through ALSA's `pulse` plugin. `remove_virtual_sink` deletes exactly this
    block by its markers, never touching anything else a user has in this file.

    The `hint` sub-block is not decoration. Confirmed against a real CI run: without it, a
    custom `pcm.NAME {}` is fully openable by name but invisible to `snd_device_name_hint`
    (the call PortAudio's ALSA enumeration is built on) — the file was there, the type was
    valid, and the device still never appeared in `sounddevice.query_devices()`, which only
    ever saw ALSA's two built-in hinted entries, `pulse` and `default`. `show on` is what
    actually asks to be listed; the `description` shows up as PortAudio's device name.
    """
    begin, end = _markers(sink_name)
    block = (
        f"\n{begin}\n"
        f'pcm.{pcm_name} {{\n'
        f'    type pulse\n'
        f'    device "{sink_name}"\n'
        f'    hint {{\n'
        f'        show on\n'
        f'        description "{pcm_name}"\n'
        f'    }}\n'
        f'}}\n'
        f'pcm.{pcm_name}_monitor {{\n'
        f'    type pulse\n'
        f'    device "{sink_name}.monitor"\n'
        f'    hint {{\n'
        f'        show on\n'
        f'        description "{pcm_name}_monitor"\n'
        f'    }}\n'
        f'}}\n'
        f"{end}\n"
    )

    path = _asoundrc_path()
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(existing + block, encoding="utf-8")


def _get_default_sink() -> str | None:
    result = _run_pactl("get-default-sink")
    name = result.stdout.strip()
    return name if result.returncode == 0 and name else None


def _get_default_source() -> str | None:
    result = _run_pactl("get-default-source")
    name = result.stdout.strip()
    return name if result.returncode == 0 and name else None


def _set_default_sink(name: str) -> None:
    result = _run_pactl("set-default-sink", name)
    if result.returncode != 0:
        detail = result.stderr.strip() or "no reason given"
        raise LinuxVirtualSinkError(f"pactl set-default-sink {name} failed: {detail}")


def _set_default_source(name: str) -> None:
    result = _run_pactl("set-default-source", name)
    if result.returncode != 0:
        detail = result.stderr.strip() or "no reason given"
        raise LinuxVirtualSinkError(f"pactl set-default-source {name} failed: {detail}")


def _remove_asoundrc_block(sink_name: str) -> None:
    path = _asoundrc_path()
    if not path.exists():
        return

    begin, end = _markers(sink_name)
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\n?" + re.escape(begin) + r".*?" + re.escape(end) + r"\n?", re.DOTALL
    )
    path.write_text(pattern.sub("", text), encoding="utf-8")


def create_virtual_sink(name: str, channels: int = 2, samplerate: int = 48000) -> VirtualSinkHandle:
    """
    Load a `module-null-sink` and bridge it into ALSA so PortAudio can see it.

    Raises `LinuxVirtualSinkError` on any real failure — no pactl, no Pulse/PipeWire
    server running, a rejected sink name — never returns a handle for a sink that was
    not actually loaded. The `~/.asoundrc` bridge is the one part of this design that
    could not be exercised on the machine that wrote it (see the plan): it needs a live
    PulseAudio server to prove for real, which is what the `linux_virtual_sink`-marked
    test and its new CI step are for.
    """
    if not virtual_sink_supported():
        raise LinuxVirtualSinkError(
            f"Linux virtual sinks need pactl and a running PulseAudio/PipeWire server; "
            f"unavailable on {platform.system()}"
        )

    sink_name = f"tonesphere_{_sanitize(name)}"
    pcm_name = sink_name

    result = _run_pactl(
        "load-module", "module-null-sink",
        f"sink_name={sink_name}",
        f"sink_properties=device.description={sink_name}",
        f"rate={samplerate}",
        f"channels={channels}",
    )

    if result.returncode != 0 or not result.stdout.strip().isdigit():
        detail = result.stderr.strip() or result.stdout.strip() or "no reason given"
        raise LinuxVirtualSinkError(f"pactl load-module module-null-sink failed: {detail}")

    module_id = int(result.stdout.strip())

    # Best-effort and non-fatal: gives ALSA-native applications a way to select the sink
    # by name, but PortAudio itself does not see it either way (see the module docstring),
    # so a failure here does not block the mechanism that actually matters below.
    try:
        _write_asoundrc_block(sink_name, pcm_name)
    except OSError as e:
        logger.debug(f"Could not write the ~/.asoundrc bridge for '{sink_name}': {e}")

    previous_sink = _get_default_sink()
    previous_source = _get_default_source()

    try:
        _set_default_sink(sink_name)
        _set_default_source(f"{sink_name}.monitor")
    except LinuxVirtualSinkError:
        _run_pactl("unload-module", str(module_id))
        raise

    logger.info(
        f"Created Linux virtual sink '{sink_name}' (pactl module {module_id}), "
        f"now the default sink/source (was '{previous_sink}'/'{previous_source}')"
    )

    return VirtualSinkHandle(
        name=name,
        sink_name=sink_name,
        module_id=module_id,
        channels=channels,
        sample_rate=samplerate,
        monitor_name=f"{sink_name}.monitor",
        asoundrc_pcm=pcm_name,
        previous_default_sink=previous_sink,
        previous_default_source=previous_source,
    )


def remove_virtual_sink(handle: VirtualSinkHandle) -> None:
    """
    Restore the previous default sink/source, remove the `~/.asoundrc` bridge, then unload
    the pactl module.

    Defaults are restored first: retargeting every Pulse client back to what it was before
    ToneSphere took over, before the sink they might currently be mid-write to disappears
    out from under them.
    """
    if not virtual_sink_supported():
        raise LinuxVirtualSinkError(
            f"Linux virtual sinks need pactl and a running PulseAudio/PipeWire server; "
            f"unavailable on {platform.system()}"
        )

    if handle.previous_default_sink:
        _set_default_sink(handle.previous_default_sink)
    if handle.previous_default_source:
        _set_default_source(handle.previous_default_source)

    try:
        _remove_asoundrc_block(handle.sink_name)
    except OSError as e:
        logger.debug(f"Could not remove the ~/.asoundrc bridge for '{handle.sink_name}': {e}")

    result = _run_pactl("unload-module", str(handle.module_id))
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no reason given"
        raise LinuxVirtualSinkError(f"pactl unload-module {handle.module_id} failed: {detail}")

    logger.info(f"Removed Linux virtual sink '{handle.sink_name}' (pactl module {handle.module_id})")
