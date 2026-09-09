"""
Which applications are playing audio, and capturing what they play.

Replaces `utils/app_detector.py`, which guessed. Its heuristic matched a keyword list
containing "media", "game" and "eq" against every window title, and its fallback was
literally: if `OpenProcess` succeeds, assume the process has audio — true of essentially
every process on the machine. It also spawned a PowerShell subprocess *per PID* during
device enumeration, and its one "real" query used `New-Object -ComObject MMDeviceEnumerator`,
which is not a valid ProgID and therefore always threw.

Here the operating system is asked instead. On Windows the Audio Session API knows exactly
which processes hold a session, because that is what it is for.

Capture
-------
Windows 10 build 20348 and later can capture a single process's output through
`ActivateAudioInterfaceAsync` with `VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK`. That covers
most of why people install a virtual cable — pulling one app's audio somewhere else —
with no driver installed at all. It needs a C-level COM call PortAudio does not expose,
which `engine/process_capture.py` now makes; this module reports whether the platform
supports it and enumerates what there is to capture.
"""

import platform
import subprocess
from dataclasses import dataclass

from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)

# Per-process loopback landed in Windows 10 build 20348 (Server 2022 / 21H2 client).
_PROCESS_LOOPBACK_MIN_BUILD = 20348


@dataclass(frozen=True)
class AudioSession:
    """An application the OS reports as holding an audio session."""
    pid: int
    name: str
    display_name: str
    is_active: bool
    is_muted: bool
    volume: float

    def describe(self) -> str:
        state = "playing" if self.is_active else "idle"
        return f"{self.display_name} ({state})"


def _windows_sessions() -> list[AudioSession]:
    """
    Query the Windows Audio Session API through PowerShell.

    One subprocess for the whole query, not one per process. The old code ran a PowerShell
    invocation for every PID it had found, with a 0.5 second timeout each, inside device
    enumeration — on a machine with 200 processes that is a minute and a half of stalling
    every time the device list refreshed.
    """
    script = r'''
$ErrorActionPreference = 'SilentlyContinue'
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDeviceEnumerator {
    int EnumAudioEndpoints(int dataFlow, int stateMask, out IntPtr devices);
    int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice endpoint);
}

[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IMMDevice {
    int Activate(ref Guid iid, int clsCtx, IntPtr activationParams, out IntPtr iface);
}

[Guid("BFA971F1-4D5E-40BB-935E-967039BFBEE4"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioSessionManager2 {
    int NotUsed1(); int NotUsed2();
    int GetSessionEnumerator(out IAudioSessionEnumerator sessionEnum);
}

[Guid("E2F5BB11-0570-40CA-ACDD-3AA01277DEE8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioSessionEnumerator {
    int GetCount(out int count);
    int GetSession(int index, out IAudioSessionControl session);
}

[Guid("F4B1A599-7266-4319-A8CA-E70ACB11E8CD"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioSessionControl {
    int GetState(out int state);
    int GetDisplayName(out IntPtr name);
}

[Guid("BFB7FF88-7239-4FC9-8FA2-07C950BE9C6D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IAudioSessionControl2 {
    int GetState(out int state);
    int GetDisplayName(out IntPtr name);
    int SetDisplayName(string value, ref Guid ctx);
    int GetIconPath(out IntPtr path);
    int SetIconPath(string value, ref Guid ctx);
    int GetGroupingParam(out Guid group);
    int SetGroupingParam(ref Guid group, ref Guid ctx);
    int RegisterAudioSessionNotification(IntPtr n);
    int UnregisterAudioSessionNotification(IntPtr n);
    int GetSessionIdentifier(out IntPtr id);
    int GetSessionInstanceIdentifier(out IntPtr id);
    int GetProcessId(out uint pid);
    int IsSystemSoundsSession();
}

public class AudioSessions {
    [DllImport("ole32.dll")]
    static extern int CoCreateInstance(ref Guid clsid, IntPtr outer, int ctx,
                                       ref Guid iid, out IntPtr obj);

    public static uint[] GetPids() {
        Guid clsid = new Guid("BCDE0395-E52F-467C-8E3D-C4579291692E");
        Guid iidEnum = new Guid("A95664D2-9614-4F35-A746-DE8DB63617E6");
        Guid iidMgr  = new Guid("77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F");

        IntPtr raw;
        if (CoCreateInstance(ref clsid, IntPtr.Zero, 1, ref iidEnum, out raw) != 0)
            return new uint[0];

        var enumerator = (IMMDeviceEnumerator)Marshal.GetObjectForIUnknown(raw);
        IMMDevice device;
        if (enumerator.GetDefaultAudioEndpoint(0, 1, out device) != 0)
            return new uint[0];

        IntPtr mgrPtr;
        if (device.Activate(ref iidMgr, 1, IntPtr.Zero, out mgrPtr) != 0)
            return new uint[0];

        var manager = (IAudioSessionManager2)Marshal.GetObjectForIUnknown(mgrPtr);
        IAudioSessionEnumerator sessions;
        if (manager.GetSessionEnumerator(out sessions) != 0)
            return new uint[0];

        int count;
        sessions.GetCount(out count);

        var result = new System.Collections.Generic.List<uint>();
        for (int i = 0; i < count; i++) {
            IAudioSessionControl control;
            if (sessions.GetSession(i, out control) != 0) continue;
            var control2 = control as IAudioSessionControl2;
            if (control2 == null) continue;
            uint pid;
            if (control2.GetProcessId(out pid) == 0 && pid != 0) {
                int state;
                control2.GetState(out state);
                result.Add(pid);
                result.Add((uint)state);
            }
        }
        return result.ToArray();
    }
}
"@
$pairs = [AudioSessions]::GetPids()
for ($i = 0; $i -lt $pairs.Length; $i += 2) {
    $procId = $pairs[$i]
    $state  = $pairs[$i + 1]
    $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if ($p) {
        "{0}|{1}|{2}" -f $procId, $p.ProcessName, $state
    }
}
'''

    try:
        result = subprocess.run(
            ['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
            capture_output=True, text=True, timeout=8,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug(f"Audio session query failed: {e}")
        return []

    if result.returncode != 0:
        logger.debug(f"Audio session query returned {result.returncode}")
        return []

    sessions: list[AudioSession] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split('|')
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            name = parts[1]
            # AudioSessionState: 0 inactive, 1 active, 2 expired.
            state = int(parts[2])
        except ValueError:
            continue

        if state == 2:
            continue

        sessions.append(AudioSession(
            pid=pid,
            name=name,
            display_name=_friendly_name(name),
            is_active=(state == 1),
            is_muted=False,
            volume=1.0,
        ))

    return sessions


def _linux_sessions() -> list[AudioSession]:
    """PulseAudio and PipeWire both answer `pactl list sink-inputs`."""
    try:
        result = subprocess.run(
            ['pactl', 'list', 'sink-inputs'],
            capture_output=True, text=True, timeout=4,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []

    if result.returncode != 0:
        return []

    sessions: list[AudioSession] = []
    pid: int | None = None
    name: str | None = None
    corked = False

    for line in result.stdout.splitlines():
        stripped = line.strip()

        if stripped.startswith('Sink Input #'):
            if pid is not None and name:
                sessions.append(AudioSession(
                    pid=pid, name=name, display_name=_friendly_name(name),
                    is_active=not corked, is_muted=False, volume=1.0,
                ))
            pid, name, corked = None, None, False

        elif 'application.process.id' in stripped:
            try:
                pid = int(stripped.split('=')[1].strip().strip('"'))
            except (IndexError, ValueError):
                pass
        elif 'application.name' in stripped:
            try:
                name = stripped.split('=')[1].strip().strip('"')
            except IndexError:
                pass
        elif stripped.startswith('Corked:'):
            corked = 'yes' in stripped.lower()

    if pid is not None and name:
        sessions.append(AudioSession(
            pid=pid, name=name, display_name=_friendly_name(name),
            is_active=not corked, is_muted=False, volume=1.0,
        ))

    return sessions


def _friendly_name(process_name: str) -> str:
    """Tidy an executable name for display. Cosmetic only — never used to detect audio."""
    cleaned = process_name.replace('.exe', '').strip()

    known = {
        'chrome': 'Google Chrome', 'msedge': 'Microsoft Edge', 'firefox': 'Firefox',
        'spotify': 'Spotify', 'vlc': 'VLC', 'discord': 'Discord', 'slack': 'Slack',
        'teams': 'Microsoft Teams', 'ms-teams': 'Microsoft Teams', 'zoom': 'Zoom',
        'obs64': 'OBS Studio', 'obs': 'OBS Studio', 'reaper': 'REAPER',
        'ableton live': 'Ableton Live', 'guitar rig': 'Guitar Rig',
        'audacity': 'Audacity', 'foobar2000': 'foobar2000',
    }

    lowered = cleaned.lower()
    if lowered in known:
        return known[lowered]

    return cleaned[:1].upper() + cleaned[1:] if cleaned else process_name


def list_audio_sessions() -> list[AudioSession]:
    """
    Applications the OS reports as holding an audio session.

    Empty means either nothing is playing or the query is unsupported here — not that
    every running process might have audio, which is what the old heuristic concluded.
    """
    system = platform.system()

    if system == "Windows":
        return _windows_sessions()
    if system == "Linux":
        return _linux_sessions()

    # macOS has no supported per-application enumeration without a helper. Reporting
    # nothing is honest; guessing from process names was not.
    return []


def process_loopback_supported() -> bool:
    """
    Whether this OS can capture a single application's output without a driver.

    Windows 10 build 20348 and later, through `ActivateAudioInterfaceAsync` with
    `VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK`.
    """
    if platform.system() != "Windows":
        return False

    try:
        build = int(platform.version().split('.')[-1])
    except (ValueError, IndexError):
        return False

    return build >= _PROCESS_LOOPBACK_MIN_BUILD


def system_loopback_devices() -> list[str]:
    """
    Loopback devices PortAudio can already capture from — whole-system output capture.

    WASAPI exposes each render endpoint as a capture device for loopback, which records
    everything playing rather than one application. That works today, unlike per-process
    capture, so the two are reported separately instead of being conflated.
    """
    try:
        import sounddevice as sd
    except (ImportError, OSError):
        return []

    if platform.system() != "Windows":
        return []

    found = []
    try:
        host_apis = sd.query_hostapis()
        for device in sd.query_devices():
            api_name = host_apis[device['hostapi']]['name']
            if 'WASAPI' in api_name and device['max_output_channels'] > 0:
                found.append(str(device['name']))
    except Exception as e:
        logger.debug(f"Could not enumerate loopback devices: {e}")

    return found


def capture_status() -> dict:
    """
    What application capture can and cannot do here.

    `process_loopback_implemented` tracks `process_loopback_supported()` exactly, because
    `engine/process_capture.py` implements the whole of what the platform offers — where
    Windows can do it, ToneSphere can. It is deliberately still two separate fields: on
    Linux and macOS the answer is no, and reporting a blanket "implemented" on a machine
    that cannot do it would be the same lie in the other direction.
    """
    sessions = list_audio_sessions()
    supported = process_loopback_supported()

    return {
        'platform': platform.system(),
        'sessions': [
            {'pid': s.pid, 'name': s.display_name, 'active': s.is_active}
            for s in sessions
        ],
        'session_count': len(sessions),
        'system_loopback_available': bool(system_loopback_devices()),
        'process_loopback_supported': supported,
        'process_loopback_implemented': supported,
        'note': (
            "Per-application capture uses ActivateAudioInterfaceAsync with "
            "VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK (Windows 10 build 20348+). "
            "Whole-system loopback works through PortAudio on any Windows."
            if supported else
            "Per-application capture needs ActivateAudioInterfaceAsync with "
            "VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, which exists only on Windows 10 "
            "build 20348 and later. This machine cannot do it."
        ),
    }
