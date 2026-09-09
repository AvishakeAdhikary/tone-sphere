"""
Track 2 — the Linux virtual sink.

Per `docs/VIRTUAL_AUDIO_DRIVER.md` and the plan this implements, an OS-visible sink on
Linux is "configuration, not code": `pactl load-module module-null-sink` plus a small
`~/.asoundrc` bridge so PortAudio's ALSA backend enumerates it like any sound card. That
bridge is the one part of this design that could not be exercised on the machine that
wrote it — it needs a live PulseAudio/PipeWire server to prove for real.

So this file has two honesty tiers, matching `tests/test_process_capture.py`'s split for
Track 1:

- Everything below `TestRealLinuxSink` runs on every platform, including this one (Windows),
  and asserts the *honest* behaviour off Linux: a platform mismatch is refused with a real
  reason, never silently ignored or given a fake success.
- `TestRealLinuxSink` is the actual proof — create a sink, have the OS independently confirm
  it, route a known tone through it, capture its `.monitor` back with a second raw PortAudio
  stream, and tear it down — and is marked `linux_virtual_sink` rather than `hardware`,
  because unlike a real sound card, CI *can* give it a dummy Pulse server (see the setup step
  added to `.github/workflows/ci.yml`). It is skipped outright off Linux; nothing here claims
  to prove the OS-level behaviour anywhere but there.
"""

import math
import subprocess
import sys

import numpy as np
import pytest

RATE = 48000
BLOCK = 256


def sine(frames: int, freq: float = 1000.0, rate: int = RATE,
         amplitude: float = 0.5, channels: int = 2, phase: float = 0.0) -> np.ndarray:
    """Same tone helper as `tests/test_engine_audio.py` and `tests/test_process_capture.py`."""
    t = (np.arange(frames, dtype=np.float64) + phase) / rate
    wave = (amplitude * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)
    return np.repeat(wave.reshape(-1, 1), channels, axis=1)


def dominant_frequency(block: np.ndarray, rate: int = RATE) -> float:
    mono = block[:, 0] if block.ndim > 1 else block
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    return float(np.fft.rfftfreq(len(mono), 1.0 / rate)[int(np.argmax(spectrum))])


def _run(coroutine):
    """Call a REST endpoint function directly, the same shortcut `test_process_capture.py` uses."""
    import asyncio
    return asyncio.run(coroutine)


class TestPlatformGuardIsHonest:
    """
    `virtual_sink_supported()` and the two pactl-driving functions must all agree with
    `platform.system()` — the same "platform-only assertion, testable everywhere" pattern
    `tests/test_honesty.py::TestCaptureStatusMatchesWhatIsImplemented` already holds
    Track 1 to for `process_loopback_supported()`.
    """

    def test_supported_flag_matches_the_real_platform(self):
        import platform

        from tonesphere.engine.linux_virtual import virtual_sink_supported

        assert virtual_sink_supported() == (platform.system() == "Linux")

    def test_create_refuses_off_linux_with_a_real_reason(self):
        import platform

        from tonesphere.engine.linux_virtual import (
            LinuxVirtualSinkError,
            create_virtual_sink,
            virtual_sink_supported,
        )

        if virtual_sink_supported():
            pytest.skip("this machine is Linux; see TestRealLinuxSink instead")

        with pytest.raises(LinuxVirtualSinkError) as raised:
            create_virtual_sink("test-sink")

        assert platform.system() in str(raised.value)

    def test_remove_refuses_off_linux_with_a_real_reason(self):
        from tonesphere.engine.linux_virtual import (
            LinuxVirtualSinkError,
            VirtualSinkHandle,
            virtual_sink_supported,
        )

        if virtual_sink_supported():
            pytest.skip("this machine is Linux; see TestRealLinuxSink instead")

        handle = VirtualSinkHandle(
            name="test-sink", sink_name="tonesphere_test_sink", module_id=0,
            channels=2, sample_rate=RATE, monitor_name="tonesphere_test_sink.monitor",
            asoundrc_pcm="tonesphere_test_sink",
        )

        from tonesphere.engine.linux_virtual import remove_virtual_sink

        with pytest.raises(LinuxVirtualSinkError):
            remove_virtual_sink(handle)

    def test_module_imports_with_no_linux_only_top_level_import(self):
        """
        `tests/test_imports.py` already walks every module and would catch a hard import
        failure, but this pins down the actual reason it must never happen: nothing in
        this module may assume Linux at import time, only at call time.
        """
        import importlib

        module = importlib.import_module("tonesphere.engine.linux_virtual")
        assert hasattr(module, "create_virtual_sink")
        assert hasattr(module, "remove_virtual_sink")
        assert hasattr(module, "virtual_sink_supported")


class TestEngineRefusesHonestlyOffLinux:
    """
    `AudioEngine.create_linux_system_sink`/`remove_linux_system_sink`, exercised for real
    on whatever platform this test runs on — not mocked, so on this Windows machine the
    real refusal path actually runs end to end.
    """

    def test_create_returns_none_rather_than_a_placeholder_id(self):
        import platform

        from tonesphere.core.engine import AudioEngine
        from tonesphere.engine.linux_virtual import virtual_sink_supported

        if virtual_sink_supported():
            pytest.skip(f"this machine ({platform.system()}) is Linux; "
                        f"see TestRealLinuxSink instead")

        engine = AudioEngine()
        engine.initialize()

        try:
            assert engine.create_linux_system_sink("test-sink") is None
            assert engine._system_virtual_devices == {}
        finally:
            engine.cleanup()

    def test_remove_reports_an_unknown_id_rather_than_success(self):
        from tonesphere.core.engine import AudioEngine

        engine = AudioEngine()
        engine.initialize()

        try:
            assert engine.remove_linux_system_sink(999_999) is False
        finally:
            engine.cleanup()

    def test_cleanup_tears_down_every_tracked_sink(self):
        """
        No OS resource this engine created is left behind by `cleanup()` — same
        obligation `_stop_all_process_captures` already holds for capture threads,
        applied here to a `pactl load-module` instead of a thread.
        """
        from tonesphere.core.engine import AudioEngine

        engine = AudioEngine()
        engine.initialize()

        removed = []
        engine._system_virtual_devices[123] = {"name": "fake", "handle": None}
        engine.remove_linux_system_sink = lambda device_id: removed.append(device_id) or True

        engine.cleanup()

        assert removed == [123]


class TestOriginLabelling:
    """
    The mirror image of `tests/test_honesty.py::TestBusesAreNotSystemDevices`: a device
    tracked in `_system_virtual_devices` must be labelled `os_virtual_endpoint`, never
    `in_process_bus` or bare `hardware`, regardless of platform — this is pure dict
    bookkeeping in `get_devices()`, not a Linux syscall, so it is fully provable here.
    """

    def test_tracked_device_is_labelled_os_virtual_endpoint(self):
        from tonesphere.core.engine import AudioEngine

        engine = AudioEngine()
        engine.initialize()

        # `get_devices()` filters to the active backend by default (the same device
        # enumerated once per host API), so the relabelled id has to be one that
        # actually survives that filter, not merely any key of `_device_by_id`.
        visible_ids = [d["id"] for d in engine.get_devices()]
        if not visible_ids:
            pytest.skip("no enumerated devices on this machine to relabel")

        target_id = visible_ids[0]
        engine._system_virtual_devices[target_id] = {"name": "fake sink", "handle": None}

        origins = {d["id"]: d["origin"] for d in engine.get_devices()}

        assert origins[target_id] == "os_virtual_endpoint"

    def test_bus_is_never_labelled_os_virtual_endpoint(self):
        from tonesphere.core.engine import AudioEngine

        engine = AudioEngine()
        bus_id = engine.create_virtual_input("a bus", channels=2)

        origins = {d["id"]: d["origin"] for d in engine.get_devices()}

        assert origins[bus_id] == "in_process_bus"

    def test_untracked_hardware_device_is_labelled_hardware(self):
        from tonesphere.core.engine import AudioEngine

        engine = AudioEngine()
        engine.initialize()

        devices = engine.get_devices()
        if not devices:
            pytest.skip("no enumerated devices on this machine")

        for entry in devices:
            if entry["id"] not in engine._system_virtual_devices:
                assert entry["origin"] == "hardware"


class TestExposureIsActuallyWired:
    """
    The CLI and REST surface, proved reachable rather than assumed — same pattern
    `test_process_capture.py::TestExposureIsActuallyWired` uses for Track 1.
    """

    def test_cli_dispatches_the_linuxsink_command(self):
        import inspect

        from tonesphere.cli.interface import AudioEngineCLI

        source = inspect.getsource(AudioEngineCLI.run_interactive_mode)

        assert "linuxsink" in source
        assert "manage_linux_sink" in source

    def test_cli_reports_the_honest_refusal_off_linux(self, capsys):
        import platform

        from tonesphere.cli.interface import AudioEngineCLI
        from tonesphere.engine.linux_virtual import virtual_sink_supported

        if virtual_sink_supported():
            pytest.skip(f"this machine ({platform.system()}) is Linux")

        cli = AudioEngineCLI()
        cli.initialize_engine()

        try:
            cli.manage_linux_sink()
        finally:
            if cli.engine:
                cli.engine.cleanup()

        output = capsys.readouterr().out
        assert "unavailable" in output.lower()
        assert platform.system() in output

    def test_api_create_refuses_off_linux_with_a_reason(self):
        import platform

        from fastapi import HTTPException

        import tonesphere.api.server as server
        from tonesphere.api.models import CreateLinuxSinkRequest
        from tonesphere.core.engine_factory import UnifiedAudioEngine
        from tonesphere.engine.linux_virtual import virtual_sink_supported

        if virtual_sink_supported():
            pytest.skip(f"this machine ({platform.system()}) is Linux")

        engine = UnifiedAudioEngine()
        engine.initialize()
        server.audio_engine = engine

        try:
            with pytest.raises(HTTPException) as raised:
                _run(server.create_linux_virtual_sink(
                    CreateLinuxSinkRequest(name="test-sink")))
        finally:
            server.audio_engine = None
            engine.cleanup()

        assert raised.value.status_code == 400
        assert platform.system() in raised.value.detail

    def test_api_remove_reports_an_unknown_device_rather_than_success(self):
        from fastapi import HTTPException

        import tonesphere.api.server as server
        from tonesphere.core.engine_factory import UnifiedAudioEngine

        engine = UnifiedAudioEngine()
        engine.initialize()
        server.audio_engine = engine

        try:
            with pytest.raises(HTTPException) as raised:
                _run(server.remove_linux_virtual_sink(999_999))
        finally:
            server.audio_engine = None
            engine.cleanup()

        assert raised.value.status_code == 404


@pytest.mark.linux_virtual_sink
@pytest.mark.skipif(sys.platform != "linux", reason="needs Linux with a PulseAudio/PipeWire server")
class TestRealLinuxSink:
    """
    The proof: a sink the OS itself agrees exists, carrying a real tone.

    Run in CI on Linux only (see `.github/workflows/ci.yml`'s PulseAudio setup step), or
    locally on Linux with `uv run pytest -m linux_virtual_sink`.
    """

    def _pactl_sink_names(self) -> list[str]:
        result = subprocess.run(
            ["pactl", "list", "short", "sinks"], capture_output=True, text=True, timeout=5,
        )
        assert result.returncode == 0, f"pactl itself failed: {result.stderr}"
        return [line.split("\t")[1] for line in result.stdout.splitlines() if line.strip()]

    def test_create_route_capture_teardown(self):
        import time

        import sounddevice as sd

        from tonesphere.core.engine import AudioEngine

        engine = AudioEngine(sample_rate=RATE, buffer_size=BLOCK, exclusive=False)
        engine.initialize()

        device_id = engine.create_linux_system_sink("citest", channels=2)
        assert device_id is not None, "sink was not created — see the log for why"

        try:
            handle = engine._system_virtual_devices[device_id]["handle"]

            # The OS itself, not just our own bookkeeping, has to see it.
            assert handle.sink_name in self._pactl_sink_names()

            origins = {d["id"]: d["origin"] for d in engine.get_devices()}
            assert origins[device_id] == "os_virtual_endpoint"

            tone_bus = engine.create_virtual_input("tone", channels=2)
            success, message = engine.create_routing(tone_bus, device_id, volume=0.8)
            assert success, message

            engine.start_engine()

            monitor_pcm = f"{handle.asoundrc_pcm}_monitor"
            devices = sd.query_devices()
            monitor_index = next(
                (i for i, d in enumerate(devices) if monitor_pcm in d["name"]), None
            )
            assert monitor_index is not None, (
                f"'{monitor_pcm}' never appeared as a PortAudio input device"
            )

            captured: list[np.ndarray] = []

            def on_block(indata, frames, time_info, status):
                captured.append(indata.copy())

            with sd.InputStream(
                device=monitor_index, channels=2, samplerate=RATE,
                blocksize=BLOCK, dtype="float32", callback=on_block,
            ):
                phase = 0
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    phase += engine.write_to_bus(tone_bus, sine(BLOCK, amplitude=0.8, phase=phase))
                    time.sleep(BLOCK / RATE / 2)
                time.sleep(0.2)

            assert captured, "the monitor delivered nothing at all"
            audio = np.concatenate(captured)

            middle = audio[audio.shape[0] // 3: audio.shape[0] // 3 + 16384]
            assert middle.shape[0] == 16384, "not enough audio captured to measure"

            rms = float(np.sqrt((middle[:, 0] ** 2).mean()))
            assert rms > 0.01, f"captured RMS {rms:.5f} is silence, not a tone"

            measured_freq = dominant_frequency(middle)
            assert measured_freq == pytest.approx(1000.0, abs=30.0), (
                f"captured tone is at {measured_freq:.1f} Hz, not 1 kHz"
            )
        finally:
            engine.stop_engine()
            assert engine.remove_linux_system_sink(device_id) is True
            engine.cleanup()

        assert handle.sink_name not in self._pactl_sink_names(), (
            "pactl still lists the sink after teardown"
        )
