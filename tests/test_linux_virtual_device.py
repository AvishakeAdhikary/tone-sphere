"""
Track 2 — the Linux virtual sink.

Per `docs/VIRTUAL_AUDIO_DRIVER.md` and the plan this implements, an OS-visible sink on
Linux is "configuration, not code": `pactl load-module module-null-sink` makes it. How
ToneSphere itself reaches it through PortAudio changed after real CI proved the first
design wrong — see `engine/linux_virtual.py`'s module docstring for the full story; in
short, PortAudio never enumerates a custom-named ALSA PCM in this environment no matter
how it is bridged, but it always exposes a generic `pulse` device, so `create_virtual_sink`
now points PulseAudio's default sink/source at the sink instead and `pulse` is what's used.

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

import subprocess
import sys

import numpy as np
import pytest

from tests.signals import dominant_frequency, sine

RATE = 48000
BLOCK = 256


def _run(result):
    """
    An endpoint called directly. The endpoints are plain functions FastAPI runs in its
    threadpool, so the call has already happened; calling them directly skips the lifespan
    hook, which would open real streams on a machine that may not have any.
    """
    return result


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
            previous_default_sink=None, previous_default_source=None,
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

        from tonesphere.core.engine import LOOPBACK_ID_BASE

        for entry in devices:
            if entry["id"] >= LOOPBACK_ID_BASE:
                # An output's whole-system loopback: a source ToneSphere derives from the
                # hardware, and labelled as that, not as another piece of hardware.
                assert entry["origin"] == "loopback" and entry["name"].endswith("(loopback)")
            elif entry["id"] not in engine._system_virtual_devices:
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

            # Both directions go through the same 'pulse' device now: `create_virtual_sink`
            # pointed PulseAudio's default *sink* at this sink and its default *source* at
            # this sink's monitor, so opening `device_id` for input is what reaches the
            # monitor — there is no separate named monitor device to look up any more (see
            # `engine/linux_virtual.py`'s module docstring for why the original per-device
            # design was replaced).
            captured: list[np.ndarray] = []

            def on_block(indata, frames, time_info, status):
                captured.append(indata.copy())

            with sd.InputStream(
                device=device_id, channels=2, samplerate=RATE,
                blocksize=BLOCK, dtype="float32", callback=on_block,
            ):
                phase = 0
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    phase += engine.write_to_bus(tone_bus, sine(BLOCK, amplitude=0.8, phase=phase))
                    time.sleep(BLOCK / RATE / 2)
                time.sleep(0.2)

            assert captured, "the monitor delivered nothing at all"
            audio = np.concatenate(captured)

            # 8192 frames is still ~6 Hz of FFT resolution at this rate -- easily enough to
            # tell 1 kHz apart from noise -- and a size CI has actually delivered, unlike
            # 16384: this environment's playback/capture pace is real but not guaranteed to
            # be realtime-exact, so demanding more than a modest, comfortably-measurable
            # window is asserting a throughput guarantee this test does not need to make.
            required = 8192
            middle = audio[audio.shape[0] // 3: audio.shape[0] // 3 + required]
            assert middle.shape[0] == required, "not enough audio captured to measure"

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


@pytest.mark.linux_virtual_sink
@pytest.mark.skipif(sys.platform != "linux", reason="needs Linux with a PulseAudio/PipeWire server")
class TestRealLinuxBusRouting:
    """
    Device -> bus -> device on the PortAudio host, through a real sound server.

    Two null sinks, A and B. PulseAudio's default source is A's monitor and its default
    sink is B, so PortAudio's `pulse` device reads A and plays into B. Another program
    (`pacat`) plays 1 kHz into A, and `parec` records B's monitor. ToneSphere routes the
    `pulse` input through a bus as wide as the device to its output, at a route gain of
    0.5: the tone must come out of B at its frequency and at exactly half its level. (A bus
    narrower than the device would lose the channels PulseAudio up-mixed the stereo source
    into, and its down-mix would read that as a level change.) One engine per process: the
    ALSA bridge to PulseAudio under WSL stalls a second stream opened on `pulse` by the same
    process for about two seconds, whichever route it carries. Before the PortAudio host's
    buses rendered, this route was silent.
    """

    def _pactl(self, *args: str) -> str:
        result = subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, f"pactl {' '.join(args)}: {result.stderr}"
        return result.stdout.strip()

    def _through(self, through_bus: bool) -> np.ndarray:
        import threading
        import time

        from tonesphere.core.engine import AudioEngine

        engine = AudioEngine(sample_rate=RATE, buffer_size=BLOCK, exclusive=False, host_backend='portaudio')
        engine.initialize()
        recorder = None
        recorded: list[bytes] = []
        try:
            pulse = [d for d in engine.get_devices() if d['name'] in ('pulse', 'pulse (In)', 'pulse (Out)')]
            if len({d['direction'] for d in pulse}) < 2:
                pytest.skip("PortAudio has no duplex 'pulse' device here (libasound2-plugins missing?)")
            pulse_id = pulse[0]['id']
            if through_bus:
                # As wide as the device: the bus then changes nothing PulseAudio up- or down-mixes.
                bus = engine.create_virtual_input("through", channels=min(pulse[0]['channels'], 8))
                assert engine.create_routing(pulse_id, bus)[0]
                assert engine.create_routing(bus, pulse_id, 0.5)[0]
            else:
                assert engine.create_routing(pulse_id, pulse_id, 0.5)[0]
            engine.start_engine()
            assert engine.host.is_running, engine.get_performance_stats()['problems']

            raw = ["--format=float32le", "--rate=48000", "--channels=2", "--raw"]
            recorder = subprocess.Popen(["parec", "--device=ts_b.monitor", *raw], stdout=subprocess.PIPE)
            # Read as it records: a pipe holds 64 KB, about 8000 frames, and a recorder
            # blocked on a full pipe stops recording.
            reader = threading.Thread(target=lambda: recorded.extend(iter(lambda: recorder.stdout.read(8192), b'')),
                                      daemon=True)
            reader.start()
            player = subprocess.Popen(["pacat", "--playback", "--device=ts_a", *raw], stdin=subprocess.PIPE)
            player.stdin.write(sine(RATE * 3, amplitude=0.5).astype('<f4').tobytes())
            player.stdin.close()
            player.wait(timeout=10)
            time.sleep(0.5)
        finally:
            if recorder is not None:
                recorder.terminate()
            engine.cleanup()
        reader.join(5)
        data = b''.join(recorded)
        audio = np.frombuffer(data[:len(data) // 8 * 8], dtype='<f4').reshape(-1, 2)
        loud = np.nonzero(np.abs(audio[:, 0]) > 0.005)[0]
        assert len(loud) > RATE, f"B carried {len(loud)} loud frames: nothing came through"
        return audio[loud[0] + RATE // 4: loud[0] + RATE // 4 + 8192, 0]

    def test_a_device_through_a_bus_to_a_device(self):
        previous_sink = self._pactl("get-default-sink")
        previous_source = self._pactl("get-default-source")
        modules = [self._pactl("load-module", "module-null-sink", f"sink_name=ts_{n}",
                               f"sink_properties=device.description=ts_{n}") for n in ("a", "b")]
        try:
            self._pactl("set-default-source", "ts_a.monitor")
            self._pactl("set-default-sink", "ts_b")
            bused = self._through(through_bus=True)
        finally:
            for module in modules:
                self._pactl("unload-module", module)
            # PulseAudio's placeholder sink (auto_null, all a CI runner has) goes away while a
            # real sink is loaded and comes back by itself: a default is restored only if it
            # still exists to be restored.
            for kind, previous in (("sink", previous_sink), ("source", previous_source)):
                listed = [line.split("\t")[1] for line in self._pactl("list", "short", f"{kind}s").splitlines()
                          if line.strip()]
                if previous in listed:
                    self._pactl(f"set-default-{kind}", previous)

        level = float(np.sqrt((bused ** 2).mean()))
        expected = 0.5 * 0.5 / np.sqrt(2)   # the tone's rms, times the route gain out of the bus
        print(f"\npulse -> bus -> pulse: {dominant_frequency(bused):.1f} Hz, rms {level:.4f} "
              f"({20 * np.log10(level / expected):+.2f} dB from the {expected:.4f} expected)")
        assert dominant_frequency(bused) == pytest.approx(1000.0, abs=30.0)
        assert abs(20 * np.log10(level / expected)) < 0.5
