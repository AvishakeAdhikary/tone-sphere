"""
Per-process capture, proved by capturing a tone this process really played.

The whole point of Track 1 is that this is testable for real here. The main test renders a
1 kHz sine out of a real output device through the engine's own audio path, captures this
process by pid, and asserts the captured samples are that tone: right frequency, right
duration, real amplitude, and nothing else in the spectrum. If Windows delivered silence,
a different rate, or the wrong channel interleaving, the assertions fail — which is the
only way to know this works, because every COM call in the path returns `S_OK` whether or
not audio moves.

Marked `hardware` because it needs a real output device to render to; skipped entirely
where the platform cannot do process loopback at all.
"""

import asyncio
import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from tonesphere.engine.app_capture import process_loopback_supported

RATE = 48000
BLOCK = 256

pytestmark = pytest.mark.skipif(
    not process_loopback_supported(),
    reason="per-process loopback needs Windows 10 build 20348 or later",
)


def sine(frames: int, freq: float = 1000.0, rate: int = RATE,
         amplitude: float = 0.5, channels: int = 2, phase: float = 0.0) -> np.ndarray:
    """Same tone helper as `tests/test_engine_audio.py`, for the same reasons."""
    t = (np.arange(frames, dtype=np.float64) + phase) / rate
    wave = (amplitude * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)
    return np.repeat(wave.reshape(-1, 1), channels, axis=1)


def dominant_frequency(block: np.ndarray, rate: int = RATE) -> float:
    mono = block[:, 0] if block.ndim > 1 else block
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    return float(np.fft.rfftfreq(len(mono), 1.0 / rate)[int(np.argmax(spectrum))])


def dead_pid() -> int:
    """A process id that genuinely existed and genuinely does not any more."""
    process = subprocess.Popen([sys.executable, '-c', 'pass'])
    process.wait()
    return process.pid


def _run(coroutine):
    """
    Call a REST endpoint directly.

    The endpoints are plain async functions, so this skips FastAPI's lifespan hook, which
    would open real streams on a machine that may not have any.
    """
    return asyncio.run(coroutine)


def playing_engine():
    """
    An engine rendering a 1 kHz tone out of a real output device.

    Process loopback taps a render stream, so something in this process has to actually be
    rendering for there to be anything to capture. Quiet on purpose: this comes out of the
    speakers for real.
    """
    from tonesphere.core.engine import AudioEngine

    engine = AudioEngine(sample_rate=RATE, buffer_size=BLOCK, exclusive=False)
    engine.initialize()

    output_id = engine.default_output_id()
    if output_id is None:
        pytest.skip("no default output device")

    tone_bus = engine.create_virtual_input("tone", channels=2)
    success, message = engine.create_routing(tone_bus, output_id, volume=0.1)
    if not success:
        pytest.skip(f"could not open an output on this machine: {message}")

    engine.start_engine()
    return engine, tone_bus


def render_tone(engine, tone_bus, seconds: float) -> None:
    """
    Feed the tone bus for `seconds`, keeping the rendered tone phase-continuous.

    The phase only advances by the frames the bus actually accepted. Writing faster than
    the audio clock is deliberate — it is how the existing hardware tests keep the bus
    fed — but a rejected block whose phase was advanced anyway would put a 120 degree
    discontinuity into the rendered tone every few milliseconds. Measured: that alone
    smears the spectrum enough to drop a pure sine's in-band energy to 39%, which would
    look exactly like the capture mangling it.
    """
    phase = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        phase += engine.write_to_bus(tone_bus, sine(BLOCK, amplitude=0.5, phase=phase))
        time.sleep(BLOCK / RATE / 2)


class TestActivationFailuresAreReported:
    """
    Measured on Windows 11 build 26200: `ActivateAudioInterfaceAsync`,
    `IAudioClient::Initialize` and `Start` all return `S_OK` for a process id that has
    exited *and* for one that never existed, after which the capture delivers silence
    forever. A capture built by trusting those HRESULTs would report itself healthy while
    carrying nothing — the exact failure `tests/test_honesty.py` exists to prevent — so
    these assert the refusal actually happens.
    """

    def test_exited_process_is_refused(self):
        from tonesphere.engine.process_capture import ProcessCapture, ProcessCaptureError

        capture = ProcessCapture(dead_pid())

        with pytest.raises(ProcessCaptureError) as raised:
            capture.start(lambda fmt: (lambda block: len(block)))

        message = str(raised.value)
        assert 'has exited' in message or 'no process' in message, message
        assert capture.is_running is False
        assert capture.statistics()['frames_captured'] == 0

    def test_never_existing_pid_is_refused(self):
        from tonesphere.engine.process_capture import ProcessCapture, ProcessCaptureError

        with pytest.raises(ProcessCaptureError, match="no process with id"):
            ProcessCapture(999_999).start(lambda fmt: (lambda block: len(block)))

    def test_refused_capture_leaves_no_bus_behind(self):
        """A bus that nothing feeds is a device the UI would offer and silence would fill."""
        from tonesphere.core.engine import AudioEngine
        from tonesphere.engine.process_capture import ProcessCaptureError

        engine = AudioEngine(sample_rate=RATE, buffer_size=BLOCK)
        engine.initialize()

        before = len(engine.list_virtual_devices())

        with pytest.raises(ProcessCaptureError):
            engine.start_process_capture(999_999)

        assert engine.process_capture_status() == []
        assert len(engine.list_virtual_devices()) == before


class TestExposureIsActuallyWired:
    """
    The CLI and REST surface, proved reachable rather than assumed.

    `tests/test_cli.py` exists because `_delete_virtual_device` printed "not yet
    implemented" for a feature the engine had had since Phase 1 — a command that looked
    wired and was not. These are the same check for the new surface, and they use a dead
    pid so they need no hardware: what matters is that the call arrives and the refusal
    comes back, not that audio moves.
    """

    def test_cli_dispatches_the_appcapture_command(self):
        """A command absent from the dispatch table is a feature that does not exist."""
        import inspect

        from tonesphere.cli.interface import AudioEngineCLI

        source = inspect.getsource(AudioEngineCLI.run_interactive_mode)

        assert 'appcapture' in source
        assert 'manage_app_capture' in source

    def test_cli_start_reports_the_real_refusal(self, capsys):
        from unittest.mock import patch

        from tonesphere.cli.interface import AudioEngineCLI

        cli = AudioEngineCLI()
        cli.initialize_engine()

        try:
            with patch('builtins.input', side_effect=[str(dead_pid()), '', 'y']):
                cli._start_app_capture()
        finally:
            if cli.engine:
                cli.engine.cleanup()

        output = capsys.readouterr().out
        assert 'Capturing process' not in output
        assert 'has exited' in output or 'no process' in output, output

    def test_cli_shows_no_captures_rather_than_a_blank_table(self, capsys):
        from tonesphere.cli.interface import AudioEngineCLI

        cli = AudioEngineCLI()
        cli.initialize_engine()

        try:
            cli._show_app_captures()
        finally:
            if cli.engine:
                cli.engine.cleanup()

        assert 'No captures running' in capsys.readouterr().out

    def test_api_reports_status_without_claiming_more_than_the_platform_offers(self):
        import tonesphere.api.server as server
        from tonesphere.engine.app_capture import capture_status

        status = _run(server.get_app_capture_status())

        assert status['process_loopback_implemented'] == \
            capture_status()['process_loopback_supported']

    def test_api_refuses_a_dead_pid_with_a_reason(self):
        from fastapi import HTTPException

        import tonesphere.api.server as server
        from tonesphere.api.models import StartProcessCaptureRequest
        from tonesphere.core.engine_factory import UnifiedAudioEngine

        engine = UnifiedAudioEngine()
        engine.initialize()
        server.audio_engine = engine

        try:
            with pytest.raises(HTTPException) as raised:
                _run(server.start_app_capture(
                    StartProcessCaptureRequest(pid=dead_pid())))
        finally:
            server.audio_engine = None
            engine.cleanup()

        assert raised.value.status_code == 400
        assert raised.value.detail

    def test_api_stop_reports_an_unknown_bus_rather_than_success(self):
        from fastapi import HTTPException

        import tonesphere.api.server as server
        from tonesphere.core.engine_factory import UnifiedAudioEngine

        engine = UnifiedAudioEngine()
        engine.initialize()
        server.audio_engine = engine

        try:
            with pytest.raises(HTTPException) as raised:
                _run(server.stop_app_capture(999_999))
        finally:
            server.audio_engine = None
            engine.cleanup()

        assert raised.value.status_code == 404


@pytest.mark.hardware
class TestSelfCapture:
    """
    The proof. Render a known tone from this process, capture this process, compare.

    Run with `uv run pytest -m hardware -k process_capture`.
    """

    def test_captures_the_tone_this_process_plays(self):
        from tonesphere.engine.process_capture import ProcessCapture

        engine, tone_bus = playing_engine()
        captured: list[np.ndarray] = []

        capture = ProcessCapture(os.getpid(), sample_rate=RATE, channels=2)
        fmt = capture.start(lambda f: (lambda block: (captured.append(block), len(block))[1]))

        try:
            assert fmt.sample_rate == RATE, \
                f"Windows delivered {fmt.sample_rate} Hz, not the engine's {RATE}"
            assert fmt.channels == 2

            render_tone(engine, tone_bus, seconds=2.0)
            time.sleep(0.2)
            stats = capture.statistics()
        finally:
            capture.stop()
            engine.stop_engine()

        assert stats['error'] is None, f"capture failed: {stats['error']}"
        assert captured, "no audio was captured at all"

        audio = np.concatenate(captured)
        assert audio.shape[1] == 2, f"expected 2 channels, got {audio.shape[1]}"
        assert audio.shape[0] > RATE, \
            f"only {audio.shape[0]} frames captured ({audio.shape[0] / RATE:.2f}s)"
        assert stats['frames_captured'] == audio.shape[0]

        # The delivered rate, measured rather than trusted: `GetMixFormat` is E_NOTIMPL on
        # a loopback client, so counting frames against the clock is the only real check
        # that Windows honoured the rate it was asked for. A 44.1 kHz delivery — the
        # obvious way for this to be quietly wrong — is 8% out and fails here.
        assert stats['measured_sample_rate'] == pytest.approx(RATE, rel=0.01), \
            f"measured {stats['measured_sample_rate']} Hz against a requested {RATE}"

        # Skip the leading packets: the capture starts before the tone does, so the first
        # fraction of a second is legitimately silence.
        middle = audio[audio.shape[0] // 3: audio.shape[0] // 3 + 16384]
        assert middle.shape[0] == 16384

        rms = float(np.sqrt((middle[:, 0] ** 2).mean()))
        peak = float(np.abs(middle).max())

        assert rms > 0.01, f"captured RMS {rms:.5f} is silence, not a tone"
        assert peak <= 1.0001, f"captured peak {peak:.4f} is clipped"

        # Amplitude is a band rather than a ratio to what was played, and the frequency
        # tolerance is 30 Hz rather than 1. Both because of one measured fact about this
        # machine: the render endpoint runs its own dynamics processing, which applies a
        # level-dependent gain (a played 0.1 came back at 0.42 peak, a played 0.4 at 0.81)
        # and modulates the tone enough to put sidebands ~38 Hz either side of it, pulling
        # the spectral peak to 987-990 Hz. Asserting 1000.0 Hz exactly, or a fixed gain,
        # would be asserting a property of one machine's audio enhancements rather than of
        # the capture. What process loopback genuinely promises is the render stream as the
        # Windows audio engine mixed it, and that is what is checked below.
        measured_freq = dominant_frequency(middle)
        assert measured_freq == pytest.approx(1000.0, abs=30.0), \
            f"captured tone is at {measured_freq:.1f} Hz, not 1 kHz"

        # Nearly all the energy sits around the tone. Measured 93.8-94.8% across segments,
        # so 0.85 has real margin — and silence, noise, a mangled interleave or a wrong
        # sample rate all fail it, which is the point.
        spectrum = np.abs(np.fft.rfft(middle[:, 0] * np.hanning(len(middle))))
        freqs = np.fft.rfftfreq(len(middle), 1.0 / RATE)
        power = spectrum ** 2
        in_band = float(power[(freqs > 900.0) & (freqs < 1100.0)].sum())
        total = float(power.sum())

        assert in_band / total > 0.85, \
            f"only {in_band / total:.1%} of the captured energy is near 1 kHz"

        assert stats['glitch_count'] == 0, \
            f"{stats['glitch_count']} discontinuities in the captured stream"

    def test_engine_feeds_a_capture_into_a_bus_it_sized_itself(self):
        """
        The engine wiring: a bus whose channel count came from the negotiated format, fed
        by a live capture, reporting only what it measured.
        """
        engine, tone_bus = playing_engine()

        capture_bus = engine.start_process_capture(os.getpid(), name="self")

        try:
            render_tone(engine, tone_bus, seconds=1.5)
            time.sleep(0.2)
            status = engine.process_capture_status(capture_bus)[0]
            buses = {b['id']: b for b in engine.list_virtual_devices()}
        finally:
            engine.stop_process_capture(capture_bus)
            engine.stop_engine()

        assert status['error'] is None, status['error']
        assert status['running'] is True
        assert status['pid'] == os.getpid()
        assert status['bus_id'] == capture_bus
        assert status['resampled'] is False, \
            f"needed resampling from {status['sample_rate']} Hz"

        assert status['frames_captured'] > RATE, \
            f"only {status['frames_captured']} frames reached the bus"
        assert status['buffer_frames'] is not None, "buffer size was never read back"

        # The bus was created inside on_format, so its width is the delivered width by
        # construction rather than by assumption.
        assert buses[capture_bus]['channels'] == status['channels']

        # Nothing is routed out of the capture bus in this test, so every write found no
        # destination. That is counted as its own thing, not as a capture fault — the two
        # mean completely different things to whoever reads these numbers.
        assert status['unrouted_writes'] > 0
        assert status['frames_rejected_by_sink'] == 0

    def test_a_stopped_capture_stops_claiming_to_run(self):
        """
        Same failure class as `test_no_polling_thread_pretends_to_carry_audio`: state that
        outlives the thread it describes.
        """
        from tonesphere.engine.process_capture import ProcessCapture

        capture = ProcessCapture(os.getpid(), sample_rate=RATE)
        fmt = capture.start(lambda f: (lambda block: len(block)))

        assert capture.is_running is True
        assert fmt.sample_rate == RATE
        assert capture.statistics()['running'] is True

        capture.stop()

        assert capture.is_running is False
        assert capture.statistics()['running'] is False
        assert capture.statistics()['error'] is None

    def test_engine_teardown_leaves_no_capture_thread_running(self):
        """No capture thread outlives the engine that started it."""
        engine, tone_bus = playing_engine()
        capture_bus = engine.start_process_capture(os.getpid())
        capture = engine._process_captures[capture_bus]['capture']

        assert capture.is_running is True

        engine.cleanup()

        assert capture.is_running is False
        assert engine.process_capture_status() == []
