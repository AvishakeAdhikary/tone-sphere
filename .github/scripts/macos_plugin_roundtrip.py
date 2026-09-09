#!/usr/bin/env python3
"""
The proof for Track 3: does the ToneSphere CoreAudio HAL plug-in actually carry audio?

This is the only thing in the repository that can answer that. The plug-in's C source was
written on a Windows machine and has never been compiled, installed or listened to by its
author (see native/coreaudio-plugin/README.md), so every claim about it rests on this
script running green on a real macOS runner in the `build-macos-plugin` CI job, after that
job has built the bundle, ad-hoc signed it, installed it into
/Library/Audio/Plug-Ins/HAL and restarted coreaudiod.

It proves two separate things, in order:

  1. The device round-trips real audio. Open an OutputStream and an InputStream on the
     device from this same process, write a 1 kHz sine into the output, capture the input,
     and assert the captured signal's dominant frequency and RMS match. Finding a name in
     `sd.query_devices()` proves only that a plug-in loaded; this proves samples crossed
     coreaudiod's mix graph and came back.
  2. ToneSphere's own wiring finds it. `AudioEngine.create_macos_system_device()` returns
     a real device id and `get_devices()` labels that id `os_virtual_endpoint` — the part
     of Track 3 that unit tests can only exercise against a fabricated registry entry.

The tone helpers are imported from tests/test_engine_audio.py rather than redefined, so
this measures a signal with exactly the same idiom the rest of the suite uses.

Usage (from the repository root): uv run python .github/scripts/macos_plugin_roundtrip.py
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import sounddevice as sd

from tests.test_engine_audio import dominant_frequency, sine
from tonesphere.engine.macos_virtual import MACOS_DEVICE_NAME

RATE = 48000
BLOCK = 256
TONE_HZ = 1000.0
AMPLITUDE = 0.5
CAPTURE_SECONDS = 2.0

# Wide, because this is asking "did the tone survive at all", not "is the converter
# transparent" — a real failure here is silence or noise, not a decibel of error.
MIN_RMS = 0.05
MAX_RMS = 0.6
FREQUENCY_TOLERANCE_HZ = 30.0

# Enough of the middle of the capture to resolve 1 kHz cleanly and to skip the ramp-up
# while coreaudiod's own buffers fill.
ANALYSIS_FRAMES = 16384


def find_device() -> int:
    print("PortAudio devices on this runner:")
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        print(f"  [{index}] {device['name']} "
              f"({device['max_input_channels']} in / {device['max_output_channels']} out) "
              f"@ {device['default_samplerate']:.0f} Hz")

    matches = [
        index for index, device in enumerate(devices)
        if MACOS_DEVICE_NAME in device["name"]
        and device["max_input_channels"] >= 2
        and device["max_output_channels"] >= 2
    ]

    if not matches:
        sys.exit(
            f"FAIL: no duplex device named '{MACOS_DEVICE_NAME}' in PortAudio's list. "
            f"The bundle loaded (or did not) and coreaudiod restarted (or did not); "
            f"check the install and kickstart steps above."
        )

    print(f"\nUsing device [{matches[0]}] for the round trip.")
    return matches[0]


def round_trip(device_index: int) -> np.ndarray:
    captured: list[np.ndarray] = []
    phase = 0

    def on_output(outdata, frames, time_info, status):
        nonlocal phase
        outdata[:] = sine(frames, freq=TONE_HZ, rate=RATE, amplitude=AMPLITUDE,
                          channels=outdata.shape[1], phase=phase)
        phase += frames

    def on_input(indata, frames, time_info, status):
        captured.append(indata.copy())

    output = sd.OutputStream(device=device_index, channels=2, samplerate=RATE,
                             blocksize=BLOCK, dtype="float32", callback=on_output)
    capture = sd.InputStream(device=device_index, channels=2, samplerate=RATE,
                             blocksize=BLOCK, dtype="float32", callback=on_input)

    with output, capture:
        time.sleep(CAPTURE_SECONDS)

    if not captured:
        sys.exit("FAIL: the input stream opened but delivered no blocks at all.")

    audio = np.concatenate(captured)
    print(f"Captured {audio.shape[0]} frames ({audio.shape[0] / RATE:.2f} s).")
    return audio


def measure(audio: np.ndarray) -> int:
    start = audio.shape[0] // 3
    if audio.shape[0] - start < ANALYSIS_FRAMES:
        sys.exit(
            f"FAIL: only {audio.shape[0]} frames captured, need at least "
            f"{start + ANALYSIS_FRAMES} to measure the middle of the tone."
        )

    middle = audio[start:start + ANALYSIS_FRAMES]
    rms = float(np.sqrt((middle[:, 0] ** 2).mean()))
    frequency = dominant_frequency(middle, rate=RATE)

    print(f"Measured RMS {rms:.4f} (expected {MIN_RMS}-{MAX_RMS}, "
          f"a {AMPLITUDE} sine is ~{AMPLITUDE / np.sqrt(2):.3f})")
    print(f"Measured dominant frequency {frequency:.1f} Hz "
          f"(expected {TONE_HZ:.0f} +/- {FREQUENCY_TOLERANCE_HZ:.0f})")

    failures = []
    if not (MIN_RMS <= rms <= MAX_RMS):
        failures.append(
            f"RMS {rms:.4f} is outside {MIN_RMS}-{MAX_RMS}: "
            f"{'silence, so nothing crossed the device' if rms < MIN_RMS else 'far too loud'}"
        )
    if abs(frequency - TONE_HZ) > FREQUENCY_TOLERANCE_HZ:
        failures.append(
            f"dominant frequency {frequency:.1f} Hz is not {TONE_HZ:.0f} Hz — the device "
            f"is running at the wrong rate, or what came back is not the tone that went in"
        )

    for failure in failures:
        print(f"FAIL: {failure}")

    return 1 if failures else 0


def check_engine_wiring() -> int:
    """The Python half of Track 3, against the real device instead of a fake registry."""
    from tonesphere.core.engine import AudioEngine

    engine = AudioEngine(sample_rate=RATE, buffer_size=BLOCK, exclusive=False)
    engine.initialize()

    try:
        device_id = engine.create_macos_system_device()
        if device_id is None:
            print("FAIL: create_macos_system_device() returned None with the plug-in "
                  "installed and its device enumerable — see the log above for its reason.")
            return 1

        origins = {d["id"]: d["origin"] for d in engine.get_devices()}
        if origins.get(device_id) != "os_virtual_endpoint":
            print(f"FAIL: device {device_id} is labelled {origins.get(device_id)!r}, "
                  f"not 'os_virtual_endpoint'.")
            return 1

        print(f"OK: attached as device {device_id}, labelled 'os_virtual_endpoint'.")

        if not engine.remove_macos_system_device(device_id):
            print("FAIL: remove_macos_system_device() refused an id it had just handed out.")
            return 1

        return 0
    finally:
        engine.cleanup()


def main() -> int:
    device_index = find_device()

    print("\n--- 1/2: round-tripping a 1 kHz sine through the device ---")
    status = measure(round_trip(device_index))

    print("\n--- 2/2: ToneSphere's own wiring against the real device ---")
    status |= check_engine_wiring()

    print("\nPASS: the plug-in built, installed and carried audio."
          if status == 0 else "\nFAILED.")
    return status


if __name__ == "__main__":
    sys.exit(main())
