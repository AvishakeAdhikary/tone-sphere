---
title: Testing
layout: default
permalink: /testing/
description: How ToneSphere establishes that a feature works — signal tests, the hardware suite, the evidence levels, benchmarks, the soak test and the virtual driver's VM procedure.
---

# Testing

**A feature does not exist until a test proves it moves audio.** A return value of `True`
proves that a function returned. The tests here put a known signal in — a sine at a known
frequency and level, an impulse, a sweep, noise with a fixed seed — and assert on what
comes out: its frequency, its RMS, its peak, its delay, or its exact samples.

## Running them

```
uv sync --all-groups
uv run python scripts/fetch_sdks.py      # Windows: the Steinberg SDKs (never committed)
uv run python scripts/build_native.py    # Windows: the native engine, ASIO host, test plugins
uv run pytest -m "not hardware"          # everything that needs no audio device (what CI runs)
uv run pytest -m hardware                # real devices, run locally before trusting an audio change
uv run ruff check .
```

`-m hardware` needs a working default output and input. Tests needing something more
specific — FlexASIO for the ASIO tests, Surge XT for the third-party VST3 tests, the
ToneSphere driver for the virtual-device tests — skip with a sentence saying what is
missing, and never pass by default.

## The suites

| Where | What | Needs |
|---|---|---|
| `tests/native/` | the C++ engine through its C ABI: rings under two-thread stress, plan swaps under a running audio thread, mixing, channel mapping, gain ramps, strips, EQ against the Python reference sample for sample, compressor, delay, limiter, resampler, sample conversion, meters, statistics, allocation counting, the VST3 host with the deterministic test plugin (bit-exact gain and delay, reported latency equal to measured, state, a plugin that crashes on load and one that crashes on the audio thread), the ASIO host's conversions and message handling, `AudioEngine` on the native host with no device | the built DLLs (Windows) |
| `tests/unit/`, `tests/test_*.py` | the control plane, the PortAudio host's mixer with synthetic signals, DSP formulas, presets, config store, network transport and jitter buffer, the UI offscreen, i18n catalogues, legal documents, packaging, and `test_honesty.py` | nothing |
| `tests/hardware/` | WASAPI render into loopback (bit-exact), exclusive-mode negotiation, three independent clocks, process loopback; ASIO against FlexASIO, including FlexASIO on a USB interface; a USB interface's own clock in exclusive and shared mode, its input's real signal, and a monitoring path (`test_interface.py`, `TONESPHERE_TEST_INTERFACE`); the round-trip measurement on the digital path, and through an interface cable when one is connected; VST3 with Surge XT; `AudioEngine` through a plugin to the speaker, heard back by process loopback (level within 2 %, balance, bypass, per-channel meters); the inserts dialog driving a real plugin; the virtual driver | a real audio device; FlexASIO; Surge XT; the driver (VM only) |

About 880 tests in all; `uv run pytest --collect-only -q` gives the current number.

## House idiom

`tests/signals.py` holds the signal generators (`sine`, `impulse`, `channel_impulses`,
`log_sweep`, `white_noise`, `pink_noise`) and the measurements (`dominant_frequency`, `rms`,
`peak`, `assert_finite`). A new backend, effect or routing feature gets a test of this shape:

```python
tone = sine(RATE, 1000.0, amplitude=0.2)
out = run_through_the_new_thing(tone)
assert dominant_frequency(out) == pytest.approx(1000.0, abs=3.0)
assert rms(out) == pytest.approx(rms(tone) * expected_gain, rel=0.02)
```

This has caught real bugs nothing else would have: a filter that diverged to infinity, a
fan-out that silently dropped a destination, a limiter's attack wrong by a factor of 256,
asymmetric integer scaling at the device boundary, a resampler that clicked below unity
ratio, and a driver adding 9.7 dB of "enhancement" that turned out not to be ToneSphere.

A test that needs the engine to *report* something (a meter, a statistic) is paired with
one that measures the audio independently, so the report is checked against reality rather
than against itself.

## Evidence levels

Every entry in [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) carries one:

| Level | Meaning | Needs |
|---|---|---|
| NOT IMPLEMENTED | no code | — |
| IMPLEMENTED | code exists | nothing proves it moves audio yet |
| UNVERIFIED | code exists and was exercised, but not in the conditions that matter | a stated reason |
| VERIFIED | a signal test passes in CI or offline | the test's name |
| HARDWARE VERIFIED | a signal test passes against real devices on a named machine | the test's name, the device, the result |

A claim is never upgraded past its evidence. "Round-trips audio in CI" is not "works on
macOS"; "verified against FlexASIO" is not "works with your interface".

## Honesty checks

`tests/test_honesty.py` enforces the rule that a measurement not taken is `--`, never `0`:
stopped engines report empty meters, not zero dBFS; unmeasured latency is `None`; the
round trip is `None` until a measurement at the current rate and block exists, and a
loopback measurement is never reported as the round trip (`tests/test_ui_views.py`). The
PortAudio host's bus defect was a strict `xfail`, so fixing it (M17) failed the suite until
the documents that published it were corrected; `test_device_to_bus_to_device_carries_the_signal`
now passes, and `TestRealLinuxBusRouting` proves the route through a real PulseAudio server.

## Benchmarks and the soak test

```
uv run python benchmarks/bench_engine.py --json out.json         # offline cost per block
uv run python benchmarks/soak.py --minutes 30 --json soak.json   # the live path, then 25 restarts
```

The benchmark times the engine inside `run_block` at 44.1, 48 and 96 kHz and 32–512 frames,
with a representative guitar chain. The soak test runs the UI's own path on the default
output for as long as asked, sampling the engine's statistics, the process's private bytes
and handle count, and then stops and restarts the engine repeatedly. Its output is silent by
design (see [REALTIME.md](REALTIME.md)). Results are kept under `benchmarks/results/`.

## The virtual audio driver

Tested only inside a Hyper-V VM with test-signing on, never on a development machine. On
a host with Hyper-V, elevated:

1. `scripts\vm\new_driver_vm.ps1 -Iso <Windows 11 ISO>` builds the VM once: the image applied
   to a VHDX, test-signing on in the VM disk's own boot store, Secure Boot off, an answer
   file, Windows Update off in the guest, and a `clean` checkpoint taken once OOBE's own
   restart is over.
2. `uv run python scripts/build_driver.py` builds and test-signs the package and assembles
   `driver/windows_virtual_audio/x64/Release/vm_kit/`.
3. `scripts\vm\run_driver_tests.ps1` restores a checkpoint, copies in the tree and the built
   binaries, installs the driver, runs `tests/hardware/test_virtual_driver.py`, uninstalls,
   checks nothing is left, and brings every log back. The first pass also takes a `deps`
   checkpoint with Python and the locked dependencies installed.

It passed on 2026-09-30, 7 of 7, and the driver is HARDWARE VERIFIED in the VM
([VIRTUAL_AUDIO_DRIVER.md](VIRTUAL_AUDIO_DRIVER.md)).

## CI

`.github/workflows/ci.yml` runs lint and the CI-safe suite on Windows, Linux and macOS; on
Windows it fetches the SDKs and builds the native DLLs first, so `tests/native` runs there.
It builds the frozen application on all three and smoke-tests it, builds the macOS plug-in
and round-trips audio through it, and on a version tag publishes the executables — the
Windows one with its Corresponding Source archive. It does not build the driver: whether the
hosted runners carry a WDK that can has not been checked.
