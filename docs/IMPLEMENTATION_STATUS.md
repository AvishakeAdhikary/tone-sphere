---
title: Implementation Status
layout: default
permalink: /implementation-status/
description: What ToneSphere actually does today, subsystem by subsystem, with the evidence for each claim and the level it has reached.
---

# Implementation status

This is the record of what exists. The README summarises it; when they disagree, this
file is right and the README is a bug. Levels are defined in `AGENTS.md`:

- **NOT IMPLEMENTED** — does not exist.
- **UNVERIFIED** — code exists; real-world behaviour not demonstrated.
- **IMPLEMENTED** — code exists and an automated CI-safe test demonstrates it.
- **VERIFIED** — plus a signal-based integration test end to end.
- **HARDWARE VERIFIED** — exercised on real hardware/software, result recorded below.

Evidence is a test name, a file:line, or a recorded run. "Target" is the architecture in
`AGENTS.md` this subsystem is being migrated to.

## Baseline (2026-09-29, before the native migration)

Machine: Windows 11 Pro 25H2 (build 26200), Intel i7-1165G7, Realtek ALC257 (speakers,
headphone out, mic), Intel SST mic array, Bluetooth headset. No ASIO drivers installed,
no VST3 plugins installed, no C/C++ toolchain.

- `uv run pytest -m "not hardware"`: **660 passed, 2 skipped** (Linux PulseAudio test;
  `makeappx` schema check), 7 deselected — before any change on this branch.
- `uv run pytest -m hardware`: **7 passed** — output stream on the default Realtek
  device without xruns; process-loopback self-capture of a 1 kHz tone.

## Reality matrix

| Subsystem | Existing implementation | Target | Level | Evidence / notes |
|---|---|---|---|---|
| **Audio I/O (all platforms)** | One sounddevice/PortAudio stream per device, Python callbacks (`engine/host.py:895/916/972`) | Windows: native backend, no Python on the audio thread. Linux/macOS: keep, labelled not real-time safe | HARDWARE VERIFIED (output only) | `TestRealHardware::test_output_stream_runs_without_xruns` on Realtek WASAPI. Callback is Python under the GIL; see *Real-time safety* below. |
| **WASAPI** | Via PortAudio: shared, and exclusive through `sd.WasapiSettings(exclusive=True)` (`host.py:607-625`); exclusive→shared fallback only at open, not at validate (`host.py:552-558`) | Native event-driven shared/exclusive, MMCSS, endpoint IDs, `IMMNotificationClient` | HARDWARE VERIFIED (via PortAudio) | Same test as above. No native WASAPI code. |
| **Device enumeration** | PortAudio only; key `host_api::name` (`devices.py:89-92`); no endpoint IDs; refresh is manual and never re-initialises PortAudio | Native MMDevice enumeration, stable endpoint IDs, arrival/removal/default-change events | IMPLEMENTED (PortAudio) / device-change handling NOT IMPLEMENTED | `TestRealHardware::test_devices_are_enumerated`. |
| **ASIO** | Enum value, preference order and messages only (`devices.py:31,52,110,312`); the PyPI PortAudio build has no ASIO | Native ASIO host, GPLv3 `native/asio/` | NOT IMPLEMENTED | No ASIO SDK code anywhere. No ASIO driver on this machine (`HKLM\SOFTWARE\ASIO` empty). |
| **VST3** | `pedalboard.load_plugin` in `engine/effects.py:623-650`, processed in the Python callback (`effects.py:697-725`); top-level `*.vst3` glob only; no parameters, no state, no editor | Native host on the Steinberg VST3 SDK 3.8 (MIT) | UNVERIFIED | No UI, API or CLI caller reaches `AudioEngine.load_plugin` (`core/engine.py:1241`). No test loads a real VST3; `TestPluginHosting` uses pedalboard's built-in `Gain` and stand-in objects. Plugins are lost on any stream restart. pedalboard is GPLv3. |
| **Mixer** | Python/NumPy in the callback (`host.py:986-1204`, `dsp.py`): -3 dB constant-power pan, per-block linear gain ramps, per-channel trim/mute/solo/polarity/swap, output limiter | Native mixer on atomic control targets | VERIFIED (offline) | `test_mixer_integration.py`, `TestBusMixing` drive the real `_mix_into` with synthetic signals. Known defects below. |
| **Routing graph** | Immutable `RoutingGraph` (`graph.py:111`), DFS feedback check on the control thread (`graph.py:247`), per-route rings (`host.py:433-491`) | Compiled execution plan, native, atomic swap | VERIFIED (device→device) | `TestGraph`, `TestBusMixing`. Buses do **not** forward — see defects. |
| **Buses** | In-process summing points | Native buses with real forwarding | **BROKEN** for device→bus→device | Nothing reads a device→bus ring into the bus's outgoing routes; `(bus → Y)` rings are filled only by `write_bus` (`host.py:1271`). Bus→device works when a producer calls `write_bus`. |
| **Ring buffer** | NumPy SPSC with GIL-atomic indices (`ringbuffer.py`); overrun drops the newest frames, underrun zero-fills | Native SPSC, acquire/release atomics | IMPLEMENTED | `TestRingBuffer` incl. a two-thread stress test. SPSC is not enforced for bus sources: `write_bus` is called from several threads (network playout, process capture, main thread). |
| **Metering** | Peak/RMS/clip per channel computed in the callback (`meters.py:160-188`), read by the UI every 33 ms | Native, seqlock-published | IMPLEMENTED (with a defect) | `…::in` and `…::out` keys map to the same id (`core/engine.py:1339`), so input and output meters overwrite each other in the UI. |
| **Latency reporting** | PortAudio-*reported* stream latency + plugin latency (`host.py:1414-1446`) | Nominal / reported / plugin / **measured** round trip, separately | IMPLEMENTED as *reported*; measured round trip NOT IMPLEMENTED | Until this branch it was labelled "measured"; it is now `reported_latency_ms`, and `measured_round_trip_ms` stays None (`tests/test_honesty.py::TestUnmeasuredValuesAreNotZero`). It excludes ring and resampler delay on cross-device routes. |
| **Performance statistics** | `stream.cpu_load` from PortAudio; xruns = callbacks with any status flag | Callback min/mean/max/percentile from QPC, load = worst ÷ period | IMPLEMENTED (PortAudio's figures only) | No callback timing exists. |
| **Process loopback (Windows)** | ctypes COM `ActivateAudioInterfaceAsync` (`process_capture.py`, `wasapi_com.py`), a Python capture thread writing into a bus | Native WASAPI capture into a native SPSC port | HARDWARE VERIFIED | `TestSelfCapture::test_captures_the_tone_this_process_plays` on this machine, 2026-09-29. |
| **Whole-system loopback** | None. Previously *claimed* available because render endpoints existed | Native WASAPI loopback | NOT IMPLEMENTED | `test_whole_system_loopback_is_not_claimed_until_something_opens_it`. |
| **Audio-session detection** | PowerShell `Add-Type` C# block querying the default render endpoint's sessions (`app_capture.py:51-211`) | Native or ctypes IAudioSessionManager2 | IMPLEMENTED | Volume and mute fields are hard-coded (`app_capture.py:207-208`). |
| **Windows virtual device** | None | PortCls/WaveRT loopback-cable driver, OS-visible endpoints | NOT IMPLEMENTED | — |
| **Linux virtual sink** | `pactl` null sink + ALSA `pulse` PCM | Kept as is | VERIFIED (CI) | `TestRealLinuxSink` in the Linux CI job. |
| **macOS virtual device** | AudioServerPlugIn in `native/coreaudio-plugin/` | Kept as is | VERIFIED (CI only) | `build-macos-plugin` job round-trips a 1 kHz sine. Day-to-day use UNVERIFIED. |
| **Built-in DSP** | Python: RBJ biquad EQ with a per-sample Python loop (`effects.py:174`), soft-knee compressor, delay, limiter without lookahead | Native ports of the same formulas | VERIFIED (offline) | `test_effects.py`, `test_dsp.py` signal tests. Reachable only through `AudioEngine.enable_*`; no UI/API/CLI caller. |
| **Network audio** | UDP transport with jitter buffer and paced send worker; TCP receive only. Threads touch rings, never the callback | Kept; feeds native external-port rings | VERIFIED (localhost) | `test_network_send_wiring.py` (1 kHz sine across two engines over localhost UDP). |
| **UI** | PySide6 mixer strips, patchbay, status bar; synchronous engine calls on the GUI thread | Control plane only, plus device config, plugin browser/editor, diagnostics views | IMPLEMENTED | `test_ui.py` (offscreen). The strip pan control has no audible effect (defect below). No plugin, effect or network UI. |
| **Persistence** | SQLite settings (`utils/config.py`), YAML presets keyed on device names | Stable endpoint IDs, plugin chains and state | IMPLEMENTED | `test_config_store.py`, `test_presets.py`. Presets hold no plugin or effect state. |
| **Diagnostics (`main.py test`)** | Plays a 440 Hz tone to the default output, checks callback count/errors/xruns, prints reported latency and PortAudio CPU load | Real signal verification and measured round trip | IMPLEMENTED | It never captures its own output, and writes the tone ~3× faster than it is consumed, so the tone it plays is choppy by construction. |
| **Packaging** | PyInstaller one-file (Releases) and one-folder (MSIX); MSIX manifest and pack script | Plus native DLLs; driver has its own installer | IMPLEMENTED | CI `package` job smoke-tests the frozen GUI. MSIX never signed, installed or submitted. |

## Real-time safety of the existing path

The existing callback is Python under the GIL, so none of the `AGENTS.md` real-time
rules can hold for it; this is recorded so the native engine is measured against it,
not so it can be patched into compliance.

- **Allocation per block:** strings/tuples/sets per connection (`host.py:1000-1037`),
  `source * -1.0` for invert (`host.py:1102`), gain-ramp temporaries (`host.py:1110-1115`),
  `DriftResampler` fancy indexing (`dsp.py:441-446`), biquad `astype`/`empty_like` and a
  per-sample Python loop (`effects.py:174-187`), `np.abs` in limiter/compressor/meters,
  `np.ascontiguousarray(block.T)` and pedalboard's return copies (`effects.py:708-720`).
- **Logging in the callback:** `logger.error` on a plugin exception (`effects.py:717`).
- **Cross-thread mutation without snapshots:** `PluginChain` and `ParametricEQ` read three
  parallel lists with `zip(strict=True)` while the control thread mutates them in steps
  (`effects.py:645-663`, `:258-282`) — a torn read raises in the callback. The graph and
  route table are published separately (`host.py:862`), so a callback can see a new graph
  with an old route table (a route skipped, or mixed 3 dB hot for one block).
- **No thread priority:** no MMCSS, no GC control, no switch-interval tuning.

## Known defects in the existing path

| Defect | Where | Consequence |
|---|---|---|
| Buses never forward their input | `host.py` (no reader of device→bus rings) | device→bus→device carries silence; the rings overflow |
| Strip pan stored, never applied | `dsp.py:128-236` (`_pans` unused in `process`) | the UI pan knob does nothing |
| In/out meter keys collide | `core/engine.py:1339-1344` | one meter overwrites the other |
| Stream restart drops channel controls and plugins | `core/engine.py:1129-1135` | settings silently revert after reconfiguring |
| Stereo balance steps 3 dB off centre | `host.py:1186-1190`, `dsp.py:253-255` | level jump when a stereo pan leaves centre |
| Graph-level solo never populated | `core/engine.py:1003-1009` | only per-device channel solo works |
| Duplex in→out refused as feedback | `core/engine.py:407-437` | the guitar path through `AudioEngine` cannot be made on one device id |
| Limiter has no lookahead, ramp starts at the old envelope | `dsp.py:340-360` | first over-threshold block passes unreduced; with `clip_off=True` samples >1.0 can reach the driver |
| `pan_law_minus_6db` is not a -6 dB law | `dsp.py:27-52` | mislabelled option |
| Dead code | `core/sample_rate_converter.py` (never imported), `_bus_pending`, `GAIN_RAMP_FRACTION` | — |

These are fixed by the native engine (Windows) and, where the legacy host stays in use
(Linux/macOS), in that host during migration step M8.

## Migration progress

| Milestone | Scope | Status |
|---|---|---|
| M0 | `AGENTS.md` contract, `CLAUDE.md` pointer | done |
| M1 | This matrix, honesty fixes (reported vs measured latency, loopback claim, plugin and bus claims), dependency audit, shared test signals | done |
| M2 | Native foundation: toolchain, C ABI, SPSC, snapshots, offline `process_block` | not started |
| M3 | Native graph, mixer, DSP | not started |
| M4 | Native WASAPI backend | not started |
| M5 | Native ASIO host | not started |
| M6 | Measured round-trip latency | not started |
| M7 | Native VST3 host | not started |
| M8 | Engine migration, persistence | not started |
| M9 | Windows virtual audio driver | not started |
| M10 | UI | not started |
| M11 | Validation, benchmarks, documentation | not started |
