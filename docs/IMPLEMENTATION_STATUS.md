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
| **WASAPI** | Legacy: via PortAudio (`host.py:607-625`). **Native:** `native/windows_audio/wasapi.cpp` — event-driven shared (IAudioClient3 low-latency where possible) and exclusive with format negotiation, raw mode, MMCSS, master + satellite clocks with drift correction | Native, driving the engine | Native: **HARDWARE VERIFIED** (render, loopback, process loopback, exclusive, multi-clock) | `tests/hardware/test_wasapi.py`; bit-exact render→process loopback; details and limits in `docs/WINDOWS_AUDIO.md`. Not yet used by `AudioEngine` (M8) |
| **Device enumeration** | Legacy: PortAudio, key `host_api::name`. **Native:** MMDevice enumeration with endpoint IDs, default roles, formats, periods, raw support; `IMMNotificationClient` events | Native | Native enumeration HARDWARE VERIFIED; notifications IMPLEMENTED; behaviour on removal while streaming NOT VERIFIED | `test_endpoints_have_stable_ids_names_and_formats`, `test_device_notifications_can_be_watched` |
| **ASIO** | Legacy: enum/preference only. **Native:** `native/asio/asio_host.cpp` (GPLv3 DLL) — discovery, STA driver thread, negotiation, buffer switch driving the engine, `asioMessage` | Native ASIO host | Host **HARDWARE VERIFIED against FlexASIO 1.10b** (a software ASIO driver); boundary VERIFIED; **hardware-interface ASIO verification: NOT AVAILABLE ON THIS MACHINE** | `tests/native/test_asio.py`; `tests/hardware/test_asio.py` against FlexASIO: query, 80/80 buffer switches at 48 kHz/882, 0 xruns, output captured at 1 kHz. `docs/ASIO.md` |
| **VST3** | Legacy pedalboard chain **removed**. **Native:** `native/vst3/vst3_host.cpp` on the VST3 SDK 3.8.1 — out-of-process scan, load, processing as an engine insert, parameters, state, editor, SEH fault isolation | Native VST3 host | VERIFIED with ToneSphere's deterministic test plugin; **HARDWARE VERIFIED with Surge XT 1.3.4**; commercial plugins NOT TESTED | `tests/native/test_vst3.py` (bit-exact gain+delay, measured latency = reported, parameters, state, crash isolation on load and on the audio thread); `tests/hardware/test_vst3_third_party.py` (Surge XT delay echo at 250.06 ms, bypass, state, editor). Not yet reachable from UI/API. `docs/VST3.md` |
| **Mixer** | Python/NumPy in the callback (`host.py:986-1204`, `dsp.py`): -3 dB constant-power pan, per-block linear gain ramps, per-channel trim/mute/solo/polarity/swap, output limiter | Native mixer on atomic control targets | VERIFIED (offline) | `test_mixer_integration.py`, `TestBusMixing` drive the real `_mix_into` with synthetic signals. Known defects below. |
| **Routing graph** | Immutable `RoutingGraph` (`graph.py:111`), DFS feedback check on the control thread (`graph.py:247`), per-route rings (`host.py:433-491`) | Compiled execution plan, native, atomic swap | VERIFIED (device→device) | `TestGraph`, `TestBusMixing`. Buses do **not** forward — see defects. |
| **Buses** | In-process summing points | Native buses with real forwarding | **BROKEN** for device→bus→device | Nothing reads a device→bus ring into the bus's outgoing routes; `(bus → Y)` rings are filled only by `write_bus` (`host.py:1271`). Bus→device works when a producer calls `write_bus`. |
| **Ring buffer** | NumPy SPSC with GIL-atomic indices (`ringbuffer.py`); overrun drops the newest frames, underrun zero-fills | Native SPSC, acquire/release atomics | IMPLEMENTED | `TestRingBuffer` incl. a two-thread stress test. SPSC is not enforced for bus sources: `write_bus` is called from several threads (network playout, process capture, main thread). |
| **Metering** | Peak/RMS/clip per channel computed in the callback (`meters.py:160-188`), read by the UI every 33 ms | Native, seqlock-published | IMPLEMENTED (with a defect) | `…::in` and `…::out` keys map to the same id (`core/engine.py:1339`), so input and output meters overwrite each other in the UI. |
| **Latency reporting** | Legacy host: PortAudio-*reported* stream latency + plugin latency (`host.py:1414-1446`), labelled `reported_latency_ms`; `measured_round_trip_ms` stays None there. **Native:** `tonesphere/native/roundtrip.py` sends an exponential sweep, tees it in the engine, and times the capture by GCC-PHAT, refusing below a confidence threshold | Nominal / reported / plugin / **measured** round trip, separately | Measurement method VERIFIED (synthetic) and HARDWARE VERIFIED on the digital path; acoustic round trip on this machine: `--` | `tests/unit/test_roundtrip_estimator.py`; `tests/hardware/test_roundtrip.py`: render→loopback **62.35 ms (2993 frames), confidence 26.9, repeatable to the frame** — identical to the independent cross-correlation in `test_wasapi.py`. Speakers→microphone on this laptop: no path above the confidence threshold (the Realtek mic heard silence; the Intel array heard nothing correlated), so no acoustic figure exists. Not yet wired into `main.py test` / the UI (M8, M10) |
| **Performance statistics** | `stream.cpu_load` from PortAudio; xruns = callbacks with any status flag | Callback min/mean/max/percentile from QPC, load = worst ÷ period | IMPLEMENTED (PortAudio's figures only) | No callback timing exists. |
| **Process loopback (Windows)** | ctypes COM `ActivateAudioInterfaceAsync` (`process_capture.py`, `wasapi_com.py`), a Python capture thread writing into a bus | Native WASAPI capture into a native SPSC port | HARDWARE VERIFIED | `TestSelfCapture::test_captures_the_tone_this_process_plays` on this machine, 2026-09-29. |
| **Whole-system loopback** | Legacy: none (previously claimed). **Native:** `loopback` stream kind | Native WASAPI loopback | Native: HARDWARE VERIFIED; not yet reachable from `AudioEngine`, which is why `capture_status()` still reports it unimplemented | `test_raw_render_comes_back_whole_at_the_level_sent[loopback]` |
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

## Native engine (`native/`, `tonesphere/native/`)

| Capability | Level | Evidence |
|---|---|---|
| Toolchain: EWDK (MSVC 14.50, SDK 10.0.28000) + CMake/Ninja from PyPI; `/W4 /WX` clean | IMPLEMENTED | `scripts/build_native.py`; `docs/BUILDING_WINDOWS.md` |
| C ABI (`native/include/tonesphere_native.h`), ctypes bindings, ABI version check | IMPLEMENTED | `tests/native/*` load and drive it |
| SPSC frame ring (wait-free with one producer and one consumer) | VERIFIED | `tests/native/test_ring.py` — wraparound, full, empty, 8-channel framing, two real threads moving 200 000 frames of sine+noise sample-exact |
| Plan validation, topological ordering, cycle rejection | VERIFIED | `TestPlanValidation`; a refused plan leaves the running plan intact |
| Plan swap by atomic exchange + hazard pointer, no audio-thread blocking | VERIFIED | `test_plans_swap_while_another_thread_processes` — 300 swaps under a concurrently running audio thread, every block finite and bounded |
| Mixing: routes, gain ramps, mute, invert, master, buses **with forwarding**, fan-out, channel mapping, constant-power pan, continuous stereo balance | VERIFIED (offline) | `TestAudioMovesThroughTheGraph`, `TestGainMuteInvert`, `TestChannelMapping` — known signals in, exact samples out |
| Ring ports (Python producer/consumer ↔ audio thread) with overrun/underrun counts and events | VERIFIED | `TestRingPorts` |
| Non-finite guard at sinks | VERIFIED | `test_a_nan_never_reaches_an_output` |
| Meters (peak since reset, block RMS, clip latch) | VERIFIED | `test_meters_report_what_was_processed`, `test_clipping_latches_until_reset` |
| Callback timing (min/mean/max, histogram p99, load = worst ÷ period) | IMPLEMENTED | `test_statistics_measure_real_callback_time`; `None` until a block runs |
| Zero heap allocations on the audio thread | VERIFIED | `test_the_audio_thread_allocates_nothing` — 500 blocks with rings, buses, ramps, swaps: `rt_allocations == 0` (this DLL's allocations only; plugin modules are not counted) |
| Channel strip: per-channel trim and polarity, fader, mute — persisted across plan swaps | VERIFIED (offline) | `tests/native/test_dsp.py::TestChannelStrip`, incl. `test_strip_settings_survive_a_plan_change` |
| Parametric EQ (peaking, shelves, HP, LP — 8 bands) | VERIFIED (offline) | `test_matches_the_python_reference_sample_for_sample`: all five filter types equal the proven Python `Biquad` within 2e-6 on noise; filter memory survives a plan swap bit-exactly; NaN input cannot poison it |
| Compressor (soft knee, per-sample detector) | VERIFIED (offline) | `TestCompressor`: -6 dBFS into -18 dB/4:1 settles at -15 dBFS with -9 dB reported; attack lags the transient |
| Limiter (sample-accurate, instant attack, no lookahead) and per-sink safety limiter | VERIFIED (offline) | `TestLimiter`: a +6 dBFS tone never exceeds the threshold by one sample; below threshold is bit-exact. Fixes the legacy limiter's unreduced first block |
| Delay with feedback | VERIFIED (offline) | `TestDelay`: echoes exactly at k·d with amplitude mix·feedback^(k-1), nothing between |
| RoutingGraph → native plan compiler (device in/out split, solo, stable ids) | VERIFIED (offline) | `tests/native/test_plan_compiler.py`; the duplex in→out monitoring path is no longer refused as feedback (`graph.would_feedback`) |
| Drift resampler across device clocks | VERIFIED (offline) + HARDWARE VERIFIED | `tests/native/test_boundary.py`: ±100 and ±500 ppm absorbed with bounded fill, no underrun, no discontinuity (a click at ratio < 1 was found and fixed here); on hardware in the three-clock test |
| Sample-format conversion at the device boundary | VERIFIED | `TestConversion`: int16/24/32, float32/64 round trips within half a step; over-range clips instead of wrapping |
| WASAPI backend | HARDWARE VERIFIED | `docs/WINDOWS_AUDIO.md` |
| ASIO backend (separate GPLv3 DLL, attached through the external-backend C ABI) | HARDWARE VERIFIED against FlexASIO; hardware interface NOT AVAILABLE | `docs/ASIO.md` |

## Engine on the native host (Windows)

`tonesphere/engine/native_host.py` gives `AudioEngine` the host interface it always used,
backed by the native engine, so the UI, API, CLI and `main.py test` run on it unchanged.
`get_performance_stats()['backend']` says which host is in use; the PortAudio host remains
for Linux/macOS, and on Windows only if the native DLL is missing (logged as a warning) or
asked for (`host_backend='portaudio'`).

| Capability | Level | Evidence |
|---|---|---|
| Device enumeration from MMDevice (or ASIO drivers) with stable endpoint-ID keys; old name-based preset keys still resolve | IMPLEMENTED | `native_devices`, `DeviceInfo.key`/`name_key`, `PresetManager._map_devices` |
| Routing compiled to native plans and swapped live; streams restarted only when the set of devices changes | VERIFIED | the 822-test suite passes on it |
| Device → bus → device carries audio (the PortAudio host's defect) | VERIFIED (offline) | `tests/native/test_engine.py::test_device_to_bus_to_device_carries_the_tone`; the legacy defect stays pinned by its strict xfail against the PortAudio host |
| Bus-only / network-only routing run by the native clock, sample-exact | VERIFIED | `tests/native/test_engine_native_host.py` |
| AudioEngine → native host → VST3 → WASAPI, level exact; fader and plugin survive a restart; preset restores the plugin and its state | HARDWARE VERIFIED | `tests/hardware/test_engine_native_host.py`: 0.07071 RMS heard for 0.07071 expected; after restart at half fader, again exact |
| Per-block load judged against each block's own period; one engine run per device callback | VERIFIED | `test_load_is_judged_per_block_not_against_the_last_block`; `main.py test` on the development machine: exclusive 128 frames, worst callback 0.025 ms = 0.8 % of the period, mean 0.2 %, 0 xruns, 0 allocations |
| Satellite cushion trimmed to its target on priming | VERIFIED (hardware) | digital round trip 61.35–65.35 ms over six starts (within one 480-frame period), fixed within a run |
| Strip channel pan and channel swap on the native host | NOT IMPLEMENTED | recorded by the strip, not applied (pan lives on routes natively); the PortAudio host's strip pan never worked either |
| Per-process capture on the native host | IMPLEMENTED (Python capture thread into a bus feed) | `tests/test_process_capture.py` hardware tests pass on the native host; the native `process_loopback` stream kind is not yet used by `AudioEngine` |
| Device removal while running | NOT VERIFIED | notifications are registered; `AudioEngine` does not yet react to them automatically |

## Measurements

**Offline engine cost** — `benchmarks/bench_engine.py`, results in
`benchmarks/results/m3_offline_i7-1165G7.json`. Measured 2026-09-29 on the development
machine (i7-1165G7, Windows 11 25H2, MSVC 14.50 release build), benchmark thread registered
with MMCSS "Pro Audio", 5 s of audio per configuration. Times are taken inside `run_block`;
they are what ToneSphere's own processing costs, not a latency and not a device run.

| Scenario, 48 kHz | Block | Period | Mean | p99 | Max | Worst load |
|---|---|---|---|---|---|---|
| Guitar chain (3-band EQ, compressor, bus, delay, safety limiter) | 32 | 667 µs | 4.1 µs | 8.0 µs | 152.8 µs | 22.9 % |
| | 128 | 2667 µs | 13.9 µs | 19.0 µs | 148.9 µs | 5.6 % |
| | 256 | 5333 µs | 23.9 µs | 32.0 µs | 39.7 µs | 0.7 % |
| 16 stereo sources, EQ each, 2 buses, 2 limited outputs | 32 | 667 µs | 6.2 µs | 11.3 µs | 142.9 µs | 21.4 % |
| | 128 | 2667 µs | 17.8 µs | 26.9 µs | 166.1 µs | 6.2 % |
| | 256 | 5333 µs | 33.1 µs | 45.3 µs | 165.2 µs | 3.1 % |

Zero audio-thread allocations in every configuration (30 of them, 44.1/48/96 kHz × 32–512
frames). The worst single block is 140–230 µs in almost every configuration regardless of
workload, while p99 tracks the workload — consistent with interrupt or DPC activity on this
laptop pre-empting the thread, **not established**: it has not been traced. The tightest
budget measured (32 frames at 96 kHz, 333 µs) peaked at 49 % of the period.

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
| M2 | Native foundation: toolchain, C ABI, SPSC, snapshots, offline `process_block` | done — see *Native engine* below |
| M3 | Native graph, mixer, DSP | done — see *Native engine* and *Measurements* below |
| M4 | Native WASAPI backend | done — see `docs/WINDOWS_AUDIO.md`; switching `AudioEngine` onto it moves to M8, so the host interface is designed once, with ASIO and VST3 in view |
| M5 | Native ASIO host | done — HARDWARE VERIFIED against FlexASIO (software ASIO driver); hardware-interface ASIO not available on this machine — see `docs/ASIO.md` |
| M6 | Measured round-trip latency | done — `tonesphere/native/roundtrip.py`; digital path measured, acoustic path unavailable on this machine |
| M7 | Native VST3 host | done — see `docs/VST3.md`; not yet reachable from the UI/API (M8, M10) |
| M8 | Engine migration, persistence | done — `AudioEngine` runs on `NativeHost` on Windows; see *Engine on the native host* below |
| M9 | Windows virtual audio driver | not started |
| M10 | UI | not started |
| M11 | Validation, benchmarks, documentation | not started |
