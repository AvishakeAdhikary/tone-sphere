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
| **WASAPI** | Legacy: via PortAudio (`host.py:607-625`). **Native:** `native/windows_audio/wasapi.cpp` — event-driven shared (IAudioClient3 low-latency where possible) and exclusive with format negotiation, raw mode, MMCSS, master + satellite clocks with drift correction | Native, driving the engine | Native: **HARDWARE VERIFIED** (render, loopback, process loopback, exclusive, multi-clock) | `tests/hardware/test_wasapi.py`; bit-exact render→process loopback; details and limits in `docs/WINDOWS_AUDIO.md`. `AudioEngine` runs on it on Windows (M8) |
| **Device enumeration** | Legacy: PortAudio, key `host_api::name`. **Native:** MMDevice enumeration with endpoint IDs, default roles, formats, periods, raw support; `IMMNotificationClient` events | Native | Native enumeration HARDWARE VERIFIED; notifications IMPLEMENTED; behaviour on removal while streaming NOT VERIFIED | `test_endpoints_have_stable_ids_names_and_formats`, `test_device_notifications_can_be_watched` |
| **ASIO** | Legacy: enum/preference only. **Native:** `native/asio/asio_host.cpp` (GPLv3 DLL) — discovery, STA driver thread, negotiation, buffer switch driving the engine, `asioMessage` | Native ASIO host | Host **HARDWARE VERIFIED against FlexASIO 1.10b** (a software ASIO driver), including **FlexASIO driving the Audio Array AI-04 USB interface** in WASAPI exclusive; boundary VERIFIED; **an interface manufacturer's own ASIO driver: NOT AVAILABLE** (the AI-04 has none — it is driver-free, on Windows' class driver) | `tests/native/test_asio.py`; `tests/hardware/test_asio.py` against FlexASIO: query, 80/80 buffer switches at 48 kHz/882, 0 xruns, output captured at 1 kHz (2026-09-29); through FlexASIO to the AI-04 at 144 frames: 499 buffer switches in 1.5 s, none missed, max 18.2 µs, 0 xruns, input 1 carrying the guitar's 50.10 Hz hum at −19.5 dBFS (2026-09-30). `docs/ASIO.md` |
| **VST3** | Legacy pedalboard chain **removed**. **Native:** `native/vst3/vst3_host.cpp` on the VST3 SDK 3.8.1 — out-of-process scan, load, processing as an engine insert, parameters, state, editor, SEH fault isolation | Native VST3 host | VERIFIED with ToneSphere's deterministic test plugin; **HARDWARE VERIFIED with Surge XT 1.3.4**; commercial plugins NOT TESTED | `tests/native/test_vst3.py` (bit-exact gain+delay, measured latency = reported, parameters, state, crash isolation on load and on the audio thread); `tests/hardware/test_vst3_third_party.py` (Surge XT delay echo at 250.06 ms, bypass, state, editor). Reachable from the UI (browser, insert chains, parameters, editor, host bypass — `tests/hardware/test_engine_native_host.py::test_the_inserts_dialog_drives_a_real_plugin`); **not from the REST API or CLI**. Instruments are refused (no MIDI path). `docs/VST3.md` |
| **Mixer** | Python/NumPy in the callback (`host.py:986-1204`, `dsp.py`): -3 dB constant-power pan, per-block linear gain ramps, per-channel trim/mute/solo/polarity/swap, output limiter | Native mixer on atomic control targets | VERIFIED (offline) | `test_mixer_integration.py`, `TestBusMixing` drive the real `_mix_into` with synthetic signals. Known defects below. |
| **Routing graph** | Immutable `RoutingGraph` (`graph.py:111`), DFS feedback check on the control thread (`graph.py:247`), per-route rings (`host.py:433-491`) | Compiled execution plan, native, atomic swap | VERIFIED (device→device) | `TestGraph`, `TestBusMixing`. Buses do **not** forward — see defects. |
| **Buses** | In-process summing points | Native buses with real forwarding | Native: VERIFIED (offline, exact samples); PortAudio host: **BROKEN** for device→bus→device | Native: `test_device_through_a_bus_to_another_device`, and a device-less bus→network send sample for sample (`test_a_bus_reaches_a_network_send_sample_for_sample_with_no_device`). PortAudio host: nothing reads a device→bus ring into the bus's outgoing routes (`host.py:1271`) — strict xfail `test_device_to_bus_to_device_carries_the_signal`. |
| **Ring buffer** | NumPy SPSC with GIL-atomic indices (`ringbuffer.py`); overrun drops the newest frames, underrun zero-fills | Native SPSC, acquire/release atomics | IMPLEMENTED | `TestRingBuffer` incl. a two-thread stress test. SPSC is not enforced for bus sources: `write_bus` is called from several threads (network playout, process capture, main thread). |
| **Metering** | Native: peak/RMS/clip per node and per channel (first 8), measured on the audio thread; PortAudio host: per channel in the callback. `AudioEngine.get_meters` reports each side of a device separately | Native, published per block | VERIFIED; HARDWARE VERIFIED per channel | `test_each_channel_is_metered_on_its_own`; `tests/hardware/test_engine_native_host.py::test_bypass_balance_and_per_channel_meters_are_what_is_heard` (meter within 0.3 dB of the tone heard per channel); `tests/test_ui_views.py` (a duplex device's input and output strips no longer overwrite each other) |
| **Latency reporting** | Legacy host: PortAudio-*reported* stream latency + plugin latency (`host.py:1414-1446`), labelled `reported_latency_ms`; `measured_round_trip_ms` stays None there. **Native:** `tonesphere/native/roundtrip.py` sends an exponential sweep, tees it in the engine, and times the capture by GCC-PHAT, refusing below a confidence threshold | Nominal / reported / plugin / **measured** round trip, separately | Measurement method VERIFIED (synthetic) and HARDWARE VERIFIED on the digital path; acoustic round trip on this machine: `--` | `tests/unit/test_roundtrip_estimator.py`; `tests/hardware/test_roundtrip.py`: render→loopback **62.35 ms (2993 frames), confidence 26.9, repeatable to the frame** — identical to the independent cross-correlation in `test_wasapi.py`. Speakers→microphone on this laptop: no path above the confidence threshold (the Realtek mic heard silence; the Intel array heard nothing correlated), so no acoustic figure exists. AI-04 output→AI-04 input (earphones on the output, a guitar on the input — no path between them): confidence 0.9–1.0, so `--`; `test_an_interface_cable_round_trip` measures it, exclusive at the interface's 3 ms period, once a cable joins the two. In the UI: Diagnostics → Measure (engine stopped; any output to any input, or the output's own loopback). A loopback result is recorded as the digital path and never reported as the round trip; a result is shown as the round trip only at the rate and block it was taken at (`tests/test_ui_views.py`, `tests/hardware/test_roundtrip.py::test_the_engine_keeps_a_loopback_measurement_apart_from_the_round_trip`) |
| **Performance statistics** | `stream.cpu_load` from PortAudio; xruns = callbacks with any status flag | Callback min/mean/max/percentile from QPC, load = worst ÷ period | IMPLEMENTED (PortAudio's figures only) | No callback timing exists. |
| **Process loopback (Windows)** | ctypes COM `ActivateAudioInterfaceAsync` (`process_capture.py`, `wasapi_com.py`), a Python capture thread writing into a bus | Native WASAPI capture into a native SPSC port | HARDWARE VERIFIED | `TestSelfCapture::test_captures_the_tone_this_process_plays` on this machine, 2026-09-29. |
| **Whole-system loopback** | Legacy: none (previously claimed). **Native:** `loopback` stream kind | Native WASAPI loopback | Native: HARDWARE VERIFIED; not yet reachable from `AudioEngine`, which is why `capture_status()` still reports it unimplemented | `test_raw_render_comes_back_whole_at_the_level_sent[loopback]` |
| **Audio-session detection** | PowerShell `Add-Type` C# block querying the default render endpoint's sessions (`app_capture.py:51-211`) | Native or ctypes IAudioSessionManager2 | IMPLEMENTED | Volume and mute fields are hard-coded (`app_capture.py:207-208`). |
| **Windows virtual device** | `driver/windows_virtual_audio/` — SimpleAudioSample (MS-PL) + a render→capture cable; endpoints "Speakers (ToneSphere Virtual Audio Cable)" and "Microphone Array (ToneSphere Virtual Audio Cable)", 48 kHz 32-bit stereo | PortCls/WaveRT loopback-cable driver, OS-visible endpoints | **HARDWARE VERIFIED in a Hyper-V test VM** (install, enumeration, audio between applications at the level sent, silence, clean uninstall); on a real desktop with a real communications app NOT VERIFIED; custom endpoint names NOT IMPLEMENTED; production signing NOT AVAILABLE | `tests/hardware/test_virtual_driver.py`, 7 of 7 in the VM (Windows 11 Enterprise LTSC 10.0.26100, test-signing), 2026-09-30, three passes in a row: PortAudio→cable→PortAudio and PortAudio→cable→ToneSphere 1 kHz within −0.009 to +0.000 dB, ToneSphere→Test Gain ×0.5→cable→PortAudio exactly half. Built and run by `scripts/vm/new_driver_vm.ps1` / `run_driver_tests.ps1`; logs in `driver/windows_virtual_audio/test-results/`. The run found the driver had never written into the cable. `docs/VIRTUAL_AUDIO_DRIVER.md` |
| **Linux virtual sink** | `pactl` null sink + ALSA `pulse` PCM | Kept as is | VERIFIED (CI) | `TestRealLinuxSink` in the Linux CI job. |
| **macOS virtual device** | AudioServerPlugIn in `native/coreaudio-plugin/` | Kept as is | VERIFIED (CI only) | `build-macos-plugin` job round-trips a 1 kHz sine. Day-to-day use UNVERIFIED. |
| **Built-in DSP** | Python: RBJ biquad EQ with a per-sample Python loop (`effects.py:174`), soft-knee compressor, delay, limiter without lookahead | Native ports of the same formulas | VERIFIED (offline) | `test_effects.py`, `test_dsp.py` signal tests. Reachable only through `AudioEngine.enable_*`; no UI/API/CLI caller. |
| **Network audio** | UDP transport with jitter buffer and paced send worker; TCP receive only. Threads touch rings, never the callback | Kept; feeds native external-port rings | VERIFIED (localhost) | `test_network_send_wiring.py` (1 kHz sine across two engines over localhost UDP). |
| **UI** | PySide6 mixer strips (per-side, per-channel meters; stereo balance; insert chains), patchbay, status bar; device configuration (backend, exclusive, buffer, sample rate); plugin browser (scan status and reason per module, custom folders); insert chain with parameters, bypass and the plugin's editor; Diagnostics (callback min/mean/p99/max, worst and mean load, xruns, audio-thread allocations, rings, nominal / driver-reported / plugin / measured latency, round-trip measurement, virtual-device presence). Scans and measurements run off the GUI thread | Control plane only | IMPLEMENTED; views VERIFIED offscreen | `test_ui.py`, `test_ui_views.py`, `test_i18n.py` (every string in en.json and hi.json). No built-in-effect (EQ/compressor) or network UI; strip channel swap is API/CLI only |
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
| Drift resampler across device clocks | VERIFIED (offline) + HARDWARE VERIFIED | `tests/native/test_boundary.py`: ±100, ±500 and ±3000 ppm absorbed with bounded fill, no underrun, no discontinuity (a click at ratio < 1 was found and fixed here); on hardware in the three-clock test. Rebuilt 2026-09-30 after the AI-04 exposed two faults: the loop steered towards a fill the packet cadence never averages to, warping every stream's first seconds (the AI-04's digital round trip failed 4 times in 10), and its ±0.1 % limit could not follow the AI-04's 44.1 kHz input, 0.2–0.3 % slow (355 frames lost in 20 s). Now: a calibration phase learns the real setpoint, a dead band keeps same-clock streams at exactly 1, and a proportional-integral loop within ±0.5 % follows real drift with the fill centred — round trip 10 of 10 at identical confidence; 44.1 kHz on the AI-04 0–65 frames lost at start, then none, the ratio following that input's ±0.15 % wander (`test_interface.py`, recorded xfail) |
| Sample-format conversion at the device boundary | VERIFIED | `TestConversion`: int16/24/32, float32/64 round trips within half a step; over-range clips instead of wrapping |
| WASAPI backend | HARDWARE VERIFIED | `docs/WINDOWS_AUDIO.md` |
| ASIO backend (separate GPLv3 DLL, attached through the external-backend C ABI) | HARDWARE VERIFIED against FlexASIO, and through FlexASIO on the AI-04 interface; a manufacturer's ASIO driver NOT AVAILABLE | `docs/ASIO.md` |

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
| Per-block load judged against each block's own period; a device period longer than the engine block is cut into equal blocks | VERIFIED | `test_load_is_judged_per_block_not_against_the_last_block`; `main.py test` on the development machine: exclusive 128 frames, worst callback 0.025 ms = 0.8 % of the period, mean 0.2 %, 0 xruns, 0 allocations. The AI-04's 144-frame exclusive period was cut 128 + 16, and the remainder's 0.33 ms budget reported 152 % load for periods that finished in under 3 % of their time; it is cut 2 × 72 now (14.0 % worst, of a 1.5 ms block), in the WASAPI and ASIO loops alike |
| Satellite cushion trimmed to its target on priming | VERIFIED (hardware) | digital round trip 61.35–65.35 ms over six starts (within one 480-frame period), fixed within a run |
| Stereo strip balance and channel swap on the native host | VERIFIED; HARDWARE VERIFIED (balance) | balance is folded into the channel trims by the engine's own law; swap is a native control crossfaded over one block (`test_swap_exchanges_the_channels_without_a_step`, `test_the_native_strip_applies_balance_and_swap`); at the speaker, balance 0.5 leaves the near side at unity and the far side at cos(π/4) within 2 % (`test_bypass_balance_and_per_channel_meters_are_what_is_heard`) |
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

## Known defects in the PortAudio host, and where the native host stands

The PortAudio host is the audio path on Linux and macOS (and on Windows only with
`TONESPHERE_HOST=portaudio`). The right-hand column is the same behaviour on the native host,
which is the default on Windows.

| Defect (PortAudio host) | Where | Consequence | Native host (Windows) |
|---|---|---|---|
| Buses never forward their input | `host.py` (no reader of device→bus rings) | device→bus→device carries silence; the rings overflow | **works** — `test_device_through_a_bus_to_another_device` (exact samples) |
| ~~Strip pan stored, never applied~~ | fixed in M10 on both hosts: a stereo strip's pan is its balance | — | works, HARDWARE VERIFIED |
| ~~In/out meter keys collide~~ | fixed in M10: meters are reported per side | — | works |
| Stream restart drops channel controls and plugins | `core/engine.py:1129-1135` | settings silently revert after reconfiguring | **works** — strips, plugins and insert state survive restarts (`test_the_engine_plays_through_a_plugin_and_keeps_it_across_a_restart`) |
| Stereo balance steps 3 dB off centre | `host.py:1186-1190`, `dsp.py:253-255` | level jump when a stereo pan leaves centre | **works** — unity at centre, far side only (`TestChannelMapping`) |
| Graph-level solo never populated | `core/engine.py:1009-1015` reads a routing-matrix field nothing sets | only per-device channel solo works | same (control plane); the native plan compiles solo correctly when it is set (`test_solo_silences_every_other_source`) |
| Duplex in→out refused as feedback | `core/engine.py:407-437` | the guitar path through `AudioEngine` cannot be made on one device id | **works** — only bus cycles are refused (`test_the_engine_refuses_nothing_it_used_to_accept`) |
| Limiter has no lookahead, ramp starts at the old envelope | `dsp.py:340-360` | first over-threshold block passes unreduced; with `clip_off=True` samples >1.0 can reach the driver | the native limiter is instant-attack and sample-accurate, so no sample above its ceiling passes (no lookahead either) — `tests/native/test_dsp.py::TestLimiter`, `test_the_sink_safety_limiter_catches_a_routing_mistake` |
| `pan_law_minus_6db` is not a -6 dB law | `dsp.py:27-52` | mislabelled option | not applicable (one pan law) |
| Dead code | `_bus_pending`, `GAIN_RAMP_FRACTION` in `host.py` | — | — |

## Migration progress

| Milestone | Scope | Status |
|---|---|---|
| M0 | `AGENTS.md` contract, `CLAUDE.md` pointer | done |
| M1 | This matrix, honesty fixes (reported vs measured latency, loopback claim, plugin and bus claims), dependency audit, shared test signals | done |
| M2 | Native foundation: toolchain, C ABI, SPSC, snapshots, offline `process_block` | done — see *Native engine* below |
| M3 | Native graph, mixer, DSP | done — see *Native engine* and *Measurements* below |
| M4 | Native WASAPI backend | done — see `docs/WINDOWS_AUDIO.md`; switching `AudioEngine` onto it moves to M8, so the host interface is designed once, with ASIO and VST3 in view |
| M5 | Native ASIO host | done — HARDWARE VERIFIED against FlexASIO (software ASIO driver), and through FlexASIO on the Audio Array AI-04; no manufacturer ASIO driver exists for that interface — see `docs/ASIO.md` |
| M6 | Measured round-trip latency | done — `tonesphere/native/roundtrip.py`; digital path measured, acoustic path unavailable on this machine |
| M7 | Native VST3 host | done — see `docs/VST3.md`; reachable from the UI (M10), not the API |
| M8 | Engine migration, persistence | done — `AudioEngine` runs on `NativeHost` on Windows; see *Engine on the native host* below |
| M9 | Windows virtual audio driver | done in a test VM — installs, enumerates, carries audio between applications at the level sent, and uninstalls cleanly (2026-09-30, Hyper-V, test-signing); real-desktop use and production signing not available — `docs/VIRTUAL_AUDIO_DRIVER.md` |
| M10 | UI | done — device configuration, plugin browser, insert chains and parameter editor, diagnostics with round-trip measurement, virtual-device status; meters per side and channel; strip balance made real |
| M11 | Validation, benchmarks, documentation | done — soak and restart test (30 min, 180,024 callbacks: 0 xruns, 0 audio-thread allocations; callback mean 65.9 µs, p99 ≤ 128 µs, worst 1.10 ms (11 % of the 10 ms period); bus ring underruns after start 0; private bytes 280.3 → 280.5 MB; 25 of 25 restarts running within 55–63 ms, handle count 352 → 352; `benchmarks/soak.py`); GPLv3 Corresponding Source packaging and licence texts in the binary; legal documents revised (T&C 1.1, ToS 1.2, Privacy 1.1 — **not lawyer-reviewed**); `docs/ARCHITECTURE.md`, `REALTIME.md`, `TESTING.md`, `ENGINEERING_REPORT.md`; README rewritten. Device removal during streaming: NOT VERIFIED |
| M12 | Hardware follow-up, 2026-09-30 | done — an Audio Array AI-04 USB interface through native WASAPI (exclusive 48 kHz at a 3 ms period, 0 dropouts; its 44.1 kHz input clock recorded as a known failure) and through FlexASIO; round trip through it `--` without a cable; the drift resampler rebuilt; the virtual driver HARDWARE VERIFIED in a Hyper-V VM after four real defects were found there; the Store package made MIT-only (no ASIO); docs published by GitHub Actions |
