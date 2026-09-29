---
title: Engineering report — Windows-native migration
layout: default
permalink: /engineering-report/
description: What the Windows-native migration built, what each part was proven to do and on what, what it found, and what still stands between it and a verified, distributable product.
---

# Engineering report: the Windows-native migration

Branch `windows-native`, 2026-09-29 and 30. Development machine: Intel Core i7-1165G7
(4C/8T), Windows 11 Pro 26200, Realtek ALC257 (speakers, headphone output, microphone),
Intel SST microphone array; from 2026-09-30 also an Audio Array AI-04 USB interface with a
guitar on its input and earphones on its output. No loopback cable. The virtual driver was
tested in a Hyper-V VM on the same machine.

Every figure below was measured on that machine unless it says otherwise, and cites the test
or tool that produced it. A figure that was not measured is `--`.

## 1. Outcome

ToneSphere on Windows now runs its audio on a native real-time engine — C++20 behind a flat
C ABI, loaded by ctypes — with native WASAPI and ASIO backends and a native VST3 host. The
Python control plane (`AudioEngine`) and its three front ends (Qt UI, REST API, CLI) run on
it unchanged, through a `NativeHost` implementing the interface the PortAudio host did. The
UI gained a plugin browser, insert chains with parameters and editors, and a diagnostics
view that measures the round trip instead of reporting it. A Windows kernel driver
publishing a virtual cable carries audio between applications at exactly the level sent,
**in a test VM**; it is test-signed, so it cannot be offered to anyone until Microsoft signs
it.

| Area | Level | Evidence (section) |
|---|---|---|
| Native engine: plans, mixer, strips, DSP, rings, resampler, meters, statistics | VERIFIED | §3.1 |
| Native WASAPI (shared, exclusive, raw, loopback, process loopback, multi-clock) | HARDWARE VERIFIED | §3.2 |
| Native ASIO host (separate GPLv3 DLL) | HARDWARE VERIFIED against FlexASIO (software driver), including FlexASIO on a USB interface (Audio Array AI-04); a manufacturer's ASIO driver **not available** (the AI-04 has none) | §3.3 |
| Native VST3 host | VERIFIED (test plugin); HARDWARE VERIFIED with Surge XT; commercial plugins **not tested** | §3.4 |
| Measured round trip | method HARDWARE VERIFIED on the digital path (Realtek and AI-04); through the AI-04 or acoustically `--` — no cable from output to input | §3.5 |
| `AudioEngine` on the native host, end to end to the speaker | HARDWARE VERIFIED | §3.6 |
| UI views | VERIFIED offscreen; the controls they drive HARDWARE VERIFIED | §3.7 |
| Soak and restart | HARDWARE VERIFIED, 30 min + 25 restarts | §3.8 |
| Native WASAPI on a USB interface (AI-04: exclusive 48 kHz at 3 ms, shared, monitoring) | HARDWARE VERIFIED; its 44.1 kHz input clock a recorded known failure | §3.10 |
| Windows virtual audio driver | **HARDWARE VERIFIED in a Hyper-V test VM** (install, enumeration, audio between applications at the level sent, clean uninstall); real-desktop use NOT VERIFIED; production signing **NOT AVAILABLE** | §3.9 |

The per-item matrix is [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

## 2. What was built

Twelve commits on `windows-native` (not pushed), 159 files, about 33,700 lines added:

| Milestone | Commit | What |
|---|---|---|
| M0 | `9abc9da` | `AGENTS.md` rewritten as the project contract: real-time rules, native build rules, driver rules, licensing, evidence levels |
| M1 | `3eac2dd` | The reality matrix; "measured latency" that was driver-reported renamed and its figure withdrawn; a whole-system-loopback claim removed; `requests` dropped, packaging tools moved to a group; shared test signals |
| M2 | `9be19ee` | Toolchain (EWDK, CMake and Ninja from PyPI), the C ABI, wait-free SPSC ring, plan exchange by atomic swap and hazard pointer, allocation counter, offline `process` |
| M3 | `d97cffb` | Mixer strips, 8-band RBJ EQ, compressor, delay, limiter, drift resampler, sample conversion — ported against the Python reference |
| M4 | `b92cb9c` | Native WASAPI: enumeration with endpoint IDs, notifications, event-driven shared/exclusive, raw mode, loopback, process loopback, MMCSS, master and satellite clocks |
| M5 | `da42996`, `2275220` | Native ASIO host as a separate GPLv3 DLL on the external-backend ABI; verified against FlexASIO |
| M6 | `2333b6d` | Round-trip measurement: exponential sweep teed inside the engine, GCC-PHAT, confidence threshold |
| M7 | `5a06e18` | Native VST3 host on SDK 3.8.1; subprocess scanner; SEH fault isolation; deterministic test plugins (gain+latency, crash on load, crash on the audio thread); pedalboard (GPLv3, unreachable, never proven) removed |
| M8 | `f0b334c` | `AudioEngine` on `NativeHost`; plugins per device side; presets v2 with plugin chains and endpoint IDs |
| M9 | `532d041` | Kernel-mode loopback-cable driver from Microsoft's SimpleAudioSample; build and VM-kit scripts; install/uninstall scripts; hardware tests that skip without it |
| M10 | `b19bf1d` | UI: diagnostics, plugin browser, insert chains, parameter editor, sample rate; per-channel and per-side meters; strip balance and native channel swap made real; host bypass |
| M11 | `6727a7b` | Soak test, Corresponding Source packaging, licence texts in the binary, legal documents, architecture/real-time/testing docs, README, this report |

| M12 | this commit | The AI-04 interface on native WASAPI and through FlexASIO; the drift resampler rebuilt (calibration, dead band, proportional-integral); device periods cut into equal engine blocks; the virtual driver verified in a Hyper-V VM, with the VM built and driven by committed scripts, and the driver defects that found; the Store package made MIT-only; docs published by GitHub Actions |

## 3. Evidence

Final run on the development machine: `uv run pytest -m "not hardware"` — 836 passed, 2
skipped (a Linux-only test; `makeappx` validation, which needs the Windows SDK on PATH), 1
expected failure (the PortAudio host's bus defect, strict); `uv run pytest -m hardware` — 33
passed, 5 skipped (the virtual-driver tests: the driver is not installed on this machine,
by rule). `uv run ruff check .` clean. The suite before the migration: 660 passed.

### 3.1 Native engine (offline, CI)

148 tests in `tests/native/`, all passing, run in CI on Windows after the SDK fetch and build.
Highlights: ring integrity under a two-thread stress test with sample-exact checking; 300
plan swaps under a concurrently running audio thread with every block finite and bounded;
EQ equal to the Python `Biquad` within 2e-6 for all five filter types; a NaN at a source
never reaching an output; `rt_allocations == 0` over 500 blocks with rings, buses, ramps
and swaps.

Offline cost (`benchmarks/results/m3_offline_i7-1165G7.json`, MMCSS "Pro Audio"): a guitar
chain — EQ, compressor, bus, delay, limiter — at 48 kHz / 256 frames averages about 24 µs a
block, p99 32 µs (0.45 % / 0.6 % of the 5.33 ms period).

### 3.2 WASAPI (`tests/hardware/test_wasapi.py`)

- A tone rendered to the default output and captured back through process loopback is
  **bit-exact**, 73.4 ms later.
- Exclusive mode is granted at a 24-in-32-bit container, 480 frames.
- Three streams on three independent device clocks run together through the drift
  resamplers.
- `main.py test`, exclusive at 128 frames: callback mean 0.007 ms, p99 0.016 ms, worst
  0.020 ms (0.7 % of the device period), 0 xruns, 0 audio-thread allocations; reported
  round trip 3.0 ms, measured `--` (no loopback path in that test).
- The Realtek driver's own "enhancements" add +9.7 dB to anything rendered, proven by
  rendering the same tone through PortAudio. Raw mode, now the default, bypasses them; the
  endpoint's loopback still shows a high shelf from effects raw mode cannot remove.

### 3.3 ASIO (`tests/hardware/test_asio.py`)

Against FlexASIO 1.10b (installer SHA-256 `FE496BCC…031209`; unsigned): 80 of 80 buffer
switches at 48 kHz / 882 frames, mean 35 µs, 0 xruns, a 1 kHz output captured back at 1 kHz.
Through FlexASIO to the Audio Array AI-04 in WASAPI exclusive at 144 frames: 499 buffer
switches in 1.5 s, none missed, max 18.2 µs, 0 xruns, input 1 carrying the guitar's
50.10 Hz mains hum at −19.5 dBFS; in FlexASIO's default shared mode, the output came back
through process loopback at exactly the level sent. **No manufacturer's ASIO driver has
been tested**: the AI-04's maker publishes none.

### 3.4 VST3 (`tests/native/test_vst3.py`, `tests/hardware/test_vst3_third_party.py`)

- ToneSphere Test Gain: output bit-exact to gain × input delayed by the reported 64
  samples; measured latency equals reported; parameter changes land within a block; state
  round-trips.
- A plugin that crashes in `initialize` is reported and never inserted; one that crashes on
  its tenth `process` call is bypassed from that block on, the engine keeps running, and
  the fault is reported.
- **Surge XT 1.3.4** (Surge XT Effects, delay): echo measured at 250.06 ms for a 250 ms
  setting; bypass, state round-trip and the editor window all work.
- **Not tested:** any commercial plugin, instruments (refused), sidechains, sample-accurate
  automation.

### 3.5 Round trip (`tests/hardware/test_roundtrip.py`)

The digital path (output → its own loopback) measures 61.35–65.35 ms over six starts at
48 kHz / 480 shared, repeatable within one device period and fixed within a run
(confidence 26.9 at 62.35 ms; an independent cross-correlation in `test_wasapi.py` agrees
to the frame). Speaker → microphone: no path above the confidence threshold on this
laptop, so **the acoustic round trip is `--`**. The engine records a loopback measurement
as the digital path and never reports it as the round trip. On the AI-04 the digital path
measures 33.35 or 43.35 ms (a period apart by start phase), identical within a run, 10 of 10
at one confidence — after the resampler fix in §4; before it, 6 of 10. Through the AI-04
itself (earphones on its output, a guitar on its input) there is no path: confidence 1.0,
so `--` until a cable joins output and input.

### 3.6 End to end (`tests/hardware/test_engine_native_host.py`)

`AudioEngine` → `NativeHost` → VST3 test plugin at ×0.5 → WASAPI → Windows, heard back by
process loopback in a separate engine: RMS 0.07071 against 0.07071 expected. Survives a
restart with its fader and plugin; a preset restores the chain with the plugin's state.
Balance 0.5 leaves the near side at unity and the far side at cos(π/4) within 2 %; a
bypassed plugin leaves the signal untouched and stops counting its 64 samples of latency;
per-channel meters agree with the heard tone within 0.3 dB.

### 3.7 UI

`tests/test_ui_views.py`, `tests/test_ui.py`, `tests/test_i18n.py` (offscreen): a duplex
device's input and output strips are metered separately; a side with no reading greys out
rather than showing silence; the balance knob appears only on stereo strips; nothing
unmeasured is shown as a number; a round trip is shown only for the configuration it was
measured at; the plugin browser lists crashed and wrong-architecture modules with their
reasons and will not insert them or an instrument; every string is in both catalogues. On
hardware, the inserts dialog's slider reaches the plugin and what it then shows is the
plugin's own read-back.

### 3.8 Soak and restarts (`benchmarks/soak.py`)

The UI's own path — `AudioEngine`, native host, WASAPI shared at 48 kHz / 480 frames on the
Realtek headphone output — with a bus fed pink noise in real time by a Python thread and
Surge XT Effects then the test plugin (at gain 0, so nothing is audible) on the output, for
30 minutes, then 25 stop/start cycles (`benchmarks/results/soak_30min_i7-1165G7.json`):

| | |
|---|---|
| Callbacks | 180,024, engine running at every sample |
| Xruns | 0 |
| Audio-thread heap allocations | 0 |
| Callback mean / p99 / worst | 65.9 µs / ≤ 128 µs / 1.10 ms (11 % of the 10 ms period) |
| Bus ring underruns after start | 0 |
| Output meter | silent throughout; bus meter carried the noise throughout |
| Private bytes, first → last sample | 280.3 → 280.5 MB |
| Restarts | 25 of 25 running; 55–63 ms each; 0 xruns after each; handles 352 → 352 |

Not covered: device removal during the run (not automated), and plugins other than these two.

### 3.9 Virtual audio driver

Built with the EWDK, test-signed with the WDK test certificate, `InfVerif /w` clean — and on
2026-09-30 installed and tested in a Hyper-V VM (Windows 11 Enterprise LTSC Evaluation
10.0.26100, test-signing on in the VM disk's own boot store, Secure Boot off), built by
`scripts/vm/new_driver_vm.ps1` and driven by `scripts/vm/run_driver_tests.ps1`.
`tests/hardware/test_virtual_driver.py`, 7 of 7:

| | |
|---|---|
| Device and endpoints | `ROOT\MEDIA\0000` OK; "Speakers (ToneSphere Virtual Audio Cable)" and "Microphone Array (ToneSphere Virtual Audio Cable)", both 48 kHz stereo |
| PortAudio process → cable → PortAudio process | 1000.00 Hz, +0.000 to −0.009 dB over three passes |
| PortAudio → cable → ToneSphere's engine | 1000.00 Hz, +0.000 to −0.009 dB |
| ToneSphere → Test Gain ×0.5 → cable → PortAudio | 1000.00 Hz, rms 0.07071 = exactly half |
| Idle; a new capture after the player stopped | exact silence; nothing replayed |
| `AudioEngine.virtual_device_status()` | installed, by the names Windows gives |
| Uninstall | device, driver-store package and certificate trust gone; nothing left |

The first passes found that **the driver had never written into the cable** (the sample's
render-side writer is gated on data files, which are off), that a new capture replayed the
last 100 ms of an earlier one, that the tests could not have run (floats passed to
`Popen`), and that the app looked for names Windows does not show. Also fixed on the way:
the scripts' certificate import (refused over PowerShell Direct), the build's date check
(inf2cat against UTC), a read-only file breaking the kit rebuild, and a checkpoint that
replayed OOBE's restart. Not verified: a real desktop with Discord or OBS, sleep/resume,
many clients. Custom endpoint names are not implemented (the Windows Driver INF rules
refuse the registry route).

### 3.10 A USB interface (`tests/hardware/test_interface.py`)

The Audio Array AI-04 (USB, Windows' class driver), a guitar on input 1:

| Mode | Period | Callback max | Worst load | Dropouts |
|---|---|---|---|---|
| Exclusive 48 kHz | 144 frames, 3.0 ms reported | 209 µs | 14.0 % of a 1.5 ms block | none |
| Exclusive 44.1 kHz | 132 frames | 153 µs | 10.2 % | 0–65 input frames at start (known failure) |
| Shared 48 kHz | 480 frames | 108 µs | 1.1 % | none |

The input carried the guitar pickup's 50 Hz mains hum at −19.9 dBFS; input → gain bus →
output gave the output exactly input × 0.01, sample for sample. At 44.1 kHz this device's
input runs 0.2–0.3 % slow against its own output, and wanders ±0.15 %. Nothing listened to
the output jack.

## 4. Real defects found and fixed

Found by signal tests during the migration, each now covered by one:

- asymmetric integer scaling at the device boundary (2^(N-1) scale, clip at the positive
  maximum);
- a resampler click when the ratio dropped below 1;
- a p99 latency reported above the maximum (histogram edge);
- processing load computed against a split block's tiny remainder period, not the block's
  own — fixed by one engine run per device callback, with the device period passed apart;
  and, where a device's minimum period exceeds the engine block (the AI-04's 144 frames
  against 128), by cutting the period into equal blocks: 152 % reported before, 14 % now;
- a drift resampler that steered every stream towards a fill level packet delivery never
  averages to, time-warping its first seconds (4 in 10 digital round trips on the AI-04
  failed), and whose ±0.1 % limit could not follow a USB input 0.2–0.3 % slow (355 frames
  lost per 20 s) — now calibrated, dead-banded, proportional-integral within ±0.5 %;
- excess satellite latency from resampler priming, now trimmed to its target;
- a race on stream error strings;
- a nested audio-thread scope that reset the allocation flag;
- the strip pan knob, which moved nothing on either host;
- `swap_channels`, which did nothing on the native host;
- input and output meters of one device overwriting each other;
- the UI drawing one aggregate meter reading as two channels;
- the diagnostics claiming "shared" and "0 xruns" while stopped, when nothing had been
  measured;
- in the virtual driver, found in the VM: no audio ever entering the cable, and a new
  capture replaying the end of an earlier one.

## 5. What remains, and what blocks it

| Item | Blocker | Owner |
|---|---|---|
| Production-signed driver | EV code-signing certificate and a Partner Center hardware account (attestation signing) | **owner**, cost and identity verification |
| The driver on a real desktop, with Discord or OBS | the driver is test-signed: only a test-signed machine or VM | **owner** / after signing |
| A measured round trip through the interface | a cable from the AI-04's output to its input; then `uv run pytest tests/hardware/test_roundtrip.py -m hardware -s -k interface` | **owner** (a cable) |
| ASIO with a manufacturer's ASIO driver | an interface whose maker ships one (the AI-04 has none) | **owner** |
| Commercial VST3 plugins | licences for them | **owner** |
| Legal review of T&C 1.2, ToS 1.3, Privacy 1.2 | a lawyer | **owner** |
| Store submission | Partner Center identity, and a check of the Store's current licence-terms fields; the package is now MIT-only by decision | **owner** |
| The AI-04 at 44.1 kHz | its input clock; a larger satellite cushion would trade latency for the start-up loss | project |
| Custom endpoint names for the driver | a `KSPROPERTY_PIN_NAME` handler in the driver | project |
| Plugins from the REST API and CLI; MIDI for instruments; built-in effects in the UI; whole-system loopback in the routing; reopening a device that returns; the PortAudio host's bus defect; presets stored under the user data folder instead of the working directory | engineering time | project |
| Driver build in CI | unverified whether the hosted runners' WDK builds it | project |

## 6. Licensing and distribution state

- ToneSphere's own source: MIT. `native/asio/`: GPLv3. The driver: MS-PL. The VST3 SDK
  (MIT) and the ASIO SDK (GPLv3) are fetched at build time, pinned, never committed.
- The Windows executable bundles the ASIO DLL and is therefore GPLv3 as a whole. It now
  carries the GPLv3 text and both SDK licence files, and the release job publishes its
  Corresponding Source next to it (`scripts/package_source.py`: 38.7 MiB, the tree, the
  ASIO SDK as fetched, the compiled parts of the VST3 SDK). Checked: extracted on its own,
  it rebuilds both DLLs with no download, and the 148 native tests pass against them.
  CI first built it on 2026-09-30, for this branch's pull request (38 MiB, uploaded as a
  run artifact beside the executable); publishing both on a version tag has not happened
  yet. That same first CI run found the Windows build could not find the compiler on
  GitHub's runners (an environment-variable name compared case-sensitively); fixed, and the
  Windows job now builds the native engine and passes 833 tests there.
- **The Microsoft Store package leaves ASIO out** and is MIT only (`build_msix.ps1`), so
  the owner can set its price and terms: ToneSphere is free today and may be paid later.
  GitHub releases keep ASIO under GPLv3 with their source. A paid build with ASIO would
  need Steinberg's proprietary ASIO licence signed first. Contributions come in under MIT
  with a DCO sign-off (`CONTRIBUTING.md`).
- Terms and Conditions 1.2, Terms of Service 1.3 and Privacy Policy 1.2 describe that —
  free today, later versions or editions possibly paid, a copy already obtained keeps its
  licence, who would process a payment — as well as the GPLv3 build, and state that nothing
  in them restricts a GPLv3 right. **These are drafted by an engineer, not a lawyer; have
  them reviewed before relying on them.**

## 7. How to reproduce

```
uv sync --all-groups
uv run python scripts/fetch_sdks.py
uv run python scripts/build_native.py
uv run pytest -m "not hardware"
uv run pytest -m hardware                 # FlexASIO and Surge XT for their tests
uv run python benchmarks/bench_engine.py
uv run python benchmarks/soak.py --minutes 30
uv run python main.py test
```
