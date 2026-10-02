---
title: Engineering report — Windows-native migration
layout: default
permalink: /engineering-report/
description: What the Windows-native migration built, what each part was proven to do and on what, what it found, and what still stands between it and a verified, distributable product.
---

# Engineering report: the Windows-native migration

Branch `windows-native`, 2026-09-29 to 2026-10-01. Development machine: Intel Core i7-1165G7
(4C/8T), Windows 11 Pro 26200, Realtek ALC257 (speakers, headphone output, microphone),
Intel SST microphone array; from 2026-09-30 also an Audio Array AI-04 USB interface — first
with a guitar and earphones, then with a 6.35 mm cable from its headphone output to its input.
The virtual driver was tested in a Hyper-V VM on the same machine; Linux routing in WSL2.

Every figure below was measured on that machine unless it says otherwise, and cites the test
or tool that produced it. A figure that was not measured is `--`.

## 1. Outcome

ToneSphere on Windows now runs its audio on a native real-time engine — C++20 behind a flat
C ABI, loaded by ctypes — with native WASAPI and ASIO backends and a native VST3 host. The
Python control plane (`AudioEngine`) and its three front ends (Qt UI, REST API, CLI) run on
it unchanged, through a `NativeHost` implementing the interface the PortAudio host did. The
UI gained a plugin browser, insert chains with parameters and editors, and a diagnostics
view that measures the round trip instead of reporting it. A Windows kernel driver
publishing virtual cables — two by default, up to eight, each managed from the UI — carries
audio between applications at exactly the level sent, **in a test VM**; it is test-signed,
so it cannot be offered to anyone until Microsoft signs it. The second half of the work
(M13–M20) closed every other gap the README listed: a measured round trip through an
interface and acoustically, a thread-safe control plane, devices that come and go, whole-
system loopback as a source, built-in effects in the UI, TCP send, an adaptive jitter buffer,
Opus, bus routing on the Linux/macOS host, plugins from the REST API and CLI, and instruments.

| Area | Level | Evidence (section) |
|---|---|---|
| Native engine: plans, mixer, strips, DSP, rings, resampler, meters, statistics | VERIFIED | §3.1 |
| Native WASAPI (shared, exclusive, raw, loopback, process loopback, multi-clock) | HARDWARE VERIFIED | §3.2 |
| Native ASIO host (separate GPLv3 DLL) | HARDWARE VERIFIED against FlexASIO and ASIO4ALL, both on a USB interface (Audio Array AI-04), and ASIO4ALL in the driver VM; a manufacturer's ASIO driver **not available** (the AI-04 has none) | §3.3 |
| Native VST3 host | VERIFIED (test plugin); HARDWARE VERIFIED with Surge XT; commercial plugins **not tested** | §3.4 |
| Measured round trip | HARDWARE VERIFIED: digital path; through the AI-04 and a cable (WASAPI and ASIO); acoustically on the laptop, just above the confidence threshold | §3.5 |
| `AudioEngine` on the native host, end to end to the speaker | HARDWARE VERIFIED | §3.6 |
| UI views | VERIFIED offscreen; the controls they drive HARDWARE VERIFIED | §3.7 |
| Soak and restart | HARDWARE VERIFIED, 30 min + 25 restarts | §3.8 |
| Native WASAPI on a USB interface (AI-04: exclusive 48 kHz at 3 ms, shared, monitoring) | HARDWARE VERIFIED; its 44.1 kHz input clock a recorded known failure | §3.10 |
| Windows virtual audio driver, several cables | **HARDWARE VERIFIED in a Hyper-V test VM** (two isolated cables, audio between applications at the level sent, add/rename/disable/enable/uninstall from the UI, a cable disappearing under the engine and reopened, clean uninstall); real-desktop use NOT VERIFIED; production signing **NOT AVAILABLE** | §3.9 |
| Threads, devices that come and go, whole-system loopback, built-in effects in the UI, network (TCP, adaptive jitter, Opus), PortAudio bus routing, plugins from REST/CLI, instruments | VERIFIED; the audio paths HARDWARE VERIFIED (AI-04, WSL2 PulseAudio, the VM) | §3.11 |

The per-item matrix is [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

## 2. What was built

The milestones on `windows-native` (pushed; pull request #1):

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

| M12 | `1ecdb5c`…`11586a6` | The AI-04 interface on native WASAPI and through FlexASIO; the drift resampler rebuilt (calibration, dead band, proportional-integral); device periods cut into equal engine blocks; the virtual driver verified in a Hyper-V VM, with the VM built and driven by committed scripts, and the driver defects that found; the Store package made MIT-only; docs published by GitHub Actions |
| M13 | `5470caf` | Measured round trip through the AI-04 and a cable, WASAPI and ASIO (`measure_asio`); ASIO4ALL on the AI-04 |
| M14 | `42e1eb8` | A threaded, thread-safe control plane: one lock per engine object; the UI's engine calls on a worker and a poller; REST handlers in the threadpool |
| M15 | `c353949` | A device monitor that reopens what changed; every output's loopback a routable source; the built-in effects in the Inserts dialog |
| M16 | `095207a` | TCP send, an adaptive jitter buffer, Opus through libopus |
| M17 | `eeaac3a` | Buses forward on the PortAudio host: device → bus → device on Linux and macOS |
| M18 | `3313881` | Plugins from the REST API and CLI; VST3 instruments played by MIDI |
| M19 | this series | Several virtual cables, managed from the UI; devices disappearing, proven on the cables; ASIO4ALL in the VM |
| M20 | this series | The acoustic round trip; documentation |

## 3. Evidence

Final run on the development machine, 2026-10-01: `uv run pytest -m "not hardware"` — 916
passed, 3 skipped (two Linux-only tests; `makeappx` validation, which needs the Windows SDK
on PATH). `uv run ruff check .` clean. The hardware tests were run milestone by milestone on
this machine (the virtual-driver tests skip here by rule: the driver is installed only in the
VM) and in the VM (§3.9). The suite before the migration: 660 passed.

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

Against **ASIO4ALL 2.22** (WDM-KS) on the AI-04 with its output cabled to its input: the
output heard at +0.00 dB from native WASAPI exclusive over the same cable; a measured round
trip of 15.58 / 18.27 / 23.58 / 34.27 ms at 64 / 128 / 256 / 512 frames, identical to the
frame over three runs. In the driver VM on ToneSphere's cables: 48 kHz from the buffer switch
after a 0.4 s start, and the output through a cable at +0.00 dB; its round trip there `--`
(confidence 2.1 in one pass; 29.94 ms at 512 frames, confidence 79, in the next). Chasing a
hang there found one of ToneSphere's own: stopping a WASAPI stream whose device event never
stopped firing waited for ever, because the stop event was the second handle waited on;
fixed (`docs/ASIO.md`). ASIO4ALL was uninstalled from this machine afterwards.

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
48 kHz / 480 shared, repeatable within one device period and fixed within a run (confidence
26.9 at 62.35 ms; an independent cross-correlation in `test_wasapi.py` agrees to the frame).
The engine records a loopback measurement as the digital path and never reports it as the
round trip.

| Path | Measured round trip |
|---|---|
| AI-04 output → 6.35 mm cable → input, WASAPI exclusive, 3 ms period | 17.92 ms (860 frames; 15.94 ms in 2 of 10 starts, a USB packet group earlier) |
| the same, WASAPI shared, 480 frames | 76.9 ms |
| the same, ASIO4ALL, 64 / 128 / 256 / 512 frames | 15.58 / 18.27 / 23.58 / 34.27 ms |
| the laptop's speakers → its microphone array (acoustic, raw capture) | 74.60–75.90 ms in 9 of 10 runs, confidence 4.3–5.1 against a threshold of 4.0; the tenth `--` at 3.8 |

The cable path captured the −18 dBFS sweep at about −8 dBFS: no clipping. For the acoustic
figure the Realtek speakers, found muted, were held unmuted at 60 % for the sweeps only and
put back exactly (`tests/hardware/endpoint_volume.py`); it is a real figure, only just above
the threshold.

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

Not covered: plugins other than these two. Device removal is covered since M19, on the
virtual cables in the VM (§3.9).

### 3.9 Virtual audio driver

Built with the EWDK, test-signed with the WDK test certificate, `InfVerif /w` clean, and
tested in a Hyper-V VM (Windows 11 Enterprise LTSC Evaluation 10.0.26100, test-signing on in
the VM disk's own boot store, Secure Boot off), built by `scripts/vm/new_driver_vm.ps1` and
driven by `scripts/vm/run_driver_tests.ps1`. Each cable is its own device instance, with its
own buffer. `tests/hardware/test_virtual_driver.py`, 14 of 14 (2026-10-01):

| | |
|---|---|
| Fresh install | "ToneSphere Cable 1" and "ToneSphere Cable 2", `ROOT\MEDIA\0000` and `0001`, four endpoints at 48 kHz stereo |
| PortAudio process → each cable → PortAudio process | 1000.00 Hz, +0.000 dB on both |
| Isolation | a tone into cable 1: cable 2's peak exactly 0 |
| PortAudio → cable → ToneSphere; ToneSphere → Test Gain ×0.5 → cable → PortAudio | −0.009 dB; exactly half |
| ffmpeg (DirectShow) recording what ToneSphere plays into a cable | 1000.00 Hz, +0.000 dB |
| Idle; a new capture after the player stopped | exact silence; nothing replayed |
| Add, rename, uninstall a cable through the app | "Chat" added, carrying audio, renamed "Game" (its endpoints with it), uninstalled; cable 1 unaffected |
| The Virtual Cables dialog's own buttons | a real disable and enable of cable 2 |
| A cable disabled under a running engine | reported gone, the engine running on without it (cable 1 carrying 440 Hz), reopened on return with no user action, 1 kHz back at −0.009 dB |
| A cable in use elsewhere | disabled under a PortAudio recorder; refused cleanly while another native-engine program held it; changed once it let go |
| Uninstall | every cable, the driver-store package and the certificate trust gone; nothing left |

The passes found: the driver had never written into the cable; a new capture replayed the
end of an earlier one; the tests could not have run (floats passed to `Popen`); the app
looked for names Windows does not show; the sample's one-device guard failed every cable
after the first (`STATUS_DEVICE_BUSY`); a name set before the driver was installed was
replaced by the INF's; disabling a cable ToneSphere streamed through left it "pending a
restart" — Windows' audio engine vetoes taking a device from a native-engine stream — and a
vetoed persistent disable still recorded one; a cable's two endpoints settle hundreds of
milliseconds apart. Not verified: a real desktop with Discord or OBS, sleep/resume, many
clients. Custom pin names are not implemented; the cable's own name is in both endpoint names.

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

### 3.11 Closing the gaps (M14–M18)

- **Threads** (`tests/test_threading.py`): eight threads for five seconds of connects,
  disconnects, gains, inserts, statistics, bus writes and rebuilds on one engine; without the
  locks it fails at once (a `KeyError` mid-rebuild), with them a 1 kHz signal still arrives at
  the expected level afterwards. The Qt event loop keeps turning while an engine call takes
  two seconds.
- **Whole-system loopback** (`tests/hardware/test_loopback_source.py`): two other processes
  played 1 kHz at 0.1 and 440 Hz at 0.05 into the AI-04; its loopback, routed through a bus,
  read 0.1000 and 0.0500.
- **Built-in effects in the UI** (`tests/native/test_builtin_effects_ui.py`, driving the
  dialog's widgets): a 1 kHz high-pass takes 100 Hz down 40.0 dB and 5 kHz 0.01 dB; the
  compressor's 10.3 dB of reduction matches its 10.31 dB readout; a 100 ms delay set there was
  heard through the AI-04's cable at 100.000 ms.
- **Network**: TCP both ways, sample for sample, with backpressure; the adaptive jitter buffer
  lost 0 of 7,490 packets under ±15 ms of jitter where a fixed 10 ms buffer lost 32 %, and
  0–0.31 % through a real 0–30 ms relay; Opus 161 bytes per 10 ms at 40.4 dB SNR on a tone.
- **The PortAudio host's buses**: device → bus → device through a real PulseAudio server in
  WSL2 (Ubuntu 24.04), 0.1768 rms as expected, 5 of 5, and in the Linux CI job.
- **Plugins beyond the UI**: REST and CLI load, chain, set and bypass plugins; Surge XT and
  Dexed played in tune by MIDI from the engine, REST, the on-screen keyboard and through the
  AI-04's cable.

### 3.12 The product a user downloads (M21–M24)

v0.2.0, installed by the owner on a clean machine, opened no window, showed meters but no
sound on Monitor Input, called Guitar Rig "crashed" and added no cables. Every one of those
paths had tests; none of the tests drove what a user downloads. So:

- **Packaging** (`.github/scripts/smoke_test_release.py`, every CI run): the Windows
  installer is installed silently and the installed app started with no arguments; the
  portable zip, the AppImage and the app from the mounted `.dmg` are started the same way.
  Each writes, once its window is up, what it is running on; the test checks the window, the
  backend and (Windows) the native engine. The frozen scanner reads the MIT test plugin.
- **Guitar Rig 7.0.1** (`tests/hardware/test_guitar_rig.py`): scanned in 0.7 s where it had
  timed out; its own −12.0 dB master volume moves the output −12.04 dB; 60 s with no fault.
- **The built app, through its own window** (UI Automation from PowerShell, AI-04, guitar on
  input 1, measured from another process on the output's loopback — `first_run_probe.py`):
  Monitor Input preselected Line (AI-04), input 1 to both ears, Speakers (AI-04); on Start the
  headphones carried the guitar 3.02 dB below the input (mono law 3.01 dB; coherence 0.976);
  Guitar Rig added from the browser and set from the parameter list to −12.0 dB lowered it
  12.20 dB; after close and reopen, the route and the plugin at its setting came back
  (−15.05 dB). `tests/hardware/test_first_run.py` repeats the measured part against the
  built app on a prepared session. Record: `docs/FIRST_RUN_VERIFICATION.md`.

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
- in the virtual driver, found in the VM: no audio ever entering the cable, a new capture
  replaying the end of an earlier one, and a second cable refused by the sample's
  one-device guard;
- a network sender that read one block per late timer wake, delivering a third less audio
  than it was given; outgoing TCP connections closed as they opened;
- the PortAudio host's buses never forwarding their input;
- a monitor thread alive at interpreter exit taking the process down (0xC0000409) after
  every test had passed;
- a WASAPI stop that never returned when a device's event never stopped firing (the stop
  event was the second handle waited on, and `WaitForMultipleObjects` reports the first);
  WASAPI capture threads draining packets in an unbounded loop that never looked at the stop
  event (now bounded per wake-up); and an ASIO host that would wait for ever on a driver
  stuck in `stop()` (now abandoned after 5 s and reported);
- a disable of a cable in use leaving it "pending a restart", and a device change recorded
  while the cable's second endpoint had not yet gone;
- found by the owner on v0.2.0, then by driving the built app: no window from a no-argument
  launch; a monitor route created muted on an engine nobody started; a MASTER meter that read
  the inputs; routes made before Start that stayed silent, and strip settings not re-applied
  to a new plan; a scanner that took a plugin's teardown fault, or a helper process holding
  its pipes, for a crash; a plugin setting changed while nothing processed the plugin missing
  from its saved state (Guitar Rig 7; now flushed by a zero-sample `process()`); an Exclusive
  button that said exclusive after a shared session was restored; tests writing into the
  developer's real settings folder;
- a jitter buffer that never recovered from one sender stall longer than its depth:
  playout ran on ahead of the sender at the same rate, and every packet after the stall
  arrived just after its slot and was dropped (macOS CI, Opus over UDP: 200 received, 200
  lost). A run of eight late packets into an empty buffer now realigns it.

## 5. What remains, and what blocks it

| Item | Blocker | Owner |
|---|---|---|
| Production-signed driver | EV code-signing certificate and a Partner Center hardware account (attestation signing) | **owner**, cost and identity verification |
| The driver on a real desktop, with Discord or OBS | the driver is test-signed: only a test-signed machine or VM | **owner** / after signing |
| ASIO with a manufacturer's ASIO driver | an interface whose maker ships one (the AI-04 has none) | **owner** |
| Commercial VST3 plugins beyond Guitar Rig 7.0.1; Neural Amp Modeler | licences; NAM's installer is unsigned | **owner** |
| Code-signed downloads (no SmartScreen or Gatekeeper warning) | a code-signing certificate; an Apple Developer ID | **owner** |
| Discord as a capture source, tested | it played nothing during the run and exposes no controls to automation; per-app capture itself is HARDWARE VERIFIED | project |
| MIDI from a hardware keyboard | a MIDI device | **owner** |
| macOS day to day, and bus routing with real macOS devices | a Mac | whoever owns one |
| An intermittent exclusive-mode round trip through a virtual cable in the VM: the capture holds the whole sweep, the correlation sometimes fails on it | cause not established; `docs/VIRTUAL_AUDIO_DRIVER.md` | project |
| Plugin isolation (a plugin can still take the process down) | a plugin host process | project |
| Legal review of T&C, ToS, Privacy | a lawyer | **owner** |
| Store submission | Partner Center identity; the package is MIT-only by decision | **owner** |
| The AI-04 at 44.1 kHz | its input clock; a larger satellite cushion would trade latency for the start-up loss | project |
| Custom pin names for the driver | a `KSPROPERTY_PIN_NAME` handler in the driver | project |
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
  run artifact beside the executable). Since M21 a release is published on every green
  push to main, its source archive in the `gpl-source` release, linked from its notes. That same first CI run found the Windows build could not find the compiler on
  GitHub's runners (an environment-variable name compared case-sensitively); fixed, and the
  Windows job now builds the native engine and passes 833 tests there.
- **The Microsoft Store package leaves ASIO out** and is MIT only (`build_msix.ps1`), so
  the owner can set its price and terms: ToneSphere is free today and may be paid later.
  GitHub releases keep ASIO under GPLv3 with their source. A paid build with ASIO would
  need Steinberg's proprietary ASIO licence signed first. Contributions come in under MIT
  with a DCO sign-off (`CONTRIBUTING.md`).
- Terms and Conditions 1.3 (where the GPLv3 source is published), Terms of Service 1.4 and
  Privacy Policy 1.3 (local logs on by default, local crash reports, the session file)
  describe that — free today, later versions or editions possibly paid, a copy already
  obtained keeps its licence, who would process a payment — as well as the GPLv3 build, and
  state that nothing
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
