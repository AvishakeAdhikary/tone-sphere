---
title: ASIO
layout: default
permalink: /asio/
description: ToneSphere's native ASIO host — how it works, its licence, and exactly what has and has not been verified.
---

# ASIO

`native/asio/` builds `tonesphere_asio.dll`, a native ASIO host; `tonesphere/native/asio.py`
binds it.

## Status

| | |
|---|---|
| ASIO host implementation | **IMPLEMENTED** — driver discovery, loading, initialisation, channel/format/rate/buffer negotiation, `createBuffers`, a native buffer switch driving the engine, `asioMessage` handling, stop/dispose/release |
| Boundary tests without a driver | **VERIFIED** — `tests/native/test_asio.py`: every Windows ASIO sample type round-trips, aligned types put the sample in the low bits, over-range clips, big-endian types are refused, a missing driver fails with the reason |
| Against a real ASIO driver: **FlexASIO 1.10b** (software ASIO driver) | **HARDWARE VERIFIED** on the development machine, 2026-09-29 — see below |
| Through FlexASIO to a USB audio interface, **Audio Array AI-04**, WASAPI exclusive | **HARDWARE VERIFIED** 2026-09-30: the interface's clock drives the buffer switch at 144 frames, and its input delivers a real signal — see below |
| Against **ASIO4ALL 2.22** (a third-party WDM-KS ASIO driver) on the AI-04, with a cable from its output to its input | **HARDWARE VERIFIED** 2026-09-30: round trip measured at 64–512 frames, repeatable to the frame; the output heard through the cable at the level native WASAPI gives — see below |
| ASIO4ALL in the driver VM, on ToneSphere's virtual cables | **HARDWARE VERIFIED in the VM** 2026-10-01 — see below; a hang it exposed in ToneSphere's WASAPI stop is fixed |
| A driver that hangs in `stop()` | abandoned after 5 s, reported, no further ASIO load in that process: the engine side **IMPLEMENTED** (`tests/native/test_external_backend.py`); the ASIO host's timeout **UNVERIFIED** — no driver that hangs in `stop()` was available to prove it on |
| Against an audio interface manufacturer's own ASIO driver | **NOT AVAILABLE** — the AI-04 has none: Audio Array sells it as driver-free, a USB Audio Class device on Windows' in-box driver. Until an interface with its own ASIO driver is tested, ToneSphere's ASIO support is proven through a software ASIO driver only |

A driver name in the registry is not ASIO support. Only a driver initialised by this host,
with audio moving through its buffer switch, is — and that has now happened with FlexASIO.

## Verified with FlexASIO

FlexASIO 1.10b (MIT-licensed; installer from its GitHub release, SHA-256
`FE496BCC08D6C421C6244C8A60AC7B538560BDA138000FD1A54AB8EBCE031209`, not Authenticode-signed)
was installed on the development machine for this. It is a real ASIO driver that renders
through Windows audio APIs from inside the host process. `tests/hardware/test_asio.py`:

- **Query:** loads, initialises and unloads it: 2 inputs and 2 outputs, all float32;
  buffers 441–44100 frames (preferred 882, granularity 1); every probed rate accepted;
  reported latency 882 / 3528 frames; `outputReady` supported.
- **Buffer switch:** at 48 kHz with its preferred 882-frame buffer, 80 buffer switches in
  1.5 s (1.47 s of audio — none missed), engine callback mean 35.2 µs, max 67.5 µs, 0.4 %
  of the 18.4 ms period, 0 xruns, 0 audio-thread allocations. Both inputs delivered frames
  (the microphone's; their content was not asserted).
- **Output content:** a 1 kHz tone played by the ASIO host was captured back through this
  process's loopback at exactly 1000 Hz. Its level came back **+9.2 dB** high — the same
  driver enhancement effect `docs/WINDOWS_AUDIO.md` measures for any stream that does not
  ask for raw mode: FlexASIO opens its own stream without it. It is not the ASIO host's.

What this does not establish: behaviour with a hardware interface's driver (different
threading, sample types, buffer behaviour, reset requests), and anything acoustic.

## Verified through FlexASIO on the AI-04

The Audio Array AI-04 (2 in / 2 out, USB, C-Media VID 0D8C PID 0269, Windows' class driver;
a guitar on input 1, earphones on the output) with FlexASIO configured for WASAPI exclusive
on `Line (AI-04)` and `Speakers (AI-04)` and 144-frame buffers (`%USERPROFILE%\FlexASIO.toml`,
removed afterwards). `tests/hardware/test_asio.py`, 2026-09-30:

- **Query:** 2 in / 2 out, all Int24 LSB (in shared mode FlexASIO reports float32; in
  exclusive mode the device format passes through); rates 44.1–192 kHz; buffer 144 fixed;
  reported latency 576 / 720 frames at query, 42.7 / 48.7 ms once running (FlexASIO's
  report, not a measurement).
- **Buffer switch:** 499 buffer switches of 144 frames in 1.5 s — 1.497 s of audio, none
  missed — callback mean 7.9 µs, max 18.2 µs, 0.6 % of the 3 ms period, 0 xruns, 0
  audio-thread allocations.
- **Input content:** input 1 carried the guitar's pickup hum at 50.10 Hz, −19.5 dBFS rms
  (Kolkata mains is 50 Hz); input 2, with nothing plugged in, its noise floor at −63.1 dBFS.
  A real signal from a physical source crossed the interface's ADC, the class driver,
  FlexASIO and ToneSphere's buffer switch. Recorded, not asserted.
- **Output content:** with a cable from the AI-04's output to its input (later the same
  day), a 1 kHz tone played through FlexASIO's outputs came back on its input 1 at 999.67 Hz,
  **−0.03 dB** from the level the same tone gives through native WASAPI exclusive over the
  same cable (`test_the_output_reaches_an_interface_cable_at_the_level_native_wasapi_gives`).
- **Duplex in exclusive mode:** FlexASIO opening input and output exclusively in one
  stream damages the audio — 2106 sample discontinuities in 2.7 s of a sweep, round-trip
  confidence 1.7–5. Each direction alone is clean (0 discontinuities: FlexASIO exclusive
  input under a native render, and FlexASIO exclusive output under a native capture), and
  native WASAPI duplex exclusive on the same device is clean, so this is FlexASIO's duplex
  path through PortAudio, not the ASIO host. Use ASIO4ALL, or FlexASIO in shared mode, for
  duplex on this interface.

With FlexASIO left at its defaults (shared mode, 882-frame buffers) on the AI-04, the output
test does run: the 1 kHz tone came back through process loopback at exactly the level sent
(rms 0.03536 against 0.03536), and the inputs again carried the guitar's hum. The +9.2 dB
measured with the Realtek above was that driver's enhancement; the AI-04 applies none.

This is the ASIO host driving real interface hardware, through a software ASIO driver. It
is not a test of any manufacturer's ASIO driver.

## Verified with ASIO4ALL on the AI-04

ASIO4ALL 2.22 (freeware by Michael Tippach; `ASIO4ALL_2_22.exe` from asio4all.org, SHA-256
`0d4f0c63bf5df077e4c74f18372f72b8e875a9abea499fea78aaa9f56022ac7e`, Authenticode-signed by
Michael Tippach) installed silently (`/S`) on the development machine, and switched in its
own panel from the Realtek to the AI-04. It talks to the device through kernel streaming
(WDM-KS), a path entirely separate from WASAPI and from FlexASIO. A 6.35 mm cable joined
the AI-04's headphone output to its input. 2026-09-30:

- **Query:** 2 in / 2 out ("AI-04 1/2"), Int32 LSB; buffers 64–2048 (granularity 8); rates
  22.05–192 kHz; reported latency 747 / 814 frames at query, 6.23 / 7.63 ms once running at
  64 frames.
- **Buffer switch at 64 frames (1.33 ms):** 3659 buffer switches in 5 s, 0 xruns, callback
  mean 5.0 µs, max 95.4 µs, 0 audio-thread allocations.
- **Output content:** a 1 kHz tone through ASIO4ALL's outputs came back on its input 1 at
  1000.35 Hz, **+0.00 dB** from native WASAPI exclusive over the same cable.
- **Measured round trip** (`tonesphere.native.roundtrip.measure_asio`: the sweep played and
  captured from the one buffer switch, so no drift cushion is in the path), three runs each,
  identical to the frame:

  | Buffer | Measured round trip | Driver-reported (in + out) |
  |---|---|---|
  | 64 frames | **15.58 ms** (748 frames) | 13.85 ms |
  | 128 frames | 18.27 ms (877 frames) | — |
  | 256 frames | 23.58 ms (1132 frames) | — |
  | 512 frames | 34.27 ms (1645 frames) | — |

  The captured sweep peaked at −8.0 dBFS for −18 dBFS sent: the headphone output into the
  line input gains about 10 dB, well clear of clipping.

ASIO4ALL is a genuine ASIO driver from a third party and its path to the hardware is
kernel streaming, but it is still not an interface manufacturer's driver. It was uninstalled
from the development machine afterwards (its own uninstaller, `/S`); `HKLM\SOFTWARE\ASIO`
lists FlexASIO alone again.

## ASIO4ALL in the driver VM, on ToneSphere's virtual cables

`scripts/vm/run_driver_tests.ps1` installs ASIO4ALL 2.22 silently in the Hyper-V test VM,
where the only audio devices are ToneSphere's own cables, and runs `test_asio.py` and
`test_roundtrip.py` against "ToneSphere Cable 1" as the interface (2026-10-01):

- **Query:** 2 in / 2 out, Int32 LSB, buffers 64–2048, rates 44.1–192 kHz.
- **Buffer switch at 512 frames:** the first switch arrives about 0.4 s after `start()`
  returns; from then on the driver runs at 47,776–48,113 frames/s, 0 xruns, callback mean
  14–26 µs. (The test used to count the start-up gap against the rate, and failed on it.)
- **Output content through a cable** (the cable's render endpoint loops to its capture
  endpoint inside the driver): a 1 kHz tone from the ASIO host came back at 1000.41 Hz, rms
  0.03536, **+0.00 dB** from native WASAPI exclusive through the same cable.
- **Round trip from the buffer switch:** `--`. `measure_asio` found no path above its
  confidence threshold (2.1 against 4.0) through ASIO4ALL on the cable, where native WASAPI
  exclusive on the same cable measured 12.02 ms, identically twice, at confidence 79.
  Recorded, not explained.
- ASIO4ALL asks for a reset (`kAsioResetRequest`) during its first stream there; the host
  flags it in the stream status, as it should, for the control plane to act on.

**A hang, found here and fixed.** After the cable tests, with ASIO4ALL playing into a cable
through kernel streaming while a second engine held a WASAPI shared stream on the same cable,
stopping that WASAPI engine never returned and two cores spun. The cause was ToneSphere's:
each WASAPI stream thread waited on `{device event, stop event}`, and `WaitForMultipleObjects`
reports the lowest-index handle signalled — so a device event that never stops firing (as it
does for a shared stream fighting kernel streaming for the same pins) hid the stop request for
ever, and `stop()` waited to join a thread that never saw it. The stop event now comes first.
With that, the test finishes in the VM (it skips, correctly: ASIO4ALL does not render through
WASAPI). Two more defects were fixed on the way:

- the WASAPI capture threads drained packets in an unbounded loop that never looked at the
  stop event; the drain is now bounded per wake-up (`kMaxPacketsPerWake`);
- the ASIO host waited for ever on a driver's own `stop()`. It now gives the driver 5 seconds,
  then abandons it — the driver, its thread and the stream are leaked rather than freed under
  it — reports "the driver did not return from stop() within 5 s and was abandoned" (the
  engine's stop returns that error), and loads no ASIO driver again in that process, as a
  precaution: ASIO drivers are in-process and often single-instance.
  `tests/native/test_external_backend.py` proves the engine side without a driver; the
  host's timeout itself has not met a driver that hangs.

An ASIO driver also stays loaded in the process after release (an in-process COM object is
not unloaded), and ASIO4ALL's left-over state disturbed WASAPI round trips measured on the
same cable afterwards in the same process. The VM runner therefore runs each test file in a
process of its own.

## Licence

The Steinberg ASIO SDK is dual-licensed: a proprietary licence that needs an agreement
signed by Steinberg before publishing, or **GPL version 3** (its `LICENSE.txt` says
"Version 3", without "or later"). ToneSphere uses the GPLv3 option. So:

- `native/asio/` is GPLv3 (`native/asio/LICENSE`) and builds to its own DLL. The rest of
  ToneSphere's source stays MIT, and `tonesphere_native.dll` never includes an ASIO header.
- The ASIO DLL reaches the engine only through `tonesphere_native.h`'s C ABI (the
  external-backend entry points).
- Any binary distribution that includes `tonesphere_asio.dll` is distributed under GPLv3
  as a whole. The Windows executable does: `tonesphere.spec` bundles the GPLv3 text and
  both SDKs' licence files into it, and the release job publishes its Corresponding
  Source as `ToneSphere-<version>-windows-source.zip` in the rolling `gpl-source` release,
  linked from that release's notes (`scripts/package_source.py`: the
  tree at that commit, the ASIO SDK exactly as fetched, and the parts of the VST3 SDK the
  build compiles), because the tag's own source archive has neither SDK. Extracted on its
  own, the archive rebuilds both DLLs with nothing downloaded, and `tests/native` passes
  against them (checked 2026-09-29). For the Store
  package, see `docs/MICROSOFT_STORE.md` section 7.
- The SDK is fetched by `scripts/fetch_sdks.py` from Steinberg's official URL, pinned by
  SHA-256, and never committed (`sdks/README.md`).
- "ASIO" is a Steinberg trademark; it is not part of ToneSphere's name, and the
  ASIO-compatible logo is not used.

## How it works

**Discovery.** `asio.drivers()` reads `HKLM\SOFTWARE\ASIO` — each subkey's `CLSID` and
`Description` — and checks that the CLSID's `InprocServer32` DLL actually exists, so a
half-uninstalled driver shows as broken rather than failing obscurely on load.

**One thread per driver.** ASIO drivers are in-process COM objects, and many assume every
control call arrives on the thread that created them, and that it pumps messages. Each
loaded driver gets a dedicated STA thread with a hidden top-level window (some drivers
parent their control panel to it) and a message loop; every control call is marshalled
onto it.

**Starting.** `asio.start(engine, name, input_node=, inputs=, output_node=, outputs=,
buffer_frames=)`:

1. Loads the driver (`CoCreateInstance` with the driver's CLSID as both class and
   interface ID — ASIO's convention) and calls `init` with the hidden window.
2. Checks every requested channel exists.
3. Requires the engine's sample rate: `canSampleRate`, then `setSampleRate` if needed. A
   driver that cannot run at the engine rate is refused, never resampled behind the user's
   back.
4. Validates the buffer size against the driver's min/max/granularity (including
   power-of-two granularity), defaulting to the driver's preferred size.
5. Refuses any channel whose sample type it cannot convert, naming the type. Supported:
   Int16/24/32 LSB, Float32/64 LSB, and Int32 LSB with 16/18/20/24-bit alignment. MSB
   (big-endian) types and DSD are refused.
6. `createBuffers`, zeroes both halves of every output buffer, reads the driver's reported
   latencies, checks for `outputReady` support, and `start`s.

**The buffer switch** (`bufferSwitch` / `bufferSwitchTimeInfo`, on the driver's thread)
converts each input channel to float, runs the engine through `ts_engine_run_block` in
blocks of at most the engine's block size, converts outputs back, and calls
`outputReady()` where supported. It joins MMCSS on first use. It never locks, allocates,
logs or calls Python.

**Driver messages.** `kAsioResetRequest` and `kAsioBufferSizeChange` are *flagged*, never
acted on inside the callback; the stream status then says "the driver requested a reset:
restart the stream", and the control plane restarts it. `kAsioResyncRequest` and
`kAsioOverload` are counted (overloads also count as engine xruns). `kAsioLatenciesChanged`
is flagged. `kAsioSupportsTimeInfo` is answered yes.

**One driver per process.** ASIO's callbacks carry no context pointer, so a host can run
one driver at a time. A second `start` while one runs is refused.

## Latency

The stream status's `reported_latency_ms` is what the driver's `getLatencies` reports for
the negotiated buffer — the driver's claim, not a measurement. A measured round trip needs
a loopback path: `tonesphere/native/roundtrip.py` measures WASAPI paths; with the AI-04 and
no cable from its output to its input it reports `--` (confidence 1.0 against a threshold of
4).

## Verifying on a machine with a driver

```
uv run python scripts/fetch_sdks.py asio && uv run python scripts/build_native.py
uv run pytest tests/hardware/test_asio.py -m hardware -s
```

With FlexASIO (an open-source ASIO driver that renders through WASAPI inside the host
process), the output test also captures what the ASIO host played through process
loopback and checks it. With a hardware interface's driver, output content can only be
checked with a loopback cable.
