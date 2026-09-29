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
- **Output content:** not checked. FlexASIO holds the device exclusively, so its output
  bypasses the Windows mixer and no loopback can hear it; the output test skips with that
  reason. Checking it needs a cable from the output to the input (`test_roundtrip.py`'s
  cable test).

With FlexASIO left at its defaults (shared mode, 882-frame buffers) on the AI-04, the output
test does run: the 1 kHz tone came back through process loopback at exactly the level sent
(rms 0.03536 against 0.03536), and the inputs again carried the guitar's hum. The +9.2 dB
measured with the Realtek above was that driver's enhancement; the AI-04 applies none.

This is the ASIO host driving real interface hardware, through a software ASIO driver. It
is not a test of any manufacturer's ASIO driver.

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
  Source next to it as `ToneSphere-windows-source.zip` (`scripts/package_source.py`: the
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
