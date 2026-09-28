---
title: Windows Audio
layout: default
permalink: /windows-audio/
description: How ToneSphere's native WASAPI backend works, what it has been proven to do on real hardware, and what it has not.
---

# Windows audio (WASAPI)

`native/windows_audio/wasapi.cpp`, bound from Python by `tonesphere/native/wasapi.py`.
The Python PortAudio host (`tonesphere/engine/host.py`) remains for Linux and macOS; on
Windows the native backend replaces it for the audio path (the switch-over in
`AudioEngine` is migration step M8 — until then the native backend is driven directly).

## Endpoints

`wasapi.endpoints()` enumerates active render and capture endpoints through
`IMMDeviceEnumerator`. Each is identified by its **MMDevice endpoint ID**
(`{0.0.0.00000000}.{guid}`) — the string Windows itself uses, stable across reboots,
re-enumeration and unplug/replug — rather than a PortAudio index, which shifts whenever a
device comes or goes. Reported per endpoint: friendly name, flow, which default roles it
holds (console, multimedia, communications), the shared-mode mix format, the default and
minimum device periods, the IAudioClient3 low-latency shared-mode period range, and
whether it supports **raw** processing.

`wasapi.watch()` registers an `IMMNotificationClient`; `wasapi.poll_events()` returns
arrival, removal, state and default-device changes. Notifications arrive on Windows'
threads and are queued there, never on the audio thread.

## Streams and threads

`NativeEngine.start_wasapi([StreamSpec, ...], master=i)` opens every stream on its own
thread (MTA, MMCSS "Pro Audio"). The **master** stream's device clock drives the engine:
on each device event its thread runs exactly as many frames as the device wants, in
blocks of at most the engine's block size. Every other stream is a **satellite** on its
own clock: it meets the master through a wait-free ring, read through the native drift
resampler on the consuming side, which holds the ring's fill level at two device periods
by consuming within ±0.1 % of real time.

Stream kinds: `render`, `capture`, `loopback` (everything a render endpoint plays) and
`process_loopback` (one process's audio, Windows 10 build 20348+).

**Shared mode.** When the device's mix format is float32 at the engine rate and width,
the stream uses `IAudioClient3::InitializeSharedAudioStream` at the smallest period that
covers one engine block (on the development machine, 480 frames = 10 ms). Otherwise it
asks Windows to convert (`AUTOCONVERTPCM`).

**Exclusive mode.** Formats are tried in order — float32, int32 with 24 valid bits, int32,
packed int24, int16 — at the engine rate, and the stream reports which it got. A refusal
falls back to shared mode only if the spec allows it, and the status then says
`exclusive: False` with the refusal's reason; it is never silent.

**Raw mode, on by default.** A shared stream asks for raw processing
(`AUDCLNT_STREAMOPTIONS_RAW`), which bypasses the driver's enhancement effects on that
stream. This matters: on the development machine, a stream *without* raw mode came back
**+9.7 dB louder, rising over the first second** — the Realtek driver's loudness
processing — measured by playing a tone through PortAudio (not ToneSphere) and capturing
it, so it is the driver, not this code. Whether the endpoint honoured raw mode is in the
stream status.

## What is proven, on what

Development machine, 2026-09-29: Windows 11 Pro 25H2 (build 26200), Realtek ALC257
(`Realtek HD Audio 2nd output`, the default render endpoint; `Microphone`), Intel Smart
Sound Technology microphone array. All figures are from `tests/hardware/test_wasapi.py`.

| Claim | Level | Evidence |
|---|---|---|
| Enumeration: stable endpoint IDs, names, formats, periods, default roles, raw support | HARDWARE VERIFIED | `test_endpoints_have_stable_ids_names_and_formats` — 4 endpoints, all raw-capable |
| Shared render → process loopback, **bit-exact** | HARDWARE VERIFIED | `test_raw_render_is_bit_exact_through_process_loopback`: 1 s of white noise, max sample error 0.00e+00. Skips itself on a run where the drift resampler engaged (interpolated samples are correct audio, not identical samples) |
| Shared render → whole-endpoint loopback, level exact at 1 kHz, no gaps | HARDWARE VERIFIED | `test_raw_render_comes_back_whole_at_the_level_sent`, both loopback kinds |
| Render-to-capture delay | **measured** | 3521 frames = **73.4 ms** (process loopback), 2993 frames = 62.4 ms (endpoint loopback), by cross-correlation. This is the digital path through the Windows audio engine, not an acoustic round trip |
| Endpoint effects after the mix | recorded | whole-endpoint loopback shows a high shelf on this machine: 0.00 dB to 1 kHz, +0.08 (1–4 kHz), +0.61 (4–12 kHz), +0.92 dB (12–20 kHz) — the endpoint's own processing, which raw mode on one stream cannot bypass |
| Exclusive mode | HARDWARE VERIFIED (runs; output not captured) | Realtek granted exclusive: int32 container / 24 valid bits, 480-frame buffer, reported latency 10.0 ms, 0 glitches, 0 xruns. Loopback cannot capture an exclusive stream, so the audio itself was not verified this way |
| Three clocks: microphone master + render and loopback satellites | HARDWARE VERIFIED | `test_a_capture_master_drives_render_and_loopback_satellites`: tone arrives at the sent level (±1 %), no gaps, no satellite underrun |
| Audio thread on the device: callback time, allocations | measured | render master at 480 frames: mean 15–19 µs, max 26–34 µs per block of 10 ms; 0 xruns; 0 audio-thread allocations |
| Start/stop cycles | HARDWARE VERIFIED | 5 cycles, no threads left behind |
| A missing endpoint fails with a reason | HARDWARE VERIFIED | `test_a_bad_endpoint_fails_with_a_reason_not_silence` |
| Device notifications registered and polled | IMPLEMENTED | `test_device_notifications_can_be_watched` |
| Device removal/arrival while streaming | **NOT VERIFIED** | no device was unplugged during a test; a stream whose device disappears is designed to fail with `AUDCLNT_E_DEVICE_INVALIDATED` in its status, but that has not been exercised |
| Capture of a real microphone signal's content | NOT VERIFIED | the capture master test proves the microphone *clock* drives the engine; nothing asserted what the microphone heard |

## Latency figures and what they mean

Every stream reports `reported_latency_ms` (`IAudioClient::GetStreamLatency`) — what
Windows says, not a measurement. The render-to-loopback delay above *is* a measurement,
but of a digital path that stops before the DAC. The acoustic round trip (output → air or
cable → input) needs a loopback cable or a microphone near a speaker, and is measured by
the round-trip tool of migration step M6; until then it is `--`.
