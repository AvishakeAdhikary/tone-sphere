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
on each device event its thread runs exactly as many frames as the device wants, cut into
**equal** blocks of at most the engine's block size — a 144-frame exclusive period with a
128-frame engine becomes 2 × 72, never 128 + 16, because load is judged per block against
that block's own period and a 16-frame remainder's 0.33 ms budget made fixed per-block
costs read as a 152 % overload that never happened. Every other stream is a **satellite**
on its own clock: it meets the master through a wait-free ring, read through the native
drift resampler on the consuming side.

**The drift resampler** primes the ring to two device periods, then spends 64 blocks at
ratio 1 learning the fill this pair of streams actually settles at (sampled before each
read, a packet-fed ring averages below its primed level; steering at the primed level
corrects a drift that is not there, and on the AI-04 that warped the first seconds of
every stream). After that a proportional-integral loop steers the fill back to that
setpoint within ±0.5 % of real time, with a ±5 % dead band so two streams on one clock
run at a ratio of exactly 1 (`native/engine/resampler.h`).

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

## On a USB interface: Audio Array AI-04

2026-09-30, the same machine with an Audio Array AI-04 attached (USB, C-Media VID 0D8C
PID 0269, Windows' class driver — the maker publishes no driver) and made the default
device: a guitar on input 1, earphones on the output. `tests/hardware/test_interface.py`
(set `TONESPHERE_TEST_INTERFACE` for another interface), 20 s per run:

| Mode | Device period | Callback mean / p99 / max | Worst load | Dropouts |
|---|---|---|---|---|
| Exclusive 48 kHz, 24-bit | 144 frames (3.0 ms reported) | 5.8 / 16.0 / 209.4 µs | 14.0 % of a 72-frame (1.5 ms) block | 0 glitches, 0 xruns, 0 underruns |
| Exclusive 44.1 kHz, 24-bit | 132 frames (3.0 ms reported) | 4.7 / 16.0 / 152.7 µs | 10.2 % of a 66-frame block | 0 xruns; **input underruns 0–65 frames per run, at start** (below) |
| Shared 48 kHz, float32 | 480 frames (10 ms) | 25.1 / 45.3 / 107.5 µs | 1.1 % | 0 xruns; 1 capture discontinuity flag at stream start (WASAPI's own) |

In every mode, 0 audio-thread allocations. What the input delivered is a real signal from
a physical source: input 1, the guitar, carried its pickup's mains hum at 50.0 Hz
(Kolkata mains is 50 Hz), −19.9 dBFS rms; input 2, empty, its noise floor at −63.5 dBFS.
**Monitoring** input → gain bus → output at −40 dB gave the output exactly the input × 0.01,
sample for sample (`test_the_monitoring_path_carries_the_input_at_the_gain_set`). The
**render → loopback** digital path measures 33.35 or 43.35 ms (1601 or 2081 frames — one
480-frame period apart, the phase of each start), identical within a run, confidence 34.3.
The AI-04's endpoint applies no processing of its own: loopback came back at +0.00 dB in
every band, raw or not (the Realtek applies a high shelf).

**Its 44.1 kHz input clock.** At 44.1 kHz the AI-04's input delivers 0.2–0.3 % fewer frames
than its output consumes (792 frames short in 308,748 at the first measurement), and the
shortfall wanders by ±0.15 % over seconds; at 48 kHz the two sides agree exactly. With the
first drift resampler (a ±0.1 % limit) that was 355 frames lost every 20 s. Now the
resampler follows it: no ongoing loss, but a run can lose a few frames (0–65 measured)
while it converges, and the ratio follows the input's wander — a pitch change of at most
±2.6 cents, over seconds. That case is recorded as a known failure in the test, not hidden.
Use 48 kHz on this interface.

**Measured round trip through the interface.** Later on 2026-09-30 a 6.35 mm cable joined
the AI-04's headphone output to its input (`test_roundtrip.py`), a −18 dBFS sweep each time,
captured back at a −8.0 dBFS peak (the path gains about 10 dB; nothing clips):

| Path | Measured round trip | Runs | Nominal / reported |
|---|---|---|---|
| Native WASAPI exclusive, 144-frame period | **17.92 ms** (860 frames); 15.94 ms (765) in 2 of 10 starts | 10, confidence 16–24 | 3.0 ms / 6.0 ms |
| Native WASAPI exclusive, engine block 96 / 128 / 256 | 17.92 / 15.94 / 25.15 ms | 1 each | |
| Native WASAPI shared, 480-frame block | **76.9 ms** (3688–3692 frames) | 7, confidence 13–19 | 10 ms / -- |
| ASIO4ALL 2.22 at 64 frames | **15.58 ms** (748 frames) | 3, identical | [ASIO.md](ASIO.md) |
| FlexASIO defaults (shared, 882 frames) | 119.75 ms | 1 | |

Within one start the exclusive figure is fixed to the frame; between starts it lands on one
of two values 96 frames (two USB packet groups of 1 ms) apart, set by the phase the capture
stream starts in. The same tone through the cable came back at the same level through
native WASAPI, ASIO4ALL (+0.00 dB) and FlexASIO (−0.03 dB), which also establishes that
sound leaves the output jack.

## Whole-system loopback, as a source

Every WASAPI output is offered in the routing as "<output> (loopback)": everything every
application plays on it, after the Windows mixer. It is shared mode only (an exclusive
stream bypasses the mixer, and the loopback with it), and it has no clock: Windows sends a
loopback nothing while the output is silent. So when a loopback is the only device in a
plan, the host also opens a shared-mode render stream on the same output, playing nothing,
as the plan's clock; that stream also keeps the loopback delivering packets through
silence. Routing a loopback back into its own output, directly or through buses, is refused
as feedback. On the AI-04, two other processes playing 1 kHz at 0.1 and 440 Hz at 0.05 came
back through the loopback, a bus and a network send's ring at 0.1000 and 0.0500
(`tests/hardware/test_loopback_source.py`).

## Devices that come and go

`engine/device_monitor.py` drains Windows' endpoint notifications on a control thread and,
300 ms after the last of a burst, has the engine re-enumerate. If a device the routing uses
left, arrived, or has a failed stream, the engine restarts with what is present: other
devices carry on after a gap of a few blocks (the backend opens and closes its streams
together), and a route to the missing device is kept, so when it returns it is reopened
under the same id with no user action. What is proven without hardware is the decision
(`tests/test_device_monitor.py`); a real device disappearing and returning is exercised on
the ToneSphere virtual cable in the driver VM.

## Latency figures and what they mean

Every stream reports `reported_latency_ms` (`IAudioClient::GetStreamLatency`) — what
Windows says, not a measurement. The render-to-loopback delay above *is* a measurement,
but of a digital path that stops before the DAC. The acoustic round trip (output → air or
cable → input) needs a loopback cable or a microphone near a speaker, and is measured by
the round-trip tool of migration step M6; until then it is `--`.
