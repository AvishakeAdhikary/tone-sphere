---
title: Real-time rules
layout: default
permalink: /realtime/
description: The rules ToneSphere's audio thread follows, why each exists, and the test or measurement that checks it.
---

# Real-time rules

An audio callback has a deadline: at 48 kHz and 128 frames, 2.67 ms, every time, for as long
as the stream runs. Missing it once is an audible click. Nothing that *usually* finishes
quickly is allowed on that thread, because the problem is never the usual case.

These rules apply to the native engine's audio thread (`native/engine`, the WASAPI and ASIO
backends, and the VST3 host's `process` path). The PortAudio host used on Linux and macOS
breaks several of them by construction — it runs Python in the callback — and says so
wherever it reports itself.

## The rules, and how each is enforced

| Rule | Why | How it is checked |
|---|---|---|
| **No heap allocation** | the allocator takes a lock, and can page-fault | `operator new`/`delete` are replaced in the DLL; a thread-local flag set by `AudioThreadScope` counts every allocation made on the audio thread. `test_the_audio_thread_allocates_nothing` runs 500 blocks with rings, buses, ramps and plan swaps and requires `rt_allocations == 0`; every hardware test and the soak test read the same counter from a live device thread. Plugin modules allocate through their own CRT and are not counted |
| **No locks, no waiting on the control thread** | a lock held by a preempted thread is an unbounded wait | plans are published by atomic exchange and retired through a hazard pointer; controls are atomics; rings are wait-free SPSC with 64-bit indices. `test_plans_swap_while_another_thread_processes` swaps 300 plans under a concurrently running audio thread |
| **No Python** | the GIL is held by whoever the scheduler likes | the audio thread is native; Python calls the engine only from control threads, through ctypes |
| **No I/O and no logging** | a file write or a console write can block for milliseconds | the audio thread posts fixed-size events to a preallocated ring; the control thread logs them. A ring that overflows says so once, and counts what it lost |
| **No scanning, loading or unloading plugins** | module loads run arbitrary code and take the loader lock | scanning is in a subprocess; loading and `setActive` happen on the plugin thread before an instance is inserted into a plan; a faulted module is leaked, not unloaded |
| **No unbounded work** | the budget is per block | every loop is bounded by the block size and the plan's fixed node, route and insert counts (`TS_MAX_*`) |
| **Nothing non-finite reaches a filter or a driver** | a NaN in a feedback path never leaves it; a NaN at a DAC is full-scale noise | sources refuse non-finite input and sinks silence a non-finite block, each reporting once (`test_a_nan_never_reaches_an_output`) |
| **No discontinuities from control changes** | a stepped gain is a click | gains ramp per sample over one block; new routes fade in from silence; channel swap crossfades over one block (`test_swap_exchanges_the_channels_without_a_step`) |
| **Device threads run as "Pro Audio"** | ordinary priority loses to everything | every WASAPI thread registers with MMCSS; the benchmark records whether registration succeeded |

## Measured on the development machine

Intel Core i7-1165G7, Windows 11, Realtek ALC257. The figures are the engine's own timing,
taken inside `run_block` on the audio thread.

| Measurement | Result | Source |
|---|---|---|
| Offline, 48 kHz / 256 frames, EQ + compressor + bus + delay + limiter | mean 23.9 µs, p99 32 µs, worst 39.7 µs per block (0.45 % / 0.6 % / 0.74 % of 5.33 ms) | `benchmarks/results/m3_offline_i7-1165G7.json` |
| `main.py test`, WASAPI exclusive, 128 frames | mean 0.007 ms, p99 0.016 ms, worst 0.020 ms = 0.7 % of the device period, 0 xruns | `main.py test` output, 2026-09-29 |
| 30-minute soak, WASAPI shared 48 kHz / 480 frames, Surge XT Effects + test plugin on the output | 30 min, 180,024 callbacks: 0 xruns, 0 audio-thread allocations; callback mean 65.9 µs, p99 ≤ 128 µs, worst 1.10 ms (11 % of the 10 ms period); bus ring underruns after start 0; private bytes 280.3 → 280.5 MB; 25 of 25 restarts running within 55–63 ms, handle count 352 → 352 | `benchmarks/results/soak_30min_i7-1165G7.json` |
| Audio-thread heap allocations, every run above | 0 | the counter above |

### Soak

`benchmarks/soak.py` runs the path the UI drives — `AudioEngine`, the native host, WASAPI
on the default output — for 30 minutes with a bus fed in real time by a Python thread
(pink noise) and two VST3 plugins on the output side: Surge XT Effects, then ToneSphere's
test plugin at gain 0 so the room hears nothing while the chain does real work. It samples
the engine's statistics every 10 s, then stops and restarts the engine 25 times. The
recorded result, on the Realtek headphone output: the table above, and in full in
`benchmarks/results/soak_30min_i7-1165G7.json`. The output meter read silence for all 30
minutes and the bus meter read the noise, so the chain carried signal the whole time. The
one 1.10 ms block is the worst of 180,024; it is inside the budget, and whether it was the
engine, a plugin or the scheduler is not distinguished by these statistics. The p99 is a
histogram bucket edge, so it is an upper bound at about 19 % resolution.

## What is not covered

- **Plugin code.** A VST3 plugin runs on the audio thread and can allocate, lock or block
  as its author chose. ToneSphere measures the whole block, plugin included, but cannot
  make someone else's code real-time safe.
- **The driver and the OS.** The engine's timing starts when the backend calls `run_block`;
  time spent in the driver, the Windows audio engine and scheduling jitter before the call
  is not in it. Xruns reported by the device catch the consequences.
- **The PortAudio host** (Linux, macOS) — see above.
