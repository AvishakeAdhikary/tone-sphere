---
title: Architecture
layout: default
permalink: /architecture/
description: How ToneSphere is put together — the Python control plane, the native real-time engine behind a C ABI, the device backends, the plugin host and the virtual audio driver — and where each licence boundary falls.
---

# Architecture

ToneSphere is two programs sharing one process. A **control plane** in Python owns
everything a person or another program can ask for: devices, routes, buses, presets,
plugins, the UI, the REST API and the CLI. A **real-time engine** in C++ owns one thing:
turning the current routing into audio, block by block, on a thread that never waits for
the control plane. The two meet at a flat C ABI, and nothing crosses it in the direction
of the audio thread except immutable plans and atomic values.

This is the Windows design. On Linux and macOS the control plane is the same, but the
audio runs on the older PortAudio host (Python and NumPy inside PortAudio's callback),
which is labelled "not real-time safe" wherever it reports itself.

## The layers

```
UI (PySide6)   REST API + WebSocket (FastAPI)   CLI   main.py test
        \               |                        /
         `---- AudioEngine (tonesphere/core/engine.py) -------------------------.
               routing matrix, buses, channel controls, presets, network,       |
               plugins, round-trip measurement                                  |
                         |                                                      |
               NativeHost (tonesphere/engine/native_host.py)       AudioHost (engine/host.py)
               PlanCompiler: RoutingGraph -> flat plan             PortAudio, Linux/macOS
                         |  ctypes, control thread only
   ======================|====================  C ABI: native/include/tonesphere_native.h
                         |
   tonesphere_native.dll (MIT, C++20, static CRT)
     engine/     plan exchange, mixer, strips, DSP inserts, meters, stats, rings, resampler
     windows_audio/  WASAPI: MMDevice enumeration, event-driven shared/exclusive, raw mode,
                     loopback, process loopback, MMCSS; a clock backend for device-less plans
     vst3/       VST3 host (Steinberg VST3 SDK 3.8.1, MIT); plugin thread, SEH isolation
                         |  external-backend ABI (ts_engine_attach_backend / run_block)
   tonesphere_asio.dll (GPLv3, separate)  ASIO host on Steinberg's ASIO SDK
```

Outside the process:

- `driver/windows_virtual_audio/` — a kernel-mode loopback-cable driver (MS-PL, derived from
  Microsoft's SimpleAudioSample). ToneSphere reaches it through its ordinary WASAPI
  backend; there is no private kernel/user channel. Development and test only; see
  [VIRTUAL_AUDIO_DRIVER.md](VIRTUAL_AUDIO_DRIVER.md).
- `native/coreaudio-plugin/` — the macOS HAL plug-in, proven in CI only.
- The plugin scanner — `main.py scan-plugin <path>`, one subprocess per VST3 module, so a
  module whose load code crashes takes down the scanner and not the application.

## The control plane

`AudioEngine` is the only object the front ends talk to (`UnifiedAudioEngine` is a
compatibility wrapper that forwards to it). It keeps the user's intent — routes with gain,
mute, pan and polarity; buses; per-channel strip settings; plugin chains; network sends and
receives — as plain Python state, and after every change publishes an immutable
`RoutingGraph` to the host.

`NativeHost` implements the same `AudioHost` interface the PortAudio host does, so the UI,
API and CLI did not change when the engine underneath them did. It:

- enumerates devices through the native MMDevice (WASAPI) or registry (ASIO) enumeration,
  and keys each by its endpoint ID, which survives renames, reboots and replugging;
- compiles the graph with `PlanCompiler` (`tonesphere/native/plan.py`) into nodes (a
  source and a sink per device side, one node per bus, network and capture ports), routes
  and inserts, and hands the plan to the engine;
- restarts device streams only when the *set of devices* changes; a route, gain or plugin
  change is a plan swap and never interrupts audio;
- carries channel-strip state, plugin instances and insert parameters across restarts and
  across sample-rate or block-size changes (which rebuild the engine).

## The real-time engine

One `Engine` per open configuration. Its audio thread belongs to whichever backend is
running — a WASAPI render or capture thread, the ASIO driver's `bufferSwitch`, or a
high-resolution timer when the plan has no device at all — and that thread calls
`run_block` once per device period.

**Plans.** A plan is built entirely on the control thread: validated, topologically sorted
(Kahn's algorithm; a cycle is refused with `TS_ERR_CYCLE`), every buffer allocated. It is
published by one atomic exchange. The audio thread announces the plan it is using through a
single hazard pointer; the control thread frees a retired plan only when the hazard no
longer names it. Neither side ever waits for the other.

**A block.** For every node in topological order: a source copies its device buffer or ring
into per-channel buffers (refusing non-finite samples at the door); any other node sums its
incoming routes, each with a per-sample gain ramp and its pan or balance; then the node's
strip runs — polarity and trim, channel swap, inserts in slot order (built-in EQ,
compressor and delay, or a VST3 plugin), fader and mute. A sink then applies the master
gain, its safety limiter and a NaN guard, and writes to its device or ring. Every node is
metered last, so an output's meter reads exactly what the device was given.

**State that outlives plans.** Channel strips, insert processors (filter memory, delay
lines, plugin instances) and rings are keyed by node id or (node, slot), not by plan, so a
plan swap that keeps a node keeps its sound. New routes fade in from silence over one block.

**Clocks.** One device is the master: its callback drives the engine. Every other device
runs its own thread and crosses into the master's clock through a wait-free SPSC ring and a
drift resampler that consumes exactly the frames each block needs and trims its priming
excess, so the added latency is a fixed cushion rather than whatever accumulated at start.

**Talking back.** Meters, callback timing (min, mean, a log-scale histogram for p99, max),
per-block load against its own period, xruns, ring over/underruns and audio-thread heap
allocations are written by the audio thread into atomics and read by the control plane.
Anything the audio thread needs to report — a ring that starved, a non-finite sample, a
plugin that faulted — goes into a preallocated event ring, drained and logged by the
control thread. The audio thread never logs.

## Plugins

The VST3 host (`native/vst3/`) loads a module, creates the component and controller, and
wires them the way the SDK's own hosting code does. Every call into plugin code runs under
structured exception handling. A plugin that faults while loading is reported and never
inserted; one that faults on the audio thread is bypassed from that block on, its module is
deliberately leaked rather than unloaded (unloading faulted code is how a second crash
happens), and the fault is reported. Controller and editor calls run on one plugin thread
with its own message loop. Parameter changes cross to the audio thread through a
fixed-capacity queue into the block's `IParameterChanges`. Processing is in-process for
now; the reasoning, and what would change it, is in [VST3.md](VST3.md).

## Licence boundaries

| Part | Licence | Why it is separate |
|---|---|---|
| Everything under `tonesphere/`, `native/engine`, `native/windows_audio`, `native/vst3`, `native/test_plugin`, `scripts/`, `tests/` | MIT | the project's own code |
| `native/asio/` → `tonesphere_asio.dll` | GPLv3 | built from Steinberg's ASIO SDK, which is offered under GPLv3. Kept in its own directory and binary, loaded dynamically, so the core builds and runs without it. A build that bundles it is distributed under GPLv3 as a whole, with its Corresponding Source (`scripts/package_source.py`) |
| `driver/windows_virtual_audio/` | MS-PL | Microsoft's sample, modified; never linked with anything else |
| Steinberg VST3 SDK (fetched, not committed) | MIT | compiled into `tonesphere_native.dll`; its notice ships with the binary |
| Steinberg ASIO SDK (fetched, not committed) | GPLv3 | see above |

Neither SDK is ever committed; `scripts/fetch_sdks.py` fetches pinned versions and checks
the ASIO SDK's checksum and licence text. See [DEPENDENCIES.md](DEPENDENCIES.md) for every
third-party component.

## Where to look next

- [REALTIME.md](REALTIME.md) — the rules the audio thread follows, and how each is checked.
- [WINDOWS_AUDIO.md](WINDOWS_AUDIO.md), [ASIO.md](ASIO.md), [VST3.md](VST3.md) — each backend in detail.
- [TESTING.md](TESTING.md) — how "it works" is established.
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) — what is proven, at which level.
