---
title: VST3
layout: default
permalink: /vst3/
description: ToneSphere's native VST3 host — how plugins are found, opened, processed and isolated, and exactly what has been proven with which plugins.
---

# VST3 hosting

`native/vst3/vst3_host.cpp` (in `tonesphere_native.dll`, MIT) hosts VST3 plugins on the
Steinberg VST3 SDK 3.8.1 (MIT). `tonesphere/plugins/` is the Python side: `PluginInfo`,
`PluginInstance`, `PluginParameter`, `PluginState`, and the scanner. The pedalboard-based
chain it replaces was never reachable from the UI, API or CLI and has been removed.

## Status

| Capability | Level | Evidence |
|---|---|---|
| Discovery (standard + custom folders, recursive, bundles and single-file modules) | VERIFIED | `tests/native/test_vst3.py::TestDiscovery` |
| Out-of-process scanning with timeout and cache; crash, hang, wrong architecture, non-DLL reported with the reason | VERIFIED | a module that faults in `GetPluginFactory` → `crashed`; a scanner killed outright → `crashed`; a 32-bit PE header → `wrong architecture`, without loading it |
| Load, factory and class enumeration, processor/controller creation and connection (the SDK's `PlugProvider`), bus negotiation, `setupProcessing`, activation | VERIFIED | test plugin; Surge XT |
| float32 processing on the engine's audio thread, as an engine insert | VERIFIED (test plugin), HARDWARE VERIFIED (Surge XT) | test plugin output equals input shifted by its 64 samples, bit-exact; Surge XT's default delay echoes an impulse 250.06 ms later, as its own display says |
| Plugin latency: reported vs measured | VERIFIED | the test plugin's reported 64 samples equals the delay measured by cross-correlation |
| Parameters: ID, title, short title, units, step count, default, normalised, plain, display text, flags (automatable, read-only, bypass) | VERIFIED | test plugin; Surge XT's 14 parameters with its own display strings, bypass flag detected |
| Parameter changes from Python and from the plugin's editor reach the audio thread | VERIFIED (Python side) | a change takes effect on the next block; editor edits use the same queue (`performEdit`) but no test drives an editor control |
| State (component + controller) save and restore through the SDK's state APIs; foreign state rejected | VERIFIED; HARDWARE VERIFIED (Surge XT) | round-trips into a fresh instance; bytes from elsewhere are refused and the plugin keeps working |
| Editor window (IPlugView in a native window on the plugin thread) | HARDWARE VERIFIED (opens and closes) | Surge XT's editor; nothing interacts with it in a test |
| Crash isolation: fault during initialise, during process on the audio thread | VERIFIED | initialise fault reported, host continues; process fault → plugin bypassed, dry signal passes, never called again |
| Instruments played by MIDI (M18) | **HARDWARE VERIFIED** with Surge XT 1.3.4 and Dexed 1.0.1 | An instrument goes on a bus of its own (`AudioEngine.create_instrument`, Plugins → Add Instrument) and is played by note on/off through `ts_vst3_send_midi`: pushed on the plugin thread into a single-producer queue, drained on the audio thread into a preallocated event list (512 events) handed to `process()`; controllers and the pitch wheel go through the plugin's `IMidiMapping` to its parameters. `tests/hardware/test_instruments.py`: Surge XT's note 69 at **440.63 Hz** from the engine, **440.17 Hz** over REST, C4 at **261.76 Hz** from the on-screen keyboard, and **440.63 Hz out of the AI-04 and back through its cable**; Dexed's notes 60 and 67 at 131.00 and 196.57 Hz (its default voice sits an octave below the key), an exact fifth; every note silent within two seconds of its release |
| MIDI from a hardware port (winmm) | UNVERIFIED | `engine/midi_input.py` forwards a port's notes, controllers and pitch wheel to an instrument; the development machine has no MIDI input device, so only enumeration (none) and the refusals are tested |
| Plugins from the REST API and the CLI (M18) | VERIFIED | `/plugins`, `/chains/{id}` (add a VST3 or a built-in, remove, move, bypass, parameters), `/instruments`, `/midi/{id}`; CLI `plugins`, `chain`, `effect`, `param`, `instrument`, `note`. `tests/native/test_plugins_remote.py`: the test plugin put on a bus and set to ×0.5 by parameter id over REST halves a 1 kHz tone exactly; a high-pass set from the CLI takes 100 Hz down by over 20 dB |
| Commercial plugins (Guitar Rig, Neural DSP, ...) | **NOT TESTED** | none installed on the development machine; nothing is claimed about them |
| Plugin process isolation (a worker process per plugin) | NOT IMPLEMENTED | see *Isolation* below |

Surge XT 1.3.4 (GPL-3.0; official release `surge-xt-win64-1.3.4-pluginsonly.zip`, SHA-256
`564E162C560AF07AD4ED47FE1BFCD827CF97A575DE30D06C48249AAD2E7C35E6`) was placed in the
per-user VST3 folder for these tests — no installer, no administrator rights. Results are
from `tests/hardware/test_vst3_third_party.py` on the development machine, 2026-09-29.

## How it works

**Finding plugins.** `tonesphere.plugins.scan` walks `%COMMONPROGRAMFILES%\VST3`,
`%LOCALAPPDATA%\Programs\Common\VST3` (the VST3 specification's per-user folder) and any
custom folders, recursively, collecting `.vst3` bundles (without looking inside them) and
legacy single-file modules. Before anything is loaded, the module's PE header is read: a
32-bit or ARM64 build is reported as the wrong architecture, not loaded.

**Scanning out of process.** Loading a module runs the plugin's own initialisation code,
so each module is loaded by a subprocess (`python -m tonesphere.plugins.scan <path>`, or
`ToneSphere.exe scan-plugin <path>` when frozen) with a 30-second timeout. A crash, a hang
or a load failure is recorded with the reason; the main process never loads that module
through the scanner. Results are cached by path, size and modification time.

**Opening.** `PluginInstance(info, sample_rate, max_block, channels)` runs on the plugin
thread: the SDK's `Module` loader, then its `PlugProvider` (component and controller
creation, connection points, initial state sync — exactly as Steinberg's validator does
it), then bus arrangement for the requested width, `setupProcessing` (realtime, float32,
the engine's rate and block), `setActive`, `setProcessing`. A plugin that will not take the
requested width is refused, naming the width it would take.

**Processing.** Inserted with `Insert(node, slot, VST3, plugin=instance.handle)`, the plugin
runs in the node's insert chain on the engine's audio thread. Buffers, `ProcessData`, the
process context and the parameter-change structures are preallocated; parameter changes
arrive through a wait-free queue into fixed-capacity `IParameterChanges` implemented here,
so the host never allocates on the audio thread on the plugin's behalf. What a plugin does
inside its own `process()` is its own business — ToneSphere does not claim third-party
plugins are real-time safe.

**Threads.** Every non-real-time call — open, parameters, state, editor, close — runs on
one plugin thread, OLE-initialised and pumping messages, because VST3 controllers expect a
single UI thread and editors need a message loop. `performEdit` from a plugin's own editor
is forwarded to the audio thread through the same queue as Python's changes.

**State.** `instance.state()` returns the component and controller state exactly as the
plugin's `getState` wrote them — opaque bytes, never a Python object; `PluginState.to_dict`
base64-encodes them for presets. `restore()` hands them back through `setState` and
`setComponentState`; a plugin that rejects them (another plugin's state, an incompatible
version) raises `PluginError` and keeps its current state.

## Isolation, and why plugins run in-process

A plugin that crashes in-process can take its host down. Two layers limit that here:

- Every call into plugin code is wrapped in structured exception handling. An access
  violation or a C++ exception escaping the plugin marks it crashed: on the control side
  the call fails with the reason; on the audio thread the plugin is bypassed (the dry
  signal passes) and never called again, and its module is deliberately leaked rather
  than unloaded, because code that just faulted cannot be trusted to shut down.
- Scanning — where most broken plugins are first met — runs in a subprocess.

What this cannot catch: a plugin that corrupts memory (a heap or stack overwrite) before,
or instead of, faulting. Only running each plugin in its own process would contain that.
That design was weighed and not built for now: it costs an inter-process round trip per
block (two context switches and a shared-memory handoff per plugin per block, on top of a
budget of a few milliseconds at small buffers), plus a separate editor process with window
reparenting across processes. The in-process design is what low-latency hosts commonly use;
per-plugin process isolation remains an option to measure and add if crash reports warrant
it.

Dexed 1.0.1 (GPL-3.0; the official `Dexed-1.0.1-win.zip`, MD5 `f1d47cbee77a07f5bdb4848e8bcb078e`
as the project's own `artifact_md5sum.txt` gives it, SHA-256 `1f118445…05f05`) was unzipped
into a test folder (`TONESPHERE_TEST_PLUGINS`) — nothing installed. Neural Amp Modeler, the
free amp modeller, was not tested: its Windows installer is not code-signed, and it was not
installed with administrator rights on the development machine.

## Not yet

- Sidechain and multiple buses (only the main input and output bus are used).
- Sample-accurate automation (a change applies at the start of the next block); notes play
  at the start of the next block too.
- MIDI from a hardware port is implemented and untested (no device).
