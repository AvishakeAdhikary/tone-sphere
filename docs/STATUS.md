# NexTone build status

_Last updated: 2026-10-10._

## What works (verified)

| Area | Status | Evidence |
|---|---|---|
| Real-time core: lock-free rings, varispeed resampler, drift-compensated adaptive bridges | ✅ | `engine/tests/core_tests.cpp`: SPSC ring stress; 44.1→48 kHz resampling better than −60 dB; ±300 ppm clocks stay locked within ±15 ppm over 10 simulated minutes with 0 dropouts |
| Engine graph: mixes/buses, crosspoint routing, gain/mute/solo/pan/delay/polarity, meters | ✅ | `engine/tests/engine_tests.cpp` (11 cases) |
| Built-in effects: noise suppression (RNNoise), gate, EQ, compressor, limiter, auto-duck, delay | ✅ | unit tests: limiter ceiling, gate, 4:1 compression = −15 dB |
| **VST3 plugins**: out-of-process scan, inserts on any node, plugin windows, state in config and scenes, custom folders, crash containment | ✅ | `tests/integration/plugins.mjs` 7/7 with the bundled *NexTone Test Gain* plugin (see below). Real plugin: Guitar Rig 7 scans in 2 s and inserts in 0.35 s; its window opens; CPU 1.6 %, 0 xruns |
| Windows audio devices: WASAPI shared/exclusive, IAudioClient3 low latency, hot-plug | ✅ | `tests/integration/host-audio.mjs` 10/10; render starts with 3 periods queued and grows if the device runs dry (counted as xruns) |
| Per-app capture ("copy") and per-app redirection ("isolate", app microphones) | ✅ capture verified; redirection implemented | host-audio: a playing app appears automatically and its audio reaches a mix |
| **NexTone virtual audio driver** (4 outputs + 4 mics, test-signed) | ✅ in the Hyper-V VM | `tests/integration/vad-audio.mjs` 14/14 in the VM: app → Output → NexTone → Mic → app, with 0 clicks, gaps or dropouts |
| **NexTone ASIO driver** (x64 + x86, multi-client) | ✅ | `tests/integration/asio-clients.mjs` 11/11 for 64- and 32-bit hosts: two DAWs at once, exact timing, float32 |
| Engine hosting a real ASIO interface (shared between DAWs) | ✅ | `tests/integration/asio-host.mjs` 8/8 with FlexASIO looped through VB-CABLE |
| Network audio (VBAN, VoiceMeeter-compatible) | ✅ | codec unit tests + `tests/integration/vban.mjs` 4/4 |
| Recording (multitrack WAV, 32-bit float, RF64-safe) and the instant-replay buffer | ✅ | `tests/integration/recorder.mjs` 6/6 |
| Idle meters (OS peak meters for devices and apps that aren't routed) and global hotkeys | ✅ | `tests/integration/meters-hotkeys.mjs` 2/2 |
| UI (Electron + React): six bins, mixes, drag-to-route, send-to menus, matrix, inspector, effects, plugins, scenes, settings, dark mode | ✅ | `ui/tests/e2e/ui.spec.ts` 7/7 |
| **Desktop installer** (`NexTone-Setup-<ver>.exe`): app + engine + ASIO driver + virtual driver | ✅ in a pristine VM | `vm/Test-Installer.ps1`: silent install, drivers present, ASIO registered, the app starts the engine, vad-audio 14/14 against the installed build |
| **Microsoft Store package** (MSIX) | ✅ sideload-tested in a pristine VM | `vm/Test-Msix.ps1`: installs, launches, finds the engine; sign-in startup task starts the engine plus the app in the tray; `nextone-engine.exe` alias and `nextone:` link are registered |
| Windows App Certification Kit | ⚠️ 15 tests pass; no real failures found | `vm/Test-Wack.ps1` (see below) |
| CLI | ✅ | `tools/nextone-ctl.mjs` (also finds a Store-installed engine) |

## How to run

```powershell
.\scripts\build.ps1 -Test            # engine, ASIO driver, tools, test plugin + unit tests (x64)
.\scripts\build.ps1 -Arch x86        # 32-bit ASIO driver
.\drivers\win\vad\build.ps1          # kernel driver package (test-signed)
cd ui; npm run build; npx electron . # UI (starts the engine automatically when packaged)

.\tests\run-all.ps1                  # every host-safe suite (+ UI e2e); needs VB-CABLE + FlexASIO
.\vm\Test-InVM.ps1                   # virtual driver tests in the Hyper-V VM
.\vm\Test-Installer.ps1              # desktop installer in a pristine VM
.\vm\Test-Msix.ps1                   # Store package in a pristine VM
.\vm\Test-Wack.ps1                   # App Certification Kit in a pristine VM

cd ui; npm run dist:win              # desktop installer  → ui\release\
cd ui; npm run dist:store            # Store package      → ui\release-store\  (see packaging\store\README.md)
```

## Pending / known limitations

- **Virtual audio driver signing.** The driver is **test-signed**. It only loads on PCs with
  Windows test signing on; it was verified inside the Hyper-V test VM and never installed on
  the host. Public use needs Microsoft attestation signing, which requires an EV certificate.
  Without it, NexTone uses Compatibility mode (VB-CABLE etc.) for virtual devices. The ASIO
  driver is user-mode and works everywhere.
- **Store submission is yours to do.** Follow `packaging/store/README.md`: identity values
  from Partner Center, listing text, privacy policy, and a GPLv3 source link.
- **WACK.** Run in a pristine VM, 15 tests pass. The other results are not real failures:
  - **Six metadata tests** can't start in the VM ("platform metadata directory does not
    exist") because only part of the SDK is copied there. They validate WinRT `.winmd`
    components, and NexTone ships none.
  - **"Blocked executables" and "Platform appropriate files"** are optional tests. The
    blocked-executable findings are string matches such as "cmd" inside Chromium resources,
    expected for every Electron app.
  - **DPI awareness** is a warning only.

  To get a fully clean report, run WACK on a PC with the Windows SDK installed. It installs the
  package and needs the dev certificate trusted, so it was not run on this host.
- **Audio under heavy CPU load.** With the test VM saturating all cores, the VB-CABLE loopback
  suites (host-audio, asio-host) can record clicks that the engine itself doesn't see (no ring
  underrun, no device starvation). With the VM idle, every suite passes.
  - A real start-up issue was found and fixed: only 2 WASAPI periods were queued when a stream
    started, which starved VB-CABLE once in roughly a third of runs. The queue now starts at 3
    periods and adapts.
  - Device starvation is now detected and reported as xruns.
- **VST3 limitations.**
  - Plugins run inside the engine. A fault while processing switches the plugin off instead of
    crashing the engine, but a plugin can still hang the engine.
  - There are no generic sliders for plugin parameters; use the plugin window.
  - Plugin latency isn't compensated.
  - Plugin windows are DPI-unaware, so Windows scales them.
  - Plugin windows are Windows-only for now.
- **Not implemented yet:**
  - Linux PipeWire backend (planned M8), macOS (experimental).
  - MIDI/OSC control.
  - Round-trip latency measurement (`latency.measure`).
  - FLAC recording (WAV only).
- **Not run yet:** Driver Verifier soak of the virtual driver (1 h).
