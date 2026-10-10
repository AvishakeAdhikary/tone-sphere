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
| **Linux (PipeWire)**: physical devices + hot-plug, persistent NexTone virtual outputs/mics usable by any app, app capture | ✅ in WSL Ubuntu 24.04, PipeWire 1.0.5 | `tests/integration/linux-pipewire.mjs` 11/11 (×4 runs; one click seen once at stream start in an earlier run). Engine unit tests pass on GCC 13 |
| **Linux desktop app** (AppImage) | ✅ launches under WSLg | `scripts/package-linux.sh`: the app starts its bundled engine and connects |
| Round-trip latency measurement (chirp + cross-correlation, Settings › Audio) | ✅ | unit test: delay found within 0.5 sample, sub-sample refinement, no-signal rejected; `tests/integration/latency.mjs` 6/6 through VB-CABLE (~150 ms, VB-CABLE's own buffer; varies by up to ~20 ms between device restarts) |
| "Install drivers" button (Settings › Devices, desktop build; one UAC prompt via `nextone-setup install-all`) | ✅ in a pristine VM | `vm/Test-DriverInstall.ps1`: with the engine already running, `install-all` installs both drivers in 4 s and the engine picks up the ASIO driver and all 8 virtual devices by itself |
| **Driver soak under Driver Verifier** (1 h, 54 cycles of the full audio test + hot-unplug while streaming) | ✅ no crash | `vm/Test-Soak.ps1`: 0 bugchecks, no dumps; device and engine survived every disable/enable. Audio clean in 45 of 54 iterations. Iteration 1: the engine started before the driver (now fixed: the VAD backend polls for its rings). Iterations 2–4 overlapped host builds. 2–6 clicks in 5 later iterations; iterations 28–54 all clean |
| OSC remote control | ✅ | unit tests (messages, bundles, junk); `tests/integration/osc.mjs` 5/5 with real UDP packets |
| MIDI controllers (mappings + learn, Windows) | ⚠️ logic tested only | unit tests for the mapping logic; no MIDI input on this PC, so the winmm input path is untested with hardware |
| UI fonts | ✅ | Segoe UI Variable for text; JetBrains Mono (bundled, OFL) for numbers, ports and paths |
| **Portable presets** (import/export routing maps that find devices by name; placeholders for devices not plugged in yet; re-binding when a device returns under a new id) | ✅ | unit tests (matching rules); `tests/integration/presets.mjs` 9/9: import with a missing device, placeholder keeps its connection, portable export, idempotent re-import, `node.forget` |
| **Guitar Rig live scenario** (`presets/guitar-rig-obs-ai04.nextone.json`) | ✅ core flow in the VM; ⏳ your check with the AI-04 | `vm/Test-GuitarRig.ps1`: the real Guitar Rig 7 (this PC's copy) on NexTone ASIO joins the preset in ~25 s; OBS and speaker stand-ins get its sound; virtual devices unplugged → shown disconnected, connections kept; Guitar Rig closed/reopened → sound returns clean; 0 engine xruns. On this PC the preset matched Line (AI-04), Speakers (Realtek(R) Audio) and CABLE Output at import |
| **Skins** (Fluent, Midnight, Studio, Neon, Paper; import/export as JSON) | ✅ | UI e2e: picking Studio applies its tokens and forces dark mode; Fluent restores |
| Virtual devices notice the driver going away while streaming (driver marks rings as gone) | ✅ in the VM | Guitar Rig scenario unplug step |
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
wsl -e bash scripts/build-linux.sh   # Linux engine + unit tests (WSL)
node tests\integration\linux-pipewire.mjs   # Linux end-to-end (PipeWire in WSL)
wsl -e bash scripts/package-linux.sh # Linux AppImage → ui\release\
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
  - Linux: per-app output redirection and app microphones (only app capture exists); plugin windows; MIDI. macOS: not started.
  - MIDI/OSC control.
  - FLAC recording (WAV only).
- **Reconnect transients.** Re-enabling the virtual audio driver in the VM makes Windows re-create all 8 endpoints, and the audio stack glitches for roughly 10–15 s (the test recorder itself loses data). The sound comes back by itself, and later measurements are clean, with no xruns. Streams that have just started can click during their first seconds. Unplugging the AI-04 over USB is a different, single-device path; that's for the owner's manual test.
- **Intermittent:** asio-host "shared interface" clicked once (13 clicks, −4.95 dB) in a `run-all` that started right after a full rebuild. It passed 5 times in a row afterwards on an idle machine.
