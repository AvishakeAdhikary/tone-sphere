# ToneSphere build status

_Last updated: 2026-10-15 (version 1.0.0)._

ToneSphere 1.0 is the engine and interface first developed under the working name NexTone,
renamed and merged into this repository, which until 0.2.1 held the Python implementation
(tagged `legacy-python`). Every row below was re-verified after the rename unless it says
otherwise.

## What works (verified)

| Area | Status | Evidence |
|---|---|---|
| Real-time core: lock-free rings, varispeed resampler, drift-compensated adaptive bridges (with measured device-rate feed-forward up to ±3 %) | ✅ | `engine/tests/core_tests.cpp` (10 cases, 1,435 assertions): SPSC ring stress; 44.1→48 kHz resampling better than −60 dB; ±300 ppm and 1 %-slow devices stay locked. Real AI-04 USB interface running −10,300 ppm: locked at 15 ms latency (verified before the rename) |
| **Multi-core DSP** (auto, one core, or N cores; `dspThreads`) | ✅ | `engine/tests/parallel_tests.cpp` (1.28 M assertions): parallel output bit-identical to the serial order over 400 blocks while switching modes, block sizes and a solo mix, with feedback loops and sidechains both ways; auto engages for a heavy graph and not a light one; 20,000 pool rounds run every item exactly once. **ThreadSanitizer**: no races (Linux, GCC 13, `setarch -R`) |
| Multi-core speed-up | ✅ measured | `tonesphere-dsp-bench`, 8 inputs with RNNoise + EQ + compressor into 8 outputs, 128-frame blocks, i7-1165G7 (4 cores / 8 threads): one core 740–1,450 µs per block (the laptop's power state varies through the day), auto mode 290–330 µs, about 4.5× faster. LTO: one core within noise, auto ~10 % faster |
| Denormals flushed on every audio thread (FTZ/DAZ) | ✅ | `enableFlushToZero()` on the clock thread and every DSP worker |
| Engine graph: mixes, crosspoints, gain/mute/solo/pan/delay/polarity, meters | ✅ | `engine/tests/engine_tests.cpp` (17 cases) |
| Built-in effects: RNNoise, gate, EQ, compressor, limiter, auto-duck, delay | ✅ | unit tests: limiter ceiling, gate, 4:1 compression = −15 dB |
| **VST3 effects**: out-of-process scan, inserts, windows, state in config and scenes | ✅ | `tests/integration/plugins.mjs` 7/7 (ToneSphere Test Gain). Guitar Rig 7 scans in 2 s, inserts in 0.35 s (before the rename) |
| **VST3 instruments played by MIDI** + on-screen keyboard | ✅ | `tests/integration/instruments.mjs` 7/7 with the bundled *ToneSphere Test Synth*: listed as an instrument, A4 at full velocity −6.02 dBFS, CC 7 = 64 lowers it by the expected 5.9 dB (IMidiMapping), note off silences it; e2e: the keyboard sends C4/C3/E3 on and off |
| Windows audio devices: WASAPI shared/exclusive, IAudioClient3, hot-plug | ✅ | `tests/integration/host-audio.mjs` 10/10 |
| Per-app capture and redirection | ✅ capture verified | host-audio: a playing app appears automatically and its audio reaches a mix |
| **ToneSphere ASIO driver** (x64 + x86, multi-client, new CLSID) | ✅ | `tests/integration/asio-clients.mjs` 11/11, 64- and 32-bit hosts |
| Engine hosting an ASIO interface driver | ✅ FlexASIO; ⏳ ASIO4ALL | `tests/integration/asio-host.mjs` 8/8 with FlexASIO looped through VB-CABLE |
| Network audio (VBAN) | ✅ | `tests/integration/vban.mjs` 4/4 |
| Recording and instant replay | ✅ | `tests/integration/recorder.mjs` 6/6 |
| Idle meters and global hotkeys | ✅ | `tests/integration/meters-hotkeys.mjs` 2/2 |
| Meters only while someone watches (`meters.subscribe`; the hidden window unsubscribes; no subscriber = no meter work on the audio thread) | ✅ | `tests/integration/meters-subscribe.mjs` 4/4 |
| Round-trip latency measurement | ✅ | `tests/integration/latency.mjs` 6/6 through VB-CABLE |
| OSC remote control (and the old `/nextone/` prefix) | ✅ | `tests/integration/osc.mjs` 5/5; unit test for the alias |
| HTTP JSON-RPC for scripts (`POST /rpc`, token, refuses web pages) | ✅ | `tests/integration/http-rpc.mjs` 6/6 |
| Crash reports (engine handler + notice in the app) | ✅ | `tests/integration/crash-report.mjs`: Windows 4/4 (report + MDMP minidump), Linux 3/3 (SIGSEGV); e2e: the notice shows once |
| Portable presets (and old `nextone-preset` files) | ✅ | `tests/integration/presets.mjs` 9/9 |
| Moving from NexTone: settings, presets, plugin cache, window profile copied once; old preset/skin formats accepted | ✅ | unit test (copy once, never over an existing config, skips runtime files) |
| **UI** (Electron + React): bins, matrix, inspector, effects, instruments, scenes, settings, skins | ✅ | `ui/tests/e2e/ui.spec.ts` 17/17 |
| **Splash** (the README banner, re-encoded to 610 KB WebM) and **single instance** | ✅ | e2e: the splash hands over to the main window; a second launch exits and leaves one window |
| "Hear my instrument" wizard | ✅ | e2e: picks the ASIO interface and the headphones, input 1 into both ears at −12 dB |
| Window shortcuts (Ctrl+S/O/,/M/1/2, F5, Ctrl+Shift+M) | ✅ | e2e, including "not while typing" |
| **Hindi** interface (233 strings, machine translation, unreviewed) | ✅ | `npm run check:locales` 233/233; e2e switches to Hindi and back |
| **VB-CABLE through Settings › Devices › Install drivers** | ✅ signature check; see the VM row | `tests/integration/vbcable-signature.mjs` 6/6: the official download is accepted (signer *BUREL VINCENT*), one changed byte, a Microsoft-signed file and a missing file are refused |
| ToneSphere virtual audio driver (renamed, new endpoint GUIDs) | ✅ builds, catalogs and test-signs; ⏳ VM re-run | `drivers/win/vad/build.ps1` with `DriverVer` stamped from `VERSION` |
| Desktop installer (public: ASIO driver, no virtual audio driver) | ✅ builds; ⏳ VM / CI smoke | `npm run dist:win` without the VAD package: drivers folder holds the ASIO DLLs and `tonesphere-setup.exe` only |
| Microsoft Store package | ✅ builds unsigned | `electron-builder --win appx`: "AppX is not signed (Windows Store only build)" |
| **Linux** engine (PipeWire) | ✅ in WSL Ubuntu 24.04 | build + unit tests 3/3; runs and answers without a PipeWire server; `linux-pipewire.mjs` 11/11 before the rename |
| **Linux AppImage** | ✅ | `scripts/package-linux.sh` + `.github/scripts/smoke-linux.sh`: the AppImage's engine starts and answers (1.0.0) |
| Single version source, release script | ✅ | `node --test scripts/` 6/6: first release v1.0.0 after the Python v0.2.1, then patch bumps |
| CI workflow | ✅ lints (actionlint); ⏳ first run on GitHub | `.github/workflows/ci.yml`, `pages.yml` |
| Secrets | ✅ | gitleaks over the whole merged history (120 commits): no findings. The VM password is generated per machine outside the repository |

## How to run

```powershell
.\scripts\setup-dev.ps1              # toolchain from a fresh clone (see docs/DEVELOPMENT.md)
.\scripts\build.ps1 -Test            # engine, ASIO driver, tools, test plugins + unit tests (x64)
.\scripts\build.ps1 -Arch x86        # 32-bit ASIO driver
.\drivers\win\vad\build.ps1          # kernel driver package (test-signed)
.\tests\run-all.ps1                  # every host-safe suite + UI e2e; needs VB-CABLE + FlexASIO
.\vm\Test-VbCable.ps1                # public installer + VB-CABLE "Install drivers" in a pristine VM
.\vm\Test-DriverInstall.ps1 ; .\vm\Test-InVM.ps1   # virtual audio driver in the VM
build\x64-release\bin\tonesphere-dsp-bench.exe     # multi-core speed-up
```

## Pending / known limitations

- **Virtual audio driver signing.** Test-signed only; public builds leave it out until Microsoft
  attestation-signs it (needs an EV certificate). Virtual devices come from VB-CABLE meanwhile.
- **VB-CABLE install is VB-Audio's installer.** Its licence forbids building it into another
  installer, so ToneSphere downloads, verifies and opens it; the user clicks *Install Driver*.
- **Store submission is the owner's** (`packaging/store/README.md`).
- **MIDI from hardware ports** (controllers and instrument keyboards): the logic is unit-tested and
  instruments are verified through `fx.midi`, but no MIDI device was connected to test the winmm
  input path itself.
- **ASIO4ALL** as the engine's interface driver: not yet run here (FlexASIO is verified).
- **VST3:** plugins run inside the engine (a fault switches the plugin off, a hang can still stall
  the engine); no generic parameter sliders; plugin latency isn't compensated; plugin windows are
  Windows-only.
- **Not implemented:** macOS; Linux per-app redirection, plugin windows and MIDI; FLAC recording.
  From the old Python ToneSphere, deliberately not carried over: its custom UDP/TCP/Opus network
  protocol (VBAN interoperates with existing tools instead), the PortAudio fallback host and the
  macOS CoreAudio plug-in (a whole platform port), and its own kernel cable driver (superseded).
- **Audio under heavy CPU load.** With the host saturated (builds, the VM busy), the VB-CABLE
  loopback suites can record clicks the engine itself doesn't see; run them on an idle machine.
- **Hindi** is a machine translation, not yet reviewed by a native speaker.
