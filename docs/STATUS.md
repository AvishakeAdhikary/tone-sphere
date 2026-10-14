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
| **Multi-core DSP** (auto, one core, or N cores; `dspThreads`) | ✅ | `engine/tests/parallel_tests.cpp` (1.28 M assertions): parallel output bit-identical to the serial order over 400 blocks while switching modes, block sizes and a solo mix, with feedback loops and sidechains both ways; auto engages for a heavy graph and not a light one; 20,000 pool rounds run every item exactly once; pools used or destroyed before their threads start finish, on 1, 2, 4 and 8 CPUs (these hung before the fix, and hung CI for 80 minutes). **ThreadSanitizer**: no races (Linux, GCC 13, `setarch -R`) |
| Multi-core speed-up | ✅ measured | `tonesphere-dsp-bench`, 8 inputs with RNNoise + EQ + compressor into 8 outputs, 128-frame blocks, i7-1165G7 (4 cores / 8 threads): one core 740–1,450 µs per block (the laptop's power state varies through the day), auto mode 290–330 µs, about 4.5× faster. LTO: one core within noise, auto ~10 % faster |
| Denormals flushed on every audio thread (FTZ/DAZ) | ✅ | `enableFlushToZero()` on the clock thread and every DSP worker |
| Engine graph: mixes, crosspoints, gain/mute/solo/pan/delay/polarity, meters | ✅ | `engine/tests/engine_tests.cpp` (17 cases) |
| Built-in effects: RNNoise, gate, EQ, compressor, limiter, auto-duck, delay | ✅ | unit tests: limiter ceiling, gate, 4:1 compression = −15 dB |
| **VST3 effects**: out-of-process scan, inserts, windows, state in config and scenes | ✅ | `tests/integration/plugins.mjs` 7/7 (ToneSphere Test Gain). Guitar Rig 7 scans in 2 s, inserts in 0.35 s (before the rename) |
| **VST3 instruments played by MIDI** + on-screen keyboard | ✅ | `tests/integration/instruments.mjs` 7/7 with the bundled *ToneSphere Test Synth*: listed as an instrument, A4 at full velocity −6.02 dBFS, CC 7 = 64 lowers it by the expected 5.9 dB (IMidiMapping), note off silences it; e2e: the keyboard sends C4/C3/E3 on and off |
| Windows audio devices: WASAPI shared/exclusive, IAudioClient3, hot-plug | ✅ (⚠️ render gaps under load) | `tests/integration/host-audio.mjs` 10/10 on an idle host. On 2026-10-15 the render path had one gap in 4 of 8 runs **with the committed engine and with the clock-master engine alike** (0 engine dropouts): the host was not idle (a background CPU miner, see the limitations below) |
| Per-app capture and redirection | ✅ capture verified | host-audio: a playing app appears automatically and its audio reaches a mix |
| **ToneSphere ASIO driver** (x64 + x86, multi-client, new CLSID) | ✅ | `tests/integration/asio-clients.mjs` 11/11, 64- and 32-bit hosts |
| **Real audio interface: Audio Array AI-04** (USB, class-compliant, no manufacturer ASIO driver) | ✅ on the host | Hyper-V can't hand a USB audio device to the VM, so these ran on the host (user-mode only, nothing installed). `tests/integration/interface.mjs` 33/33: 44.1, 48, 96 and **192 kHz**, shared and exclusive, both directions, 0 dropouts; smallest exclusive period 144 frames (3 ms). `interface-latency.mjs`: best input-to-output ~28 ms on the internal clock, **~21 ms** with the AI-04 leading the clock (output ~8 ms, input ~13 ms). `interface-daws.mjs` 12/12: the AI-04 exclusive as clock master while two DAWs share ToneSphere ASIO (3 runs, exact pitch and level, 0 clicks, both on time). `interface-asio.mjs FlexASIO` 7/7: FlexASIO configured for the AI-04 (WASAPI exclusive, 144 frames) hosted by the engine, 0 dropouts and 0 device glitches, the interface's input reaches a mix, with the two DAWs alongside |
| **Device clock master** (`clockMaster`: auto, internal, or a device) | ✅ | The engine's timer follows the master device's own position (delay-locked loop, ±0.5 %, ~0.5 s); every other device keeps drift compensation. Auto picks a physical WASAPI output in exclusive mode; shared-mode devices never lead (Windows' mixer clock clicked). `tests/integration/clock-master.mjs` 8/8: shared VB-CABLE keeps the internal timer, FlexASIO chosen explicitly leads and audio through it is clean, and when the master goes away the engine carries on on its own timer. FlexASIO isn't picked automatically: as auto master it made the ToneSphere ASIO client path click in 2 of 3 runs |
| Engine hosting an ASIO interface driver | ✅ FlexASIO; ⚠️ ASIO4ALL | `tests/integration/asio-host.mjs` 8/8 with FlexASIO looped through VB-CABLE, on the host and in the VM; FlexASIO on the AI-04 above. **ASIO4ALL 2.22** (VM, `vm/Test-Asio4All.ps1`): listed (32 in / 16 out), exact loopback level (−12.000 dB), continuous, two DAWs share it on time, but its output bridge overruns (~100 per 10 s, fill 80 ms against a 26.5 ms target) and the shared path clicked (36 and 5 clicks in two runs). Not diagnosed yet; FlexASIO in the same VM is clean, so it is specific to ASIO4ALL there. ASIO4ALL with a real interface is untested |
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
| **UI** (Electron + React): bins, matrix, inspector, effects, instruments, scenes, settings, skins | ✅ | `ui/tests/e2e/ui.spec.ts` 18/18 |
| **Splash** (the README banner, re-encoded to 610 KB WebM) and **single instance** | ✅ | e2e: the splash hands over to the main window; a second launch exits and leaves one window |
| "Hear my instrument" wizard, with **Lowest delay** | ✅ | e2e: picks the ASIO interface and the headphones, input 1 into both ears at −12 dB; with a WASAPI interface, "Lowest delay" puts it (and its output, when it is the same device) in exclusive mode with a 3 ms buffer, so it leads the clock |
| Window shortcuts (Ctrl+S/O/,/M/1/2, F5, Ctrl+Shift+M) | ✅ | e2e, including "not while typing" |
| **Hindi** interface (246 strings, machine translation, unreviewed) | ✅ | `npm run check:locales` 246/246; e2e switches to Hindi and back |
| **VB-CABLE through Settings › Devices › Install drivers** | ✅ in a pristine VM | `tests/integration/vbcable-signature.mjs` 6/6 (the official download is accepted, signer *BUREL VINCENT*; one changed byte, a Microsoft-signed file and a missing file are refused). `vm/Test-VbCable.ps1`: public installer installed silently and launched like a user; install-all registered ASIO, verified VB-Audio's signature and opened VB-Audio's installer (clicked by `vm/vbcable-click.ps1`); the running engine found the cable by itself. After the restart VB-Audio asks for: host-audio 10/10 through the cable; a second run had one start-up gap on the render path (0 engine dropouts). Before that restart the render path clicks, as VB-Audio's readme warns |
| ToneSphere virtual audio driver (renamed, new endpoint GUIDs) | ✅ in the Hyper-V VM | `vm/Test-DriverInstall.ps1`: install-all installs ASIO + the driver in 3 s and the running engine finds all 8 devices (ToneSphere Output/Mic 1–4) by itself; `vm/Test-InVM.ps1` vad-audio 14/14 (mic, output and cable paths: exact frequency and level, 0 clicks, 0 gaps, 0 dropouts) |
| Desktop installer (public: ASIO driver, no virtual audio driver) | ✅ in a pristine VM and on GitHub | CI "Install and launch on a clean Windows" passes (run 38645375561). `vm/Test-VbCable.ps1`: silent install, ASIO registered, no driver package inside, the app starts its engine on a clean Windows with no developer tools |
| Microsoft Store package | ✅ builds unsigned | `electron-builder --win appx`: "AppX is not signed (Windows Store only build)" |
| **Linux** engine (PipeWire) | ✅ in WSL Ubuntu 24.04 | build + unit tests 3/3; runs and answers without a PipeWire server; `linux-pipewire.mjs` 11/11 before the rename |
| **Linux AppImage** | ✅ | `scripts/package-linux.sh` + `.github/scripts/smoke-linux.sh`: the AppImage's engine starts and answers (1.0.0) |
| Test VM | ✅ | `vm/New-TestVM.ps1` rebuilt from the Windows 11 Enterprise evaluation (26H2); the `clean` checkpoint is now taken after first sign-in's pending restart (it used to reboot the VM a minute into every test) |
| Single version source, release script | ✅ | `node --test scripts/release-version.test.mjs` 6/6: first release v1.0.0 after the Python v0.2.1, then patch bumps |
| CI workflow | ✅ green on GitHub (branch) | Run 38645375561 on `tonesphere-v1`: secret scan, version, Windows build + unit + e2e + integration + packages, VAD compile and sign check, Linux + ThreadSanitizer + AppImage, install-and-launch smoke. The release job runs only on `main` and hasn't run yet. `pages.yml` deploys |
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
- **ASIO4ALL** as the engine's interface driver: works, but overruns on its output in the VM loopback (see above); to investigate, ideally with a real interface.
- **VST3:** plugins run inside the engine (a fault switches the plugin off, a hang can still stall
  the engine); no generic parameter sliders; plugin latency isn't compensated; plugin windows are
  Windows-only.
- **Not implemented:** macOS; Linux per-app redirection, plugin windows and MIDI; FLAC recording.
  From the old Python ToneSphere, deliberately not carried over: its custom UDP/TCP/Opus network
  protocol (VBAN interoperates with existing tools instead), the PortAudio fallback host and the
  macOS CoreAudio plug-in (a whole platform port), and its own kernel cable driver (superseded).
- **Audio under heavy CPU load.** With the host saturated (builds, the VM busy), the VB-CABLE
  loopback suites can record clicks the engine itself doesn't see; run them on an idle machine.
  The 2026-10-15 `run-all` (every other suite passed, clock-master 8/8, e2e 18/18) ran next to a
  CPU miner found on the development PC (not part of this project; reported to the owner), which
  explains the host-audio gaps above.
- **ASIO4ALL with a real interface**, and **the AI-04 inside the VM**: Hyper-V can't pass a USB
  audio device through, so the AI-04 was tested on the host, with WASAPI and FlexASIO only.
- **Hindi** is a machine translation, not yet reviewed by a native speaker.
