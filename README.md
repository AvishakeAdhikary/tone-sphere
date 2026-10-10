# NexTone

**Route any audio anywhere: simple for everyone, deep enough for studios.**

NexTone is a free, open-source audio router for Windows (Linux too; macOS later). Everything on
your computer that makes or takes sound is sorted into six bins:

| Sources (sound comes from) | Destinations (sound goes to) |
|---|---|
| **Physical inputs**: microphones, line-ins, audio interfaces | **Physical outputs**: speakers, headphones, interface outputs |
| **Virtual outputs**: virtual speakers that other apps play into | **Virtual inputs**: virtual microphones that other apps record from |
| **Application outputs**: what each app plays (game, browser, Spotify, a DAW) | **Application inputs**: the microphone each app hears (Discord, Zoom, OBS, a DAW) |

To connect two things, drag a source onto a destination. Everything else is optional.

NexTone is a clean-room successor to *ODeus ASIO Link Pro*: it keeps the power and drops the
confusion. See [docs/parity.md](docs/parity.md) for the feature-by-feature comparison.

## Features

- **Routing.** Any source to any destination. Mixes (e.g. "Stream mix", "Monitor mix") have
  per-route volume, a channel matrix, mute/solo, pan, polarity and delay.
- **Clock sync.** Every device stays in sync through adaptive drift compensation, so there's no
  slow drift and no clicks between devices.
- **Applications.** Capture any app's sound, give an app its own output, or feed a mix into an
  app's microphone.
- **NexTone ASIO driver.** Several DAWs can share one audio interface at once (64- and 32-bit
  hosts).
- **Effects.** Built-in effects: noise suppression (RNNoise), gate, EQ, compressor, limiter,
  auto-duck and delay. **VST3 plugins** work on any input, output or mix, with their own
  windows.
- **Network audio.** VBAN, compatible with VoiceMeeter.
- **Recording.** Multitrack recording and an instant-replay buffer.
- **Control.** Scenes with global hotkeys, push-to-talk, OSC and MIDI controllers, and a
  command-line tool.
- **Diagnostics.** Round-trip latency measurement and per-device latency, drift and dropout
  statistics.

## Install

| Platform | Package | Notes |
|---|---|---|
| Windows (recommended) | Microsoft Store | Free, signed by Microsoft. Virtual devices use *Compatibility mode* (an installed cable such as VB-CABLE). |
| Windows (full) | `NexTone-Setup-<version>.exe` (Driver Pack) | Adds the NexTone ASIO driver and the NexTone virtual audio driver. |
| Linux | `NexTone-<version>-x86_64.AppImage` | Needs PipeWire. NexTone's virtual devices are native PipeWire nodes. |

> **Note.** The NexTone *virtual audio driver* is currently **test-signed**, so Windows only loads
> it with test signing turned on. Public builds need Microsoft attestation signing. Everything
> else, including the ASIO driver, works on any PC. See [docs/STATUS.md](docs/STATUS.md).

## Build from source

**Requirements**
- **Windows:** Visual Studio 2022 or later with C++, Windows SDK and WDK 10.0.26100, CMake,
  Ninja, and Node.js 22+.
- **Linux:** GCC 13+, CMake, Ninja, pkg-config, `libpipewire-0.3-dev`, and Node.js 22+.

```powershell
# Windows
.\scripts\build.ps1 -Test            # engine, ASIO driver, tools, test plugin + unit tests (x64)
.\scripts\build.ps1 -Arch x86        # 32-bit ASIO driver
.\drivers\win\vad\build.ps1          # virtual audio driver package (test-signed)
cd ui; npm ci; npm run dev           # UI against the engine (or: npm run mock-engine)
cd ui; npm run dist:win              # desktop installer → ui\release\
cd ui; npm run dist:store            # Microsoft Store package → ui\release-store\
```

```bash
# Linux (or WSL)
scripts/build-linux.sh               # engine + unit tests
scripts/package-linux.sh             # AppImage → ui/release/
```

## Tests

```powershell
.\tests\run-all.ps1                         # host-safe suites + UI end-to-end (needs VB-CABLE + FlexASIO)
node tests\integration\linux-pipewire.mjs   # Linux engine inside WSL
.\vm\Test-InVM.ps1                          # virtual audio driver, inside a Hyper-V test VM
.\vm\Test-Soak.ps1                          # 1 h driver soak under Driver Verifier (VM)
.\vm\Test-Installer.ps1 ; .\vm\Test-Msix.ps1 ; .\vm\Test-Wack.ps1
```

Kernel-driver tests only ever run inside the Hyper-V VM (`vm/`), never on the development PC.
What is verified and what is still pending is tracked in [docs/STATUS.md](docs/STATUS.md).

## Repository layout

```
engine/        C++20 audio engine: graph, bridges, effects, VST3 host, backends (WASAPI, ASIO,
               PipeWire, VBAN), OSC/MIDI, WebSocket JSON-RPC
ui/            Electron + React app (bins, matrix, inspector, settings) and packaging
drivers/win/   NexTone ASIO driver (user mode) and virtual audio driver (kernel, WaveRT)
shared/        Engine ⇄ UI protocol (PROTOCOL.md; TypeScript types live in ui/src/shared)
tools/         nextone-ctl (CLI), nextone-setup (driver installer), audio-probe, asio-test-host
tests/         Integration suites (Node) that drive the real engine with real audio
vm/            Hyper-V test VM automation
packaging/     Microsoft Store listing, privacy policy, submission guide
docs/          STATUS.md (verified / pending), parity.md
```

## Remote control

**OSC.** Turn on *Settings › Control › Allow OSC control* (UDP port 9000 by default). Then send:

```
/nextone/volume "Stream mix" 0.75     /nextone/mute "Microphone"      (toggle)
/nextone/gain   "Speakers"  -6.0      /nextone/scene "Gaming"
/nextone/record 1                     /nextone/replay
```

**MIDI.** In *Settings › Control*, choose an action, press **Learn** and move a knob, fader or
pad.

**Command line.** Use `node tools/nextone-ctl.mjs state`, plus `connect`, `tone`, `meters` and
raw `call`.

## License

NexTone is free software under the **GNU General Public License v3.0 or later** (see
[LICENSE](LICENSE)). Third-party components:
- ASIO SDK (GPLv3 option), VST3 SDK (MIT), RNNoise (BSD-3), nlohmann/json (MIT), Catch2 (BSL-1.0).
- JetBrains Mono (SIL OFL 1.1).
- The virtual audio driver is based on Microsoft's SimpleAudioSample (MS-PL; see
  `drivers/win/vad/LICENSE-MS-PL.txt`).

ASIO is a trademark of Steinberg Media Technologies GmbH. NexTone is not affiliated with ODeus
or Steinberg.
