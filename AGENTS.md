# AGENTS.md: working on NexTone

Working rules for coding agents (and humans) changing this repository. Start with
[README.md](README.md) for what NexTone is. Read [docs/STATUS.md](docs/STATUS.md) for what is
verified and what is pending.

## Safety rules (non-negotiable)

- **Kernel driver work happens only in the Hyper-V test VM** (`vm/`, VM name
  `NexTone-DriverTest`). This covers installing it, Driver Verifier and soak tests. Never install
  `nextonevad.sys` on the development PC, and never reboot the host. Rebooting the VM is fine:
  `Restore-TestVM` reverts to the `clean` checkpoint.
- Don't run `AsioLinkProToolPatcher`, and don't disassemble `asiolinkpro.exe`. NexTone is a
  clean-room design based on the public manual and install script. These files are the owner's
  property and are git-ignored.
- **Don't submit anything to Partner Center or publish releases.** The owner does that
  (see `packaging/store/README.md`).
- Certificates and keys (`*.pfx`, `*.cer`) are local test material and are git-ignored. Never
  commit them.
- **Host changes.** Installing the MSIX on the host or trusting certificates machine-wide
  changes the owner's PC, so do it in the VM instead (`vm/Test-Msix.ps1`, `vm/Test-Wack.ps1`).

## Layout and build

| Area | Path | Build / run |
|---|---|---|
| Engine (C++20) | `engine/` | `.\scripts\build.ps1 [-Test] [-Arch x86]` → `build\x64-release\bin` |
| Linux engine | same sources | `wsl -e bash scripts/build-linux.sh` → `~/nt-build/bin` |
| UI (Electron + React + Vite) | `ui/` | `npm run build`, `npm run typecheck`, `npx playwright test` |
| ASIO driver | `drivers/win/asio/` | part of `build.ps1` (x64 and x86) |
| Virtual audio driver | `drivers/win/vad/` | `.\drivers\win\vad\build.ps1` (cl/link + Inf2Cat + test signing) |
| Packages | `ui/` | `npm run dist:win` (NSIS), `npm run dist:store` (MSIX), `scripts/package-linux.sh` (AppImage) |

`scripts/build.ps1` sets up the MSVC environment itself (`scripts/vsenv.ps1`).
- **Stop a running engine before rebuilding.** Otherwise the linker can't overwrite
  `nextone-engine.exe`: run `Get-Process nextone-engine | Stop-Process`.
- **VS Code terminals set `ELECTRON_RUN_AS_NODE=1`.** The UI scripts and tests remove it. Do the
  same if you launch Electron yourself.

## Architecture rules

- **The audio path never blocks.**
  - Nothing on audio threads may lock, allocate or do I/O. That covers `processProgram`,
    bridges, backend callbacks and plugin `process()`.
  - Structural changes publish a new immutable `Program` (RCU). Retire old objects with
    `Engine::retireLocked()`, never by deleting them directly.
  - Live parameters are atomics.
- **The engine lock (`Engine::mu_`).**
  - Every RPC handler runs under it, except `latency.measure`, `midi.learn` and `midi.devices`,
    which wait for real time and dispatch before taking the lock (`Engine::call`).
  - Anything that waits for audio or for the user must do the same.
- **Main (UI) thread.**
  - VST3 plugins are created, edited, saved and destroyed there, through `ui::run()`
    (`util/ui_thread.h`). Their windows also live there.
  - The main thread must **never** take the engine lock. Callers hold it while they wait on
    `ui::run`.
  - Background work (power events, hotkeys, OSC/MIDI) runs on its own threads.
- **Devices.**
  - Backends implement `Backend`/`DeviceStream` (`engine/src/backends/device.h`).
  - Streams talk only to `CaptureBridge`/`RenderBridge`, which do drift compensation.
  - Report device-side starvation through `DeviceStream::deviceGlitches()`, so xruns stay
    honest.
- **The protocol has four copies, kept in step.** When you add a method or a field, update all
  of them in the same change:
  - `shared/protocol/PROTOCOL.md`,
  - `ui/src/shared/protocol.ts`,
  - the engine (`core/rpc.cpp`, `core/state.cpp`, `core/config.cpp` for persisted settings),
  - `ui/mock-engine/server.mjs`.
- **Platform code** lives under `backends/win`, `backends/linux` and so on, behind `#ifdef`s or
  CMake conditions. Linux must keep building (`scripts/build-linux.sh`).

## Style

- **C++.**
  - C++20, MSVC with a static CRT. Use the 4-space style of the surrounding code, `nt::`
    namespace, and `NT_INFO`/`NT_WARN` for logging.
  - Error messages that reach the UI are plain sentences a non-expert understands.
  - Comments explain *why*, as in the existing code.
- **TypeScript/React.**
  - Strict typecheck (`npm run typecheck`). Keep components accessible: ARIA labels, keyboard
    use, `role="status"` for results.
  - Fonts: Segoe UI Variable (system) for text, and JetBrains Mono (`--mono`, bundled via
    `@fontsource-variable/jetbrains-mono`) for numbers, ports, paths and code.
- **Line endings.** Many source files are CRLF. Use exact-match editing tools rather than
  LF-only string replacements. Shell scripts are LF (`.gitattributes`).

## Testing

**Every behaviour change needs a test that exercises it for real.**

| Suite | What it covers |
|---|---|
| `engine/tests/*.cpp` (Catch2, `build.ps1 -Test`) | DSP, the graph, parsers, estimators |
| `tests/integration/*.mjs` | The real engine with real audio through VB-CABLE, FlexASIO, VBAN, the test VST3 plugin, OSC and so on. `tests/run-all.ps1` runs them all. |
| `ui/tests/e2e` (Playwright, mock engine) | UI flows |
| `vm/*.ps1` | Drivers, installers, Store package and WACK, in the VM |

- **Integration suites need an idle machine.** They produce real audio through VB-CABLE, and
  heavy load on the host (compiling, the VM running WACK or a soak) causes clicks the engine
  can't see. Don't run them while building or while the VM is busy.
- **Record outcomes honestly in `docs/STATUS.md`.** Mark each item verified (with the evidence)
  or pending, including known limitations and flaky results.

## Commits

- Commit on `main` in small, verified steps, with messages that say what changed and why.
- End commit messages with the attribution trailer the session's instructions give, if any.
