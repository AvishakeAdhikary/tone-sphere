# AGENTS.md: working on ToneSphere

Working rules for coding agents (and humans) changing this repository. Start with
[README.md](README.md) for what ToneSphere is. Read [docs/STATUS.md](docs/STATUS.md) for what is
verified and what is pending, and [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) to set up a machine
from a fresh clone (`scripts/setup-dev.ps1`).

The one rule above all others, carried over from the first ToneSphere:

> **A feature does not exist until a test proves it moves audio.**
> And a measurement you did not take is reported as `--`, never as `0`.

## Safety rules (non-negotiable)

- **Kernel driver work happens only in the Hyper-V test VM** (`vm/`, VM name
  `ToneSphere-DriverTest`). This covers installing it, Driver Verifier and soak tests. Never install
  `tonespherevad.sys` on the development PC, and never reboot the host. Rebooting the VM is fine:
  `Restore-TestVM` reverts to the `clean` checkpoint. Its password lives outside the repository
  (`%LOCALAPPDATA%\ToneSphereVM`); never write one into a script.
- Don't run `AsioLinkProToolPatcher`, and don't disassemble `asiolinkpro.exe`. ToneSphere is a
  clean-room design based on the public manual and install script. These files are the owner's
  property and are git-ignored.
- **Releases come only from CI.** Every green push to `main` tags and publishes one
  (`.github/workflows/ci.yml`, `scripts/release-version.mjs`). Don't create tags or releases by
  hand, and don't submit anything to Partner Center: the owner does that
  (`packaging/store/README.md`).
- **VB-CABLE is VB-Audio's.** Its licence allows passing the package on unmodified, not building
  it into another installer: never bundle it or install it silently. The app downloads the
  official package on request, checks VB-Audio's signature and opens VB-Audio's own installer.
- Certificates and keys (`*.pfx`, `*.cer`) are local test material and are git-ignored. Never
  commit them.
- **Host changes.** Installing the MSIX on the host or trusting certificates machine-wide
  changes the owner's PC, so do it in the VM instead (`vm/Test-Msix.ps1`, `vm/Test-Wack.ps1`).

## Layout and build

| Area | Path | Build / run |
|---|---|---|
| Engine (C++20) | `engine/` | `.\scripts\build.ps1 [-Test] [-Arch x86]` → `build\x64-release\bin` |
| Linux engine | same sources | `wsl -e bash scripts/build-linux.sh` → `~/ts-build/bin` |
| UI (Electron + React + Vite) | `ui/` | `npm run build`, `npm run typecheck`, `npx playwright test` |
| ASIO driver | `drivers/win/asio/` | part of `build.ps1` (x64 and x86) |
| Virtual audio driver | `drivers/win/vad/` | `.\drivers\win\vad\build.ps1` (cl/link + Inf2Cat + test signing) |
| Packages | `ui/` | `npm run dist:win` (NSIS), `npm run dist:store` (MSIX), `scripts/package-linux.sh` (AppImage) |
| Version | `VERSION` | The floor of the release version; CI stamps the real one (`scripts/release-version.mjs`) |
| Icons, splash | `assets/images/` | `npx electron scripts/make-icons.cjs` in `ui/` derives every icon from `ToneSphere.png` |

`scripts/build.ps1` sets up the MSVC environment itself (`scripts/vsenv.ps1`).
- **Stop a running engine before rebuilding.** Otherwise the linker can't overwrite
  `tonesphere-engine.exe`: run `Get-Process tonesphere-engine | Stop-Process`.
- **VS Code terminals set `ELECTRON_RUN_AS_NODE=1`.** The UI scripts and tests remove it. Do the
  same if you launch Electron yourself.

## Architecture rules

- **The audio path never blocks.**
  - Nothing on audio threads may lock, allocate or do I/O. That covers `processProgram`,
    bridges, backend callbacks and plugin `process()`.
  - Structural changes publish a new immutable `Program` (RCU). Retire old objects with
    `Engine::retireLocked()`, never by deleting them directly.
  - Live parameters are atomics.
- **Multi-core DSP** (`core/dsp_pool`, `computeLevels` in `core/graph.cpp`). Nodes of one
  dependency level run on several cores at once, so a node may only read another node's buffer
  through a dependency `computeLevels` knows about: routes, solo mixes and ducker sidechains
  (`ProgramNode::sidechainFrom`). A new kind of cross-node read must be added there, with a test
  in `engine/tests/parallel_tests.cpp` proving the parallel result is bit-identical to the serial
  one. Every audio thread calls `enableFlushToZero()`.
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
  - C++20, MSVC with a static CRT. Use the 4-space style of the surrounding code, `ts::`
    namespace, and `TS_INFO`/`TS_WARN` for logging.
  - Error messages that reach the UI are plain sentences a non-expert understands.
  - Comments explain *why*, as in the existing code.
- **TypeScript/React.**
  - Strict typecheck (`npm run typecheck`). Keep components accessible: ARIA labels, keyboard
    use, `role="status"` for results.
  - Fonts: Segoe UI Variable (system) for text, and JetBrains Mono (`--mono`, bundled via
    `@fontsource-variable/jetbrains-mono`) for numbers, ports, paths and code.
  - Every user-visible string goes through `t()` (`ui/src/renderer/i18n.ts`) and into each
    catalog in `ui/src/renderer/locales`; `npm run check:locales` (and CI) fails on a missing one.
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
| CI (`.github/workflows/ci.yml`) | All of the above that needs no audio hardware, ThreadSanitizer on Linux, a secret scan, and the downloads installed and launched on clean runners |

- **Integration suites need an idle machine.** They produce real audio through VB-CABLE, and
  heavy load on the host (compiling, the VM running WACK or a soak) causes clicks the engine
  can't see. Don't run them while building or while the VM is busy.
- **Record outcomes honestly in `docs/STATUS.md`.** Mark each item verified (with the evidence)
  or pending, including known limitations and flaky results.

## Commits

- Commit on `main` in small, verified steps, with messages that say what changed and why.
- End commit messages with the attribution trailer the session's instructions give, if any.
