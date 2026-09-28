# Agent instructions for Tone Sphere

This file is the engineering contract for this repository. `CLAUDE.md` only points here.
If this file and any other document disagree, this file wins — and the disagreement is a
bug to fix in the other document.

## The one rule that governs everything here

**A feature does not exist until a test proves it moves audio.** And a measurement you did
not take is reported as `--`, never as `0`. This is stated in README.md and enforced by
`tests/test_honesty.py`.

This project was rebuilt from an earlier version whose entire audio layer was fake: eight
"drivers" that returned `np.zeros()` and called it a stream, "virtual devices" that were
`queue.Queue` objects, a UI that displayed `CPU: 0% | Latency: 0ms` permanently. Every rule
below exists to prevent a regression back to that state. Before adding a capability or a
status field, ask: does this reflect something actually measured, and is there a real test
(not a mock that always passes) proving the audio actually moved?

## Product direction

ToneSphere is a **Windows-first, Python-first, low-latency virtual audio mixer, router,
ASIO host and VST3 host**. The reference platform is **Windows 10/11 x64**.

- The Windows architecture is the reference implementation. Linux and macOS code is kept
  where it is clean and tested (the PortAudio host, the Linux null sink, the macOS
  CoreAudio plug-in), but it must never constrain the Windows real-time engine, and
  parity with it is not a goal.
- The target, and the order it is built in, is the migration plan recorded in
  `docs/IMPLEMENTATION_STATUS.md`. That file — not the README, not a commit message, not
  a test name — is the record of what is actually done.

## Architecture: two worlds

**Control plane — Python.** Application lifecycle, configuration, routing and mixer
state, presets, device selection, plugin discovery and metadata, plugin parameter UI, the
Qt UI, the REST/WebSocket API, the CLI, persistence, logging, diagnostics, tests,
benchmark orchestration.

**Real-time plane — native (C++20 behind a C ABI).** The device callback (ASIO
`bufferSwitch`, WASAPI event thread), preallocated buffers, the compiled routing plan,
the mixer, metering, built-in DSP, VST3 `process()`, format conversion at the device
boundary.

The boundary between them:

- `native/include/tonesphere_native.h` is a **flat C ABI**. Python loads the DLL with
  **ctypes** (the house idiom — `engine/wasapi_com.py` already does raw COM this way). No
  pybind11, no CPython extension, no C++ types crossing the boundary. A C ABI does not
  need rebuilding per Python version and cannot leak interpreter state into native code.
- **Python is never called from the audio thread.** No Python callback, no ctypes
  `CFUNCTYPE` invoked by the real-time path, no GIL acquisition. ctypes calls into the
  DLL happen on control threads only.
- Python prepares, native executes. Routing changes are compiled by Python into a flat
  execution plan; native validates it again, preallocates everything **off** the audio
  thread, and publishes it by atomic pointer swap. The audio thread never parses,
  allocates for, or analyses topology.
- Continuous controls (gain, pan, mute, solo, trim, polarity) are atomic targets smoothed
  on the audio thread. Plugin parameter changes cross in bounded SPSC queues. Meters and
  statistics come back through preallocated slots the control plane polls.
- A missing native DLL is an explicit error with the load failure's reason. It is never
  silently replaced by something that looks like it works.

## Real-time safety rules

Inside the audio callback and anything it calls, none of the following:

- Python, the GIL, Python objects
- heap allocation or free (`new`, `malloc`, `std::vector` growth, `std::string`,
  `std::function` captures that allocate) after the plan is activated
- blocking locks, condition variables, waiting on events, `Sleep`
- filesystem, registry, network, console or GUI access
- logging — errors go into a preallocated event ring or atomic counter, and the control
  plane logs them later
- plugin scanning, module loading, device enumeration, configuration parsing
- unbounded loops or unbounded queues

Synchronisation uses atomics, immutable snapshots, and lock-free structures whose
guarantee is stated precisely. **Do not call something "lock-free" or "wait-free" unless
the implementation justifies that exact word**, and say whether it is SPSC, MPSC or MPMC.
A structure that is only correct with one producer must document and, where practical,
enforce that.

Freeing an old plan or buffer happens on the control thread after the audio thread has
acknowledged the new one — never on the audio thread.

Third-party VST3 plugins are not assumed to be real-time safe. Say so where it matters,
and never extend a guarantee about ToneSphere's own code to someone else's plugin.

## Windows audio requirements

- **ASIO:** a real native host (`native/asio/`) — driver discovery from
  `HKLM\SOFTWARE\ASIO`, loading by CLSID, channel/format/rate/buffer negotiation,
  `createBuffers`, a native `bufferSwitch` driving the native graph, `asioMessage`
  handling (reset requests are serviced on the control thread), latency reporting,
  clean stop/dispose/exit. A registry entry or a device name is not ASIO support; ASIO
  support means a driver was initialised and audio moved through `bufferSwitch`.
- **WASAPI:** native shared and exclusive, event-driven, MMCSS-registered; MMDevice
  enumeration with stable endpoint IDs; `IMMNotificationClient` for arrival, removal and
  default changes; process loopback and whole-system loopback.
- **VST3:** a real native host on the official Steinberg SDK — module load, factory and
  class enumeration, component/controller lifecycle, bus negotiation, `setupProcessing`,
  float32 `process()`, parameters, state through the SDK's state APIs, latency, clean
  unload. Finding a `.vst3` file is discovery, not hosting.
- **Virtual device:** an endpoint Windows itself enumerates, which another application
  can open. It needs a kernel driver (`driver/windows_virtual_audio/`). An in-process bus,
  a queue, or a named object is not a virtual device and must never be labelled as one.

**Fallbacks** are allowed only between genuine alternatives, and they must be visible:
"ASIO unavailable → WASAPI exclusive" is fine if the UI says `Backend: WASAPI Exclusive`
and `ASIO: unavailable`. Pretending the requested backend is running is forbidden.

## Measurement and honesty rules

- Unknown is `--` (`utils/formatting.UNKNOWN`), never `0`, never a plausible guess.
- Latency is reported as **separate** quantities, each labelled for what it is:
  configured buffer and sample rate; **nominal** (buffer ÷ rate, arithmetic); **reported**
  (what the driver or PortAudio says); plugin latency; **measured** round trip (a real
  signal emitted and captured). A number computed or reported must never be labelled
  "measured". Without a loopback path, measured round trip is `--`.
- Performance statistics come from the callback itself: callback duration min / mean /
  max (and a percentile), buffer period, processing load = worst callback ÷ buffer
  period, xruns, overruns, underruns. Averages alone are not acceptable — real-time
  stability is governed by the worst case.
- Benchmark numbers are recorded from actual runs, with the machine and configuration.
  Never invent, extrapolate or round a number up.

## Evidence levels

Every capability in `docs/IMPLEMENTATION_STATUS.md` and the README carries exactly one:

| Level | Meaning | Evidence required |
|---|---|---|
| **NOT IMPLEMENTED** | The capability does not exist. | — |
| **UNVERIFIED** | Code exists; real-world behaviour has not been demonstrated. | The code. |
| **IMPLEMENTED** | Code exists and automated tests demonstrate it. | A test that runs in `-m "not hardware"`. |
| **VERIFIED** | Implemented, plus an integration test demonstrates the behaviour end to end with a real signal. | A signal-based integration test (known input, asserted frequency/amplitude at the output). |
| **HARDWARE VERIFIED** | Exercised against real Windows hardware or real third-party software, with the result recorded. | A `hardware`-marked test run locally, the machine and device named, and the output recorded in the docs. |

Never move a feature up a level because it compiles, because a similar feature works,
or because a test with a promising name exists. Never move it down because external
verification is unavailable — record both halves instead, e.g.
`ASIO host: IMPLEMENTED` / `ASIO hardware verification: NOT AVAILABLE ON THIS MACHINE`.
When a verification is missing, name the reason (no device, no licence, no certificate).

## Testing rules

- Real signals over mocks. The house idiom is a `sine()` helper and a
  `dominant_frequency()`/RMS assertion (shared helpers live in `tests/signals.py`) —
  prove a known signal survives a code path at the expected frequency and amplitude.
  `assert engine.running` is not an audio test.
- Test vectors: silence, impulse, 440 Hz and 1 kHz sines, swept sine, white/pink noise,
  multichannel impulses. Check frequency, RMS, peak, polarity, channel mapping, latency,
  clipping, NaN/inf and buffer corruption as appropriate.
- Categories:
  - **unit** — no hardware, no native DLL required.
  - **native** — needs the built `tonesphere_native.dll`; runs the graph offline through
    the ABI's `process_block` entry point, so it needs no audio device and runs in CI.
  - **hardware** — `@pytest.mark.hardware`; needs a real device, driver or OS feature.
    Excluded from CI, expected to be run locally before trusting an audio-path change.
- If you add a backend, an effect or a plugin path, add a test that asserts a known
  signal comes out the other side at the expected amplitude. This has caught real bugs
  (a filter that diverged to infinity, a fan-out that silently dropped a destination, a
  limiter's attack time wrong by a factor of 256) that nothing else would have.
- Never delete, skip, weaken or mock out a failing test to make a suite pass. Fix the
  code, or record the failure honestly.

## Dependency rules

- Every runtime dependency must answer: why is it needed, and could the standard
  library, existing code or the native layer do it instead? The answers live in
  `docs/DEPENDENCIES.md`; update them when you add or remove one.
- Build/packaging-only tools go in dependency groups, not runtime dependencies.
- The runtime must not require .NET, WinUI, WPF, WinForms, Electron, JUCE, a game engine
  or a heavyweight framework. PySide6 stays as the UI.
- Do not add a Python package to do what a few lines of ctypes or native code does.

## Native build rules

- Toolchain: the Microsoft **EWDK** (self-contained MSVC + Windows SDK + WDK), with
  CMake and Ninja from the `native` dependency group. Do not install the Visual Studio
  IDE or unrelated workloads. If something else is genuinely needed, install the
  smallest supported piece and record exactly what and why in
  `docs/BUILDING_WINDOWS.md`.
- `uv run python scripts/build_native.py` builds everything native; the DLLs land where
  `tonesphere/native` and PyInstaller find them.
- The native layer stays small and focused: callback, ASIO, VST3, Windows audio, virtual
  device transport, interop. ToneSphere does not become a C++ desktop application.
- SDKs are fetched by `scripts/fetch_sdks.py` into the git-ignored `sdks/` directory,
  pinned by tag or checksum. **SDK files are never committed.** `sdks/README.md` says
  where each comes from and under what licence.

## Driver development rules

- Kernel-mode code is high-risk code: keep it small, deterministic, heavily documented,
  and free of anything that can live in user mode. No routing, mixing or DSP in the
  driver — it exposes endpoints and moves samples.
- **Driver installs are tested only inside a Hyper-V VM** with test-signing enabled. The
  development host is never put into test-signing mode.
- A compiled `.sys` is not a virtual device. The driver is done only when it installs,
  Windows enumerates the endpoints, another application opens them, audio crosses the
  boundary (asserted by a signal test), and uninstall removes it cleanly.
- Never claim production distribution. Production needs attestation signing through
  Partner Center, which needs an EV certificate; until that exists, the status is
  `production-signed deployment: not available`. Never ask users to disable Secure Boot
  or enable test-signing.

## Licensing rules

- ToneSphere's own source is **MIT** (`LICENSE`).
- `native/asio/` is **GPLv3**, because the Steinberg ASIO SDK is used under its GPLv3
  option. It builds to a separate DLL. Any binary distribution that includes it is
  distributed under GPLv3 as a whole, with source available — the README and legal docs
  must say so.
- The **VST3 SDK (3.8+) is MIT**; keep its notice in distributions.
- `driver/windows_virtual_audio/` is derived from Microsoft's Windows-driver-samples
  (**MS-PL**); it keeps that licence in its own directory and is a separate binary.
- Check the licence before copying any third-party code, and record it next to the code.

## Code style

- No comments except ones explaining a non-obvious *why* (a hidden constraint, a
  workaround, a subtle invariant) — never comments describing *what* the code does. The
  existing comments are almost entirely of the "why" kind; match that density.
- Don't add abstractions, fallbacks, feature flags or error handling for scenarios that
  can't happen. Don't design for hypothetical future requirements.
- Prefer editing existing modules over creating new ones, unless the work genuinely
  introduces a new concern — a new subsystem gets its own file or directory, following
  the existing module boundaries.
- UI strings go through `tonesphere.i18n.tr()` with keys in both `locale/en.json` and
  `locale/hi.json` (`tests/test_i18n.py` enforces it).

## Commands

- `uv sync --all-groups` — install dependencies (including the `native` build tools).
- `uv run python scripts/fetch_sdks.py` — fetch the pinned VST3 and ASIO SDKs into `sdks/`.
- `uv run python scripts/build_native.py` — build the native DLLs (needs the EWDK).
- `uv run pytest -m "not hardware"` — the CI-safe suite.
- `uv run pytest -m hardware` — tests needing a real device; run locally before trusting
  an audio-path change.
- `uv run ruff check .` — lint (120-char lines; rules `E, F, W, B, UP, I`).
- `uv run python main.py server|gui|cli|test` — the four entry points. `test` is a real
  diagnostic (opens a stream, plays a tone, reports what it measured), not a placeholder.

## Architecture map

What exists today versus what is being built is in `docs/IMPLEMENTATION_STATUS.md`;
check it before assuming something works, and update it when you change a status.

- `tonesphere/core` — the control-plane model: `AudioEngine` (routing matrix, buses,
  presets, network wiring), `engine_factory.py` (`UnifiedAudioEngine`), `presets.py`.
- `tonesphere/engine` — `graph.py` (the immutable routing graph and feedback check),
  `host.py` (the PortAudio host: Python callbacks, **not real-time safe**, kept for
  Linux/macOS), `dsp.py`/`effects.py` (reference implementations of the mixer and DSP
  formulas), `devices.py`, `wasapi_com.py`/`process_capture.py`/`app_capture.py`
  (Windows COM via ctypes).
- `tonesphere/native` — ctypes bindings to the native DLLs.
- `tonesphere/plugins` — the Python view of VST3 plugins (info, parameters, state, scan
  cache).
- `tonesphere/network` — UDP/TCP audio streaming; its threads only touch rings, never
  the callback.
- `tonesphere/api`, `tonesphere/cli`, `tonesphere/ui` — REST/WebSocket, terminal, Qt.
- `native/` — `include/` (the C ABI), `engine/` (plan executor, mixer, meters, rings),
  `windows_audio/` (WASAPI), `asio/` (GPLv3), `vst3/`, `test_plugin/` (a deterministic
  MIT VST3 used by the tests), and `coreaudio-plugin/` (macOS AudioServerPlugIn, proven
  only by the `build-macos-plugin` CI job — don't upgrade "round-trips audio in CI" into
  "works on macOS").
- `driver/windows_virtual_audio/` — the Windows virtual audio driver (MS-PL-derived).
- `sdks/` — fetched SDKs, git-ignored except `README.md`.
- `benchmarks/` — measured performance runs.

`docs/VIRTUAL_AUDIO_DRIVER.md` documents what an OS-visible device needs on each
platform and the signing situation; read it before working on the driver.
