# Agent instructions for Tone Sphere

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

## Commands

- `uv sync --all-groups` — install dependencies.
- `uv run pytest -m "not hardware"` — the CI-safe test suite (no real audio device needed).
- `uv run pytest -m hardware` — tests needing a real audio device; skipped on CI, run these
  locally before trusting a change to the audio path.
- `uv run ruff check .` — lint (120-char lines; rules `E, F, W, B, UP, I`; see `pyproject.toml`
  for the reasoning behind the line length and per-file ignores).
- `uv run python main.py server|gui|cli|test` — the four entry points. `test` is a real
  diagnostic (opens a stream, plays a tone, measures latency/xruns) — not a placeholder.

## Architecture map

- `tonesphere/core` — the control-plane model: `AudioEngine` (routing matrix, virtual
  buses, presets, network wiring), `engine_factory.py` (the public-facing
  `UnifiedAudioEngine`), `presets.py` (YAML save/recall, partial-recall-safe).
- `tonesphere/engine` — the real-time audio path: `host.py` (PortAudio stream lifecycle,
  the routing graph's actual execution), `dsp.py`/`effects.py` (biquad EQ, compressor,
  delay, `DriftResampler`), `ringbuffer.py` (lock-free fan-out), `graph.py` (the
  device/bus node model), `devices.py` (real PortAudio enumeration), `app_capture.py`
  (real Windows Audio Session API session detection).
- `tonesphere/network` — `audio_router.py`: TCP audio streaming, deliberately not
  realtime (see its own docstring).
- `tonesphere/api` — FastAPI REST + WebSocket server.
- `tonesphere/cli` — interactive terminal interface.
- `tonesphere/ui` — PySide6/Qt desktop interface.
- `native/coreaudio-plugin` — the one piece of C in the repository: a macOS
  AudioServerPlugIn publishing a "ToneSphere Audio" loopback device. Built and proven only
  by the `build-macos-plugin` CI job; it was written on Windows and never compiled by its
  author, and its own README is explicit about what that does and does not establish.
  Don't upgrade "round-trips audio in CI" into "works on macOS".

`README.md`'s Roadmap table is the source of truth for what phase/feature is actually done
versus in progress — check it before assuming something works, and update it when you
finish something that changes its status.

## Code style

- No comments except ones explaining a non-obvious *why* (a hidden constraint, a workaround,
  a subtle invariant) — never comments describing *what* the code does. This codebase's
  existing comments are almost entirely of the "why" kind; match that density, don't add more.
- Don't add abstractions, fallbacks, feature flags, or error handling for scenarios that
  can't happen. Don't design for hypothetical future requirements.
- Prefer editing existing modules over creating new ones, unless the work genuinely
  introduces a new concern (a new subsystem gets its own file, following the existing
  `engine/`, `network/` module boundaries).

## Testing convention

- Real signals over mocks. The house idiom (see `tests/test_engine_audio.py`) is a
  `sine()` helper and a `dominant_frequency()`/RMS assertion — prove a known signal
  survives a code path at the expected frequency/amplitude, don't just check a return value.
- `@pytest.mark.hardware` (defined in `pyproject.toml`) marks anything needing a real audio
  device or OS feature; excluded from CI (`-m "not hardware"`) but expected to be run
  locally, on real hardware, before trusting a change.
- If you add a backend or an effect, add a test that asserts a known signal comes out the
  other side at the expected amplitude — this has caught real bugs (a filter that diverged
  to infinity, a fan-out that silently dropped a destination, a limiter's attack time wrong
  by a factor of 256) that nothing else would have.

## Where the hard constraints live

`docs/VIRTUAL_AUDIO_DRIVER.md` documents exactly what a real OS-visible virtual audio device
needs on each platform (Windows, Linux, macOS) and why the Windows kernel-mode driver route
is deliberately not attempted here — it needs a paid EV code-signing certificate and weeks
of specialized kernel-driver work. Read it before proposing more work in that area, and
don't claim that route is done or close to done.
