---
title: Dependencies
layout: default
permalink: /dependencies/
description: Every dependency ToneSphere has, why it is there, and what could replace it.
---

# Dependencies

Every dependency answers the same questions: why does ToneSphere need it, could the
standard library, existing code or the native layer do it instead, and is it needed at
runtime or only to build. Update this file whenever `pyproject.toml` changes.

## Runtime (`[project] dependencies`)

| Package | Why it is needed | Could something smaller replace it? | Plan |
|---|---|---|---|
| **numpy** | Audio blocks everywhere in the Python control plane and the legacy PortAudio host; ring buffers; test signal analysis | No — and the native engine's buffers are handed to Python as NumPy views for tests and metering | Keep |
| **sounddevice** | PortAudio (via CFFI) for the audio host on Linux and macOS, and on Windows until the native WASAPI backend replaces it | On Windows, yes: the native backend. On Linux/macOS it is the backend | Keep for Linux/macOS; Windows stops depending on it for audio once M4 lands |
| **pyside6** | The desktop UI (Qt). The largest dependency by far | Not without replacing the UI; `AGENTS.md` rules out WinUI/WPF/Electron/.NET. Unused Qt modules are excluded in `tonesphere.spec` | Keep |
| **fastapi** | The REST/WebSocket control API (`tonesphere/api/server.py`) | The standard library `http.server` could serve REST, but not WebSockets or request validation without reimplementing both | Keep |
| **pydantic** | Request/response models for the API (`tonesphere/api/models.py`); a FastAPI requirement anyway | No, while FastAPI stays | Keep |
| **uvicorn** | ASGI server that runs the API | No, while FastAPI stays | Keep |
| **websockets** | Not imported by ToneSphere directly: uvicorn's WebSocket protocol implementation for `/ws/events` (`tonesphere.spec` hidden import `uvicorn.protocols.websockets.auto`) | uvicorn can use `wsproto` instead; no smaller | Keep |
| **pyyaml** | Presets and the default config are YAML (`core/presets.py`, `utils/config.py`) | JSON from the standard library, at the cost of the presets' human-editable format and every existing preset file | Keep |

## Build and development groups

| Group | Package | Why | Notes |
|---|---|---|---|
| `dev` | pytest | the test suite | |
| `dev` | ruff | lint | |
| `dev` | httpx | FastAPI's `TestClient` backend; never imported directly | |
| `packaging` | pyinstaller | freezes the app for Releases and the MSIX package (`tonesphere.spec`) | Was a runtime dependency; nothing at runtime imports it |
| `packaging` | pillow | generates the Store's logo sizes (`packaging/msix/generate_assets.py`) and checks them in `tests/test_msix_packaging.py` | Was a runtime dependency; the application never imports it |
| `native` | cmake | generates the native build (`scripts/build_native.py`) | From PyPI so no system install is needed |
| `native` | ninja | the build tool CMake drives | From PyPI |

All three groups are in `[tool.uv] default-groups`, so a plain `uv sync` still gives a
checkout that can test, build and package. The split exists so that the runtime list above
is honest, not to add install steps.

## Removed

| Package | Why it was removed |
|---|---|
| requests | Nothing imported it (checked with a repository-wide import search, 2026-09-29) |
| pedalboard | Replaced by the native VST3 host (`docs/VST3.md`). Its plugin chain was unreachable from the UI, API and CLI, never loaded a real plugin in a test, and made the frozen app a GPLv3 binary |

## Native and SDK dependencies

These are not Python packages; they are fetched or installed by the build, never committed.

| Component | Source | Licence | Used for |
|---|---|---|---|
| EWDK (MSVC build tools, Windows SDK, WDK) | Microsoft, official ISO | Microsoft EWDK licence (driver development and supporting components; see `docs/BUILDING_WINDOWS.md`) | Compiling the native DLLs and the driver |
| VST3 SDK 3.8.x | `github.com/steinbergmedia/vst3sdk`, pinned tag, via `scripts/fetch_sdks.py` | MIT | The native VST3 host and the test plugin |
| ASIO SDK 2.3.x | steinberg.net, pinned checksum, via `scripts/fetch_sdks.py` | GPLv3 (Steinberg's open-source option) | `native/asio/` only — see `sdks/README.md` |
| Windows-driver-samples (SimpleAudioSample) | `github.com/microsoft/Windows-driver-samples` | MS-PL | Starting point for `driver/windows_virtual_audio/` |

The native layer must not pull in JUCE, .NET, a GUI toolkit or any framework beyond the
two Steinberg SDKs and the Windows SDK.
