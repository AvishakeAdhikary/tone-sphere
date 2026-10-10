# NexTone — build status

_Last updated: 2026-10-10 (overnight autonomous build)._

## What works (verified)

| Area | Status | Evidence |
|---|---|---|
| Real-time core (lock-free rings, varispeed resampler, drift-compensated adaptive bridges) | ✅ | `engine/tests/core_tests.cpp`: ring SPSC stress, 44.1→48 kHz resampling better than −60 dB, ±300 ppm clock lock within ±15 ppm over 10 simulated minutes with 0 dropouts |
| Engine graph, mixes/buses, crosspoint routing, gain/mute/solo/pan/delay/polarity, meters | ✅ | `engine/tests/engine_tests.cpp` |
| Built-in effects: noise suppression (RNNoise), gate, EQ, compressor, limiter, auto-duck, delay | ✅ | unit tests (limiter ceiling, gate, 4:1 compression = −15 dB) |
| Windows audio devices (WASAPI shared/exclusive, IAudioClient3 low latency, hot-plug) | ✅ | `tests/integration/host-audio.mjs` 10/10 |
| Per-app capture (process loopback, "copy") and per-app redirection ("isolate", app microphones) | ✅ capture verified; redirection implemented | host-audio test: a playing app appears automatically and its audio reaches a mix |
| **NexTone virtual audio driver** (4 outputs + 4 mics, test-signed) | ✅ in the Hyper-V VM | `tests/integration/vad-audio.mjs` 14/14 inside the VM: app→Output→NexTone→Mic→app, 0 clicks/gaps/dropouts |
| **NexTone ASIO driver** (x64 + x86, multi-client) | ✅ | `tests/integration/asio-clients.mjs` 11/11 for 64- and 32-bit hosts: two DAWs at once, exact timing, float32 |
| Engine hosting a real ASIO interface (shared between DAWs) | ✅ | `tests/integration/asio-host.mjs` 8/8 with FlexASIO looped through VB-CABLE |
| Network audio (VBAN, VoiceMeeter-compatible) | ✅ | codec unit tests (spec header bytes) + `tests/integration/vban.mjs` 4/4 |
| UI (Electron + React): six bins, mixes, drag-to-route, send-to menus, matrix, inspector, effects, scenes, settings, dark mode | ✅ | `ui/tests/e2e/ui.spec.ts` 7/7 (mock engine) + screenshots against the real engine |
| CLI | ✅ | `tools/nextone-ctl.mjs` |

## How to run

```powershell
.\scripts\build.ps1 -Test            # engine, ASIO driver, tools + unit tests (x64)
.\scripts\build.ps1 -Arch x86        # 32-bit ASIO driver
.\drivers\win\vad\build.ps1          # kernel driver package (test-signed)
cd ui; npm run build; npx electron . # UI (starts the engine automatically when packaged)
```

## Pending / known limitations
See the bottom of this file (updated as work continues).

- The virtual audio driver is **test-signed**: it only loads on PCs in Windows test-signing mode
  (verified inside the Hyper-V test VM, never installed on the host). Public distribution requires
  Microsoft attestation signing (EV certificate). Without it, NexTone uses Compatibility mode
  (VB-CABLE etc.) for virtual devices.
