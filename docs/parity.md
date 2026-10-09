# ASIO Link Pro → NexTone feature parity

This list was compiled from the ODeus *ASIO Link Technology* manual (v2.4.x) and from the installer's script and INF, without disassembling any binaries. Every original capability is mapped to its NexTone equivalent.

| ASIO Link Pro | What it does | NexTone equivalent | Bin / place in UI |
|---|---|---|---|
| ASIOVADPRO Speakers 01–04 (or 16 stereo) | WDM playback endpoints; whatever Windows plays there becomes an input | **Virtual Outputs** (NexTone VAD render endpoints, count/channels configurable, default 4 × stereo + 1 × 7.1) | Virtual Outputs |
| ASIOVADPRO Mix 01–04 | WDM recording endpoints fed from the matrix | **Virtual Inputs** (NexTone VAD capture endpoints) | Virtual Inputs |
| ASIO DRIVER IN | Real ASIO interface inputs | Physical Inputs (ASIO host or WASAPI) | Physical Inputs |
| ASIO DRIVER OUT MIX | Pre-mix to the real ASIO outputs | Physical Outputs (each output is a destination with its own mix) | Physical Outputs |
| ASIO HOST IN MIX | Pre-mix feeding the DAW's ASIO inputs | **NexTone ASIO** client inputs, with one per connected DAW | Application Inputs |
| Host ASIO outputs (always mixed into DRIVER OUT) | DAW playback | NexTone ASIO client outputs, routed anywhere and not hard-wired | Application Outputs |
| NETWORK IN / OUT (16 ch, ×2 racks), latency wheel, jitter readout, auto-synchronise (adaptive resampling) | Proprietary UDP audio on port 5050 | **VBAN** in/out streams (unlimited), with jitter buffer, latency readout and adaptive resampling | Sources / Destinations ("Network" chips inside the Virtual bins) |
| LOOPER IN / OUT racks | One-buffer internal loopback | **Mixes/Buses**: any bus can be both a destination and a source | Mixes panel |
| Drag-drop channel connections, 1-1 quick map, X clear | Routing UX | Drag a card onto a card (auto 1-1 map), the "Map 1-1" and "Clear" actions, and an Advanced matrix | Simple + Advanced views |
| Mapping window: crosspoint gain −∞ to +6.02 dB, down/up-mix | Per-crosspoint gain | Matrix crosspoint gain (−∞ to +12 dB), with channel-mapping presets for mono↔stereo and 5.1/7.1 → stereo | Advanced matrix |
| Profiles 1–8, Load/Save | Routing presets | **Scenes** (unlimited), hotkeys, auto-switch rules | Top bar |
| Record to file (8 ch FLAC) | Recording | Multitrack WAV/FLAC of any bus or source, plus a replay buffer | Record panel |
| Multi-client mode (up to 25 instances), "Route to ASIO IN" | Share one ASIO driver among apps | Built in: the engine owns the interface, so any number of NexTone ASIO clients can connect and each is routable | Application bins |
| Pick driver / ASIO CPanel / R (reset) | Hosted-driver control | Device settings: choose the ASIO driver, open its panel, restart the device | Device card menu |
| WinMM CPanel | Open the Windows sound panel | "Windows sound settings" link | Settings |
| Show performance / max values | DSP timing | Per-node CPU, callback time, xrun counters, and per-path latency | Status bar + Inspector |
| Show audio flow | Draw connection lines | Connection highlighting in the Simple view; the Advanced matrix shows everything | — |
| Latency wheel per WDM device ("stream 22 ms") | Virtual device latency / safety offset | Per-endpoint safety-offset slider and live latency readout | Device inspector |
| Use strict ASIO clock / DPC smoothing | VAD clock source | Clock master selection, with adaptive drift compensation on all others | Settings → Clock |
| Auto-set wave format, auto-set default device, allow 32-bit PCM | VAD format management | Engine pushes its format to the VAD; optional "make NexTone default"; float32/24/16 supported | Settings |
| Tool: sample rate, buffer size, channels in/out, Init/Start/Stop ASIO, test tone (440 Hz), Send IN to OUT, WDM test tone (880 Hz) | Configuration + test host | Engine settings, tone generator on any destination, and latency probe | Settings / Tools |
| Start with Windows, start minimised | Autostart | MSIX startupTask or tray autostart, starting minimised | Settings |
| Enable second NETWORK IN/OUT, LOOPER racks | Optional racks | Unlimited network streams and buses | — |
| ASIO Link Pro Null Driver (no ASIO present) | Network-only fallback | Engine runs on WASAPI or an internal clock with no ASIO required | — |
| Licensing / product key | Commercial | None: GPLv3, free | — |

## Beyond the original (NexTone additions)
- **Application bins**: per-app capture (isolate/copy) and per-app microphone assignment, which ASIO Link Pro could not do.
- **Physical WASAPI devices** of any kind used at the same time with drift compensation; the original only used a single ASIO driver.
- **Engine independent of DAWs**: routing runs at login with no host app needed.
- Built-in DSP (gate, RNNoise, EQ, compressor, limiter, ducking), VST3 inserts, LUFS meters.
- Global hotkeys, MIDI/OSC control, CLI, and a modern UI with light/dark themes and accessibility support.
- Windows 11-compatible driver model, with the driver tested inside Hyper-V VMs.
