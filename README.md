# Tone Sphere

![Tone Sphere Banner](./assets/images/ToneSphereBanner.gif)

> **Status: pre-alpha. ToneSphere does not pass audio yet.**
> The routing matrix, mixer state, API and UI exist and work. The part that carries
> samples to and from your audio interface is being built. Don't install this expecting
> to play guitar through it today — see [Where it actually is](#where-it-actually-is).

This is not your usual readme.

I am a guitarist, and I recently bought an audio interface, only to find out that playing or recording guitar requires me to pay again.
Honestly it was all cool and all, but I use Guitar Rig, which already is an expensive piece of software, not to mention the guitar itself is expensive enough, now I have a microphone and an audio interface.
But then comes the audio software bs that comes with it.
I was bored and frustated. I tried both Voicemeeter and Odeus Asio Link Pro.
Voicemeeter ran great, until it asked me to donate again and again, it felt a lot intrusive. I mean, why would I wait for 5 minutes if it is a donationware? Ffs it's donation ware, you either ask nicely or leave, not make people wait.
When I still waited, it became high latency out of nowhere. With VB audio cable of course.
Then I tried odeus asio link pro with asio. And honestly bro, wtf. That stuff was ancient and still no alternative to the date, it was an absolute chad.
But tbh I didn't like the approach, it was too complicated for beginners to grasp, to steep of a curve to just plug your guitar in to your audio interface and start playing.
I've had enough, so being a coder myself, I coded one on my own.

This project is all about that.

## Where it actually is

An earlier version of this README and CLI advertised ASIO, WASAPI, ALSA, PulseAudio, JACK,
PipeWire and CoreAudio support, virtual audio devices, real-time effects and low-latency
processing. None of that was working. The driver classes existed but every one of them
returned a NumPy array of zeros instead of talking to an audio API, and the "virtual audio
devices" were Python queues that no other application could ever see. This section exists so
nobody loses an evening to that again.

### Works today

- Routing matrix: create, remove, volume, mute, solo — as state, and it persists correctly
- Per-device channel controls: volume, mute, solo, pan, phase invert (state)
- Virtual buses **inside the ToneSphere process** — audio genuinely flows between them
- Device/routing CRUD over a FastAPI REST API, plus a WebSocket stats feed
- Tkinter GUI, system tray, interactive routing canvas, CLI
- YAML configuration, rotating structured logs
- `python main.py test` — honest diagnostics that fail loudly instead of printing checkmarks

### Does not work yet

- **Audio to or from real hardware.** Nothing you route reaches your interface or speakers.
- **Virtual devices other applications can select.** The buses above are in-process only.
  A device that shows up in Discord's or your DAW's device list needs a signed kernel-mode
  driver; that is planned, not present.
- **ASIO / WASAPI / ALSA / PulseAudio / JACK / PipeWire / CoreAudio.** No backend is wired up.
- **Effects.** `AudioProcessor` is not in any audio path, and its EQ/reverb are placeholders.
- **Network audio streaming.** The framing is broken and the transport is wrong for realtime.
- **Application audio capture.** The current detector flags nearly every running process.

## Roadmap

| Phase | Goal |
|---|---|
| 0 | Remove the false claims and dead code, add tests and CI ← **you are here** |
| 1 | Real audio I/O: one PortAudio duplex callback, real device enumeration, a 1 kHz tone provably in and out with zero xruns |
| 2 | Real mixer: summing buses, gain smoothing, pan law, solo, peak/RMS metering, measured latency and CPU |
| 3 | Professional GUI in PySide6: proper mixer strips, dB-scaled faders, node-graph routing |
| 4 | VST3 hosting (so Guitar Rig runs *inside* ToneSphere), real DSP, per-application capture |
| 5 | Packaging, signed installer, and the WDM virtual audio driver |

Latency target for Phase 1–2 is 5–10 ms round trip at a 256-frame buffer. ASIO needs
PortAudio built against Steinberg's ASIO SDK, which cannot be redistributed, so ASIO arrives
as a self-built wheel after WASAPI exclusive mode is working.

## Setup

Just run `pip install uv` if you already don't have `uv` installed.
Then run `uv sync`. That will install everything.

Next just run `uv run main.py gui`.

Check what your machine can actually do first:

```
uv run main.py test
```

That prints PASS/WARN/FAIL per capability and exits non-zero if something is broken. On a
machine with no audio backend wired up yet, expect `[FAIL] Physical devices: 0` — that is
the honest current state, not a bug in your setup.

## Contributing

Contributions welcome, especially on Phase 1. One rule, learned the hard way: **a feature
does not exist until a test proves it moves audio.** No more classes that satisfy an
interface and return zeros. If you add a backend, add a test that asserts a known signal
comes out the other side.

Run the tests with `uv run pytest`.

P.S. This is not a rickroll.
I'll work on executables in the future, for now I am busy working in corporate. Inviting others to contribute.
Have fun.
