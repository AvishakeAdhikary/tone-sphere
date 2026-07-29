# Tone Sphere

![Tone Sphere Banner](./assets/images/ToneSphereBanner.gif)

Low-latency audio routing and mixing for Windows, Linux and macOS.
**5.7 ms measured round trip** at a 128-frame buffer on WASAPI exclusive, ~5% DSP load.

![ToneSphere](./assets/images/screenshot.png)

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

## Setup

```
pip install uv        # if you don't have it
uv sync
uv run main.py gui
```

**Linux only:** the `sounddevice` package is a thin ctypes wrapper — unlike the
Windows/macOS wheels, the Linux wheel does not bundle PortAudio's shared library. Install
it from your distro first, or every audio feature reports `PortAudio library not found`:

```
sudo apt install libportaudio2      # Debian/Ubuntu
sudo dnf install portaudio          # Fedora
sudo pacman -S portaudio            # Arch
```

Check what your machine can do first:

```
uv run main.py test
```

That reports PASS/WARN/FAIL per capability, plays a test tone through your default output,
and exits non-zero if anything is broken. On this machine:

```
[INFO] Active backend:   Windows WASAPI
[PASS] Stream open on Speakers (Realtek(R) Audio)
[PASS] No callback errors
[PASS] No dropouts (xruns)
[INFO] Measured latency: 5.7 ms round trip
[INFO] Nominal latency:  2.7 ms (buffer arithmetic only)
[INFO] DSP load:         5.1%
```

## What works

**Audio path.** One PortAudio stream per device, mixing inside the driver's own callback.
A device used in both directions gets a single duplex stream, so input and output share one
clock and cannot drift — that is the guitar path, and the lowest-latency configuration
available. Cross-device routes pass through lock-free ring buffers with drift correction.

**Backends.** WASAPI (shared and exclusive), WDM-KS, DirectSound, MME on Windows; ALSA and
JACK on Linux; CoreAudio on macOS. Exclusive mode is tried first and falls back to shared
per device when refused, telling you which it got.

Measured on one machine, same hardware, same test:

| Backend | Round trip | vs. nominal |
|---|---|---|
| WASAPI exclusive | **8.3 ms** | 5.3 ms |
| WDM-KS | 17.0 ms | 5.3 ms |
| WASAPI shared | 22.0 ms | 5.3 ms |
| MME | 96.0 ms | 5.3 ms |
| DirectSound | 120.0 ms | 5.3 ms |

That table is why ToneSphere always shows measured latency next to the nominal figure.
Nominal is buffer ÷ sample rate — arithmetic about our own contribution, and off by up to
20× from what you actually hear.

**Mixing.** Constant-power pan, polarity invert, per-channel trim/mute/solo, channel swap,
gain smoothing on everything so nothing clicks, and a limiter on each output so a routing
mistake sounds like a compressed mix rather than a burst of digital noise.

**Effects.** Biquad EQ (peaking, shelves, high/low pass), a compressor with real attack,
release, knee and makeup, and a delay with feedback.

**VST3 / AU plugins.** Load Guitar Rig, Neural DSP or anything else you own directly onto a
channel and monitor through it. Plugin latency is added to the reported round trip.

**Patchbay.** Drag a port to a port to connect. Feedback loops are refused before they
happen. Cables show their gain; muted and broken routes look different.

**Presets.** Patch and mixer state saved as plain YAML, keyed on device names rather than
indices, so they survive plugging something in. Partial recall: a preset saved with an
interface attached loads without it and tells you what was missing.

**Also:** system-wide loopback capture, real audio-session detection (which applications
are actually playing), a REST API with a WebSocket stats feed, and a CLI.

## What does not work yet

- **Virtual devices other applications can select.** The buses are in-process summing
  points; nothing outside ToneSphere can see them. That needs a signed kernel driver —
  [docs/VIRTUAL_AUDIO_DRIVER.md](docs/VIRTUAL_AUDIO_DRIVER.md) covers exactly what and how
  much.
- **Per-application capture.** Windows 10 build 20348+ supports it without a driver, and
  ToneSphere detects that, but the native call is not written yet. Whole-system loopback
  works today.
- **ASIO.** Not in the PyPI PortAudio build — Steinberg's SDK cannot be redistributed. It
  appears automatically if you supply a PortAudio built against it. WASAPI exclusive is
  within a few ms anyway.
- **Network streaming** is TCP, which is the wrong transport for realtime. Fine for moving
  audio between machines, not for monitoring. Realtime needs UDP with a jitter buffer.

## Roadmap

| Phase | | |
|---|---|---|
| 0 | Remove false claims, delete dead code, add tests and CI | done |
| 1 | Real audio I/O: PortAudio callback, lock-free graph, real enumeration | done |
| 2 | Real mixer: pan law, polarity, limiter, drift resampling, metering | done |
| 3 | Qt interface: mixer strips, dB faders, node-graph patchbay | done |
| 4 | VST3 hosting, real DSP, honest app detection, presets | done |
| 5 | Packaging, per-process capture, virtual audio driver | in progress |

## A note on how this was rebuilt

An earlier version of this README advertised ASIO, WASAPI, ALSA, PulseAudio, JACK, PipeWire
and CoreAudio support, virtual audio devices, real-time effects and low-latency processing.
None of it worked. The eight driver classes each returned an array of zeros instead of
talking to an audio API; the "virtual audio devices" were Python queues; routing reported
success into an empty registry; and the UI displayed `CPU: 0% | Latency: 0ms` permanently,
which was not a reading of anything.

So there is one rule here now, and the test suite enforces it:

> **A feature does not exist until a test proves it moves audio.**
> And a measurement you did not take is reported as `--`, never as `0`.

That second half matters more than it sounds. `0.0` renders as a real, healthy-looking
number. Rendering `--` is the difference between a UI that tells you what it knows and one
that tells you what you want to hear.

## Contributing

Contributions welcome, especially on Phase 5. Run the tests with `uv run pytest`
(hardware-dependent tests: `uv run pytest -m hardware`).

If you add a backend or an effect, add a test that asserts a known signal comes out the
other side at the expected amplitude. Several real bugs in this rebuild — a filter that
diverged to infinity, a fan-out that silently dropped one destination, a limiter whose
attack time was wrong by a factor of 256 — were caught by exactly that kind of test and by
nothing else.

P.S. This is not a rickroll.
I'll work on executables in the future, for now I am busy working in corporate. Inviting others to contribute.
Have fun.
