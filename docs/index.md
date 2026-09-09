---
title: ToneSphere
layout: default
permalink: /
publisher: Neural Nexus Studios
description: ToneSphere is a local desktop audio router and mixer for Windows, Linux and macOS, published by Neural Nexus Studios.
---

# ToneSphere

**Low-latency audio routing and mixing for Windows, Linux and macOS.**
Published by Neural Nexus Studios, Kolkata, West Bengal, India.

Plug a guitar into an audio interface, run it through the plugins you already own, and hear
it in your headphones — without a chain of virtual cables, a donation timer, or a driver
panel from 2006. That is the whole idea.

## What it does

- Routes and mixes audio between the devices on your own machine, one stream per device,
  mixing inside the driver's callback. A device used in both directions gets a single duplex
  stream, so its input and output cannot drift apart.
- Hosts the **VST3 and AU plugins you already own**, on any channel, and adds their latency
  to the reported round trip instead of hiding it.
- Gives you a mixer and a patchbay: constant-power pan, polarity, trim, mute, solo, smoothed
  gain, a limiter on every output, and cables you make by dragging a port onto a port.
- Ships real effects: biquad EQ, a compressor with actual attack, release, knee and makeup,
  and a delay with feedback.
- Captures **one application's audio on Windows** (Windows 10 build 20348 and later, no
  driver and no virtual cable), and captures a whole output device system-wide.
- Creates **a Linux sink other applications can select**, and publishes a **CoreAudio
  loopback device on macOS** that is proven in continuous integration and unverified in
  day-to-day use — stated that way on purpose.
- Saves presets as plain YAML keyed on device names, so they survive you plugging something
  in, and tells you what was missing when it loads one without your interface attached.
- Reports **measured** latency next to the nominal buffer arithmetic. On one machine, at a
  128-frame buffer on WASAPI exclusive: 5.7 ms measured round trip at roughly 5% DSP load.
  Your hardware will give you different numbers, which is exactly why the number shown is
  the one it measured.

There is one rule in this project, and its test suite enforces it:

> A feature does not exist until a test proves it moves audio. And a measurement you did not
> take is reported as `--`, never as `0`.

The repository's [README](https://github.com/AvishakeAdhikary/tone-sphere#readme) is the
authoritative list of what works, what does not work yet, and why — including the Windows
virtual device that deliberately is not attempted, and the reasoning behind that in the
[virtual audio driver notes](VIRTUAL_AUDIO_DRIVER.md).

## Download

**[Latest release downloads](https://github.com/AvishakeAdhikary/tone-sphere/releases/latest)**
— built and smoke-tested on Windows, Linux and macOS.

On Linux, install PortAudio from your distribution first (`libportaudio2` on Debian and
Ubuntu, `portaudio` on Fedora, `portaudio` on Arch); the Linux Python wheel does not bundle
it. Run `main.py test` before anything else — it opens a stream, plays a tone, measures the
round trip and tells you PASS, WARN or FAIL per capability.

## Sponsor ToneSphere

ToneSphere is free, and every feature in it is available without paying anything. Nothing is
time-limited, nothing nags, and no function waits behind a donation prompt — that experience
is the reason this project exists at all.

If it saved you the price of a routing utility, or a re-purchase of software you already
owned, you can put that toward its development:

[![Sponsor on GitHub](https://img.shields.io/badge/GitHub%20Sponsors-Sponsor-ea4aaa?style=for-the-badge&logo=githubsponsors&logoColor=white)](https://github.com/sponsors/avishakeadhikary)
[![Patreon](https://img.shields.io/badge/Patreon-Become%20a%20patron-f96854?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/avishakeadhikary)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-Buy%20a%20coffee-ff5e5b?style=for-the-badge&logo=kofi&logoColor=white)](https://ko-fi.com/avishakeadhikary)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Support-ffdd00?style=for-the-badge&logo=buymeacoffee&logoColor=black)](https://www.buymeacoffee.com/avishake69)

- **GitHub Sponsors** — [github.com/sponsors/avishakeadhikary](https://github.com/sponsors/avishakeadhikary)
- **Patreon** — [patreon.com/avishakeadhikary](https://www.patreon.com/avishakeadhikary)
- **Ko-fi** — [ko-fi.com/avishakeadhikary](https://ko-fi.com/avishakeadhikary)
- **Buy Me a Coffee** — [buymeacoffee.com/avishake69](https://www.buymeacoffee.com/avishake69)

Sponsorship is voluntary. It buys no feature, no priority and no private support channel —
see section 10 of the [Terms of Service](legal/terms-of-service.md) — it just funds the time
that goes into the next release.

## Legal

- [Terms and Conditions](legal/terms-and-conditions.md) — the agreement governing your use of
  the application: grant of use, acceptable use, third-party plugins, warranties, liability,
  and governing law.
- [Terms of Service](legal/terms-of-service.md) — what the application provides and does not
  provide, availability and support, updates, the optional network feature, and distribution
  through the Microsoft Store.
- [Privacy Policy](legal/privacy-policy.md) — no accounts and no collected information, what
  stays on your own machine, and the one optional feature that sends audio off it.

## Project and contact

- Source, issues and releases: [github.com/AvishakeAdhikary/tone-sphere](https://github.com/AvishakeAdhikary/tone-sphere)
- Questions, bugs and anything about the documents above:
  [open an issue](https://github.com/AvishakeAdhikary/tone-sphere/issues). Issues are public,
  so keep confidential details out of them.
