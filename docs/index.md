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

- Routes and mixes audio between the devices on your own machine. On Windows the audio runs
  on a native real-time engine with WASAPI and ASIO backends; on Linux and macOS, on a
  PortAudio host.
- Hosts **VST3 effect plugins on Windows** — a browser that scans each plugin in a separate
  process, insert chains on any device's input or output, parameters, bypass and the
  plugin's own editor. Proven with the project's own test plugin, with Surge XT, and with
  Native Instruments' Guitar Rig 7.
- Gives you a mixer and a patchbay: pan and balance, polarity, trim, mute, solo, smoothed
  gain, a limiter on every output, per-channel meters, and cables you make by dragging a
  port onto a port.
- Ships real effects: biquad EQ, a compressor with actual attack, release, knee and makeup,
  and a delay with feedback.
- Captures **one application's audio on Windows** (Windows 10 build 20348 and later, no
  driver and no virtual cable).
- Creates **a Linux sink other applications can select**, and publishes a **CoreAudio
  loopback device on macOS** that is proven in continuous integration and unverified in
  day-to-day use — stated that way on purpose.
- Comes back as you left it: the session — routes, mixer, plugins and their settings — is
  saved as you work and restored when you open it. Presets are the same plain YAML, keyed on
  stable device identifiers, and say what was missing when your interface is not attached.
- Keeps latency figures apart: ToneSphere's own buffer arithmetic, what the drivers and
  plugins report, and — on Windows — a round trip it **measures** by playing a sweep and
  timing its return. Until that measurement has been taken on your own output and input,
  at your current settings, it is shown as `--`.

There is one rule in this project, and its test suite enforces it:

> A feature does not exist until a test proves it moves audio. And a measurement you did not
> take is reported as `--`, never as `0`.

The repository's [README](https://github.com/AvishakeAdhikary/tone-sphere#readme) is the
authoritative list of what works, what does not work yet, and why — including the Windows
virtual device, whose driver carries audio between applications in a test VM but is not yet
signed for anyone else's machine, and exactly where it stands in the
[virtual audio driver notes](VIRTUAL_AUDIO_DRIVER.md).

## Download

**[Latest release downloads](https://github.com/AvishakeAdhikary/tone-sphere/releases/latest)**
— a Windows installer (per user, no administrator prompt) or portable zip, a Linux AppImage
and a macOS disk image, each launched by continuous integration the way you would launch it
before it is published. A new release is published on every change that passes.

They are not code-signed yet: on Windows choose **More info → Run anyway** when SmartScreen
asks; on macOS right-click the app → **Open** the first time. Then press **Monitor Input**:
it picks your interface and its first input, in both ears. How that was checked with a
guitar, a commercial amp simulator and the release build is in the
[first-run verification](FIRST_RUN_VERIFICATION.md).

The Windows build on GitHub includes ASIO support built from Steinberg's ASIO SDK under the
GNU GPL version 3, and is distributed under that licence; its complete source, SDKs
included, is published with every release in the `gpl-source` release and linked from the
release notes. The Microsoft Store package leaves ASIO out, and it and the Linux and macOS
builds are MIT-licensed.

## Engineering documentation

How ToneSphere is built, and the evidence for every claim above. Each capability carries one
evidence level, from NOT IMPLEMENTED to HARDWARE VERIFIED, and a figure nobody measured is
shown as `--`.

- [Engineering report](ENGINEERING_REPORT.md) — what the Windows-native rebuild delivered,
  with every figure's source.
- [Implementation status](IMPLEMENTATION_STATUS.md) — the record of what is done, capability
  by capability.
- [First-run verification](FIRST_RUN_VERIFICATION.md) — the release build installed and used
  through its own window, with a guitar plugged in, and the output measured.
- [Architecture](ARCHITECTURE.md) — the Python control plane, the native real-time engine,
  and the C ABI between them.
- [Real-time rules](REALTIME.md) — what the audio thread may and may not do, and the
  measured callback timings.
- [Windows audio](WINDOWS_AUDIO.md), [ASIO](ASIO.md), [VST3](VST3.md) and the
  [virtual audio driver](VIRTUAL_AUDIO_DRIVER.md) — each backend, and exactly what has been
  verified against what.
- [Testing](TESTING.md) — how "it works" is established, with real signals.
- [Building on Windows](BUILDING_WINDOWS.md) and [dependencies](DEPENDENCIES.md) — the
  toolchain, and why each dependency is there.

## Sponsor ToneSphere

ToneSphere is free today, and every feature in it is available without paying anything. Nothing is
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
