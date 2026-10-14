---
title: Terms of Service
layout: default
permalink: /legal/terms-of-service/
version: "2.0"
effective_date: "2026-10-14"
publisher: Neural Nexus Studios
description: What the ToneSphere application provides and does not provide, how it is distributed, updated and supported, and the terms attaching to its optional network features.
---

# Terms of Service

- **Application:** ToneSphere
- **Publisher:** Neural Nexus Studios, an individual developer, Kolkata, West Bengal, India
- **Document version:** 2.0
- **Effective date:** 14 October 2026

## 1. What this document covers

This document describes **what ToneSphere offers you as a working application**: the
capabilities it provides, the ones it does not, how it reaches you, how it is updated, and
what support and availability you can expect.

It is the service-facing half of the agreement. The other half, the
[Terms and Conditions](terms-and-conditions.md), is the agreement governing your use of the
software itself: the grant of use, acceptable use, third-party plugins, warranties,
liability and governing law. Those subjects are not repeated here. The
[Privacy Policy](privacy-policy.md) covers information handling.

By using ToneSphere you accept this document together with the Terms and Conditions.

Version 1.0 of ToneSphere replaced the earlier Python implementation (versions up to 0.2.1)
with a new engine and interface. This version of the document describes the new application.

## 2. What ToneSphere provides

ToneSphere is a local desktop application for routing, mixing and processing audio on your
own computer, for Windows 10/11 and Linux (PipeWire). Provided you have working audio
hardware and drivers, it offers:

- **Routing between everything that makes or takes sound**: physical inputs and outputs,
  virtual inputs and outputs, and individual applications' sound and microphones, plus mixes
  that combine several sources. One device's clock drives the engine; every other device is
  kept in step by adaptive drift compensation.
- **Mixer controls** on every input, output, mix and connection: volume, mute, solo, pan,
  polarity, delay and a channel matrix.
- **Built-in effects**: noise suppression (RNNoise), noise gate, equaliser, compressor,
  limiter, auto-duck and delay.
- **VST3 plugins you already own**, effects and instruments, scanned in a separate process
  so that a broken plugin cannot take the application down while scanning; with their own
  windows, saved state, an on-screen keyboard and MIDI input for instruments.
- **The ToneSphere ASIO driver** on Windows, so that several DAWs and Guitar Rig can play
  into and record from ToneSphere at once, 64- and 32-bit; and the engine can use any
  installed ASIO driver for your interface (verified with FlexASIO and ASIO4ALL).
- **Per-application capture and redirection on Windows** through the operating system's own
  audio interfaces.
- **Recording** of any inputs, outputs and mixes to WAV files, and an instant-replay buffer.
- **Scenes and presets**: snapshots of your routing, and shareable JSON files that find
  devices by name.
- **Multi-core processing** that spreads heavy effect chains over several processor cores
  when it helps.
- **Control** through global hotkeys, OSC, MIDI controllers, a command-line tool and a local
  HTTP/WebSocket API for scripts.
- **Diagnostics**: round-trip latency measurement, per-device latency, drift and dropout
  statistics, and crash reports kept on your own computer.
- **Optional network features**, described in section 5.

The project's `docs/STATUS.md` is the authoritative statement of which features are verified,
with the evidence, and which are not yet. Where this document and that file disagree, the
file is newer and the file is right.

## 3. What ToneSphere does not provide

This section exists so that you do not have to discover these by trying them.

- **No accounts, no cloud, no hosted service.** There is no sign-in, no online workspace, no
  server-side storage, no cross-device sync, and no online licence check. ToneSphere runs
  entirely on your machine.
- **No signed ToneSphere virtual audio driver yet.** The project contains its own virtual
  audio driver, verified in a test virtual machine, but Windows only loads it once Microsoft
  has signed it, so the builds we distribute leave it out. Virtual devices come from
  VB-CABLE, VB-Audio's free virtual cable, which ToneSphere can download from vb-audio.com
  and open the installer of when you ask it to (Settings › Devices). VB-CABLE is VB-Audio's
  software under VB-Audio's own licence; it is not part of ToneSphere and we do not
  distribute it.
- **No macOS version**, and no AU plugins.
- **No manufacturer's ASIO driver tested.** ASIO hosting has been verified with FlexASIO and
  ASIO4ALL. **The Microsoft Store package contains no drivers**, because Store applications
  cannot install them.
- **No guaranteed latency, dropout-free operation, or plugin compatibility.** These depend
  on your hardware, drivers, buffer size and plugins. See section 8 of the
  [Terms and Conditions](terms-and-conditions.md).

## 4. Platform features and their evidence

Every feature in `docs/STATUS.md` is listed with the test that proves it, measured on real
devices where that is possible (for example, audio played through VB-CABLE and measured on
the other side). Features that are not verified are listed there as pending, with the
reason. They are provided on exactly those terms, and no representation is made that they
work outside the conditions described.

## 5. The optional network features

ToneSphere can send audio to, and receive audio from, another computer, and can be
controlled from other devices on your network. These features are part of the application,
not a service we operate, and these terms apply to them:

- **They are off by default and entirely user-initiated.** Nothing is sent until you add a
  network stream (VBAN) and enter the address of the computer you want to reach, and nothing
  is accepted from your network until you add a receiving stream or turn on OSC control.
- **We operate no server, relay or intermediary.** The connection is between your machine
  and the peer you nominate. Your audio does not pass through us, is not stored by us, and
  cannot be observed by us.
- **The transport is unencrypted and unauthenticated.** VBAN carries audio as PCM in UDP
  datagrams, compatible with VoiceMeeter; OSC carries commands in UDP datagrams. Anyone who
  can observe the network path could listen to the audio, and anyone who can reach a port
  you have opened could send audio or commands to it.
- **Use them on networks you trust.** They are designed for machines you control, on a
  network you control. They are not a hardened remote-collaboration service, and they should
  not be exposed to the public internet.
- **You are responsible for what you send.** Sending audio to a peer is a transmission you
  are making. Section 4 of the [Terms and Conditions](terms-and-conditions.md) applies.

The application's own window talks to its audio engine over a local connection bound to
`127.0.0.1` and protected by a random token that changes every time the engine starts; it
accepts no connections from other computers, and refuses requests from web pages.

## 6. Availability

There is nothing to be up or down. ToneSphere provides no hosted service, so we make no
uptime, availability or continuity commitment, and none is needed: an installed copy keeps
working whether or not we are reachable, whether or not the project's repository exists, and
whether or not development continues.

The only network activity is what you configure yourself under section 5, and the VB-CABLE
download from vb-audio.com when you ask for it. Their availability depends on your network,
your peer and VB-Audio, not on us.

## 7. Support

Support is provided on a best-effort, no-commitment basis through the project's issue
tracker:

[https://github.com/AvishakeAdhikary/tone-sphere/issues](https://github.com/AvishakeAdhikary/tone-sphere/issues)

- There is no service-level agreement, no guaranteed response time, and no undertaking that
  a given issue will be investigated, reproduced or fixed.
- There is no private or paid support channel, and no support channel operated by us other
  than the issue tracker. Sponsorship does not create one; see section 10.
- Issues are public. Do not include confidential information, credentials or licence keys in
  one. The engine's log (`engine.log`) and any crash report (the `crashes` folder) in
  `%LOCALAPPDATA%\ToneSphere` are the useful things to include, and both are local files you
  can review before posting.

## 8. Updates and versions

- Every change that passes the project's automated tests on the main branch is published as
  a release on GitHub Releases, built and smoke-tested on Windows and Linux before
  publication: installed on a clean machine, launched and checked the way a user would. The
  current downloads are at
  [https://github.com/AvishakeAdhikary/tone-sphere/releases/latest](https://github.com/AvishakeAdhikary/tone-sphere/releases/latest).
- A copy obtained from the Microsoft Store, once ToneSphere is available there, is updated
  through the Store according to your Windows update settings. A copy downloaded from GitHub
  Releases does not update itself, and the application performs no update check of its own.
- We may add, change, deprecate or remove features between versions, and are under no
  obligation to maintain any particular feature indefinitely. Changes that affect what the
  application claims to do are reflected in `docs/STATUS.md`.
- Settings, scenes and presets are plain JSON and are read tolerantly, but compatibility of a
  preset across versions is not guaranteed. Keep a copy of a preset that matters to you.
- We may discontinue distribution or development at any time. If that happens, copies
  already installed continue to work under the [Terms and Conditions](terms-and-conditions.md).

## 9. Distribution through the Microsoft Store and GitHub

ToneSphere is distributed for Windows and Linux through **GitHub Releases**, and is being
prepared for distribution for Windows through the **Microsoft Store**. The terms in this
section apply to a copy acquired from the Store once it is available there.

- Acquisition, installation, updates, ratings, refunds and any billing relating to a copy
  obtained from the Microsoft Store are handled by Microsoft under Microsoft's own terms,
  including the Microsoft Store Terms of Sale and the Standard Application License Terms.
  Those terms apply to the acquisition in addition to this document, and where they
  conflict with this document in respect of the acquisition itself, Microsoft's terms
  govern it.
- Microsoft is not a party to this agreement, does not endorse the Application, and has no
  obligation to support it. Support is as described in section 7.
- Downloading from GitHub is subject to GitHub's terms. GitHub is likewise not a party to
  this agreement.
- Information handling by both channels is described in section 6 of the
  [Privacy Policy](privacy-policy.md).

## 10. Price, sponsorship and donations

ToneSphere is currently offered at no charge, and every feature it has is available without
paying for it. No function is withheld, delayed, time-limited, nagged for, or gated behind a
payment or a donation prompt.

Future versions or editions may be offered for a price, through the Microsoft Store or
another channel. That does not change the terms of a copy you already have, and a price is
always shown before anything is charged; section 3 of the
[Terms and Conditions](terms-and-conditions.md) sets this out.

You can support the project's continued development through GitHub Sponsors, Patreon, Ko-fi
or Buy Me a Coffee. If you do:

- The contribution is voluntary and is a contribution to development, not the purchase of a
  product, a licence, a subscription or a support contract.
- It grants no additional rights, features, priority, warranty or service commitment, and
  creates no obligation on us to deliver anything specific.
- Payments are processed by that platform under its own terms, and refunds, cancellations
  and billing questions are handled by that platform, not by us.

## 11. Changes to this document

We may update these Terms of Service, in particular as features move from in progress to
working, or as platform support changes. Every version carries a version number and an
effective date at the top of the document, and the current version is published at the
address where you are reading it. Continued use of the Application after an updated version
takes effect is acceptance of it. Superseded versions remain in the project repository's
history.

## 12. Governing law

These Terms of Service are governed by the laws of **India**, and the courts at **Kolkata,
West Bengal, India** have exclusive jurisdiction, on the terms set out in section 13 of the
[Terms and Conditions](terms-and-conditions.md).

## 13. Contact

[https://github.com/AvishakeAdhikary/tone-sphere/issues](https://github.com/AvishakeAdhikary/tone-sphere/issues)

That is the project's contact channel for everything in this document. Issues are public,
so do not include anything confidential in one.
