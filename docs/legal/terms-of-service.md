---
title: Terms of Service
layout: default
permalink: /legal/terms-of-service/
version: "1.0"
effective_date: "2026-09-09"
publisher: Neural Nexus Studios
description: What the ToneSphere application provides and does not provide, how it is distributed, updated and supported, and the terms attaching to its optional network streaming feature.
---

# Terms of Service

- **Application:** ToneSphere
- **Publisher:** Neural Nexus Studios, an individual developer, Kolkata, West Bengal, India
- **Document version:** 1.0
- **Effective date:** 9 September 2026

## 1. What this document covers

This document describes **what ToneSphere offers you as a working application** — the
capabilities it provides, the ones it does not, how it reaches you, how it is updated, and
what support and availability you can expect.

It is the service-facing half of the agreement. The other half, the
[Terms and Conditions](terms-and-conditions.md), is the agreement governing your use of the
software itself: the grant of use, acceptable use, third-party plugins, warranties,
liability and governing law. Those subjects are not repeated here. The
[Privacy Policy](privacy-policy.md) covers information handling.

By using ToneSphere you accept this document together with the Terms and Conditions.

## 2. What ToneSphere provides

ToneSphere is a local desktop application for routing, mixing and processing audio on your
own computer. Provided you have working audio hardware and drivers, it offers:

- **Routing and mixing between the audio devices on your machine**, with one audio stream
  per device and mixing performed inside the driver callback. A device used for both input
  and output is opened as a single duplex stream, so that its input and output share one
  clock.
- **Mixer controls** — constant-power pan, polarity inversion, per-channel trim, mute and
  solo, channel swap, smoothed gain changes, and a limiter on each output.
- **Built-in effects** — biquad equalisation (peaking and shelving filters, high-pass and
  low-pass), a compressor with attack, release, knee and makeup gain, and a delay with
  feedback.
- **Hosting of VST3 and AU plugins that you already own**, loaded onto a channel and
  monitored through. Plugin latency is added to the reported round-trip figure rather than
  hidden from it.
- **A patchbay** in which routes are made by dragging a port to a port, and feedback loops
  are refused rather than created.
- **Presets** saved as plain YAML and keyed on device names rather than device indices, so
  that they survive hardware being plugged in or removed. A preset saved with an interface
  attached will load without it and report what was missing.
- **Latency reporting that distinguishes measured from nominal.** The application reports
  the round-trip latency it actually measured next to the figure that is only buffer
  arithmetic, and reports a measurement it did not take as unavailable rather than as zero.
- **Multiple audio backends**, selected automatically or by you: WASAPI (shared and
  exclusive), WDM-KS, DirectSound and MME on Windows; ALSA and JACK on Linux; CoreAudio on
  macOS. Exclusive mode is attempted first and falls back to shared mode per device when it
  is refused, and the application tells you which it obtained.
- **Per-application audio capture on Windows**, capturing one process's output through the
  operating system's process-loopback interface, on Windows 10 build 20348 and later. No
  driver and no virtual cable is required for this.
- **System-wide loopback capture** of what is playing on an output device, and detection of
  which applications are currently producing sound.
- **A Linux audio sink that other applications can select**, created through PulseAudio or
  PipeWire and bridged into the audio engine as an ordinary output.
- **A macOS CoreAudio plug-in publishing a loopback device**, on the terms set out in
  section 4 below — which are narrower than the others in this list, and deliberately so.
- **A local REST API with a WebSocket statistics feed, and an interactive command-line
  interface**, in addition to the desktop interface.
- **An optional network streaming feature**, described in section 5.

The project's `README.md` and its roadmap table are the authoritative statement of which
features are complete and which are in progress at any given version. Where this document
and that table disagree, the table is newer and the table is right.

## 3. What ToneSphere does not provide

This section exists so that you do not have to discover these by trying them.

- **No accounts, no cloud, no hosted service.** There is no sign-in, no online workspace, no
  server-side storage, no cross-device sync, and no online licence check. ToneSphere runs
  entirely on your machine.
- **No virtual audio device that other applications can select, on Windows.** The
  application's buses are summing points inside its own process; nothing outside ToneSphere
  can see them. A selectable Windows device requires a signed kernel-mode driver, which this
  project does not attempt and does not claim. The
  [virtual audio driver notes](../VIRTUAL_AUDIO_DRIVER.md) set out exactly what that route
  would require.
- **No ASIO** in the builds we distribute. The redistributable PortAudio build does not
  include it, because Steinberg's SDK cannot be redistributed. ASIO appears automatically if
  you supply your own PortAudio built against that SDK.
- **No recording to file.** ToneSphere routes and monitors audio; it does not capture it to
  disk. If you want to record, record in your DAW or recorder of choice, downstream of
  ToneSphere.
- **No compressed network audio.** The network transport carries uncompressed PCM. A codec
  identifier is reserved for Opus, and both encoding and decoding refuse it rather than
  quietly substituting PCM.
- **No sending over the TCP transport.** The TCP path receives only; its send side refuses
  with a message saying so. UDP is the direction that is wired.
- **No adaptive jitter buffering.** The network jitter buffer's target latency is a single
  fixed setting that you can change, not a value estimated from measured jitter.
- **No guaranteed latency, dropout-free operation, or plugin compatibility.** These depend
  on your hardware, drivers, buffer size, backend and plugins. See section 8 of the
  [Terms and Conditions](terms-and-conditions.md).

## 4. Platform features and their evidence

ToneSphere is offered for Windows, Linux and macOS, and the platform-specific features
differ in how well they are established. We would rather label that difference than average
it away:

| Feature | Platform | Status |
|---|---|---|
| Audio routing, mixing, effects, plugin hosting | Windows, Linux, macOS | Working and covered by automated tests |
| Per-application capture | Windows 10 build 20348 and later | Working, proven by a test that captures a known tone and measures it |
| Selectable virtual sink | Linux, with PulseAudio or PipeWire running | Working |
| Selectable virtual device | Windows | Not provided — see section 3 |
| "ToneSphere Audio" loopback device | macOS | **Proven in continuous integration only.** The plug-in builds, installs and round-trips a measured tone on a macOS CI runner. Its behaviour in day-to-day use — appearing in System Settings and in other applications' device pickers, surviving sleep and wake, and installing past Gatekeeper on an end user's machine — is unverified. Treat it as untested in real use until the project says otherwise. |

Features in that last row are provided on that basis, and no representation is made that
they work outside the conditions described.

## 5. The optional network streaming feature

ToneSphere can send audio to, and receive audio from, another machine over a network. This
feature is part of the application, not a service we operate, and these terms apply to it:

- **It is off by default and entirely user-initiated.** Nothing is sent until you enable the
  feature and enter the address and port of the peer you want to reach.
- **We operate no server, relay or intermediary.** The connection is between your machine
  and the peer you nominate. Your audio does not pass through us, is not stored by us, and
  cannot be observed by us.
- **The transport is unencrypted and unauthenticated.** Audio is carried as PCM in UDP
  datagrams with a binary header, sequence numbers and a jitter buffer that reorders and
  paces playout, alongside a TCP receive path for bulk transfer. There is no cipher and no
  verification of the peer. Anyone who can observe the network path could listen to the
  audio, and anyone who can reach the port you are listening on could send audio into it.
- **Use it on networks you trust.** It is designed for monitoring across machines you
  control, on a network you control. It is not a hardened remote-collaboration service, and
  it should not be exposed to the public internet.
- **You are responsible for what you send.** Sending audio to a peer is a transmission you
  are making. Section 4 of the [Terms and Conditions](terms-and-conditions.md) applies.

The local REST API and WebSocket feed are subject to the same caution: they bind to
`127.0.0.1` by default and have no authentication, so changing that bind address exposes
control of your audio engine to anyone who can reach the port.

## 6. Availability

There is nothing to be up or down. ToneSphere provides no hosted service, so we make no
uptime, availability or continuity commitment, and none is needed: an installed copy keeps
working whether or not we are reachable, whether or not the project's repository exists, and
whether or not development continues.

The only network activity is what you configure yourself under section 5, and its
availability depends on your network and your peer, not on us.

## 7. Support

Support is provided on a best-effort, no-commitment basis through the project's issue
tracker:

[https://github.com/AvishakeAdhikary/tone-sphere/issues](https://github.com/AvishakeAdhikary/tone-sphere/issues)

- There is no service-level agreement, no guaranteed response time, and no undertaking that
  a given issue will be investigated, reproduced or fixed.
- There is no private or paid support channel, and no support channel operated by us other
  than the issue tracker. Sponsorship does not create one — see section 10.
- Issues are public. Do not include confidential information, credentials or licence keys in
  one. Diagnostic output from `main.py test` and your ToneSphere logs is the useful thing to
  include, and both are local files you can review before posting.

## 8. Updates and versions

- Releases are published to GitHub Releases when a version tag is pushed, and are built and
  smoke-tested on Windows, Linux and macOS before publication. The current downloads are at
  [https://github.com/AvishakeAdhikary/tone-sphere/releases/latest](https://github.com/AvishakeAdhikary/tone-sphere/releases/latest).
- A copy obtained from the Microsoft Store, once ToneSphere is available there, is updated
  through the Store according to your Windows update settings. A copy downloaded from GitHub
  Releases does not update itself, and the application performs no update check of its own.
- We may add, change, deprecate or remove features between versions, and are under no
  obligation to maintain any particular feature indefinitely. Changes that affect what the
  application claims to do are reflected in the `README.md` roadmap.
- Configuration files and presets are plain YAML and are read tolerantly, but compatibility
  of a preset across versions is not guaranteed. Keep a copy of a preset that matters to you.
- We may discontinue distribution or development at any time. If that happens, copies
  already installed continue to work under the [Terms and Conditions](terms-and-conditions.md).

## 9. Distribution through the Microsoft Store and GitHub

ToneSphere is distributed for Windows, Linux and macOS through **GitHub Releases**, and is
being prepared for distribution for Windows through the **Microsoft Store**. The terms in
this section apply to a copy acquired from the Store once it is available there.

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

## 10. Sponsorship and donations

ToneSphere is offered at no charge, and every feature it has is available without paying
for it. No function is withheld, delayed, time-limited, nagged for, or gated behind a
payment or a donation prompt.

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
