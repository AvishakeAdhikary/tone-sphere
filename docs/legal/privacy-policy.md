---
title: Privacy Policy
layout: default
permalink: /legal/privacy-policy/
version: "1.0"
effective_date: "2026-09-09"
publisher: Neural Nexus Studios
description: ToneSphere collects no user information and has no user accounts. This policy states what stays on your machine, and what the one optional feature that sends audio off it actually does.
---

# Privacy Policy

- **Application:** ToneSphere
- **Publisher:** Neural Nexus Studios, an individual developer, Kolkata, West Bengal, India
- **Policy version:** 1.0
- **Effective date:** 9 September 2026

## 1. Summary

ToneSphere collects no user information.

There are no user accounts, no sign-in, no registration, no telemetry, no analytics, no
advertising identifiers, no crash or usage reporting, and no third-party tracking of any
kind. ToneSphere is a desktop audio router: it takes audio from the devices and
applications on your computer, processes it, and sends it to other devices on the same
computer. That audio is handled in memory, in real time, and is neither stored nor
transmitted by ToneSphere.

There is exactly one feature that sends audio away from your machine, and it is worth
stating plainly rather than leaving it to be discovered: the **optional network streaming**
feature. It is switched off by default, it does nothing until you turn it on and enter the
address of a peer, and when you do, the audio you route to it travels directly to that
peer. Section 4 describes it in full. No other part of ToneSphere makes a network
connection of its own.

## 2. What ToneSphere does with your audio

Audio entering ToneSphere is mixed, filtered and routed inside the running process and
delivered to the outputs you have patched. It exists in memory buffers for as long as it
takes to play it — a matter of milliseconds — and is then overwritten.

ToneSphere contains no recording feature. It writes no audio files. Nothing in the
application saves, uploads, transcribes, analyses or fingerprints the sound passing
through it.

This applies equally to the capture features:

- **Per-application capture on Windows**, which captures the audio of one process that you
  select by name or process id.
- **System-wide loopback capture**, which captures what is currently playing through an
  output device.
- **Audio session detection**, which lists the applications that are currently playing
  sound, so that you can pick one. This reads the operating system's own audio session
  list; it does not read the contents of those applications or anything else about them.

Captured audio is treated exactly like any other input: routed, mixed, and discarded. It
stays on your computer unless you have configured the optional network stream described in
section 4 and routed it there yourself.

## 3. Information stored on your computer

ToneSphere stores its own settings on your machine, in ordinary files that you can read,
edit and delete. Nothing in these files is sent anywhere.

| What | Where | Contains |
|---|---|---|
| Configuration | `config/default_config.yaml`, alongside the application | Sample rate, buffer size, backend choice, port numbers, logging switches |
| Presets | the `presets` folder, alongside the application | Patch and mixer state as plain YAML, keyed on audio device names |
| Log files | the `logs` folder, and only when you enable file logging, which is off by default | Diagnostic lines about streams, devices and errors |

Two of these record names rather than only numbers, and it is fair to say so. Presets
record the **names of your audio devices** — for example "Focusrite Scarlett 2i2" or
"Speakers (Realtek Audio)" — because a preset keyed on a device index breaks the moment
you plug something in. Logs may record device names, the file paths of plugins you loaded,
and error messages. Both are local files under your control. Deleting those folders
removes them, and uninstalling the application leaves nothing about you behind on any
service, because there is no service.

## 4. The one feature that sends audio off your machine

ToneSphere can stream audio over a network to a peer that you nominate. This is off by
default (`network.enabled: false` in the configuration) and stays inert until you enable it
and supply an address.

When you do enable it:

- **You choose the destination.** The audio goes to the host and port you enter, and
  nowhere else.
- **It is a direct connection between your machine and that peer.** Neural Nexus Studios
  operates no server for ToneSphere. There is no relay, no intermediary, no cloud
  component, and no copy of your audio held by anyone. We cannot see this traffic, and
  there is nowhere for us to see it from.
- **Only the audio you route there is sent.** The network destination is an ordinary
  destination in the routing matrix. Whatever you patch into it is sent; nothing else is.
- **The transport applies no encryption and no authentication.** Audio travels as PCM in
  UDP datagrams, or over the TCP receive path, without a cipher and without a check on who
  is at the other end. Anyone able to observe the network path between the two machines
  could listen to that audio, and anyone able to reach your listening port could send audio
  into it. Use this feature on networks you trust, and do not treat it as a private channel
  across the public internet.
- **The peer learns your IP address**, as it must for any direct connection, and you learn
  the peer's. That is a property of connecting two machines, not something ToneSphere adds.

The local REST API and WebSocket interface deserve the same candour. When you run
ToneSphere in server mode it listens on `127.0.0.1` by default, which means only your own
machine can reach it. If you change that host to `0.0.0.0` in order to control ToneSphere
from another computer, then any machine able to reach that port can also change your
routing and read your engine statistics; the interface has no authentication. That is your
decision to make, and the default is the private one.

## 5. Third-party plugins you load

ToneSphere hosts VST3 and AU plugins that you already own, on channels you choose. Those
plugins are software written by other companies, running inside the ToneSphere process at
your instruction.

A plugin can do things ToneSphere does not: check a licence over the internet, contact its
vendor's servers, write files, or collect information about your machine. What a plugin
does is governed by that vendor's own privacy policy and licence terms, not by this one.
Neural Nexus Studios does not supply these plugins, cannot inspect their behaviour, and
receives nothing from them. If you want to know what a plugin transmits, its vendor is the
only authority on that.

## 6. Distribution channels

ToneSphere itself has no analytics, but obtaining a copy of it involves someone else's
service, and those services keep their own records:

- **Microsoft Store.** ToneSphere is being prepared for distribution through the Microsoft
  Store for Windows. Microsoft's own data collection when you browse, acquire, install or
  update an application from the Store is described in the Microsoft Privacy Statement and
  is Microsoft's, not ours. Microsoft provides publishers with aggregated reports —
  acquisition counts, ratings, reliability summaries — that do not identify individual
  users. That aggregate is the entirety of what Neural Nexus Studios receives about
  installations, and it is not personal information.
- **GitHub.** Releases are also published on GitHub. Downloading a file from GitHub, or
  opening an issue there, is processed under the GitHub Privacy Statement. Publicly filed
  issues are visible to everyone, including the GitHub username you write them under.
  Neural Nexus Studios sees the public issue and aggregate download counts, nothing more.

Neither channel reports anything to ToneSphere, and the application does not contact either
of them while running. It performs no update check and makes no outbound request of its own.

## 7. Children

ToneSphere collects no information from anyone, of any age, so it holds no information about
children. It is a general-purpose audio utility and is not directed at children.

## 8. Your rights over your information

Data-protection rights — access, correction, export, erasure, objection, withdrawal of
consent — are exercised against an organisation that holds your personal data. Neural
Nexus Studios holds none, because ToneSphere collects none. There is no account to close,
no profile to export, and no record to delete.

The only ToneSphere-related data that exists is the local files listed in section 3, on
your own computer. You can read them in a text editor, change them, or delete them at any
time without asking anyone.

## 9. Security

Because nothing is collected, there is no central store of user data to breach. What
remains is local, and worth understanding:

- Configuration, presets and logs are plain text and are not encrypted. Anyone with access
  to your user account on that computer can read them.
- The optional network stream is unencrypted and unauthenticated, as described in section 4.
- The local REST interface is unauthenticated and is bound to `127.0.0.1` by default.

We would rather document those three facts than describe the application as secure and
leave you to discover them.

## 10. Changes to this policy

This policy carries a version number and an effective date at the top of the document. A
material change raises both, and the updated policy is published at the same address, so
that the URL registered with the Microsoft Store always resolves to the current text.
Earlier versions remain in the project repository's history.

## 11. Contact

Questions about this policy, or about anything ToneSphere does with audio on your machine,
should be raised as an issue in the project repository:

[https://github.com/AvishakeAdhikary/tone-sphere/issues](https://github.com/AvishakeAdhikary/tone-sphere/issues)

That is the project's contact channel. Issues are public, so do not include anything
confidential in one.

## 12. Related documents

- [Terms and Conditions](terms-and-conditions.md) — the agreement governing your use of the
  application itself.
- [Terms of Service](terms-of-service.md) — what the application provides, how it is
  distributed, updated and supported, and the terms attaching to the optional network
  feature.
