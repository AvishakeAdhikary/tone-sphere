---
title: Privacy Policy
layout: default
permalink: /legal/privacy-policy/
version: "2.0"
effective_date: "2026-10-14"
publisher: Neural Nexus Studios
description: ToneSphere collects no user information and has no user accounts. This policy states what stays on your machine, and what the optional features that reach the network actually do.
---

# Privacy Policy

- **Application:** ToneSphere
- **Publisher:** Neural Nexus Studios, an individual developer, Kolkata, West Bengal, India
- **Policy version:** 2.0
- **Effective date:** 14 October 2026

## 1. Summary

ToneSphere collects no user information.

There are no user accounts, no sign-in, no registration, no telemetry, no analytics, no
advertising identifiers, no usage reporting, and no third-party tracking of any kind. When
the audio engine stops unexpectedly, it writes a crash report to a file on your own computer
and the application tells you where it is; it never sends it anywhere.

ToneSphere is a desktop audio router: it takes audio from the devices and applications on
your computer, processes it, and sends it to other devices and applications on the same
computer. That audio is handled in memory, in real time. It is stored only when you record
it yourself, and transmitted only by the optional network features in section 4, when you
set them up.

## 2. What ToneSphere does with your audio

Audio entering ToneSphere is mixed, filtered and routed inside the audio engine and
delivered to the outputs you have connected. It exists in memory buffers for as long as it
takes to play it, a matter of milliseconds, and is then overwritten.

- **Recording.** When you press Record, or save the instant replay, ToneSphere writes the
  audio you chose to WAV files in the folder you set (by default `Music\ToneSphere`). The
  instant-replay buffer, when you turn it on, keeps the last minutes of your mixes in memory
  only, until you save it.
- **Application capture.** ToneSphere lists the applications that are playing or recording
  sound, from the operating system's own audio session list, so that you can route them. It
  does not read the contents of those applications or anything else about them.

Nothing in the application uploads, transcribes, analyses or fingerprints the sound passing
through it.

## 3. Information stored on your computer

ToneSphere stores its own files on your machine, in ordinary files that you can read, edit
and delete. Nothing in these files is sent anywhere.

| What | Where (Windows; Linux uses `~/.local/share/tonesphere`) | Contains |
|---|---|---|
| Settings and routing | `%LOCALAPPDATA%\ToneSphere\config.json` | Sample rate, buffer size, your devices, connections, mixes, effects and plugin state, scenes, hotkeys, MIDI and OSC settings, skin, language |
| Plugin scan cache | `%LOCALAPPDATA%\ToneSphere\plugins.json` | For each VST3 module found: its file path and date, and the names, vendors, versions and categories it declares |
| Log | `%LOCALAPPDATA%\ToneSphere\engine.log` | Diagnostic lines about devices, streams, plugins and errors |
| Crash reports | `%LOCALAPPDATA%\ToneSphere\crashes\` | Written only when the engine stops on an unexpected fault: the ToneSphere version, the fault, the program module it happened in, and (on Windows) a minidump, a snapshot of the engine's memory at that moment that can include audio buffers and device and plugin names |
| Connection details | `%LOCALAPPDATA%\ToneSphere\engine.json` | The local port and the random token the application uses to reach its engine; deleted when the engine stops |
| Window preferences | `%APPDATA%\ToneSphere` | The application window's own profile: remembered view and collapsed sections |
| Presets you export | wherever you save them | Your routing, with device names and plugin settings |
| Recordings | the folder you choose (default `Music\ToneSphere`) | The audio you recorded |

Some of these record names rather than only numbers, and it is fair to say so. Settings and
presets record the **names of your audio devices and applications**, and which plugins you
have and where they are installed; a plugin's own saved state is whatever that plugin
decides. Logs and crash reports may record device names, plugin file paths and error
messages. All of them are local files under your control, and you decide whether to share
one, for example by attaching it to an issue. Uninstalling the application leaves nothing
about you behind on any service, because there is no service.

## 4. Features that reach the network

ToneSphere makes no network connection of its own while running, except these, which you
start yourself:

- **VBAN network streams** (Settings, or the Network bins): audio goes to the address and
  port you enter, and nowhere else, directly between your machine and that peer. Neural
  Nexus Studios operates no server for ToneSphere: no relay, no intermediary, no cloud
  component. The VBAN transport applies **no encryption and no authentication**: anyone able
  to observe the network path could listen, and anyone able to reach a receiving stream's
  port could send audio into it. The peer learns your IP address, as it must for any direct
  connection.
- **OSC control** (Settings › Control, off by default): ToneSphere listens on the UDP port
  you choose for commands from your network (volume, mute, scenes, recording). Anyone able
  to reach that port can send those commands while it is on.
- **VB-CABLE download** (Settings › Devices › Install drivers, when you tick it): ToneSphere
  downloads VB-Audio's official package from `download.vb-audio.com` over HTTPS, as your
  browser would, and opens VB-Audio's installer. VB-Audio receives the ordinary details of
  any download (your IP address and the request); its own privacy terms apply.
- **Links you click** (for example "About VB-CABLE") open in your web browser.

The application window and its audio engine talk to each other over a local connection
bound to `127.0.0.1`, protected by a random token that changes every time the engine starts.
It accepts no connections from other computers and refuses requests from web pages.

## 5. Third-party plugins you load

ToneSphere hosts VST3 plugins that you already own. Those plugins are software written by
other companies, running inside the ToneSphere engine at your instruction. Scanning for
plugins loads each one briefly, in a separate process, to read what it declares.

A plugin can do things ToneSphere does not: check a licence over the internet, contact its
vendor's servers, write files, or collect information about your machine. What a plugin
does is governed by that vendor's own privacy policy and licence terms, not by this one.
Neural Nexus Studios does not supply these plugins, cannot inspect their behaviour, and
receives nothing from them.

## 6. Distribution channels

ToneSphere itself has no analytics, but obtaining a copy of it involves someone else's
service, and those services keep their own records:

- **Microsoft Store.** ToneSphere is being prepared for distribution through the Microsoft
  Store for Windows. Microsoft's own data collection when you browse, acquire, install or
  update an application from the Store is described in the Microsoft Privacy Statement and
  is Microsoft's, not ours. Microsoft provides publishers with aggregated reports
  (acquisition counts, ratings, reliability summaries) that do not identify individual
  users. That aggregate is the entirety of what Neural Nexus Studios receives about
  installations, and it is not personal information.
- **GitHub.** Releases are published on GitHub. Downloading a file from GitHub, or opening an
  issue there, is processed under the GitHub Privacy Statement. Publicly filed issues are
  visible to everyone, including the GitHub username you write them under. Neural Nexus
  Studios sees the public issue and aggregate download counts, nothing more.

If a version of ToneSphere is ever sold, the sale is processed by the store or platform it
is bought through, under that platform's privacy statement. Neural Nexus Studios receives no
card or bank details; no account with us is created or needed.

Neither channel reports anything to ToneSphere, and the application does not contact either
of them while running. It performs no update check.

## 7. Children

ToneSphere collects no information from anyone, of any age, so it holds no information about
children. It is a general-purpose audio utility and is not directed at children.

## 8. Your rights over your information

Data-protection rights (access, correction, export, erasure, objection, withdrawal of
consent) are exercised against an organisation that holds your personal data. Neural Nexus
Studios holds none, because ToneSphere collects none. There is no account to close, no
profile to export, and no record to delete.

The only ToneSphere-related data that exists is the local files listed in section 3, on
your own computer. You can read them, change them, or delete them at any time without asking
anyone.

## 9. Security

Because nothing is collected, there is no central store of user data to breach. What
remains is local, and worth understanding:

- Settings, presets, logs and crash reports are plain files and are not encrypted. Anyone
  with access to your user account on that computer can read them.
- VBAN streams and OSC control are unencrypted and unauthenticated, as described in
  section 4.
- The connection between the window and the engine is local-only and token-protected.

We would rather document those facts than describe the application as secure and leave you
to discover them.

## 10. Changes to this policy

This policy carries a version number and an effective date at the top of the document. A
material change raises both, and the updated policy is published at the same address, so
that the URL registered with the Microsoft Store always resolves to the current text.
Earlier versions remain in the project repository's history; version 1.3 described the
earlier Python implementation of ToneSphere.

## 11. Contact

Questions about this policy, or about anything ToneSphere does with audio on your machine,
should be raised as an issue in the project repository:

[https://github.com/AvishakeAdhikary/tone-sphere/issues](https://github.com/AvishakeAdhikary/tone-sphere/issues)

That is the project's contact channel. Issues are public, so do not include anything
confidential in one.

## 12. Related documents

- [Terms and Conditions](terms-and-conditions.md): the agreement governing your use of the
  application itself.
- [Terms of Service](terms-of-service.md): what the application provides, how it is
  distributed, updated and supported, and the terms attaching to the optional network
  features.
