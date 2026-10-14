# Store listing text (en-US)

## Product name
ToneSphere

## Short description (≤ 100 words)
Route any audio anywhere. ToneSphere sorts every sound source and destination on your PC into six simple bins: microphones, speakers, virtual devices and individual apps. Connect them by dragging one onto another. Mix, add effects, and share audio between apps, DAWs and the network. Every path has low latency and drift correction.

## Description
ToneSphere is a modern audio router for Windows. It is free and open source.

Everything on your PC that makes or takes sound shows up in one of six bins:
• Physical inputs: microphones, line-ins, audio interfaces
• Physical outputs: speakers, headphones
• Virtual outputs and virtual inputs: virtual cables that other apps can play into or record from
• Application outputs: what each app plays (games, browsers, music players)
• Application inputs: the microphone each app hears (Discord, Zoom, OBS, …)

Drag a source onto a destination to connect them. That's it.

When you need more:
• Mixes: build a "Stream mix" or "Monitor mix" from any sources, with per-source volume
• Effects on any input, output or mix: noise suppression, noise gate, EQ, compressor, limiter, auto-ducking, delay
• Capture any app's audio separately, or send one app to its own output
• Feed a mix into any app's microphone
• ASIO: use your audio interface in several DAWs at once with the ToneSphere ASIO driver (desktop installer)
• VST3 effects and instruments, with an on-screen keyboard and MIDI keyboards
• Uses several processor cores for heavy effect chains, automatically
• Network audio: VBAN send/receive, compatible with VoiceMeeter and VBAN apps
• Scenes with global hotkeys, push-to-talk, mute-all
• Multitrack recording and an instant-replay buffer
• Matrix view with per-channel routing for power users
• Automatic clock-drift compensation between devices, so no clicks or slow drift
• Light and dark mode, keyboard accessible, lives quietly in the tray

ToneSphere is free software under the GNU GPL v3. Source code: https://github.com/AvishakeAdhikary/tone-sphere

## Keywords
audio router, virtual audio cable, asio, mixer, streaming, obs, discord, voicemeeter alternative

## What's new in this version
First release.

## Restricted capability justification (runFullTrust)
ToneSphere is a desktop (Win32) audio routing application. It runs a native real-time audio engine that opens audio devices with WASAPI and ASIO, captures individual application audio streams, and hosts audio effects. These APIs and the engine's real-time scheduling need a full-trust desktop process.

## Notes for certification
ToneSphere needs no account or sign-in. To test it:
1. Start ToneSphere and play music in any app, e.g. a video in Microsoft Edge. The app appears under "Application outputs".
2. Drag that app's card onto a speaker or headphones card under "Physical outputs". The connection line appears, and the meters show the app's audio flowing through ToneSphere.
3. Optional tone test: open the "Matrix" tab and click the cell where "Test tone" meets your speakers. You will hear a 440 Hz tone. Click the cell again to stop it.

The optional ToneSphere desktop installer (with the ASIO driver) is a separate download. None of the app's core functions need it.

## Privacy policy
https://avishakeadhikary.github.io/tone-sphere/legal/privacy-policy/ ToneSphere does not collect, store or transmit personal data.
