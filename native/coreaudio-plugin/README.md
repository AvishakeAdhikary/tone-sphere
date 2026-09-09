# ToneSphere AudioServerPlugIn (macOS)

A CoreAudio HAL plug-in that publishes one loopback device called **ToneSphere Audio**:
anything played to it comes back out of its input side. Other applications — a DAW,
Discord, OBS, ToneSphere itself — can select it like any sound card, because once
`coreaudiod` has loaded this bundle the device is a real system device. No kernel
extension, no paid Apple Developer account for a local install.

## Status: what is proven, and where

This is the part to read before trusting anything here.

| Claim | Proven by | Where |
|---|---|---|
| It compiles | `make` | `build-macos-plugin` CI job, macos-latest |
| It signs and passes `codesign --verify` | `make sign` | same job |
| It installs and `coreaudiod` picks it up | `make install` + a device-list check | same job |
| **It actually carries audio** | a 1 kHz sine written to the device and captured back from it, asserting dominant frequency and RMS | same job, `.github/scripts/macos_plugin_roundtrip.py` |

**Not proven by anything, by anyone, yet — do not claim otherwise:**

- How it looks and behaves in **System Settings → Sound** and **Audio MIDI Setup**.
- Whether it is selectable and stable in **Discord, OBS, Logic, Ableton, Zoom** or any
  other real application's own device picker.
- **Sleep/wake**, display/dock changes, and long-running stability.
- **Gatekeeper and quarantine** behaviour for a real user's install method. CI does
  `sudo cp` from a checked-out repository, which does not carry the `com.apple.quarantine`
  attribute that a downloaded zip or `.pkg` would. Someone who downloads a prebuilt bundle
  is on a different code path than anything tested here.
- Simultaneous use by several applications at once (see *Known limitations*).

The source file was written on a Windows machine by an author who could not compile,
install, or listen to it. `ToneSphereAudio.c`'s header comment lists exactly which
structural details were verified against Apple's published `AudioServerPlugIn.h` and which
were not.

## Build and install

Requires the Command Line Tools (`xcode-select --install`); no Xcode project needed.

```sh
cd native/coreaudio-plugin
make                # builds build/ToneSphereAudio.driver (universal: x86_64 + arm64)
make sign           # ad-hoc codesign, free -- no Apple Developer account
make install        # sudo cp into /Library/Audio/Plug-Ins/HAL, then restart coreaudiod
```

`make install` asks for your password twice-ish (`sudo cp`, `sudo launchctl`). It runs:

```sh
sudo rm -rf /Library/Audio/Plug-Ins/HAL/ToneSphereAudio.driver
sudo cp -R build/ToneSphereAudio.driver /Library/Audio/Plug-Ins/HAL/
sudo chown -R root:wheel /Library/Audio/Plug-Ins/HAL/ToneSphereAudio.driver
sudo launchctl kickstart -k system/com.apple.audio.coreaudiod
```

The last line is the one people forget: `coreaudiod` only rescans that directory when it
restarts, so until it does, the device does not exist. `sudo killall coreaudiod` is the
older equivalent and is used as a fallback. **Restarting `coreaudiod` interrupts all audio
on the machine for a second or two** — every playing application will glitch or, in some
cases, need to be restarted.

Then confirm it is really there, from the OS rather than from ToneSphere:

```sh
system_profiler SPAudioDataType | grep -A3 "ToneSphere"
```

## Uninstall

```sh
cd native/coreaudio-plugin
make uninstall
```

which is:

```sh
sudo rm -rf /Library/Audio/Plug-Ins/HAL/ToneSphereAudio.driver
sudo launchctl kickstart -k system/com.apple.audio.coreaudiod
```

Nothing else is installed anywhere — no launch agent, no preference file, no receipt. If
the device is still listed after this, `coreaudiod` did not restart.

## No installer package, deliberately

A downloadable `.pkg` or zip picks up Gatekeeper's quarantine attribute, and clearing that
properly means Apple notarisation, which means a paid Developer Program membership. That
is out of scope for this project, exactly as the Windows kernel-driver route is
(`docs/VIRTUAL_AUDIO_DRIVER.md` explains that one). Build from source and install with the
commands above; that path needs no certificate at all.

## Known limitations

- **Fixed 2 channels.** `kDevice_Channels` in `ToneSphereAudio.c`; changing it is a
  recompile.
- **Four sample rates** — 44.1, 48, 88.2, 96 kHz. Rate changes go through the HAL's
  configuration-change dance, so they stop IO first rather than switching underneath a
  running stream.
- **No volume or mute control.** The device has no controls at all, so the system volume
  slider does not apply to it. This is intentional: a control that claims to attenuate and
  does not would be worse than its absence.
- **One input consumer at a time.** Frames are cleared from the loopback buffer as they
  are read, which is what stops audio repeating forever once playback stops — but it means
  a second application capturing from the device simultaneously gets whatever the first one
  did not take. Fine for the one-app-captures-the-mix case; not a general multi-client
  splitter.
- **No aggregate/multi-output device is created for you.** To *hear* audio while also
  routing it here, make an Aggregate or Multi-Output Device in Audio MIDI Setup, as with
  BlackHole or Soundflower.

## Relationship to BlackHole

[BlackHole](https://github.com/ExistentialAudio/BlackHole) (MIT) is the reference
implementation `docs/VIRTUAL_AUDIO_DRIVER.md` cites, and forking it was the recommended
route in the plan for this work. It was not forked here: a faithful fork means reproducing
several thousand lines of someone else's C exactly, which is not something that can be done
reliably from a machine that cannot fetch, diff or compile it — a half-transcribed fork
would be worse than the smaller implementation written from Apple's documented interface
and proven by CI. No BlackHole code is included, so there is nothing to attribute; if you
would rather run their far more widely exercised driver, install it and keep its MIT notice.

ToneSphere will not claim a separately installed BlackHole as its own device: the Python
side (`tonesphere/engine/macos_virtual.py`) matches only the exact device name published
by this bundle.

## Layout

```
native/coreaudio-plugin/
  ToneSphereAudio.c   the plug-in: driver interface, property tree, loopback IO
  Info.plist          CFPlugInFactories / CFPlugInTypes wiring
  Makefile            build / sign / install / uninstall
  README.md           this file
```

The device name and UID are duplicated in `tonesphere/engine/macos_virtual.py`; a test in
`tests/test_macos_virtual_device.py` reads both files and fails if they ever disagree.
