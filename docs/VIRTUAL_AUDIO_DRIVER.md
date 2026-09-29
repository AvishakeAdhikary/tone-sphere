---
title: Virtual Audio Driver
layout: default
permalink: /virtual-audio-driver/
description: What an OS-visible virtual audio device requires on Windows, Linux and macOS, what ToneSphere's Windows driver is, and exactly how far it has been verified.
---

# Virtual audio driver — what it takes

This is the claim people most want and the one ToneSphere was slowest to be able to make: a
device that appears in Discord's, OBS's or your DAW's device list. This document says exactly what that requires, because the original
codebase asserted it was already done — `devices/native_virtual.py` opened with *"Creates
actual virtual audio devices that appear in system sound settings"* above a class holding
two `queue.Queue` objects.

## Why the current buses are not it

`create_virtual_input()` makes a summing point inside the ToneSphere process. Audio flows
through it correctly and it is genuinely useful for internal routing. It is not visible
outside the process, and no amount of user-mode code can change that.

A device other applications can select must be published by the operating system's audio
stack. On Windows that means a driver loaded by the kernel. There is no user-mode API that
registers a new endpoint — the closest things (APOs, session hooks) attach to an *existing*
endpoint and cannot create one.

## What does not need a driver

| Want | Needs a driver? |
|---|---|
| Guitar in → effects → headphones out | **No.** The native engine does it (`docs/WINDOWS_AUDIO.md`). |
| Guitar in → VST3 plugin → out | **No.** The native VST3 host (`docs/VST3.md`). |
| Record everything playing on an output | **No.** WASAPI loopback, a native stream kind. |
| Capture one application's audio | **No**, on Windows 10 build 20348+ — process loopback. |
| Send ToneSphere's output *into* Discord or OBS as a microphone | **Yes.** |
| Appear as an input or output device in a DAW or any application | **Yes.** |

## The Windows driver

[`driver/windows_virtual_audio/`](https://github.com/AvishakeAdhikary/tone-sphere/tree/main/driver/windows_virtual_audio)
is a kernel-mode PortCls/WaveRT driver derived from Microsoft's **SimpleAudioSample**
(Microsoft Public License), publishing one virtual cable:

- **ToneSphere Cable Input** — a render endpoint applications play into;
- **ToneSphere Cable Output** — a capture endpoint applications record from.

What is played into the input comes out of the output. Both run 48 kHz, 32-bit PCM,
stereo. ToneSphere uses them through its ordinary WASAPI backend: render its mix into
Cable Input and Discord records Cable Output as a microphone; or let an application play
into Cable Input and capture Cable Output into ToneSphere.

### Design, and why this one

**A loopback cable, not a private channel.** The alternative is a driver whose capture
endpoint is fed by ToneSphere through a shared-memory ring mapped into both address spaces
and an event, with IOCTLs to set it up. That can shave a WASAPI period of latency, but it is
several times the kernel code — memory mapping into a user process, lifetime and security of
that mapping, a second clock. The cable needs none of it: the kernel side copies bytes
between two of its own streams, and every piece of routing, mixing, drift correction and DSP
stays in ToneSphere, in user mode. The smaller kernel surface was the deciding factor; a
direct transport remains an option if the extra period ever matters more than the risk.

**No clock of its own to drift.** Both endpoints advance on the same QueryPerformanceCounter
time base (the sample's position logic, unchanged), so producer and consumer cannot drift
apart inside the driver. ToneSphere's drift resampler handles the boundary to real hardware.

**Bounded staleness.** At most 100 ms can queue in the cable; beyond that the oldest audio is
dropped, so a capture that starts late does not begin a second behind.

**Nothing else.** No mixing, no resampling, no processing in kernel mode, and the sample's
render-to-file feature is removed entirely, including its registry override.

### Status

| | Level | Evidence |
|---|---|---|
| Driver source (SimpleAudioSample + the cable) | IMPLEMENTED | `driver/windows_virtual_audio/`; every change listed in its README |
| Builds, test-signed package (INF, SYS, CAT) with the EWDK | VERIFIED (build) | `scripts/build_driver.py` on the development machine, 2026-09-29 |
| INF meets Windows Driver requirements | VERIFIED (static) | `InfVerif /w`: no findings |
| Install / enumeration / cross-application audio / uninstall | **NOT YET VERIFIED** | needs the Hyper-V test VM; `tests/hardware/test_virtual_driver.py` and `vm_kit/driver_install.ps1`/`driver_uninstall.ps1` are ready, and skip with the reason on any machine without the driver |
| Production-signed deployment | **NOT AVAILABLE** | attestation signing needs an EV certificate and a Partner Center account; see below |
| Sleep/resume, format changes, many simultaneous clients | NOT VERIFIED | |

Until the VM run happens, the honest statement is: the driver is built and its package is
valid; nothing has shown it loading, enumerating or moving audio.

### Testing it

Only inside a Hyper-V VM with test-signing on — never on the development machine (the
driver's README has the steps). `driver_install.ps1` refuses to run with test-signing off.
Then:

```
uv run pytest tests/hardware/test_virtual_driver.py -m hardware -s
```

which checks that Windows enumerates both endpoints at one format; that a tone played by one
PortAudio process arrives at another through the cable; that audio crosses in both
directions with ToneSphere as one side, including through a VST3 plugin; and that an idle
cable is exact silence. `driver_uninstall.ps1` removes the device, the driver-store package
and the certificate trust, and fails if anything is left.

### Signing, and what it costs

Windows will not load an unsigned or test-signed kernel driver on a normal machine.

| Item | Cost | Notes |
|---|---|---|
| EV code-signing certificate | ~£250–400/year | Hardware token; identity verification takes days to weeks |
| Partner Center account | $19 one-off (individual) | Needed to submit for attestation signing |
| Attestation signing | Free | Microsoft signs the driver; needs the EV cert to submit |
| WHQL/HLK certification | Free to submit | Only needed for Windows Update distribution; a lab run is a significant effort |

Attestation signing is enough for a driver a user installs deliberately. An MSIX (Store)
package cannot install a kernel driver, so it would need a separate installer.

## macOS and Linux

**macOS** needs an AudioServerPlugIn — a user-space bundle in
`/Library/Audio/Plug-Ins/HAL`, no kernel code and no paid certificate for local
installation, though notarisation is needed for distribution. Substantially easier than
Windows. BlackHole is the reference implementation and is MIT-licensed.

ToneSphere now ships one:
[`native/coreaudio-plugin/`](https://github.com/AvishakeAdhikary/tone-sphere/tree/main/native/coreaudio-plugin), a
loopback device called **ToneSphere Audio**, with `create_macos_system_device()` on the
engine and `/virtual-devices/system/macos` on the REST API to claim it. Read that
directory's README before relying on it. In short: it is written from Apple's documented
`AudioServerPlugIn` interface rather than forked from BlackHole, it was written on a
machine with no macOS on it, and the `build-macos-plugin` CI job — which builds it, ad-hoc
signs it, installs it, restarts `coreaudiod` and round-trips a 1 kHz sine through the
device — is the only evidence that any of it works. Everything about day-to-day use
(System Settings, an application's own device picker, sleep/wake, Gatekeeper on a
downloaded bundle) is unverified, and the docs say so rather than rounding it up to
"macOS supported".

**Linux** does not need one. PipeWire and JACK already provide arbitrary virtual endpoints,
and `pactl load-module module-null-sink` creates one in a single command. The right move on
Linux is to use the platform rather than fight it.

## Where things stand

1. **Per-process WASAPI loopback capture** — done (no driver).
2. **macOS AudioServerPlugIn** — proven in CI only.
3. **Linux via PipeWire/JACK** — done (configuration, not code).
4. **Windows kernel driver** — built and statically validated; install and audio across
   applications not yet verified (test VM); production signing not available.

ToneSphere says exactly that, and nothing more, until the VM run and a signing route exist.
