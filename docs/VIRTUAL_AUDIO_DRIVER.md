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

- **Speakers (ToneSphere Virtual Audio Cable)** — a render endpoint applications play into;
- **Microphone Array (ToneSphere Virtual Audio Cable)** — a capture endpoint applications
  record from.

What is played into the render endpoint comes out of the capture endpoint. Both run
48 kHz, 32-bit PCM, stereo. ToneSphere uses them through its ordinary WASAPI backend:
render its mix into the cable and Discord records the other end as a microphone; or let an
application play into the cable and capture the other end into ToneSphere.

The names are Windows' own: the pin category ("Speakers", "Microphone Array") and the
device. Naming the endpoints "ToneSphere Cable Input/Output" needs either a
`MediaCategories` registration, which the Windows Driver INF rules refuse (InfVerif error
1321), or a pin-name property handler in the driver — **not implemented**.

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
dropped, so a capture that starts late does not begin a second behind. And a capture that
starts empties the cable: what is queued was played before anyone was listening, and must
not open someone else's recording.

**Nothing else.** No mixing, no resampling, no processing in kernel mode, and the sample's
render-to-file feature is removed entirely, including its registry override.

### Status

| | Level | Evidence |
|---|---|---|
| Driver source (SimpleAudioSample + the cable) | IMPLEMENTED | `driver/windows_virtual_audio/`; every change listed in its README |
| Builds, test-signed package (INF, SYS, CAT) with the EWDK | VERIFIED (build) | `scripts/build_driver.py` on the development machine |
| INF meets Windows Driver requirements | VERIFIED (static) | `InfVerif /w`: no findings |
| Install, enumeration, cross-application audio, silence, uninstall | **HARDWARE VERIFIED in a Hyper-V test VM** | 2026-09-30, Windows 11 Enterprise LTSC Evaluation 10.0.26100 guest, test-signing on, Secure Boot off; `tests/hardware/test_virtual_driver.py`, 7 of 7 — see below |
| Loads again after the guest reboots | observed | a guest restart mid-run left the device and both endpoints present and working |
| Custom endpoint names ("ToneSphere Cable Input/Output") | **NOT IMPLEMENTED** | see above |
| On a real desktop, with a real communications app (Discord, OBS) | NOT VERIFIED | the tests ran in the VM's PowerShell Direct session, with PortAudio processes as "other applications" |
| Production-signed deployment | **NOT AVAILABLE** | attestation signing needs an EV certificate and a Partner Center account; see below |
| Sleep/resume, format changes, many simultaneous clients | NOT VERIFIED | |

### What the VM run showed

`scripts/vm/run_driver_tests.ps1`, from the VM's `deps` checkpoint (a guest that has never
had the driver), 2026-09-30; the logs of the final run are in
`driver/windows_virtual_audio/test-results/`:

- **Install:** `driver_install.ps1` trusted the test certificate and `devcon` created
  `ROOT\MEDIA\0000`, "ToneSphere Virtual Audio Cable", status OK, with both endpoints OK.
- **Enumeration:** Windows enumerates both endpoints, 48 kHz stereo each.
- **One application to another** (two PortAudio processes, neither of them ToneSphere):
  a 1 kHz tone at 0.2 peak arrived at 1000.00 Hz, rms 0.14142 against 0.14142 sent
  (+0.000 dB; −0.009 dB in another pass).
- **An application into ToneSphere** (PortAudio plays, the native engine captures):
  1000.00 Hz, rms 0.14142 (+0.000 dB; −0.009 dB in another pass).
- **ToneSphere through a VST3 plugin into another application** (the engine renders
  through the Test Gain plugin at ×0.5; PortAudio records): 1000.00 Hz, rms 0.07071, exactly
  half (+0.000 dB against 0.07071).
- **Idle is exact silence**, and **a new capture does not replay** audio played before it
  started.
- **The engine finds it:** `AudioEngine.virtual_device_status()` reports it installed, by the
  names above.
- **Uninstall:** device removed, driver-store package deleted, certificate no longer trusted,
  no ToneSphere device or endpoint left.

The final driver passed three times in a row, twice on a VM built from scratch by
`new_driver_vm.ps1`. Getting there took seven passes before those, and each fixed something
real that nothing else would have found:

1. `Import-Certificate` is refused ("access denied") over PowerShell Direct even with an
   administrator token; the scripts use `certutil`.
2. PowerShell Direct gives a local administrator a UAC-filtered token; the VM's answer file
   lifts that for the VM (`LocalAccountTokenFilterPolicy`).
3. The tests passed floats to `subprocess.Popen`, so the cable tests had never been able to
   run.
4. **The driver never wrote into the cable.** The sample calls its render-side writer only
   when data files are enabled, and they are off; with the call left gated, every capture
   was silence. Now it is unconditional.
5. A capture started after the player stopped heard the last 100 ms of it; a capture now
   empties the cable when it starts.
6. The app looked for the INF's pin names, which Windows does not show; it now matches the
   device name.
7. OOBE restarts a new guest once, about a quarter of an hour after first logon, and a
   checkpoint taken earlier replays that restart; `new_driver_vm.ps1` now waits it out, and
   switches Windows Update off in the guest.

### Testing it

Only inside a Hyper-V VM with test-signing on — never on the development machine. On a
host with Hyper-V, from an elevated Windows PowerShell:

```
powershell -ExecutionPolicy Bypass -File scripts\vm\new_driver_vm.ps1 -Iso <Windows 11 ISO>
uv run python scripts/build_driver.py
powershell -ExecutionPolicy Bypass -File scripts\vm\run_driver_tests.ps1
```

`new_driver_vm.ps1` applies the image straight to a VHDX (no Setup, so no "press any key"
prompt and no TPM check), switches test-signing on in the **VM disk's** boot store, writes an
answer file, and creates a Generation 2 VM with Secure Boot off. It was built from the
Windows 11 Enterprise LTSC 90-day evaluation ISO (SHA-256
`67cec5865eaa037a72ddc633a717a10a2bed50778862267223ddb9c60ef5da68`). The host's
boot configuration is never touched. `run_driver_tests.ps1` restores a checkpoint, copies in
the tree and the built binaries, installs, runs `tests/hardware/test_virtual_driver.py`,
uninstalls, checks nothing is left, and brings every log back (including `setupapi.dev.log`
and any crash dump).

`driver_install.ps1` refuses to run with test-signing off. The tests check that Windows
enumerates both endpoints at one format; that a tone played by one PortAudio process
arrives at another through the cable; that audio crosses in both directions with ToneSphere
as one side, including through a VST3 plugin; that an idle cable is exact silence and a new
capture starts empty; and that the engine reports the cable. `driver_uninstall.ps1` removes
the device, the driver-store package and the certificate trust, and fails if anything is
left.

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
4. **Windows kernel driver** — installs, enumerates and carries audio between applications
   in a Hyper-V test VM, at the level sent, and uninstalls cleanly; untested on a real
   desktop with a real communications app; production signing not available.

ToneSphere says exactly that, and nothing more, until a signing route exists.
