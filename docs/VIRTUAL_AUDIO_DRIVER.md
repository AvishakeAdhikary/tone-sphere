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
(Microsoft Public License). It publishes **cables**, each one its own device — an instance of
the root-enumerated hardware ID `ROOT\ToneSphereVirtualAudio` — with two endpoints:

- **Speakers (<cable>)** — a render endpoint applications play into;
- **Microphone Array (<cable>)** — a capture endpoint applications record from.

What is played into a cable's render endpoint comes out of that cable's capture endpoint,
and out of no other. Both run 48 kHz, 32-bit PCM, stereo. A fresh install creates two cables,
"ToneSphere Cable 1" and "ToneSphere Cable 2". ToneSphere uses them through its ordinary
WASAPI backend: render its mix into a cable and Discord records the other end as a
microphone; or let an application play into a cable and capture the other end into
ToneSphere.

**Managing them from ToneSphere.** Engine → Virtual Cables… lists every cable with its state
and endpoints, and adds (up to eight), renames, disables, enables and uninstalls them one at a
time, or removes the driver (`tonesphere/engine/virtual_cables.py`, `tonesphere/ui/cables_view.py`).
Listing needs no privilege. A change runs in a separate process started through Windows' own
administrator prompt (`main.py cable-admin`), so nothing else in ToneSphere runs elevated.
Because each cable is a device of its own, Windows' disable and uninstall act on exactly one:
the others keep playing.

**A cable in use.** Windows will not take a device away from certain programs using it: its
audio engine (`audiodg.exe`) vetoes the removal (System log, Kernel-PnP event 225). In the VM
that happened for every stream ToneSphere's native engine opened — shared or exclusive,
render or capture, raw or not — and never for a PortAudio program's shared stream, which was
simply cut off; why, is not established. A plain `pnputil /disable-device` answers a veto by
leaving the cable "pending a restart": still working, and refusing every later change until
Windows restarts. So ToneSphere never does that. It lets go of its own streams on a cable
before changing it and starts them again afterwards, and it asks Windows with the "no UI"
flags, which turn a veto into a clean refusal — "a program has this cable open…" — with
nothing changed. The audio engine also opens a cable briefly by itself (when one arrives, or
becomes the default device because another left), so a veto is retried for ten seconds
before it counts.

The names are Windows' own: the pin category ("Speakers", "Microphone Array") and the
device's friendly name, which is the cable's name. Naming the endpoints "ToneSphere Cable
Input/Output" would need a `MediaCategories` registration, which the Windows Driver INF rules
refuse (InfVerif error 1321), or a pin-name property handler in the driver.

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
| Driver source (SimpleAudioSample + one cable per device) | IMPLEMENTED | `driver/windows_virtual_audio/`; every change listed in its README |
| Builds, test-signed package (INF, SYS, CAT) with the EWDK | VERIFIED (build) | `scripts/build_driver.py` on the development machine |
| INF meets Windows Driver requirements | VERIFIED (static) | `InfVerif /w`: no findings |
| Install, two cables, enumeration, cross-application audio on each, isolation, silence, uninstall | **HARDWARE VERIFIED in a Hyper-V test VM** | `tests/hardware/test_virtual_driver.py`, 14 of 14 — see below |
| Add, rename, disable, enable, uninstall a cable from the app and its dialog | **HARDWARE VERIFIED in the VM** | the same file; the dialog's own buttons drive a real disable and enable |
| A cable disappearing under a running engine, and coming back | **HARDWARE VERIFIED in the VM** | the engine reports it left, carries on without it, and reopens it on return with no user action |
| Another program's application (ffmpeg, DirectShow) recording a cable | **HARDWARE VERIFIED in the VM** | 1 kHz at +0.000 dB |
| The ASIO host through ASIO4ALL on a cable | **HARDWARE VERIFIED in the VM** (output at +0.00 dB from native WASAPI; its round trip `--`) | `tests/hardware/test_asio.py`, `test_roundtrip.py` with `TONESPHERE_TEST_INTERFACE='ToneSphere Cable 1'`; see `docs/ASIO.md` |
| Measured round trip through a cable | **HARDWARE VERIFIED in the VM**, exclusive mode **intermittent** | shared 10 of 10 (63.35 / 73.35 ms); the WASAPI **exclusive** round trip through a cable (144-frame period) measures 12.02 ms at confidence 79 most of the time, but 3 of 10 runs in one VM session found no path (confidence 1.3–2.3) — although the capture held the whole sweep, 13 ms after it was played, with no gap and the same peak. The audio crosses the cable intact; the correlation sometimes fails on it. Cause not established |
| Loads again after the guest reboots | observed | a guest restart mid-run left the device and both endpoints present and working |
| Custom pin names ("ToneSphere Cable Input/Output") | NOT IMPLEMENTED | see above; the cable's own name does appear in both endpoint names |
| On a real desktop, with a real communications app (Discord, OBS) | NOT VERIFIED | the tests ran in the VM's PowerShell Direct session (session 0), with PortAudio processes and ffmpeg as "other applications" |
| Production-signed deployment | **NOT AVAILABLE** | attestation signing needs an EV certificate and a Partner Center account; see below |
| Sleep/resume, format changes, many simultaneous clients | NOT VERIFIED | |

### What the VM runs showed

`scripts/vm/run_driver_tests.ps1`, from the VM's `deps` checkpoint (a guest that has never
had the driver), 2026-09-30 and 2026-10-01, Windows 11 Enterprise LTSC Evaluation 10.0.26100;
the logs of the final runs are in `driver/windows_virtual_audio/test-results/`:

- **Install:** `driver_install.ps1` trusted the test certificate and added the package;
  `main.py cable-admin install-cables` created `ROOT\MEDIA\0000` "ToneSphere Cable 1" and
  `ROOT\MEDIA\0001` "ToneSphere Cable 2", both working, four endpoints at 48 kHz stereo.
- **One application to another, on each cable** (two PortAudio processes): 1 kHz at rms
  0.14142 against 0.14142 sent, +0.000 dB, on both.
- **Isolation:** a tone into cable 1 arrived at 0.14128; cable 2 read a peak of exactly 0.
- **An application into ToneSphere:** 1000.00 Hz, −0.009 dB. **ToneSphere through the Test
  Gain plugin at ×0.5 into cable 2, recorded by PortAudio:** rms 0.07071, exactly half.
- **ffmpeg** recording cable 1 through DirectShow what ToneSphere played into it: 1000.00 Hz,
  +0.000 dB (it took 14.2 s to record 3 s in the VM's non-interactive session — the reason an
  earlier version of the test, whose tone lasted 5 s, heard silence).
- **Idle is exact silence**, and **a new capture does not replay** audio played before it.
- **Managed through the app** (the calls the dialog makes): a cable "Chat" added
  (`ROOT\MEDIA\0002`), carrying audio, renamed "Game" — its endpoints renamed with it —
  and uninstalled, with cable 1 still carrying audio; the dialog's own Disable and Enable
  buttons disabling and re-enabling cable 2.
- **Disappearing under a running engine:** ToneSphere captured cable 2 and played into
  cable 1; cable 2 was disabled. The engine reported both of its endpoints gone, started
  again without them, and cable 1 went on carrying a 440 Hz tone. Cable 2 was enabled: the
  engine reported it back and reopened it, and a tone another program then played into it
  reached ToneSphere at 1000.00 Hz, −0.009 dB — with no user action.
- **A cable in use elsewhere:** disabled under a PortAudio recorder; refused cleanly while a
  second native-engine program held it, the cable still working; disabled and enabled once
  that program let go.
- **Uninstall:** every cable removed, driver-store package deleted, certificate no longer
  trusted, no ToneSphere device or endpoint left.

Earlier passes each found something real that nothing else would have:

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
8. **A second cable failed to start** (`STATUS_DEVICE_BUSY`, code 10): the sample allows one
   device per driver. The guard is gone, and each device has its own cable.
9. A name set before the driver was installed was replaced by the INF's; cables are named
   after installing, then restarted so the endpoints are built under the name.
10. **Disabling a cable ToneSphere was streaming through left it "pending a restart"**, and
    every later change to it failed; see "A cable in use" above.
11. A disable that Windows vetoed with the persist flag still recorded the disable for the
    next restart, and the cable then refused to be enabled; the persistent disable is now
    asked for only once the device has stopped.
12. The two endpoints of a cable leave and arrive some hundreds of milliseconds apart, so a
    re-read straight after a change saw half a cable; the app waits for both.
13. **Stopping a WASAPI stream on a cable ASIO4ALL was driving never returned.** Each stream
    thread waited on `{device event, stop event}`, and `WaitForMultipleObjects` reports the
    lowest-index handle signalled, so a device event that never stopped firing hid the stop
    for ever. Stop now comes first (`docs/ASIO.md`). On the way: the capture threads' packet
    drain is bounded, and the ASIO host gives a driver stuck in `stop()` five seconds, then
    abandons it and says so.
14. ASIO4ALL stays loaded in a process after release, and its left-over state disturbed later
    WASAPI measurements on the same cable; the runner runs each test file in its own process.

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
the tree, the built binaries and the third-party tools it finds in `-Tools` (ASIO4ALL and
ffmpeg, downloaded once by hand), installs the driver and the two default cables, installs
ASIO4ALL silently, runs `tests/hardware/test_virtual_driver.py`, `test_asio.py` and
`test_roundtrip.py` against the cables, uninstalls, checks nothing is left, and brings every
log back (including `setupapi.dev.log` and any crash dump).

`driver_install.ps1` refuses to run with test-signing off. `driver_uninstall.ps1` stops each
cable first (retrying while the audio engine has it open for a moment), removes every cable,
the driver-store package and the certificate trust, and fails if anything is left.

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
4. **Windows kernel driver** — installs two cables, each carrying audio between applications
   at the level sent and isolated from the other; cables are added, renamed, disabled,
   enabled and uninstalled from ToneSphere; all in a Hyper-V test VM; untested on a real
   desktop with a real communications app; production signing not available.

ToneSphere says exactly that, and nothing more, until a signing route exists.
