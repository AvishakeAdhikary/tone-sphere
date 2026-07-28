# Virtual audio driver — what it takes

This is the one claim ToneSphere has never been able to make good on, and the one people
most want: a device called **ToneSphere Input 1** that appears in Discord's, OBS's or your
DAW's device list. This document says exactly what that requires, because the original
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

## What already covers most of the need

Before building a driver, note what does not need one:

| Want | Needs a driver? |
|---|---|
| Guitar in → effects → headphones out | **No.** Works today. |
| Guitar in → VST3 (Guitar Rig) → out | **No.** Works today via `pedalboard`. |
| Record everything playing on the system | **No.** WASAPI loopback captures any render endpoint. |
| Capture one specific application's audio | **No**, on Windows 10 build 20348+ — see below. |
| Send ToneSphere's output *into* Discord as a microphone | **Yes.** |
| Appear as an input device in a DAW | **Yes.** |

The last two are the real driver use cases. Everything above them is achievable without
one, which is why they were done first.

### Per-application capture without a driver

Windows 10 build 20348 and later can capture a single process's render stream through
`ActivateAudioInterfaceAsync` with `AUDIOCLIENT_ACTIVATION_PARAMS` set to
`VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK`. This replaces the most common reason people
install VB-CABLE.

PortAudio does not expose it, so it needs a small native extension calling that API
directly and feeding frames into a bus. `engine/app_capture.py` already detects whether the
platform supports it and reports `process_loopback_implemented: False` rather than
pretending. This is the highest-value item remaining and it is much smaller than a driver.

## The driver itself

### Starting point

Microsoft's **SysVAD** sample in
[microsoft/Windows-driver-samples](https://github.com/microsoft/Windows-driver-samples/tree/main/audio/sysvad)
is a complete WDM/PortCls virtual audio device. It already registers render and capture
endpoints; the work is trimming it to what ToneSphere needs and adding a channel between
kernel and user mode.

### What has to be built

1. **Endpoint topology.** How many cables, how many channels, which formats. Fixing the
   format at 48 kHz/32-bit float and letting the user-mode side resample avoids a class of
   negotiation bugs.

2. **A kernel↔user transport.** The driver's capture endpoint has to be fed by ToneSphere.
   A shared ring buffer mapped into both address spaces plus an event to signal readiness
   is the standard approach; IOCTLs per buffer are simpler and too slow.

3. **Clock and drift.** The virtual device has no crystal. It either derives timing from a
   real device or free-runs, and either way it drifts against the hardware. The existing
   `DriftResampler` handles the user-mode side, but the driver has to expose a position
   register that is honest about it.

4. **Installation.** An INF, plus a way to install and uninstall cleanly. A virtual audio
   driver that cannot be removed is worse than no driver at all.

### Signing, and what it costs

This is the part that is money rather than effort. Windows will not load an unsigned
kernel driver on a normal machine.

| Item | Cost | Notes |
|---|---|---|
| EV code-signing certificate | ~£250–400/year | Hardware token; identity verification takes days to weeks |
| Partner Center account | $19 one-off (individual) | Needed to submit for attestation signing |
| Attestation signing | Free | Microsoft signs the driver; needs the EV cert to submit |
| WHQL/HLK certification | Free to submit | Only needed for Windows Update distribution; a lab run is a significant effort |

Attestation signing is enough for a driver a user installs deliberately. WHQL is only
required to ship through Windows Update.

For development, `bcdedit /set testsigning on` allows a self-signed driver, but it puts a
watermark on the desktop and cannot be asked of users.

### Effort

Assuming familiarity with kernel-mode Windows: a few weeks for a working single-cable
driver, considerably longer to make it robust across sleep, device changes, format
negotiation and uninstall. Without that familiarity, treat it as a project in its own
right rather than a feature.

## macOS and Linux

**macOS** needs an AudioServerPlugIn — a user-space bundle in
`/Library/Audio/Plug-Ins/HAL`, no kernel code and no paid certificate for local
installation, though notarisation is needed for distribution. Substantially easier than
Windows. BlackHole is the reference implementation and is MIT-licensed.

**Linux** does not need one. PipeWire and JACK already provide arbitrary virtual endpoints,
and `pactl load-module module-null-sink` creates one in a single command. The right move on
Linux is to use the platform rather than fight it.

## Recommended order

1. **Per-process WASAPI loopback capture.** No driver, no certificate, covers the most
   common use case. Do this first.
2. **macOS AudioServerPlugIn.** No kernel code, no EV certificate for local use.
3. **Linux via PipeWire/JACK.** Configuration, not code.
4. **Windows WDM driver.** Only once the above are done and the certificate is funded.

Until step 4 exists, ToneSphere should keep saying so. That is the whole point of this
document.
