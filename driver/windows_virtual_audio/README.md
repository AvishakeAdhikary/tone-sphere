# ToneSphere virtual audio driver (Windows)

A kernel-mode audio driver that publishes a virtual audio cable Windows itself enumerates:

- **ToneSphere Cable Input**: a render endpoint. Applications play into it.
- **ToneSphere Cable Output**: a capture endpoint. Applications record from it, and hear
  what was played into the input.

Both endpoints run one fixed format, 48 kHz, 32-bit PCM, stereo, so the cable copies bytes
and never converts. ToneSphere uses the endpoints through its ordinary WASAPI backend,
with no private kernel/user channel. In the most common setup, ToneSphere renders its mix
into Cable Input and Discord or OBS records Cable Output as a microphone. In the other
direction, an application plays into Cable Input and ToneSphere captures Cable Output.

## Where it comes from, and its licence

This directory is Microsoft's **SimpleAudioSample** from
[Windows-driver-samples](https://github.com/microsoft/Windows-driver-samples)
(`audio/simpleaudiosample`, commit `2dc3fd3a0cc84a2933f2194e7ec0871584979071`), under the
**Microsoft Public License** (`LICENSE`). It is not MIT like the rest of ToneSphere, and it
is a separate binary. Modifications are marked `ToneSphere:` in the source. All of them:

| File | Change |
|---|---|
| `Source/Main/cable.cpp`, `cable.h` | **New.** The cable: a nonpaged ring (100 ms bound, oldest audio dropped first) between the render stream and the capture stream, under a spin lock |
| `Source/Main/minwavertstream.cpp` | Render: the data-file writer is replaced by `CableWrite`. Capture: the test-tone generator is replaced by `CableRead` |
| `Source/Main/adapter.cpp` | The cable is allocated in `DriverEntry` and freed in `DriverUnload`. The `DoNotCreateDataFiles` registry override is removed, so the driver never writes render audio to disk |
| `Source/Filters/speakerwavtable.h` | The render endpoint moves from 16-bit to 32-bit PCM, matching the capture endpoint |
| `Source/Main/SimpleAudioSample.inx` | Names: provider and manufacturer "Neural Nexus Studios", device "ToneSphere Virtual Audio Cable", endpoints "ToneSphere Cable Input/Output", hardware ID `ROOT\ToneSphereVirtualAudio`, service `ToneSphereVirtualAudio` |
| `Source/Main/Main.vcxproj` | Binary name `ToneSphereVirtualAudio.sys`; `cable.cpp` added |

Everything else is Microsoft's code unchanged: the topology and WaveRT miniports, the
QPC-driven position logic, power management and the INF structure. File names keep
`SimpleAudioSample` so that the diff against upstream stays readable.

## Build

```
uv run python scripts/build_driver.py
```

This needs the EWDK (see `docs/BUILDING_WINDOWS.md`). The package, test-signed with the
WDK's own test certificate, lands in `x64/Release/package/`. A kit for a test VM lands in
`x64/Release/vm_kit/`: the package, the test certificate, `devcon.exe`, and the
install/uninstall scripts. `InfVerif /w` (Windows Driver requirements) passes on the
built INF.

## Install: in a test VM only

1. Use a Hyper-V Generation 2 VM running Windows 11, with Secure Boot **off** in the VM's
   settings.
2. Inside the VM, in an elevated prompt, run `bcdedit /set testsigning on`, then reboot.
3. Copy `vm_kit/` into the VM. In an elevated PowerShell, run `.\driver_install.ps1`. It
   refuses to run if test-signing is off. It then trusts the test certificate, installs the
   root-enumerated device, and lists the device and endpoints it created.
4. To remove everything, run `.\driver_uninstall.ps1`. It removes the device, deletes the
   package from the driver store, stops trusting the test certificate, and checks that
   nothing is left.

**Never on a machine anyone depends on.** A kernel driver runs with the operating system's
privileges, and a fault in it is a blue screen.

## Production signing

**Not available.** Windows loads a kernel driver on a normal machine only if Microsoft has
signed it. For a driver a user installs themselves, that means attestation signing through
the Partner Center hardware program, which requires an EV code-signing certificate. This
project has neither. Until it does, the driver exists only for development and testing,
and nothing tells users to disable Secure Boot or enable test-signing. An MSIX package
cannot install a kernel driver either, so distribution would need its own installer.
