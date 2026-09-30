# ToneSphere virtual audio driver (Windows)

A kernel-mode audio driver that publishes virtual audio cables Windows itself enumerates.
Each cable is its own device — an instance of the root-enumerated hardware ID
`ROOT\ToneSphereVirtualAudio` — with two endpoints:

- **Speakers (<cable name>)**: a render endpoint. Applications play into it.
- **Microphone Array (<cable name>)**: a capture endpoint. Applications record from it, and
  hear what was played into that cable's render endpoint — and nothing from any other cable.

A fresh install creates two, "ToneSphere Cable 1" and "ToneSphere Cable 2"; ToneSphere's
Virtual Cables dialog adds more (up to eight), renames, disables, enables and uninstalls them
one at a time (`tonesphere/engine/virtual_cables.py`). Because each cable is a device of its
own, Windows' own disable and uninstall act on exactly one cable.

The endpoint names are Windows' own: the pin category's generic name, then the device's
friendly name, which is the cable's name. The INF's "ToneSphere Cable Input/Output" pin
names do not reach the endpoint name: that takes a `MediaCategories` registration, which the
Windows Driver INF rules forbid (InfVerif error 1321), or a `KSPROPERTY_PIN_NAME` handler in
the driver.

Both endpoints run one fixed format, 48 kHz, 32-bit PCM, stereo, so the cable copies bytes
and never converts. ToneSphere uses the endpoints through its ordinary WASAPI backend,
with no private kernel/user channel. In the most common setup, ToneSphere renders its mix
into the cable's render endpoint and Discord or OBS records its capture endpoint as a
microphone. In the other direction, an application plays into the cable and ToneSphere
captures the other end.

## Where it comes from, and its licence

This directory is Microsoft's **SimpleAudioSample** from
[Windows-driver-samples](https://github.com/microsoft/Windows-driver-samples)
(`audio/simpleaudiosample`, commit `2dc3fd3a0cc84a2933f2194e7ec0871584979071`), under the
**Microsoft Public License** (`LICENSE`). It is not MIT like the rest of ToneSphere, and it
is a separate binary. Modifications are marked `ToneSphere:` in the source. All of them:

| File | Change |
|---|---|
| `Source/Main/cable.cpp`, `cable.h` | **New.** A cable: a nonpaged ring (100 ms bound, oldest audio dropped first) between a render stream and a capture stream, under a spin lock; `CableCreate`/`CableDestroy` make one per device; `CableFlush` empties it |
| `Source/Main/common.cpp`, `Inc/common.h` | The adapter creates its device's cable in `Init` and destroys it last (`GetCable`). The sample's one-device-per-driver guard is removed — it failed every device after the first with `STATUS_DEVICE_BUSY` — and `CSaveData`'s static state is reference-counted across devices |
| `Source/Main/minwavertstream.cpp` | Render: the data-file writer is replaced by `CableWrite` on the stream's own device's cable (the miniport's device context), called on every position update (the sample only called its writer when data files were enabled, which left the cable empty). Capture: the test-tone generator is replaced by `CableRead`, and a capture stream entering RUN empties the cable, so a new recording never starts with audio played before it |
| `Source/Main/adapter.cpp` | Each device's cable is handed to its miniports as their device context. Data files are forced off, whatever the registry says, so the driver never writes render audio to disk and the devices never share a file writer |
| `Source/Filters/speakerwavtable.h` | The render endpoint moves from 16-bit to 32-bit PCM, matching the capture endpoint |
| `Source/Main/SimpleAudioSample.inx` | Names: provider and manufacturer "Neural Nexus Studios", device "ToneSphere Virtual Audio Cable", pin names "ToneSphere Cable Input/Output" (which Windows does not show; see above), hardware ID `ROOT\ToneSphereVirtualAudio`, service `ToneSphereVirtualAudio` |
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

`scripts/vm/new_driver_vm.ps1` builds that VM from a Windows 11 ISO, and
`scripts/vm/run_driver_tests.ps1` runs a whole install/test/uninstall pass in it
(`docs/VIRTUAL_AUDIO_DRIVER.md`, "Testing it"). By hand:

1. Use a Hyper-V Generation 2 VM running Windows 11, with Secure Boot **off** in the VM's
   settings.
2. Inside the VM, in an elevated prompt, run `bcdedit /set testsigning on`, then reboot.
3. Copy `vm_kit/` into the VM. In an elevated PowerShell, run `.\driver_install.ps1`. It
   refuses to run if test-signing is off. It then trusts the test certificate and adds the
   package to the driver store (`pnputil /add-driver`). The cables themselves are created by
   ToneSphere: `uv run python main.py cable-admin install-cables` makes the two default
   ones, and the Virtual Cables dialog manages them from then on.
4. To remove everything, run `.\driver_uninstall.ps1`. It removes every cable, deletes the
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
