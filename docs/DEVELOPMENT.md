---
title: Development
---

# Developing ToneSphere

Everything needed to continue work on ToneSphere is in this repository: the code, the build and
setup scripts, the test VM automation, the release pipeline, and the record of what is verified
([STATUS.md](STATUS.md)). Nothing depends on files that only exist on one PC. This page is the
way back in after a fresh clone.

## 1. From a fresh clone to a working build (Windows)

```powershell
git clone https://github.com/AvishakeAdhikary/tone-sphere.git
cd tone-sphere
.\scripts\setup-dev.ps1          # elevated PowerShell; installs the toolchain (see below)
.\scripts\build.ps1 -Test        # engine, ASIO driver, tools, test plugins, unit tests (x64)
.\scripts\build.ps1 -Arch x86    # 32-bit ASIO driver
cd ui; npm run build; npx playwright test   # UI + end-to-end tests against the mock engine
```

`setup-dev.ps1` installs, with winget:

| Tool | Why |
|---|---|
| Visual Studio 2022 Build Tools: C++, Windows SDK 10.0.26100, CMake, Ninja | Engine, ASIO driver, tools (`scripts/build.ps1` finds it with `vswhere`) |
| Windows Driver Kit 10.0.26100 | The virtual audio driver (`drivers/win/vad/build.ps1`) |
| Node.js LTS (22+), Git | UI, tests, release scripts |
| FlexASIO + `%USERPROFILE%\FlexASIO.toml` | ASIO-host integration tests (a silent VB-CABLE loopback) |

Two things it can't do for you:

- **VB-CABLE**, the audio loopback the integration tests play through. Its licence forbids
  scripted installs: install it from <https://vb-audio.com/Cable/>, or from the app
  (Settings › Devices › Install drivers).
- **Hyper-V**, for the driver test VM: `Enable-WindowsOptionalFeature -Online -FeatureName
  Microsoft-Hyper-V-All` (needs a restart).

Third-party SDKs are not committed except the ASIO SDK (`third_party/asiosdk`). CMake downloads
the rest at configure time, pinned to exact versions (VST3 SDK, RNNoise, Catch2, nlohmann/json).

## 2. Linux (or WSL)

```bash
scripts/setup-dev.sh             # packages (PipeWire, CMake, Ninja, compiler) + Node.js
scripts/build-linux.sh           # engine + unit tests into ~/ts-build
scripts/package-linux.sh         # AppImage into ui/release/
node tests/integration/linux-pipewire.mjs   # from Windows: the Linux engine inside WSL
```

## 3. The driver test VM

Kernel-driver work happens **only** in the Hyper-V VM, never on the development PC.

```powershell
.\vm\New-TestVM.ps1              # downloads the Windows 11 Enterprise evaluation ISO (~8 GB) and builds the VM
.\vm\Test-DriverInstall.ps1      # "Install drivers" path in a pristine VM
.\vm\Test-InVM.ps1               # virtual audio driver end to end
.\vm\Test-Installer.ps1          # the desktop installer on a clean Windows
.\vm\Test-Soak.ps1               # 1 h under Driver Verifier
```

The VM's local administrator password is generated on first use and kept **outside the
repository**, in `%LOCALAPPDATA%\ToneSphereVM\vm-password.txt` (`vm/vmcommon.ps1`). The VM can
always be rebuilt; nothing in it needs keeping. `Restore-TestVM` reverts to the `clean`
checkpoint.

## 4. Local-only material (never committed)

| What | Made by | Notes |
|---|---|---|
| Driver test-signing certificate (`CN=ToneSphere Test Driver Signing`) | `drivers/win/vad/build.ps1`, on first run | In `Cert:\CurrentUser\My`; the public `.cer` lands next to the driver package |
| MSIX test certificate | `packaging/store/README.md` explains how to make one | Only for sideloading a test MSIX into the VM |
| VM password | `vm/vmcommon.ps1` | See above |
| Build outputs, `node_modules`, VM disks and ISOs | the scripts | `.gitignore` keeps them out |

`*.pfx`, `*.cer` and the VM password file are git-ignored, and CI runs a secret scan
(gitleaks) over the whole history on every push.

## 5. Releasing

There is nothing to do by hand. Every push to `main` runs [CI](../.github/workflows/ci.yml); when
it is green, CI tags `v<version>` and publishes the GitHub release with the Windows installer,
the portable zip, the unsigned Store package, the Linux AppImage, the complete source and
checksums.

- The version is the latest `v*` tag plus one patch. Put `[minor]` or `[major]` in the commit
  message to bump more. `VERSION` is the floor: raise it to start a new series.
- `node scripts/release-version.mjs next` prints what the next release would be.
- Submitting to the Microsoft Store (Partner Center) stays a manual step for the owner:
  [packaging/store/README.md](../packaging/store/README.md).

## 6. Where things are

- [AGENTS.md](../AGENTS.md): the working rules (safety, architecture, style, testing, commits).
- [STATUS.md](STATUS.md): what is verified, with evidence, and what is pending.
- [shared/protocol/PROTOCOL.md](../shared/protocol/PROTOCOL.md): the engine's API.
- `git log`: why things are the way they are. Commit messages say what changed and why.
