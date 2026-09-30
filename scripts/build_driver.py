"""
Build the ToneSphere virtual audio driver and assemble a kit for installing it in a test VM.

    uv run python scripts/build_driver.py

Output, in driver/windows_virtual_audio/x64/Release/:
  package/   the driver package (INF, SYS, CAT), test-signed with the WDK's test certificate
  vm_kit/    everything a test VM needs: the package, the test certificate, devcon.exe, and
             driver_install.ps1 / driver_uninstall.ps1

This machine never installs it. AGENTS.md: test-signed drivers are installed only inside a
Hyper-V VM with test-signing on. Production distribution needs attestation signing, which
needs an EV certificate this project does not have (docs/VIRTUAL_AUDIO_DRIVER.md).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_native import capture_env, find_ewdk_root  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DRIVER = ROOT / "driver" / "windows_virtual_audio"
OUT = DRIVER / "x64" / "Release"


def main() -> int:
    ewdk = find_ewdk_root()
    if ewdk is None:
        sys.exit("The driver needs the WDK: mount the EWDK (see docs/BUILDING_WINDOWS.md).")
    env = capture_env(str(ewdk / "BuildEnv" / "SetupBuildEnv.cmd"), "amd64")
    # stampinf dates DriverVer in local time and inf2cat checks it against UTC unless told
    # otherwise, so east of UTC, between local midnight and UTC midnight, the package fails
    # as "postdated".
    subprocess.run(["msbuild", str(DRIVER / "SimpleAudioSample.sln"), "/p:Configuration=Release",
                    "/p:Platform=x64", "/p:Inf2CatUseLocalTime=true", "/m", "/v:minimal", "/nologo"],
                   env=env, check=True, shell=True)

    package = OUT / "package"
    sys_file = package / "ToneSphereVirtualAudio.sys"
    if not sys_file.is_file():
        sys.exit(f"build finished but {sys_file} is missing")

    kit = OUT / "vm_kit"
    if kit.exists():
        for f in kit.iterdir():
            f.chmod(0o666)
        shutil.rmtree(kit)
    kit.mkdir(parents=True)
    for f in package.iterdir():
        shutil.copy2(f, kit / f.name)

    # The public half of the WDK test certificate that signed the package: the VM must
    # trust it, and only the VM.
    cer = kit / 'ToneSphereTestSigning.cer'
    # Without PSModulePath: Windows PowerShell started from PowerShell 7 inherits 7's module
    # path and then cannot load its own Get-AuthenticodeSignature.
    clean = {k: v for k, v in os.environ.items() if k.upper() != 'PSMODULEPATH'}
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"$c = (Get-AuthenticodeSignature '{sys_file}').SignerCertificate; "
                    f"[IO.File]::WriteAllBytes('{cer}', $c.Export('Cert'))"], check=True, env=clean)

    tools = ewdk / "Program Files" / "Windows Kits" / "10" / "Tools"
    devcon = sorted(tools.glob("*/x64/devcon.exe"))
    if not devcon:
        sys.exit("devcon.exe not found in the EWDK tools")
    # copyfile, not copy2: the EWDK's files are read-only, and a read-only copy in the kit
    # stops the next build from replacing the kit.
    shutil.copyfile(devcon[-1], kit / "devcon.exe")
    for script in ("driver_install.ps1", "driver_uninstall.ps1"):
        shutil.copy2(ROOT / "scripts" / script, kit / script)

    print(f"driver kit: {kit}")
    for f in sorted(kit.iterdir()):
        print(f"  {f.name} ({f.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
