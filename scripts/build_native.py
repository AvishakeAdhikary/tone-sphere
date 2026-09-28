"""
Build the native DLLs into `tonesphere/native/_bin/`.

    uv run python scripts/build_native.py            # Release
    uv run python scripts/build_native.py --debug

Finds a compiler in this order, and says which it used:
  1. `cl.exe` already on PATH (a developer prompt, or CI after msvc-dev-cmd).
  2. The Enterprise WDK: `TONESPHERE_EWDK` (its root, e.g. `D:\\`), any mounted drive with
     `BuildEnv\\SetupBuildEnv.cmd`, or an EWDK ISO in `TONESPHERE_EWDK_ISO` / `C:\\SDKs\\`,
     which is mounted (no admin needed).
  3. A Visual Studio / Build Tools install found by vswhere.
CMake and Ninja come from the `native` dependency group, i.e. this interpreter's Scripts
directory, so nothing else needs installing. See docs/BUILDING_WINDOWS.md.
"""

import argparse
import os
import shutil
import string
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "native"
BUILD = NATIVE / "build"
OUTPUT = ROOT / "tonesphere" / "native" / "_bin"


def capture_env(batch: str, *args: str) -> dict[str, str]:
    command = f'call "{batch}" {" ".join(args)} >nul 2>&1 && set'
    # A string, not a list: list2cmdline would escape the quotes as \" , which cmd.exe does
    # not understand. /s makes cmd strip exactly the outer pair and keep the rest verbatim.
    result = subprocess.run(f'cmd /d /s /c "{command}"', capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        sys.exit(f"{batch} failed:\n{result.stdout}\n{result.stderr}")
    env = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            env[key] = value
    return env


def find_ewdk_root() -> Path | None:
    explicit = os.environ.get("TONESPHERE_EWDK")
    if explicit:
        return Path(explicit)
    for letter in string.ascii_uppercase:
        candidate = Path(f"{letter}:\\")
        if (candidate / "BuildEnv" / "SetupBuildEnv.cmd").is_file():
            return candidate
    iso = os.environ.get("TONESPHERE_EWDK_ISO")
    isos = [Path(iso)] if iso else sorted(Path("C:/SDKs").glob("EWDK*.iso"))
    for image in isos:
        if not image.is_file():
            continue
        print(f"mounting {image} ...")
        mounted = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Mount-DiskImage -ImagePath '{image}' -PassThru | Get-Volume).DriveLetter"],
            capture_output=True, text=True,
        )
        letter = mounted.stdout.strip()
        if mounted.returncode == 0 and len(letter) == 1:
            return Path(f"{letter}:\\")
        print(f"  could not mount: {mounted.stderr.strip()}")
    return None


def find_vs_env() -> dict[str, str] | None:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None
    found = subprocess.run(
        [str(vswhere), "-latest", "-products", "*", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
        capture_output=True, text=True,
    ).stdout.strip()
    if not found:
        return None
    vcvars = Path(found) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    return capture_env(str(vcvars)) if vcvars.is_file() else None


def add_ewdk_sdk_paths(env: dict[str, str], root: Path) -> None:
    """
    The EWDK runs vsdevcmd with `-winsdk=none` — its own driver builds find the SDK through
    MSBuild properties instead — so a plain cl/link invocation sees no Windows SDK at all.
    User-mode DLLs need it on INCLUDE/LIB/PATH, from the same kit the driver build uses.
    """
    kit = root / "Program Files" / "Windows Kits" / "10"
    version = env.get("Version_Number") or sorted(p.name for p in (kit / "Include").glob("10.*"))[-1]
    include = kit / "Include" / version
    lib = kit / "Lib" / version
    extra_include = [include / part for part in ("ucrt", "um", "shared", "winrt", "cppwinrt")]
    extra_lib = [lib / "ucrt" / "x64", lib / "um" / "x64"]
    env["INCLUDE"] = ";".join([str(p) for p in extra_include] + [env.get("INCLUDE", "")])
    env["LIB"] = ";".join([str(p) for p in extra_lib] + [env.get("LIB", "")])
    env["PATH"] = f"{kit / 'bin' / version / 'x64'};{env.get('PATH', '')}"


def toolchain_env() -> tuple[dict[str, str], str]:
    if shutil.which("cl"):
        return dict(os.environ), "cl.exe already on PATH"
    root = find_ewdk_root()
    if root:
        env = capture_env(str(root / "BuildEnv" / "SetupBuildEnv.cmd"), "amd64")
        add_ewdk_sdk_paths(env, root)
        stamp = root / "Version.txt"
        version = stamp.read_text(errors="replace").strip() if stamp.is_file() else "?"
        return env, f"EWDK at {root} ({version})"
    env = find_vs_env()
    if env:
        return env, "Visual Studio Build Tools (vswhere)"
    sys.exit("No C++ toolchain found. Mount the EWDK ISO or set TONESPHERE_EWDK; see docs/BUILDING_WINDOWS.md.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--clean", action="store_true", help="delete the build directory first")
    args = parser.parse_args()

    if sys.platform != "win32":
        sys.exit("The native engine is Windows-only for now (see AGENTS.md).")

    env, origin = toolchain_env()
    tools = Path(sys.executable).parent
    env["PATH"] = f"{tools};{env.get('PATH', '')}"
    print(f"toolchain: {origin}")

    config = "Debug" if args.debug else "Release"
    build_dir = BUILD / config.lower()
    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)

    configure = [
        "cmake", "-S", str(NATIVE), "-B", str(build_dir), "-G", "Ninja",
        f"-DCMAKE_BUILD_TYPE={config}",
        f"-DTS_OUTPUT_DIR={OUTPUT.as_posix()}",
        "-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl",
    ]
    subprocess.run(configure, env=env, check=True)
    subprocess.run(["cmake", "--build", str(build_dir)], env=env, check=True)

    for built in sorted(OUTPUT.glob("*.dll")):
        print(f"built {built.relative_to(ROOT)} ({built.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
