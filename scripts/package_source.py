"""
Build the Corresponding Source archive that goes out with every Windows release.

    uv run python scripts/package_source.py dist/ToneSphere-windows-source.zip

Why it exists: a Windows build bundles `tonesphere_asio.dll`, compiled from Steinberg's ASIO
SDK under GPLv3, so the released executable is distributed under GPLv3 as a whole
(`docs/ASIO.md`). GPLv3 section 6 then requires the Corresponding Source to go with it, and
the "Source code" archive GitHub attaches to a tag is not enough: it has neither SDK, because
neither is committed. Pointing at Steinberg's download page is not enough either, since
nothing obliges that URL to stay up for as long as the binary is offered.

The archive holds: every file tracked at this commit; the ASIO SDK exactly as fetched and
checksummed by `scripts/fetch_sdks.py`; the parts of the VST3 SDK the native build compiles
(base, pluginterfaces, public.sdk, cmake, and the top-level CMakeLists and licence; not the
154 MB of documentation or VSTGUI, which the build switches off); the GPLv3 text; and a
note saying how to rebuild and where the pinned Python dependencies come from.
"""

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VST3_PARTS = ("base", "pluginterfaces", "public.sdk", "cmake", "CMakeLists.txt", "LICENSE.txt", "README.md")

NOTE = """ToneSphere - Corresponding Source for the Windows build
=======================================================

This archive is the Corresponding Source, under GNU GPL version 3 section 6, of the
ToneSphere Windows executable it was published with. That executable includes
tonesphere_asio.dll, built from Steinberg's ASIO SDK under GPLv3, and is therefore
distributed under GPLv3 as a whole (LICENSE-GPL-3.0.txt). ToneSphere's own source files
remain under the MIT licence (LICENSE); the ASIO host in native/asio/ is GPLv3.

  tonesphere/ ...        the repository at commit {commit}
  sdks/asiosdk/          Steinberg ASIO SDK {asio}, GPLv3 (sdks/asiosdk/LICENSE.txt)
  sdks/vst3sdk/          Steinberg VST3 SDK, the parts the build compiles, MIT (LICENSE.txt)

Rebuild: docs/BUILDING_WINDOWS.md. Extract to a short path, such as C:/src: under a deep
directory the VST3 SDK's object paths exceed Windows' 260-character limit. The SDKs are
already in place, so scripts/fetch_sdks.py will find them and download nothing. Python
dependencies are pinned exactly in uv.lock and installed from PyPI by `uv sync`; the
executable is produced by `uv run pyinstaller tonesphere.spec` with ONEFILE=1.
"""


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    return [f for f in out.stdout.decode("utf-8").split("\0") if f]


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    target = Path(sys.argv[1]).resolve()
    asio = ROOT / "sdks" / "asiosdk"
    vst3 = ROOT / "sdks" / "vst3sdk"
    if not (asio / ".tonesphere-sha256").is_file() or not (vst3 / "CMakeLists.txt").is_file():
        sys.exit("the SDKs are missing: run scripts/fetch_sdks.py first (the source must be what was built)")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                            check=True).stdout.strip()
    sys.path.insert(0, str(ROOT / "scripts"))
    from fetch_sdks import ASIO_VERSION

    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in tracked_files():
            path = ROOT / name
            if path.is_file():
                archive.write(path, f"tonesphere/{name}")
                count += 1
        for path in sorted(asio.rglob("*")):
            if path.is_file():
                archive.write(path, f"tonesphere/sdks/asiosdk/{path.relative_to(asio).as_posix()}")
                count += 1
        for part in VST3_PARTS:
            root = vst3 / part
            files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
            for path in files:
                if ".git" in path.relative_to(vst3).parts:
                    continue
                archive.write(path, f"tonesphere/sdks/vst3sdk/{path.relative_to(vst3).as_posix()}")
                count += 1
        archive.write(ROOT / "native" / "asio" / "LICENSE", "LICENSE-GPL-3.0.txt")
        archive.writestr("SOURCE.txt", NOTE.format(commit=commit, asio=ASIO_VERSION))
    print(f"{target} ({target.stat().st_size / 2**20:.1f} MiB, {count} files, commit {commit[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
