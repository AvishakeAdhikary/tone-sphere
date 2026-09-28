"""
Fetch the Steinberg SDKs the native build needs into `sdks/`.

Both are pinned: the VST3 SDK by git tag, the ASIO SDK by the SHA-256 of Steinberg's
official zip. Neither is ever committed (see `sdks/README.md` for why and under which
licences). Run again at any time; an SDK already present at the pinned version is left
alone.

    uv run python scripts/fetch_sdks.py            # both
    uv run python scripts/fetch_sdks.py vst3       # just one
"""

import argparse
import hashlib
import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SDKS = ROOT / "sdks"

VST3_REPO = "https://github.com/steinbergmedia/vst3sdk.git"
VST3_TAG = "v3.8.1_build_84"

# https://www.steinberg.net/asiosdk redirects here. The checksum is of the zip as
# downloaded on 2026-09-29; a different file behind the same name is refused rather than
# silently built against.
ASIO_URL = "https://download.steinberg.net/sdk_downloads/ASIO-SDK_2.3.4_2025-10-15.zip"
ASIO_SHA256 = "d5ebf0c20dd2c5f43771fd0c1418f4b361bf52434ee670097cfa6b3a335e2eca"
ASIO_VERSION = "2.3.4"


def fetch_vst3() -> Path:
    target = SDKS / "vst3sdk"
    stamp = target / ".tonesphere-tag"
    if stamp.is_file() and stamp.read_text().strip() == VST3_TAG:
        print(f"vst3sdk {VST3_TAG} already present")
        return target

    if target.exists():
        shutil.rmtree(target)
    print(f"cloning vst3sdk {VST3_TAG} (with submodules) ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", VST3_TAG,
         "--recurse-submodules", "--shallow-submodules", VST3_REPO, str(target)],
        check=True,
    )
    stamp.write_text(VST3_TAG + "\n")
    return target


def fetch_asio() -> Path:
    target = SDKS / "asiosdk"
    stamp = target / ".tonesphere-sha256"
    if stamp.is_file() and stamp.read_text().strip() == ASIO_SHA256:
        print(f"ASIO SDK {ASIO_VERSION} already present")
        return target

    print(f"downloading ASIO SDK {ASIO_VERSION} from {ASIO_URL} ...")
    request = urllib.request.Request(ASIO_URL, headers={"User-Agent": "tonesphere-fetch-sdks"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()

    digest = hashlib.sha256(payload).hexdigest()
    if digest != ASIO_SHA256:
        sys.exit(f"ASIO SDK checksum mismatch: expected {ASIO_SHA256}, got {digest}. "
                 f"Steinberg may have published a new file; review its licence before "
                 f"updating the pin in {Path(__file__).name}.")

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        licence = archive.read("ASIOSDK/LICENSE.txt").decode("utf-8", "replace")
        if "General Public License (GPL) Version 3" not in licence:
            sys.exit("The ASIO SDK licence no longer offers GPLv3; native/asio/ relies on that "
                     "option. Stop and review the licence before building.")
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        prefix = "ASIOSDK/"
        for member in archive.infolist():
            if not member.filename.startswith(prefix) or member.is_dir():
                continue
            out = target / member.filename[len(prefix):]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(archive.read(member))

    stamp.write_text(ASIO_SHA256 + "\n")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", nargs="*", choices=["vst3", "asio"])
    which = parser.parse_args().which or ["vst3", "asio"]

    SDKS.mkdir(exist_ok=True)
    if "vst3" in which:
        print(f"-> {fetch_vst3()}")
    if "asio" in which:
        print(f"-> {fetch_asio()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
