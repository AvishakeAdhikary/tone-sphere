"""
Fetch the third-party sources the native build needs into `sdks/`.

All are pinned: the VST3 SDK by git tag, the ASIO SDK and libopus by the SHA-256 of their
official archives. None is ever committed (see `sdks/README.md` for why and under which
licences). Run again at any time; a source already present at the pinned version is left
alone.

    uv run python scripts/fetch_sdks.py            # all
    uv run python scripts/fetch_sdks.py vst3       # just one
"""

import argparse
import hashlib
import io
import shutil
import subprocess
import sys
import tarfile
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

# Xiph's release tarball (BSD 3-clause). Checksum verified against the download on
# 2026-09-30 and against the one Debian's opus_1.5.2.orig.tar.gz carries.
OPUS_URL = "https://downloads.xiph.org/releases/opus/opus-1.5.2.tar.gz"
OPUS_SHA256 = "65c1d2f78b9f2fb20082c38cbe47c951ad5839345876e46941612ee87f9a7ce1"
OPUS_VERSION = "1.5.2"


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


def fetch_opus() -> Path:
    target = SDKS / "opus"
    stamp = target / ".tonesphere-sha256"
    if stamp.is_file() and stamp.read_text().strip() == OPUS_SHA256:
        print(f"libopus {OPUS_VERSION} already present")
        return target

    print(f"downloading libopus {OPUS_VERSION} from {OPUS_URL} ...")
    request = urllib.request.Request(OPUS_URL, headers={"User-Agent": "tonesphere-fetch-sdks"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()

    digest = hashlib.sha256(payload).hexdigest()
    if digest != OPUS_SHA256:
        sys.exit(f"libopus checksum mismatch: expected {OPUS_SHA256}, got {digest}.")

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    prefix = f"opus-{OPUS_VERSION}/"
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            out = target / member.name[len(prefix):]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(archive.extractfile(member).read())

    stamp.write_text(OPUS_SHA256 + "\n")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", nargs="*", choices=["vst3", "asio", "opus"])
    which = parser.parse_args().which or ["vst3", "asio", "opus"]

    SDKS.mkdir(exist_ok=True)
    if "vst3" in which:
        print(f"-> {fetch_vst3()}")
    if "asio" in which:
        print(f"-> {fetch_asio()}")
    if "opus" in which:
        print(f"-> {fetch_opus()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
