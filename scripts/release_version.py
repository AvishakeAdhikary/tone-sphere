"""
The version a push to main releases, and the files that must carry it.

Every green push to main is a release (`.github/workflows/ci.yml`): the latest `v*` tag's
patch number goes up by one, or its minor or major number when the head commit's message
says `[minor]` or `[major]`. `pyproject.toml` holds the floor — set it ahead of the tags to
start a new series — and is otherwise left alone in the repository: CI stamps the computed
version into the build's own copy of every file that reports one.

    uv run python scripts/release_version.py next            # print the version to release
    uv run python scripts/release_version.py stamp 0.2.1     # write it into the build tree
    uv run python scripts/release_version.py notes 0.2.1 v0.2.0 <source-url>
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r'^v?(\d+)\.(\d+)\.(\d+)$')


def parse(version: str) -> tuple[int, int, int]:
    match = SEMVER.match(version.strip())
    if not match:
        raise ValueError(f"not a version: {version!r}")
    return int(match[1]), int(match[2]), int(match[3])


def next_version(tags: list[str], floor: str, message: str) -> str:
    """The version after the newest of `tags`, bumped as `message` asks; never below `floor`."""
    latest = max((parse(t) for t in tags if SEMVER.match(t.strip())), default=None)
    lowest = parse(floor)
    if latest is None:
        return '.'.join(map(str, lowest))
    major, minor, patch = latest
    if '[major]' in message:
        bumped = (major + 1, 0, 0)
    elif '[minor]' in message:
        bumped = (major, minor + 1, 0)
    else:
        bumped = (major, minor, patch + 1)
    return '.'.join(map(str, max(bumped, lowest)))


def _git(*args: str) -> str:
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def _pyproject_version() -> str:
    return re.search(r'^version = "([^"]+)"', (ROOT / 'pyproject.toml').read_text(encoding='utf-8'), re.M)[1]


def stamp(version: str):
    """Write `version` into every file that reports it, in this tree only."""
    parse(version)
    edits = [
        ('pyproject.toml', r'^version = "[^"]+"', f'version = "{version}"'),
        ('tonesphere/__init__.py', r'^__version__ = "[^"]+"', f'__version__ = "{version}"'),
        ('packaging/msix/AppxManifest.xml', r'(\n\s+)Version="\d+\.\d+\.\d+\.\d+"', rf'\g<1>Version="{version}.0"'),
    ]
    for name, pattern, replacement in edits:
        path = ROOT / name
        text = path.read_text(encoding='utf-8')
        stamped, count = re.subn(pattern, replacement, text, count=1, flags=re.M)
        if count != 1:
            raise SystemExit(f"{name}: no version to stamp")
        path.write_text(stamped, encoding='utf-8')


def notes(version: str, previous: str | None, source_url: str) -> str:
    """The release page: what changed, how to install on each system, and the GPL source."""
    span = f'{previous}..HEAD' if previous else 'HEAD'
    log = _git('log', span, '--no-merges', '--format=- %s').strip() or '- Maintenance.'
    return f"""## What's new

{log}

## Install

**Windows 10/11 (64-bit):** download `ToneSphere-{version}-Setup.exe` and run it. It installs for
your user account only, so it never asks for administrator rights, and adds ToneSphere to the
Start menu. Windows SmartScreen may say "Windows protected your PC", because the installer is not
yet code-signed: choose **More info → Run anyway**. Prefer no installer?
`ToneSphere-{version}-windows-portable.zip` runs from any folder: unzip it and open `ToneSphere.exe`.

**macOS 12 or later:** open `ToneSphere-{version}-macos.dmg` and drag ToneSphere to Applications.
The app is not notarised by Apple, so the first time, **right-click it → Open → Open**; after that
it opens normally.

**Linux (x86_64):** download `ToneSphere-{version}-x86_64.AppImage`, make it executable (in your
file manager's Properties → Permissions, or `chmod +x`), and open it.

ASIO is optional: everything works on Windows' own audio (WASAPI), and an installed ASIO driver
appears as an extra backend.

## Licence and source

ToneSphere is MIT-licensed. The Windows build includes ASIO support built from Steinberg's ASIO
SDK under the GNU GPL v3, which makes that build GPLv3 as a whole; its complete corresponding
source is here: [ToneSphere-{version}-windows-source.zip]({source_url}).
"""


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else ''
    if command == 'next':
        tags = _git('tag', '--list', 'v*').split()
        print(next_version(tags, _pyproject_version(), _git('log', '-1', '--format=%B')))
    elif command == 'stamp' and len(argv) == 3:
        stamp(argv[2])
    elif command == 'notes' and len(argv) == 5:
        print(notes(argv[2], argv[3] or None, argv[4]))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
