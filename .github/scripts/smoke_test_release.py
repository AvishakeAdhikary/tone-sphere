#!/usr/bin/env python3
"""
CI's test of what a user downloads, launched the way a user launches it.

    smoke_test_release.py windows <Setup.exe> <portable.zip> <test-plugin.vst3>
    smoke_test_release.py linux   <AppImage>
    smoke_test_release.py macos   <dmg>

Every launch is the bare executable with no arguments — a double-click, a shortcut — because
that is what v0.2.0's test did not do, and v0.2.0's exe opened nothing when double-clicked.
The app writes what its window shows to TONESPHERE_SMOKE_REPORT once it is drawing a real
device list (`MainWindow.report_when_ready`), and quits; this checks that report.

On Windows the installer runs silently for the current user, the installed app is launched,
the frozen plugin scanner is run against the MIT test plugin, and the uninstaller must leave
nothing behind. The portable zip is launched too.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

LAUNCH_TIMEOUT_S = 240


def launch(exe: Path, work: Path, expect_native: bool, env_extra: dict | None = None) -> dict:
    report = work / f'smoke-{exe.parent.name}-{time.monotonic_ns()}.json'
    env = {**os.environ, 'TONESPHERE_SMOKE_REPORT': str(report), **(env_extra or {})}
    print(f"launching {exe} (no arguments)", flush=True)
    started = time.monotonic()
    done = subprocess.run([str(exe)], env=env, timeout=LAUNCH_TIMEOUT_S, capture_output=True, text=True,
                          errors='replace')
    elapsed = time.monotonic() - started
    if not report.is_file():
        sys.exit(f"FAIL: {exe} exited with {done.returncode} after {elapsed:.0f} s and never reported a window\n"
                 f"{done.stdout[-3000:]}\n{done.stderr[-3000:]}")
    result = json.loads(report.read_text(encoding='utf-8'))
    print(json.dumps(result, indent=2), flush=True)
    problems = []
    if done.returncode != 0 or result.get('error'):
        problems.append(f"exit {done.returncode}, error {result.get('error')!r}")
    if not result.get('visible') or result.get('window_title') != 'ToneSphere':
        problems.append("the main window was not shown")
    if expect_native and (result.get('backend') != 'native' or not result.get('engine')):
        problems.append(f"the native engine did not load (backend {result.get('backend')!r})")
    if expect_native and not result.get('hosts_plugins'):
        problems.append("the VST3 host is not available")
    if problems:
        sys.exit(f"FAIL: {exe}: " + '; '.join(problems))
    print(f"OK: window up on {result['backend']} after {elapsed:.0f} s", flush=True)
    return result


def scan_plugin(exe: Path, plugin: Path, work: Path):
    """The frozen app's own scanner subprocess, as the plugin browser runs it."""
    result = work / 'scan.json'
    done = subprocess.run([str(exe), 'scan-plugin', str(plugin), '--result', str(result)], timeout=120,
                          capture_output=True, text=True, errors='replace')
    if done.returncode != 0 or not result.is_file():
        sys.exit(f"FAIL: frozen scan-plugin exited {done.returncode} without a result\n{done.stderr[-2000:]}")
    report = json.loads(result.read_text(encoding='utf-8'))
    if not report.get('classes'):
        sys.exit(f"FAIL: frozen scan-plugin found no classes in {plugin}: {report}")
    print(f"OK: frozen scanner read {[c['name'] for c in report['classes']]}", flush=True)


def windows(setup: Path, portable: Path, plugin: Path):
    work = Path(tempfile.mkdtemp())
    target = work / 'installed'
    print(f"installing {setup} for the current user", flush=True)
    subprocess.run([str(setup), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', f'/DIR={target}',
                    f'/LOG={work / "setup.log"}'], check=True, timeout=600)
    exe = target / 'ToneSphere.exe'
    if not exe.is_file():
        sys.exit(f"FAIL: the installer did not put ToneSphere.exe in {target}")
    shortcut = Path(os.environ['APPDATA']) / 'Microsoft/Windows/Start Menu/Programs/ToneSphere.lnk'
    if not shortcut.is_file():
        sys.exit(f"FAIL: no Start-menu shortcut at {shortcut}")
    launch(exe, work, expect_native=True)
    scan_plugin(exe, plugin, work)

    uninstaller = target / 'unins000.exe'
    subprocess.run([str(uninstaller), '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART'], check=True, timeout=300)
    deadline = time.monotonic() + 60
    while exe.exists() and time.monotonic() < deadline:
        time.sleep(1)  # the uninstaller finishes from a copy of itself after it exits
    if exe.exists() or shortcut.exists():
        sys.exit("FAIL: the uninstaller left ToneSphere behind")
    print("OK: uninstalled cleanly", flush=True)

    unpacked = work / 'portable'
    with zipfile.ZipFile(portable) as archive:
        archive.extractall(unpacked)
    launch(next(unpacked.rglob('ToneSphere.exe')), work, expect_native=True)


def linux(appimage: Path):
    work = Path(tempfile.mkdtemp())
    appimage.chmod(0o755)
    # CI has no FUSE; users do, and the static runtime needs nothing else from them.
    launch(appimage, work, expect_native=False,
           env_extra={'APPIMAGE_EXTRACT_AND_RUN': '1',
                      'QT_QPA_PLATFORM': os.environ.get('QT_QPA_PLATFORM', 'offscreen')})


def macos(dmg: Path):
    work = Path(tempfile.mkdtemp())
    mount = work / 'mount'
    subprocess.run(['hdiutil', 'attach', '-nobrowse', '-readonly', '-mountpoint', str(mount), str(dmg)], check=True)
    try:
        app = mount / 'ToneSphere.app'
        if not (mount / 'Applications').is_symlink():
            sys.exit("FAIL: the disk image has no Applications link to drag the app onto")
        subprocess.run(['codesign', '--verify', '--deep', '--strict', str(app)], check=True)
        installed = work / 'Applications' / 'ToneSphere.app'
        shutil.copytree(app, installed, symlinks=True)
        launch(installed / 'Contents' / 'MacOS' / 'ToneSphere', work, expect_native=False)
    finally:
        subprocess.run(['hdiutil', 'detach', str(mount)], check=False)


if __name__ == '__main__':
    kind, args = sys.argv[1], [Path(a) for a in sys.argv[2:]]
    {'windows': windows, 'linux': linux, 'macos': macos}[kind](*args)
