#!/usr/bin/env python3
"""
CI smoke test for the PyInstaller-frozen ToneSphere build.

`tonesphere.spec` builds `console=False` (windowed): there is no stdout to parse on
success, so the thing worth proving is the exact failure class the spec's own comments
were written to prevent -- a missing PortAudio/pedalboard native library making the frozen
app die on launch. That shows up as an early, silent process exit, not a printed
traceback, so this checks process liveness (does it start, and is it still running a few
seconds later) rather than output.

Usage: uv run python .github/scripts/smoke_test_frozen_app.py
"""

import platform
import subprocess
import sys
import time
from pathlib import Path

ALIVE_CHECK_SECONDS = 5
SHUTDOWN_TIMEOUT_SECONDS = 10


def frozen_executable() -> Path:
    """
    The exe this build produced, in whichever of the two layouts `tonesphere.spec`
    supports: `ONEFILE=1` puts it directly under `dist/`, the default one-folder layout
    puts it under `dist/ToneSphere/` beside `_internal`. Checking both means this script
    does not need to know which one the caller asked for.
    """
    name = "ToneSphere.exe" if platform.system() == "Windows" else "ToneSphere"
    for candidate in (Path("dist") / name, Path("dist") / "ToneSphere" / name):
        if candidate.exists():
            return candidate
    sys.exit(f"Frozen executable not found as dist/{name} or dist/ToneSphere/{name} "
              f"-- did the PyInstaller build step run first?")


def main() -> int:
    exe = frozen_executable()
    print(f"Launching {exe} gui ...")

    # 'gui' (not the no-argument default, which just prints a usage banner and returns 0
    # immediately) is the only subcommand that actually enters Qt's event loop and stays
    # alive, so it is the only one that can prove the frozen app doesn't crash on launch.
    proc = subprocess.Popen(
        [str(exe), "gui"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    time.sleep(ALIVE_CHECK_SECONDS)

    exit_code = proc.poll()
    if exit_code is not None:
        output = proc.stdout.read() if proc.stdout else ""
        print(f"FAIL: frozen app exited early with code {exit_code} (should still be running "
              f"the GUI event loop after {ALIVE_CHECK_SECONDS}s).")
        print("--- captured output ---")
        print(output)
        return 1

    print(f"OK: still running after {ALIVE_CHECK_SECONDS}s (PID {proc.pid}).")

    proc.terminate()
    try:
        proc.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)

    return 0


if __name__ == "__main__":
    sys.exit(main())
