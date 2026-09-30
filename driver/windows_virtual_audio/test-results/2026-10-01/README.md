# Driver test run, 2026-10-01

The logs of one full pass of `scripts/vm/run_driver_tests.ps1`, unedited: the multi-cable
driver built from this commit's source, installed into a Hyper-V VM that had never had it
(checkpoint `deps`), two cables created, tested, and everything removed. Same VM as on
2026-09-30 (Windows 11 Enterprise LTSC 90-day evaluation, 10.0.26100, test-signing on in the
VM disk's own boot store, Secure Boot off). ASIO4ALL 2.22 and ffmpeg were copied in from the
host's download cache and ASIO4ALL installed silently.

| File | What |
|---|---|
| `install.log` | `driver_install.ps1`: certificate trusted, package added to the driver store |
| `cables.log` | `main.py cable-admin install-cables`: "ToneSphere Cable 1" and "ToneSphere Cable 2" created |
| `devices_after_install.txt` | both cables and their four endpoints, as PnP lists them |
| `pytest.log` | `test_virtual_driver.py`, `test_asio.py`, `test_roundtrip.py` against the cables, each file in its own process: 22 passed, 1 failed, 2 skipped |
| `uninstall.log` | `driver_uninstall.ps1`: every cable removed, package deleted, nothing left |
| `devices_after_uninstall.txt` | empty: no ToneSphere device or endpoint remains |
| `summary.json` | each step's exit code, the guest OS, and the session the tests ran in (0: PowerShell Direct, not a desktop session) |

The skips: the laptop's acoustic test (no speaker or microphone in the VM), and ASIO4ALL's
output-through-process-loopback test, which correctly finds that ASIO4ALL does not render
through WASAPI. The failure is the intermittent exclusive-mode round trip through a cable
recorded in `docs/VIRTUAL_AUDIO_DRIVER.md`: the first measurement found 12.02 ms, the repeat
found no path, although its capture held the whole sweep.
