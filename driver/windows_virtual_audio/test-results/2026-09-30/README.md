# Driver test run, 2026-09-30

The logs of one full pass of `scripts/vm/run_driver_tests.ps1`, unedited: the driver built
from this commit's source, installed into a Hyper-V VM that had never had it (checkpoint
`deps`), tested, and removed. The VM was built by `scripts/vm/new_driver_vm.ps1` from the
Windows 11 Enterprise LTSC 90-day evaluation ISO (10.0.26100), with test-signing on in the
VM disk's own boot store and Secure Boot off.

| File | What |
|---|---|
| `install.log` | `driver_install.ps1`: certificate trusted, device created, driver installed |
| `devices_after_install.txt` | the device and both endpoints, as PnP lists them |
| `pytest.log` | `tests/hardware/test_virtual_driver.py -m hardware -s -v -rA`, 7 passed |
| `uninstall.log` | `driver_uninstall.ps1`: device removed, package deleted, nothing left |
| `devices_after_uninstall.txt` | empty: no ToneSphere device or endpoint remains |
| `summary.json` | each step's exit code, the guest OS, and the session the tests ran in (0: PowerShell Direct, not a desktop session) |

The same pass succeeded three times in a row on the final driver (twice on a freshly built
VM); the unity-gain paths measured between −0.009 and +0.000 dB across them.
