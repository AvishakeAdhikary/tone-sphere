---
title: Building on Windows
layout: default
permalink: /building-windows/
description: The native toolchain ToneSphere uses on Windows, exactly what was installed on the development machine and why, and how to build.
---

# Building on Windows

The Python side needs only `uv`. The native side — `tonesphere_native.dll` now; the ASIO
host, the VST3 host, the test plugin and the virtual audio driver as they land — needs a
C++ compiler, the Windows SDK and, for the driver, the Windows Driver Kit.

## Quick start

```
uv sync --all-groups                        # includes cmake and ninja (the `native` group)
uv run python scripts/fetch_sdks.py         # VST3 + ASIO SDKs into sdks/ (see sdks/README.md)
uv run python scripts/build_native.py       # -> tonesphere/native/_bin/*.dll
uv run pytest tests/native                  # the native engine, run offline with real signals
```

## The toolchain, and why this one

**The Enterprise WDK (EWDK).** Microsoft's self-contained ISO holding the Visual Studio
Build Tools (MSVC), the Windows SDK and the WDK at matching versions. It is mounted, not
installed: nothing is written to Program Files or the registry, no administrator rights
are needed, and ejecting the ISO removes it. One kit builds both the user-mode DLLs and
the kernel driver, so their SDK headers cannot drift apart.

`scripts/build_native.py` looks for a compiler in this order and prints which it used:

1. `cl.exe` already on `PATH` (a developer prompt).
2. The EWDK — `TONESPHERE_EWDK` (its root), any mounted drive containing
   `BuildEnv\SetupBuildEnv.cmd`, or an ISO in `TONESPHERE_EWDK_ISO` or `C:\SDKs\`, which
   it mounts with `Mount-DiskImage`.
3. A Visual Studio or Build Tools install found by `vswhere` — this is what the GitHub
   Actions Windows runner uses.

The EWDK runs `vsdevcmd` with `-winsdk=none` (its own driver builds find the SDK through
MSBuild properties), so the script adds the kit's `Include`, `Lib` and `bin` directories
itself.

**CMake and Ninja** come from PyPI through the `native` dependency group, so they need no
system install and are pinned in `uv.lock` like everything else.

**Static CRT.** The DLLs link the C runtime statically (`/MT`), so they load on a machine
without the Visual C++ redistributable, and so the allocation counter in
`native/engine/rt_alloc.cpp` replaces this DLL's own `operator new` rather than a shared
runtime's.

## What was installed on the development machine, and why

The machine had no C/C++ toolchain at all: no MSVC, Windows SDK, WDK, CMake, Ninja or
LLVM (checked 2026-09-29; `vswhere` found no instances, and `Windows Kits\10` did not
exist).

| What | How | Size | Why |
|---|---|---|---|
| EWDK for Windows 11 26H1, build 28000.2526, with VS Build Tools 18.3 / MSVC 14.50 | Downloaded from Microsoft (`download.microsoft.com`, linked from learn.microsoft.com's WDK page) to `C:\SDKs\EWDK_28000_260714.iso`; mounted at `D:` | 19.8 GB ISO, not extracted | Compiler, Windows SDK 10.0.28000 and WDK in one kit; no admin; needed for the driver anyway |
| cmake 4.4.3, ninja 1.13.2 | `uv sync` (PyPI, `native` group) | ~50 MB in `.venv` | Build generator and runner |

Nothing else was installed. No Visual Studio IDE, no .NET SDK, no workloads.

## Licences of the tools

The EWDK's licence grants use "to design, develop and test your device drivers and
supporting components". The virtual audio driver and its user-mode counterpart are that.
The user-mode audio engine is built with the same Build Tools, which for an individual
developer working on open-source software is also covered by Visual Studio Community's
terms. Neither licence lets the EWDK itself be redistributed, and nothing here does.

The static CRT is linked into the DLLs under the Build Tools' redistribution terms for
the runtime library.

## Driver builds

The driver additionally needs the kernel-mode headers and WDF, both of which the EWDK
has (`Include\10.0.28000.0\km`, `Include\wdf`). Driver packages are only ever installed
into a Hyper-V VM with test-signing on — never on the development host. See
`docs/VIRTUAL_AUDIO_DRIVER.md`.

`build_driver.py` passes `Inf2CatUseLocalTime`: stampinf dates the INF in local time and
inf2cat otherwise checks that date against UTC, so east of UTC, between local and UTC
midnight, the package fails as "postdated".

The test VM (`scripts/vm/`) needs Hyper-V enabled on the host and an elevated Windows
PowerShell, and nothing else installed: the image is applied with the Dism and Storage
modules and made bootable with the host's own `bcdboot` (the 26100 image's copy fails on a
26200 host). The VM was built from the Windows 11 Enterprise LTSC 90-day evaluation ISO
from Microsoft's Evaluation Center.
