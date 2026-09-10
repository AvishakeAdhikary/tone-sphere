# PyInstaller spec for ToneSphere.
#
# Replaces pyinstaller_to_program.sh, which was two lines and produced a binary that could
# not start: PyInstaller does not find PortAudio's shared library or pedalboard's plugin
# host by itself, because neither is imported as Python code — they are native libraries
# loaded at runtime through cffi and a compiled extension.
#
# Build:   uv run pyinstaller tonesphere.spec
# Output:  dist/ToneSphere/ToneSphere.exe  (or the platform equivalent)
#
# Two layouts, one spec
# ----------------------
# Default is one-folder (COLLECT): an `_internal` directory of loose DLLs next to the exe.
# That is what packaging/msix/build_msix.ps1 stages into the MSIX layout, and it must keep
# producing exactly that — nothing here changes for it.
#
# Set ONEFILE=1 to build a single self-contained executable instead, for the GitHub
# Releases download: a user fetching a "single executable" should get one, not a zip of
# 279 loose files. Trade-off, stated because it is real: a one-file build self-extracts to
# a temp directory on every launch, which costs a startup delay the one-folder build does
# not pay. That is the right trade for something downloaded and double-clicked occasionally,
# and the wrong one for what the MSIX installs permanently — hence two modes, not a switch
# of the default.

import os
import sys
from pathlib import Path

ONEFILE = os.environ.get('ONEFILE') == '1'

from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules,
)

block_cipher = None
project_root = Path(SPECPATH)

# PortAudio ships as a shared library inside `_sounddevice_data`, a separate package from
# `sounddevice` itself — asking for it from `sounddevice` warns and collects nothing,
# because that is a single module rather than a package. Without the library the frozen app
# raises OSError on first import and reports "no audio backend".
binaries = collect_dynamic_libs('_sounddevice_data')
binaries += collect_data_files('_sounddevice_data')

# pedalboard bundles a compiled VST3 host. Missing it means plugin loading fails at
# runtime with an import error rather than at build time.
try:
    binaries += collect_dynamic_libs('pedalboard')
except Exception:
    pass

datas = [
    (str(project_root / 'config'), 'config'),
    (str(project_root / 'assets' / 'images'), 'assets/images'),
    # tonesphere/i18n.py's bundled_locale_dir() looks for these under sys._MEIPASS in a
    # frozen build; without this a frozen build ships with no catalogs at all, since a
    # .py package's own directory does not survive freezing the way this data does.
    (str(project_root / 'tonesphere' / 'locale'), 'tonesphere/locale'),
]

hiddenimports = [
    # sounddevice reaches PortAudio through cffi, which PyInstaller cannot see statically.
    '_sounddevice_data',
    'cffi',
    '_cffi_backend',
    # uvicorn picks its event loop and protocol implementations by name at runtime.
    'uvicorn.logging',
    'uvicorn.loops.auto',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan.on',
]

hiddenimports += collect_submodules('tonesphere')

# Qt pulls in a large amount that a mixer never touches. Excluding it keeps the build to a
# reasonable size; each of these is verified unused by the application.
excludes = [
    'tkinter',            # the old UI; nothing imports it any more
    'matplotlib',
    'PySide6.Qt3DCore', 'PySide6.Qt3DRender', 'PySide6.Qt3DAnimation',
    'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebEngineQuick',
    'PySide6.QtQuick3D', 'PySide6.QtCharts', 'PySide6.QtDataVisualization',
    'PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets',
    'PySide6.QtPdf', 'PySide6.QtPdfWidgets',
    'PySide6.QtBluetooth', 'PySide6.QtNfc', 'PySide6.QtPositioning',
    'PySide6.QtSql', 'PySide6.QtTest', 'PySide6.QtDesigner',
]

analysis = Analysis(
    ['main.py'],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=block_cipher)

icon_path = project_root / 'assets' / 'images' / 'ToneSphere.png'

executable = EXE(
    pyz,
    analysis.scripts,
    # One-file bundles the binaries/zipfiles/datas straight into the exe; one-folder
    # leaves them out here so COLLECT can lay them beside it instead.
    analysis.binaries if ONEFILE else [],
    analysis.zipfiles if ONEFILE else [],
    analysis.datas if ONEFILE else [],
    exclude_binaries=not ONEFILE,
    name='ToneSphere',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX-compressed binaries are routinely flagged by antivirus
    console=False,      # windowed: the GUI is the primary entry point
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path) if icon_path.exists() else None,
)

if not ONEFILE:
    collection = COLLECT(
        executable,
        analysis.binaries,
        analysis.zipfiles,
        analysis.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name='ToneSphere',
    )
