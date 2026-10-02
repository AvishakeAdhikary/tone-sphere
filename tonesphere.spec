# PyInstaller spec for ToneSphere.
#
# Replaces pyinstaller_to_program.sh, which was two lines and produced a binary that could
# not start: PyInstaller does not find PortAudio's shared library by itself, because it is
# not imported as Python code — it is a native library loaded at runtime through cffi. The
# same goes for ToneSphere's own native DLLs, loaded through ctypes (see native_bin below).
#
# Build:   uv run pyinstaller tonesphere.spec
# Output:  dist/ToneSphere/ (Windows, Linux), dist/ToneSphere.app (macOS)
#
# One folder, always. What users download is built from it: the Windows installer and
# portable zip (packaging/windows), the Linux AppImage (packaging/linux), the macOS .app in a
# .dmg, and the MSIX (packaging/msix). A one-file build unpacks itself to a temp folder on
# every launch, a startup delay none of those has to pay.

import re
import sys
from pathlib import Path

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

# The native engine (scripts/build_native.py). tonesphere/native looks for it under
# sys._MEIPASS/tonesphere/native/_bin. Collected when present rather than required, so the
# Linux/macOS builds — which have no native engine yet — still freeze; on Windows the CI
# builds it first, and a frozen app without it reports the engine as unavailable.
# Linux: the sounddevice wheel carries no PortAudio, and a user's machine may have none, so
# the system's copy from the build machine travels in the bundle
# (tonesphere/engine/devices.py, use_bundled_portaudio, finds it there).
if sys.platform.startswith('linux'):
    portaudio = next((p for d in ('/usr/lib/x86_64-linux-gnu', '/usr/lib64', '/usr/lib')
                      for p in Path(d).glob('libportaudio.so.2*') if p.is_file()), None)
    if portaudio is None:
        raise SystemExit("libportaudio.so.2 not found: install libportaudio2 before freezing")
    binaries += [(str(portaudio), '.')]

native_bin = project_root / 'tonesphere' / 'native' / '_bin'
if native_bin.is_dir():
    binaries += [(str(dll), 'tonesphere/native/_bin') for dll in native_bin.glob('*.dll')]

datas = [
    (str(project_root / 'config'), 'config'),
    (str(project_root / 'assets' / 'images'), 'assets/images'),
    # tonesphere/i18n.py's bundled_locale_dir() looks for these under sys._MEIPASS in a
    # frozen build; without this a frozen build ships with no catalogs at all, since a
    # .py package's own directory does not survive freezing the way this data does.
    (str(project_root / 'tonesphere' / 'locale'), 'tonesphere/locale'),
    (str(project_root / 'LICENSE'), 'licenses/tonesphere'),
]

# Licence texts that have to travel with the binary. The MIT notices are the one condition
# of those licences; the GPLv3 text is required because a build that bundles
# tonesphere_asio.dll is distributed under GPLv3 as a whole (docs/ASIO.md).
if (native_bin / 'tonesphere_asio.dll').is_file():
    datas += [(str(project_root / 'native' / 'asio' / 'LICENSE'), 'licenses/gpl-3.0'),
              (str(project_root / 'sdks' / 'asiosdk' / 'LICENSE.txt'), 'licenses/asio-sdk')]
if (native_bin / 'tonesphere_native.dll').is_file():
    datas += [(str(project_root / 'sdks' / 'vst3sdk' / 'LICENSE.txt'), 'licenses/vst3-sdk')]
if (native_bin / 'opus.dll').is_file():
    # libopus, BSD 3-clause: its notice travels with the binary.
    datas += [(str(project_root / 'sdks' / 'opus' / 'COPYING'), 'licenses/opus')]

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

images = project_root / 'assets' / 'images'
# Windows needs an .ico for the executable and its shortcuts; PyInstaller converts the PNG
# (through Pillow, in the packaging group) for macOS.
icon_path = images / ('ToneSphere.ico' if sys.platform == 'win32' else 'ToneSphere.png')
version = re.search(r'__version__ = "([^"]+)"',
                    (project_root / 'tonesphere' / '__init__.py').read_text(encoding='utf-8')).group(1)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
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
    icon=str(icon_path),
)

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

if sys.platform == 'darwin':
    app = BUNDLE(
        collection,
        name='ToneSphere.app',
        icon=str(icon_path),
        bundle_identifier='com.neuralnexusstudios.tonesphere',
        version=version,
        info_plist={
            'CFBundleDisplayName': 'ToneSphere',
            'CFBundleShortVersionString': version,
            'NSHighResolutionCapable': True,
            # Without it macOS refuses an audio input to the app, silently: no prompt, silence.
            'NSMicrophoneUsageDescription':
                'ToneSphere routes and processes audio from your microphones and audio interfaces.',
        },
    )
