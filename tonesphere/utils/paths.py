"""
Where ToneSphere keeps per-user data, and where it finds the files packaged with it.

Nothing the app writes may live next to the executable. The Store build is an MSIX
package and MSIX mounts its install directory read-only, so the old habit of writing
`audio_engine_config.yaml` into the working directory fails outright there — silently, if
the working directory happens to be writable on the developer's machine and not on the
user's.
"""

import os
import sys
from pathlib import Path

APP_NAME = "ToneSphere"

# Store identity. Windows nests per-user data under the publisher; the Linux and macOS
# conventions do not, so the publisher appears in the Windows path only.
PUBLISHER = "Neural Nexus Studios"


def app_data_dir() -> Path:
    """
    The per-user directory for data this app writes.

    A pure path: nothing is created here, so importing a module that asks for the location
    touches no disk. Callers create what they are about to write.
    """
    if sys.platform == 'win32':
        local = os.environ.get('LOCALAPPDATA')
        base = Path(local) if local else Path.home() / 'AppData' / 'Local'
        return base / PUBLISHER / APP_NAME

    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / APP_NAME

    xdg = os.environ.get('XDG_DATA_HOME')
    base = Path(xdg) if xdg else Path.home() / '.local' / 'share'
    return base / APP_NAME


def settings_database_path() -> Path:
    """The SQLite file holding application settings."""
    return app_data_dir() / 'settings.db'


def bundled_default_config() -> Path | None:
    """
    The packaged `config/default_config.yaml`, or None if this build does not carry one.

    Two locations, neither of which is relative to the working directory: PyInstaller
    unpacks `datas` into `sys._MEIPASS` (see `tonesphere.spec`, which bundles `config/`),
    and a source checkout has it next to the package. An installed wheel has neither, so
    None is a normal answer and the caller falls back to its in-code defaults.
    """
    roots = []

    bundle = getattr(sys, '_MEIPASS', None)
    if bundle:
        roots.append(Path(bundle))
    roots.append(Path(__file__).resolve().parents[2])

    for root in roots:
        candidate = root / 'config' / 'default_config.yaml'
        if candidate.is_file():
            return candidate

    return None


def bundled_locale_dir() -> Path:
    """
    The directory holding the JSON translation catalogs.

    The same two locations as `bundled_default_config`, for the same reason: PyInstaller
    unpacks `datas` into `sys._MEIPASS` (see `tonesphere.spec`, which bundles
    `tonesphere/locale`), while a source checkout and an installed wheel carry the
    catalogs inside the package.

    Unlike the packaged defaults there is no in-code substitute for a catalog, so this
    returns where to look rather than None — a build carrying none is broken, and
    `i18n` says so with that path in the message.
    """
    bundle = getattr(sys, '_MEIPASS', None)
    if bundle:
        unpacked = Path(bundle) / 'tonesphere' / 'locale'
        if unpacked.is_dir():
            return unpacked

    return Path(__file__).resolve().parents[1] / 'locale'
