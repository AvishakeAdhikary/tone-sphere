"""
Interface translations.

JSON catalogs under `tonesphere/locale/`, one per locale, rather than Qt's `.ts`/`.qm`
toolchain. A `.qm` file is a compiled binary that needs `pyside6-lupdate` and `lrelease`
in the build — a third build step for a project whose packaging is already a PyInstaller
spec plus an MSIX layout — and a binary catalog cannot be diffed in review or asserted
against in a test. `tests/test_i18n.py` does both.

Every catalog except `en.json` records `review_status: "unreviewed"`, because that is what
they are: machine translations that no native speaker has checked. Nothing in this module
or in the interface should imply otherwise; see `tonesphere/locale/README.md`.
"""

import json
from dataclasses import dataclass
from functools import lru_cache

from tonesphere.utils.config import ConfigManager
from tonesphere.utils.logger import get_logger
from tonesphere.utils.paths import bundled_locale_dir

logger = get_logger(__name__)

BASE_LOCALE = "en"

# Where the choice lives in the settings store.
SETTINGS_SECTION = "ui"
SETTINGS_KEY = "language"

# A language whose only catalog is a regional or script variant. A Portuguese or Chinese
# system locale falling through to English, when there is a catalog its user can read,
# would be a worse answer than offering the variant.
_LANGUAGE_ALIASES = {"pt": "pt-BR", "zh": "zh-Hans"}

_config: ConfigManager | None = None
_active: str | None = None

# tr() is called from paintEvent for some labels, so an unknown key would otherwise log
# once per frame and bury everything else.
_reported_missing: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class LocaleInfo:
    """A catalog's metadata block: what it is and how much it can be trusted."""

    code: str
    name: str
    native_name: str
    rtl: bool
    review_status: str


def use_config(config_manager: ConfigManager) -> None:
    """
    Point translations at a specific settings store.

    `MainWindow` passes the store it was built with, so the interface and the language
    setting are read and written through one connection rather than two.
    """
    global _config, _active

    _config = config_manager
    _active = None


def available_locales() -> list[LocaleInfo]:
    """Every catalog this build carries, ordered by native name so the menu is stable."""
    return sorted((info for info, _ in _catalogs().values()), key=lambda info: info.native_name)


def locale_info(code: str | None = None) -> LocaleInfo:
    """The metadata for a locale, defaulting to the active one."""
    return _catalogs()[code or active_locale()][0]


def is_rtl(code: str | None = None) -> bool:
    """
    Whether a locale is written right to left.

    Read from the catalog rather than from a list of codes here, so adding a right-to-left
    language is a catalog change and not a code change.
    """
    return locale_info(code).rtl


def active_locale() -> str:
    """The locale in use: the stored choice, else the system locale, else English."""
    global _active

    if _active is None:
        _active = _resolve()
    return _active


def set_active_locale(code: str) -> bool:
    """
    Switch language, and return whether the choice reached the settings store.

    The switch itself always takes effect. The return value is only about persistence, so
    a caller can say the choice will not survive a restart instead of implying it was
    saved.
    """
    global _active

    if code not in _catalogs():
        raise KeyError(f"No catalog for locale '{code}'")

    _active = code
    return _config_manager().set(SETTINGS_SECTION, SETTINGS_KEY, code)


def tr(key: str, **kwargs: object) -> str:
    """
    The active locale's string for `key`, with `{placeholder}` substitution from `kwargs`.

    A key the active catalog lacks falls back to English and is logged: rendering the key
    itself, or an empty label, would put a developer's identifier in front of a user.
    """
    template = _template(key)

    if not kwargs:
        return template

    try:
        return template.format(**kwargs)
    except (KeyError, IndexError) as e:
        # A catalog that renamed a placeholder would otherwise raise inside a dialog
        # nobody opens until a user does. tests/test_i18n.py catches that for the
        # catalogs in this repository; this covers an edited or newer one on disk.
        logger.error(
            f"Translation '{key}' for locale '{active_locale()}' has a placeholder "
            f"the caller did not supply ({e}); showing the English string instead"
        )
        return _catalogs()[BASE_LOCALE][1][key].format(**kwargs)


# --- Internals ---


def _config_manager() -> ConfigManager:
    global _config

    if _config is None:
        _config = ConfigManager()
    return _config


def _template(key: str) -> str:
    catalogs = _catalogs()
    code = active_locale()

    strings = catalogs[code][1]
    if key in strings:
        return strings[key]

    base = catalogs[BASE_LOCALE][1]
    if key in base:
        _report_missing(key, code)
        return base[key]

    _report_missing(key, BASE_LOCALE)
    # Nothing to fall back to, so show the last path segment as words. Ugly, but a label
    # reading "add bus" is still a label, where "mixer.add_bus" or "" is a defect on show.
    return key.rsplit('.', maxsplit=1)[-1].replace('_', ' ')


def _report_missing(key: str, code: str):
    if (key, code) in _reported_missing:
        return

    _reported_missing.add((key, code))
    if code == BASE_LOCALE:
        logger.error(f"No translation key '{key}' in any catalog, including {BASE_LOCALE}")
    else:
        logger.warning(f"Translation key '{key}' is missing from catalog '{code}'; using {BASE_LOCALE}")


def _resolve() -> str:
    catalogs = _catalogs()

    stored = _config_manager().get(SETTINGS_SECTION, SETTINGS_KEY)
    if isinstance(stored, str) and stored:
        if stored in catalogs:
            return stored
        logger.warning(f"Stored language '{stored}' has no catalog; falling back to the system locale")

    system = _system_locale()
    if system is not None:
        logger.info(f"No language chosen yet; using '{system}' from the system locale")
        return system

    return BASE_LOCALE


def _system_locale() -> str | None:
    """
    The best catalog for the OS locale, or None if there is none.

    Qt is imported here rather than at module scope because the CLI and the API server
    reach this module through `ConfigManager` and must not pull in Qt to read a setting.
    """
    try:
        from PySide6.QtCore import QLocale
    except ImportError:
        return None

    system = QLocale.system()
    by_lower = {code.lower(): code for code in _catalogs()}

    # Widest tag first, dropping one subtag at a time: pt-BR matches its own catalog,
    # de-DE falls back to de, and zh-Hans-CN reaches zh-Hans.
    for tag in (system.bcp47Name(), system.name().replace('_', '-')):
        parts = [part for part in tag.split('-') if part]
        while parts:
            candidate = '-'.join(parts).lower()
            if candidate in by_lower:
                return by_lower[candidate]
            if candidate in _LANGUAGE_ALIASES:
                return _LANGUAGE_ALIASES[candidate]
            parts.pop()

    return None


@lru_cache(maxsize=1)
def _catalogs() -> dict[str, tuple[LocaleInfo, dict[str, str]]]:
    directory = bundled_locale_dir()
    loaded: dict[str, tuple[LocaleInfo, dict[str, str]]] = {}

    for path in sorted(directory.glob("*.json")):
        entry = _read_catalog(path)
        if entry is not None:
            loaded[entry[0].code] = entry

    if BASE_LOCALE not in loaded:
        raise RuntimeError(
            f"No {BASE_LOCALE}.json under {directory}: this build carries no translation "
            "catalogs, so there is nothing to render the interface in"
        )

    return loaded


def _read_catalog(path) -> tuple[LocaleInfo, dict[str, str]] | None:
    """
    One catalog, or None if it is unusable.

    A broken catalog is skipped rather than fatal: the language it offered disappears from
    the menu and the interface stays in English, which is recoverable, where refusing to
    start over one bad file is not. Contributions arrive as edits to these files.
    """
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
        meta = document['meta']
        strings = document['strings']
        info = LocaleInfo(
            code=meta['locale'],
            name=meta['name'],
            native_name=meta['native_name'],
            rtl=bool(meta['rtl']),
            review_status=meta['review_status'],
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Translation catalog {path} is unusable ({e}); that language will not be offered")
        return None

    if not isinstance(strings, dict):
        logger.error(f"Translation catalog {path} has no string table; that language will not be offered")
        return None

    if info.code != path.stem:
        logger.error(f"Translation catalog {path} declares locale '{info.code}'; that language will not be offered")
        return None

    return info, strings
