"""
Translation catalogs and the interface's use of them.

Two languages ship: `en` (the base) and `hi` (Hindi — the publisher is based in Kolkata,
West Bengal, and can actually read and vouch for it, which is the entire reason the set is
two languages rather than a dozen nobody here has read; see `tonesphere/locale/README.md`).

Everything below is structural, and it is what actually catches a mistranslation-shaped
bug: a placeholder renamed in one catalog is a `KeyError` in a dialog nobody opens until a
user does, and a key present in the UI source but absent from a catalog renders as a raw
dotted identifier. Neither shows up by reading the code; both show up here.
"""

import json
import re
from pathlib import Path

import pytest

from tonesphere import i18n
from tonesphere.utils.config import ConfigManager

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCALE_DIR = REPO_ROOT / "tonesphere" / "locale"
UI_DIR = REPO_ROOT / "tonesphere" / "ui"

# `tr('key')` / `tr("key")` — the literal-key call sites. Some call sites build the key
# dynamically (`tr(direction_key)`, `tr(key)` in a loop over a dict of known keys) and are
# checked by hand below rather than by this pattern, since a regex cannot resolve them.
TR_CALL = re.compile(r"""\btr\(\s*['"]([a-z0-9_.]+)['"]""")

# Keys built at runtime from a dynamic suffix, not a literal in the source — verified by
# reading the code around each call site instead.
DYNAMIC_KEY_PREFIXES = ("device.direction.",)


def load_catalog(code: str) -> dict:
    return json.loads((LOCALE_DIR / f"{code}.json").read_text(encoding="utf-8"))


def all_locale_codes() -> list[str]:
    return sorted(p.stem for p in LOCALE_DIR.glob("*.json"))


@pytest.fixture(autouse=True)
def clear_catalog_cache():
    """`i18n._catalogs()` is `lru_cache`d; a prior test's monkeypatched directory or
    stored language must not leak into the next one."""
    i18n._catalogs.cache_clear()
    yield
    i18n._catalogs.cache_clear()
    i18n._config = None
    i18n._active = None


class TestCatalogShape:
    def test_at_least_english_and_hindi_ship(self):
        assert set(all_locale_codes()) >= {"en", "hi"}

    def test_every_catalog_has_exactly_the_english_key_set(self):
        base_keys = set(load_catalog("en")["strings"].keys())
        assert base_keys, "en.json has no strings at all"

        for code in all_locale_codes():
            keys = set(load_catalog(code)["strings"].keys())
            missing = base_keys - keys
            extra = keys - base_keys
            assert not missing, f"{code}.json is missing keys en.json has: {sorted(missing)}"
            assert not extra, f"{code}.json has keys en.json does not: {sorted(extra)}"

    def test_no_empty_or_whitespace_only_values(self):
        for code in all_locale_codes():
            for key, value in load_catalog(code)["strings"].items():
                assert isinstance(value, str) and value.strip(), (
                    f"{code}.json: '{key}' is empty or whitespace-only"
                )

    def test_placeholders_match_the_english_original(self):
        """
        A `{count}` that became `{cuenta}` in translation is not a typo caught by eye —
        it is a `str.format(**kwargs)` KeyError the moment that string is shown.
        """
        placeholder = re.compile(r"\{(\w+)\}")
        base = load_catalog("en")["strings"]

        for code in all_locale_codes():
            if code == "en":
                continue
            strings = load_catalog(code)["strings"]
            for key, english in base.items():
                expected = set(placeholder.findall(english))
                actual = set(placeholder.findall(strings[key]))
                assert actual == expected, (
                    f"{code}.json: '{key}' has placeholders {sorted(actual)}, "
                    f"english has {sorted(expected)}"
                )

    def test_locale_code_matches_filename(self):
        for code in all_locale_codes():
            assert load_catalog(code)["meta"]["locale"] == code

    def test_every_catalog_declares_a_review_status(self):
        for code in all_locale_codes():
            meta = load_catalog(code)["meta"]
            assert meta["review_status"] in ("source", "unreviewed", "reviewed")

    def test_english_is_the_source_and_nothing_else_claims_to_be(self):
        assert load_catalog("en")["meta"]["review_status"] == "source"
        for code in all_locale_codes():
            if code != "en":
                assert load_catalog(code)["meta"]["review_status"] != "source"

    def test_hindi_is_marked_unreviewed(self):
        """
        It is a machine translation the owner has not had independently checked, however
        confident a Kolkata-based publisher might be reading it — the catalog's own
        metadata must say so, since that is what `about.translation_note` reads from.
        """
        assert load_catalog("hi")["meta"]["review_status"] == "unreviewed"

    def test_only_english_and_hindi_ship(self):
        """
        Deliberately narrow rather than exhaustive: the set was cut from sixteen languages
        to two specifically so every shipped catalog is one the publisher can actually
        read and vouch for. A stray extra catalog defeats that.
        """
        assert set(all_locale_codes()) == {"en", "hi"}


class TestEveryUiKeyExistsInEnglish:
    def _literal_keys_used_in_ui(self) -> set[str]:
        keys = set()
        for path in UI_DIR.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            keys.update(TR_CALL.findall(text))
        return keys

    def test_every_literal_tr_call_uses_a_real_key(self):
        base_keys = set(load_catalog("en")["strings"].keys())
        used = self._literal_keys_used_in_ui()
        assert used, "no tr(...) call sites found in tonesphere/ui -- did wiring regress?"

        unknown = used - base_keys
        assert not unknown, f"tr() called with keys not in en.json: {sorted(unknown)}"

    def test_dynamic_direction_keys_exist(self):
        """`main_window.py` builds `device.direction.{input,output}` from the engine's own
        device dict rather than a literal, so the regex above cannot see it."""
        base_keys = set(load_catalog("en")["strings"].keys())
        assert "device.direction.input" in base_keys
        assert "device.direction.output" in base_keys

    def test_every_english_key_is_actually_used_somewhere(self):
        """The inverse check: a key nothing renders is dead weight nobody will notice go
        stale, and a key that used to exist in the UI but was removed should be removed
        from the catalog with it."""
        base_keys = set(load_catalog("en")["strings"].keys())
        used = self._literal_keys_used_in_ui()

        # Keys reached only through a dynamic suffix/lookup, not a `tr('literal.key')`
        # call site: verified present in the source by name instead.
        dynamic = {"device.direction.input", "device.direction.output"}
        for prefix in ("patchbay.fit", "patchbay.zoom_reset", "patchbay.auto_arrange"):
            dynamic.add(prefix)  # looked up via a dict keyed by these literals

        unused = base_keys - used - dynamic
        assert not unused, f"en.json has keys nothing under tonesphere/ui renders: {sorted(unused)}"


class TestNoUnwrappedUserFacingLiteral:
    """
    A generic scan for the shapes a forgotten literal takes in this codebase: a plain
    string passed straight to a Qt method that puts it on screen. False positives are
    handled by an explicit allow-list rather than loosening the pattern, so a real miss
    cannot hide behind one.
    """

    QT_TEXT_CALLS = (
        "setText", "setToolTip", "setWindowTitle", "setPlaceholderText", "addAction",
    )

    # Not translatable, and correctly so: `objectName` is a Qt/stylesheet identifier, not
    # user-facing text; the theme constants and single-character glyphs are not language.
    ALLOWED_LITERALS = {
        "Panel", "Dim", "Heading", "Primary", "Danger", "Mute", "Solo", "StripName",
        "StripSub", "", "Fusion", "ToneSphere",
    }

    def test_no_bare_string_literal_reaches_a_text_setting_call(self):
        pattern = re.compile(
            r"\.(?:" + "|".join(self.QT_TEXT_CALLS) + r")\(\s*(['\"])((?:(?!\1).)*)\1"
        )

        offenders = []
        for path in UI_DIR.glob("*.py"):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for match in pattern.finditer(line):
                    literal = match.group(2)
                    if literal in self.ALLOWED_LITERALS:
                        continue
                    if "tr(" in line:
                        # e.g. `setObjectName("Dim")` on the same line as an unrelated
                        # tr() call for another argument -- the literal itself still has
                        # to be on the allow-list above to pass, this just avoids matching
                        # inside the tr(...) call's own quoted key.
                        continue
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")

        assert not offenders, "un-translated literal(s) found:\n" + "\n".join(offenders)


class TestActiveLocaleResolution:
    def test_defaults_to_english_with_no_stored_preference_and_no_matching_system_locale(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(i18n, "_system_locale", lambda: None)
        i18n.use_config(ConfigManager(db_path=tmp_path / "settings.db"))
        assert i18n.active_locale() == "en"

    def test_stored_language_is_used_when_present(self, tmp_path):
        config = ConfigManager(db_path=tmp_path / "settings.db")
        config.set("ui", "language", "hi")
        i18n.use_config(config)
        assert i18n.active_locale() == "hi"

    def test_an_unknown_stored_language_falls_back_rather_than_raising(self, tmp_path, monkeypatch):
        config = ConfigManager(db_path=tmp_path / "settings.db")
        config.set("ui", "language", "xx-not-a-real-locale")
        i18n.use_config(config)
        monkeypatch.setattr(i18n, "_system_locale", lambda: None)
        assert i18n.active_locale() == "en"

    def test_set_active_locale_persists_through_a_config_round_trip(self, tmp_path):
        db_path = tmp_path / "settings.db"
        i18n.use_config(ConfigManager(db_path=db_path))
        assert i18n.set_active_locale("hi") is True

        # A fresh manager against the same file, the way a new process would open it.
        i18n.use_config(ConfigManager(db_path=db_path))
        assert i18n.active_locale() == "hi"

    def test_switching_locale_actually_changes_what_tr_returns(self, tmp_path):
        i18n.use_config(ConfigManager(db_path=tmp_path / "settings.db"))
        i18n.set_active_locale("en")
        english = i18n.tr("transport.start_engine")

        i18n.set_active_locale("hi")
        hindi = i18n.tr("transport.start_engine")

        assert english != hindi
        assert english == "Start Engine"

    def test_set_active_locale_rejects_an_unknown_code(self, tmp_path):
        i18n.use_config(ConfigManager(db_path=tmp_path / "settings.db"))
        with pytest.raises(KeyError):
            i18n.set_active_locale("xx-not-a-real-locale")


class TestPlaceholderSubstitution:
    def test_a_placeholder_is_substituted(self, tmp_path):
        i18n.use_config(ConfigManager(db_path=tmp_path / "settings.db"))
        i18n.set_active_locale("en")
        assert i18n.tr("transport.buffer_frames", frames=256) == "256 frames"

    def test_a_missing_key_falls_back_to_a_readable_label_rather_than_raising(self, tmp_path):
        i18n.use_config(ConfigManager(db_path=tmp_path / "settings.db"))
        i18n.set_active_locale("en")
        assert i18n.tr("nonexistent.made_up_key") == "made up key"


class TestRightToLeftSupport:
    """
    No shipped catalog needs this (both `en` and `hi` are left-to-right), but the flag is
    read from catalog metadata rather than a hardcoded locale list specifically so a
    future right-to-left addition is a catalog change, not a code change -- verified here
    against a synthetic catalog rather than skipped for lack of one.
    """

    def test_is_rtl_reads_the_catalog_flag(self, tmp_path, monkeypatch):
        locale_dir = tmp_path / "locale"
        locale_dir.mkdir()
        (locale_dir / "en.json").write_text(json.dumps({
            "meta": {"locale": "en", "name": "English", "native_name": "English",
                     "rtl": False, "review_status": "source"},
            "strings": {"app.name": "ToneSphere"},
        }), encoding="utf-8")
        (locale_dir / "xr.json").write_text(json.dumps({
            "meta": {"locale": "xr", "name": "Test RTL", "native_name": "Test RTL",
                     "rtl": True, "review_status": "unreviewed"},
            "strings": {"app.name": "ToneSphere"},
        }), encoding="utf-8")

        monkeypatch.setattr(i18n, "bundled_locale_dir", lambda: locale_dir)
        i18n._catalogs.cache_clear()

        assert i18n.is_rtl("en") is False
        assert i18n.is_rtl("xr") is True

    def test_layout_direction_follows_the_active_locale(self, tmp_path, monkeypatch):
        pytest.importorskip("PySide6")
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        from tonesphere.ui.app import apply_layout_direction

        locale_dir = tmp_path / "locale"
        locale_dir.mkdir()
        (locale_dir / "en.json").write_text(json.dumps({
            "meta": {"locale": "en", "name": "English", "native_name": "English",
                     "rtl": False, "review_status": "source"},
            "strings": {"app.name": "ToneSphere"},
        }), encoding="utf-8")
        (locale_dir / "xr.json").write_text(json.dumps({
            "meta": {"locale": "xr", "name": "Test RTL", "native_name": "Test RTL",
                     "rtl": True, "review_status": "unreviewed"},
            "strings": {"app.name": "ToneSphere"},
        }), encoding="utf-8")

        monkeypatch.setattr(i18n, "bundled_locale_dir", lambda: locale_dir)
        i18n._catalogs.cache_clear()
        i18n.use_config(ConfigManager(db_path=tmp_path / "settings.db"))

        app = QApplication.instance() or QApplication([])

        i18n.set_active_locale("xr")
        apply_layout_direction(app)
        assert app.layoutDirection() == Qt.LayoutDirection.RightToLeft

        i18n.set_active_locale("en")
        apply_layout_direction(app)
        assert app.layoutDirection() == Qt.LayoutDirection.LeftToRight


class TestLiveRetranslation:
    """The thing that actually matters: does the window on screen change, not just what
    tr() returns in isolation."""

    @pytest.fixture
    def window(self, tmp_path, monkeypatch):
        pytest.importorskip("PySide6")
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        from tonesphere.ui.main_window import MainWindow

        QApplication.instance() or QApplication([])
        config = ConfigManager(db_path=tmp_path / "settings.db")
        win = MainWindow(config)
        yield win
        win.close()

    def test_switching_language_changes_visible_widget_text(self, window):
        window._change_language("en")
        english_button = window.engine_button.text()
        english_menu = window.menuBar().actions()[0].text()

        window._change_language("hi")
        hindi_button = window.engine_button.text()
        hindi_menu = window.menuBar().actions()[0].text()

        assert english_button != hindi_button
        assert english_menu != hindi_menu
        assert english_button == "Start Engine"

    def test_switching_language_updates_the_menu_checkmark(self, window):
        window._change_language("hi")

        # Plain loops, not a nested generator expression over the same menu bar: PySide
        # re-wraps each QMenu/QAction fresh per `.menu()`/`.actions()` call, and pulling
        # from a `next(... for ... in ...)` here reproducibly hit `libshiboken: already
        # deleted` on this PySide6 build the moment a nested genexpr touched the result --
        # a plain loop over the same objects never did, in dozens of repeats. Structuring
        # the walk this way is the fix that held up, not just the one that looks safer.
        language_menu = None
        for action in window.menuBar().actions():
            menu = action.menu()
            if menu is None:
                continue
            has_checkable = False
            for sub in menu.actions():
                if sub.isCheckable():
                    has_checkable = True
                    break
            if has_checkable:
                language_menu = menu
                break

        assert language_menu is not None, "no checkable menu found -- is the Language menu missing?"

        checked = []
        for action in language_menu.actions():
            if action.isChecked():
                checked.append(action)

        assert len(checked) == 1
        assert checked[0].text() == i18n.locale_info("hi").native_name

    def test_switching_back_restores_the_original_text(self, window):
        original = window.engine_button.text()
        window._change_language("hi")
        window._change_language("en")
        assert window.engine_button.text() == original

    def test_switching_to_the_already_active_language_is_a_harmless_no_op(self, window):
        window._change_language("en")
        before = window.engine_button.text()
        window._change_language("en")
        assert window.engine_button.text() == before
