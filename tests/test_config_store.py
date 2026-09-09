"""
The settings store: a real SQLite file, written and read back.

Two things are being defended here. First, types: settings arrive from YAML and from a Qt
dialog as bools, ints and floats, and a store that hands back the string "True" for
`exclusive_mode` has quietly broken every caller that trusts it. Second, honesty on
failure: an unreadable database must be reported as one, because defaults returned in
silence are indistinguishable from a fresh install, and the user's real settings then look
like they were never there.

Nothing here mocks sqlite3. Every test writes an actual database in `tmp_path`.
"""

import json
import logging
import sqlite3
import subprocess
import sys
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from tonesphere.utils import paths
from tonesphere.utils.config import _DEFAULTS, LEGACY_CONFIG_NAME, ConfigManager

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGED_DEFAULTS = REPO_ROOT / "config" / "default_config.yaml"


@pytest.fixture
def manager(tmp_path):
    return ConfigManager(db_path=tmp_path / "settings.db")


@pytest.fixture
def config_logs(caplog):
    """
    Capture what `tonesphere.utils.config` logs.

    The project's loggers are created with `propagate = False`, so caplog's own handler
    never sees them unless it is attached to that logger directly.
    """
    logger = logging.getLogger("tonesphere.utils.config")
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


def query(db_path: Path, sql: str, parameters: tuple = ()):
    """Read the database from outside `ConfigManager`, closing behind us.

    Windows will not let pytest remove `tmp_path` while a connection to a file in it is
    still open, so these are not left to the garbage collector.
    """
    connection = sqlite3.connect(db_path)
    try:
        return connection.execute(sql, parameters).fetchall()
    finally:
        connection.close()


def messages(logs, minimum_level: int = logging.WARNING) -> list[str]:
    return [r.getMessage() for r in logs.records if r.levelno >= minimum_level]


class TestValuesKeepTheirType:
    """JSON-encoded, one row per key, so what goes in is what comes out."""

    @pytest.mark.parametrize("value", [
        True, False, 0, 1, 48000, -1, 0.0, 1.0, 5.7, "auto", "", None,
        [128, 256, 512], [], {"nested": {"depth": 2}},
    ])
    def test_round_trip(self, manager, value):
        assert manager.set("engine", "probe", value) is True

        result = manager.get("engine", "probe")
        assert result == value
        assert type(result) is type(value)

    def test_a_stored_bool_is_not_a_string(self, manager):
        """The specific regression this encoding exists to prevent."""
        manager.set("engine", "exclusive_mode", False)

        assert manager.get("engine", "exclusive_mode") is False
        assert manager.load_config()["engine"]["exclusive_mode"] is False

    def test_a_stored_int_is_not_a_bool(self, manager):
        manager.set("engine", "buffer_size", 512)

        assert manager.get("engine", "buffer_size") == 512
        assert isinstance(manager.get("engine", "buffer_size"), bool) is False

    def test_a_value_survives_a_new_manager_on_the_same_file(self, tmp_path):
        db = tmp_path / "settings.db"
        ConfigManager(db_path=db).set("ui", "language", "de")

        assert ConfigManager(db_path=db).get("ui", "language") == "de"

    def test_setting_the_same_key_twice_replaces_it(self, manager):
        manager.set("ui", "start_minimised", True)
        manager.set("ui", "start_minimised", False)

        assert manager.get("ui", "start_minimised") is False
        rows = query(
            manager.db_path,
            "SELECT COUNT(*) FROM settings WHERE section = ? AND key = ?",
            ("ui", "start_minimised"),
        )
        assert rows[0][0] == 1


class TestDefaultsOnAnEmptyDatabase:
    def test_every_key_the_code_reads_is_present(self, manager):
        config = manager.load_config()

        assert config["engine"]["sample_rate"] == 48000
        assert config["engine"]["buffer_size"] == 128
        assert config["engine"]["preferred_driver"] == "auto"
        assert config["engine"]["exclusive_mode"] is True
        assert config["virtual_devices"]["max_inputs"] == 10
        assert config["virtual_devices"]["max_outputs"] == 10
        assert config["api"]["host"] == "127.0.0.1"
        assert config["api"]["port"] == 8080
        assert config["logging"]["level"] == "INFO"
        assert config["logging"]["enable_file_logging"] is False
        assert config["logging"]["max_file_size_mb"] == 10

    def test_no_error_is_reported_for_a_database_that_did_not_exist_yet(self, manager):
        manager.load_config()

        assert manager.storage_error is None
        assert manager.db_path.exists()

    def test_get_falls_back_to_the_default_then_to_the_caller(self, manager):
        assert manager.get("engine", "sample_rate") == 48000
        assert manager.get("engine", "no_such_key", 7) == 7
        assert manager.get("no_such_section", "no_such_key") is None

    def test_a_stored_value_does_not_wipe_its_siblings(self, manager):
        """
        The YAML layer merged a section at a time, so writing one key of `engine` dropped
        the rest of it. Per-key rows make a half-populated section impossible.
        """
        manager.set("engine", "buffer_size", 512)

        config = manager.load_config()
        assert config["engine"]["buffer_size"] == 512
        assert config["engine"]["sample_rate"] == 48000
        assert config["engine"]["exclusive_mode"] is True

    def test_the_in_code_defaults_still_match_the_packaged_file(self):
        """
        These two sets had drifted into different keys entirely, which is how a config
        file could document a setting the running code had never heard of. The in-code
        copy exists only for builds that ship without `config/`, so it has to stay equal.
        """
        assert _DEFAULTS == yaml.safe_load(PACKAGED_DEFAULTS.read_text(encoding='utf-8'))

    def test_the_store_actually_drives_the_engine(self, tmp_path):
        from tonesphere.core.engine_factory import create_audio_engine

        manager = ConfigManager(db_path=tmp_path / "settings.db")
        manager.set("engine", "buffer_size", 512)
        manager.set("engine", "sample_rate", 44100)
        manager.set("virtual_devices", "max_inputs", 4)

        engine = create_audio_engine(manager)

        assert engine.buffer_size == 512
        assert engine.sample_rate == 44100
        assert engine.max_virtual_inputs == 4


class TestSaveConfig:
    def test_a_whole_config_round_trips(self, tmp_path):
        db = tmp_path / "settings.db"

        assert ConfigManager(db_path=db).save_config({
            "engine": {"buffer_size": 256, "exclusive_mode": False},
            "api": {"port": 9999},
        }) is True

        reloaded = ConfigManager(db_path=db).load_config()
        assert reloaded["engine"]["buffer_size"] == 256
        assert reloaded["engine"]["exclusive_mode"] is False
        assert reloaded["api"]["port"] == 9999

    def test_a_top_level_scalar_is_refused_out_loud(self, manager, config_logs):
        """Settings are section -> key -> value; anything else is reported, not dropped quietly."""
        assert manager.save_config({"engine": {"buffer_size": 256}, "stray": 1}) is True

        assert any("stray" in message for message in messages(config_logs))
        assert manager.get("engine", "buffer_size") == 256


class TestLegacyYamlImport:
    def write_legacy(self, directory: Path) -> Path:
        legacy = directory / LEGACY_CONFIG_NAME
        legacy.write_text(yaml.safe_dump({
            "engine": {"buffer_size": 512, "exclusive_mode": False, "preferred_driver": "wasapi"},
            "api": {"port": 8123},
            "logging": {"level": "DEBUG", "enable_file_logging": True},
        }))
        return legacy

    def test_an_existing_yaml_is_imported_with_its_types(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self.write_legacy(tmp_path)

        config = ConfigManager(db_path=tmp_path / "settings.db").load_config()

        assert config["engine"]["buffer_size"] == 512
        assert config["engine"]["exclusive_mode"] is False
        assert config["engine"]["preferred_driver"] == "wasapi"
        assert config["api"]["port"] == 8123
        assert config["logging"]["enable_file_logging"] is True
        assert config["engine"]["sample_rate"] == 48000, "unset keys still come from the defaults"

    def test_the_import_does_not_run_twice_over_later_changes(self, tmp_path, monkeypatch):
        """
        The hazard the `meta` marker exists for: a stale YAML re-imported on every launch,
        undoing everything the user has changed since.
        """
        monkeypatch.chdir(tmp_path)
        self.write_legacy(tmp_path)
        db = tmp_path / "settings.db"

        ConfigManager(db_path=db).load_config()
        ConfigManager(db_path=db).set("engine", "buffer_size", 64)

        assert ConfigManager(db_path=db).load_config()["engine"]["buffer_size"] == 64

    def test_a_yaml_appearing_later_is_not_imported(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        db = tmp_path / "settings.db"
        ConfigManager(db_path=db).load_config()

        self.write_legacy(tmp_path)

        assert ConfigManager(db_path=db).load_config()["api"]["port"] == 8080

    def test_the_yaml_is_left_in_place(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        legacy = self.write_legacy(tmp_path)

        ConfigManager(db_path=tmp_path / "settings.db").load_config()

        assert legacy.exists(), "the user's file is not ours to delete"

    def test_what_happened_is_recorded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self.write_legacy(tmp_path)
        db = tmp_path / "settings.db"

        ConfigManager(db_path=db).load_config()

        recorded = query(db, "SELECT value FROM meta WHERE key = 'legacy_yaml_import'")
        assert recorded
        assert LEGACY_CONFIG_NAME in recorded[0][0]

    def test_an_unparseable_yaml_is_reported_and_retried(self, tmp_path, monkeypatch, config_logs):
        monkeypatch.chdir(tmp_path)
        (tmp_path / LEGACY_CONFIG_NAME).write_text("engine: {buffer_size: [unclosed\n")
        db = tmp_path / "settings.db"

        config = ConfigManager(db_path=db).load_config()

        assert config["engine"]["buffer_size"] == 128
        assert any(
            LEGACY_CONFIG_NAME in message for message in messages(config_logs, logging.ERROR)
        ), "a file we could not read must be said out loud"

        # Nothing was imported, so a corrected file must still get its chance.
        self.write_legacy(tmp_path)
        assert ConfigManager(db_path=db).load_config()["api"]["port"] == 8123

    def test_a_yaml_that_is_not_settings_is_reported_and_not_retried(
        self, tmp_path, monkeypatch, config_logs
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / LEGACY_CONFIG_NAME).write_text("just a string\n")
        db = tmp_path / "settings.db"

        assert ConfigManager(db_path=db).load_config()["api"]["port"] == 8080
        assert messages(config_logs)

        self.write_legacy(tmp_path)
        assert ConfigManager(db_path=db).load_config()["api"]["port"] == 8080


class TestDocumentAcceptance:
    def test_nothing_is_accepted_on_a_fresh_install(self, manager):
        assert manager.accepted_document_hash("eula") is None

    def test_an_acceptance_round_trips(self, tmp_path):
        db = tmp_path / "settings.db"
        digest = sha256(b"the terms as the user saw them").hexdigest()

        assert ConfigManager(db_path=db).record_document_acceptance("eula", digest) is True
        assert ConfigManager(db_path=db).accepted_document_hash("eula") == digest

    def test_documents_are_tracked_separately(self, manager):
        manager.record_document_acceptance("eula", "a" * 64)
        manager.record_document_acceptance("privacy", "b" * 64)

        assert manager.accepted_document_hash("eula") == "a" * 64
        assert manager.accepted_document_hash("privacy") == "b" * 64
        assert manager.accepted_document_hash("third_party_notices") is None

    def test_a_revised_document_no_longer_matches_the_accepted_hash(self, manager):
        """
        Why the hash is stored instead of `accepted: true`: consent covers the text that
        was shown. Rewritten text must not inherit it.
        """
        original = sha256(b"v1 terms").hexdigest()
        revised = sha256(b"v2 terms").hexdigest()
        manager.record_document_acceptance("eula", original)

        assert manager.accepted_document_hash("eula") != revised

        manager.record_document_acceptance("eula", revised)
        assert manager.accepted_document_hash("eula") == revised
        assert len(query(manager.db_path, "SELECT * FROM accepted_documents")) == 1

    def test_when_it_was_accepted_is_recorded_with_a_timezone(self, manager):
        manager.record_document_acceptance("eula", "c" * 64)

        accepted_at = query(
            manager.db_path, "SELECT accepted_at FROM accepted_documents WHERE doc_id = ?", ("eula",)
        )[0][0]

        assert datetime.fromisoformat(accepted_at).tzinfo is not None


class TestFailureIsReportedNotSwallowed:
    def test_an_unopenable_database_falls_back_to_defaults_and_says_so(self, tmp_path, config_logs):
        # A directory is not a database file, and sqlite cannot open one on any platform.
        manager = ConfigManager(db_path=tmp_path)

        config = manager.load_config()

        assert config["engine"]["sample_rate"] == 48000, "defaults, not an empty dict"
        assert manager.storage_error is not None
        assert str(tmp_path) in manager.storage_error
        errors = messages(config_logs, logging.ERROR)
        assert errors, "an unusable settings store must be logged as an error"
        assert "will not be saved" in errors[0]

    def test_a_corrupt_database_falls_back_to_defaults_and_says_so(self, tmp_path, config_logs):
        db = tmp_path / "settings.db"
        db.write_bytes(b"this is not a SQLite file, it is junk of about the right size")

        manager = ConfigManager(db_path=db)

        assert manager.load_config()["api"]["port"] == 8080
        assert manager.storage_error is not None
        assert messages(config_logs, logging.ERROR)

    def test_a_write_to_an_unusable_store_reports_that_it_did_not_happen(self, tmp_path):
        manager = ConfigManager(db_path=tmp_path)

        assert manager.set("ui", "language", "fr") is False
        assert manager.save_config({"api": {"port": 1}}) is False
        assert manager.record_document_acceptance("eula", "d" * 64) is False
        assert manager.accepted_document_hash("eula") is None

    def test_the_same_failure_is_not_logged_over_and_over(self, tmp_path, config_logs):
        manager = ConfigManager(db_path=tmp_path)

        for _ in range(5):
            manager.get("engine", "sample_rate")

        assert len(messages(config_logs, logging.ERROR)) == 1

    def test_recovery_clears_the_reported_error(self, tmp_path):
        manager = ConfigManager(db_path=tmp_path)
        manager.load_config()
        assert manager.storage_error is not None

        manager.db_path = tmp_path / "settings.db"

        assert manager.set("ui", "language", "fr") is True
        assert manager.storage_error is None


class TestTwoProcesses:
    """
    The Qt app and `main.py server` are both expected to be open at once. In the default
    rollback-journal mode a reader holding a transaction blocks every writer until the busy
    timeout expires; WAL is what makes this work.
    """

    def test_wal_is_actually_enabled_on_the_file(self, manager):
        manager.load_config()

        mode = query(manager.db_path, "PRAGMA journal_mode")[0][0]

        assert mode.lower() == "wal"

    def test_a_write_succeeds_while_another_connection_is_reading(self, manager):
        manager.load_config()

        reader = sqlite3.connect(manager.db_path, timeout=1.0)
        try:
            reader.execute("BEGIN")
            reader.execute("SELECT * FROM settings").fetchall()
            assert manager.set("ui", "language", "ja") is True
        finally:
            reader.close()

        assert manager.get("ui", "language") == "ja"

    def test_a_second_process_reads_and_writes_the_same_store(self, tmp_path):
        db = tmp_path / "settings.db"
        ConfigManager(db_path=db).set("ui", "start_minimised", True)

        result = subprocess.run(
            [sys.executable, "-c",
             "import json, sys\n"
             "from tonesphere.utils.config import ConfigManager\n"
             "manager = ConfigManager(db_path=sys.argv[1])\n"
             "manager.set('ui', 'autostart', False)\n"
             "print(json.dumps(manager.get('ui', 'start_minimised')))\n",
             str(db)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )

        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout.strip().splitlines()[-1]) is True
        assert ConfigManager(db_path=db).get("ui", "autostart") is False


class TestWhereTheDataLives:
    """
    MSIX mounts its install directory read-only, so a settings file written beside the
    executable is not a working default — it is a failure on the user's machine and nowhere
    else, which is the worst place for one to appear first.
    """

    def test_the_default_database_is_under_the_user_data_directory(self):
        assert ConfigManager().db_path == paths.app_data_dir() / "settings.db"

    def test_constructing_a_manager_writes_nothing_at_all(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        ConfigManager(db_path=tmp_path / "sub" / "settings.db")

        assert list(tmp_path.iterdir()) == [], "construction must not touch the disk"

    def test_the_windows_path_is_under_localappdata_and_the_publisher(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\someone\AppData\Local")

        expected = Path(r"C:\Users\someone\AppData\Local\Neural Nexus Studios\ToneSphere")
        assert paths.app_data_dir() == expected

    def test_the_linux_path_honours_xdg_data_home(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("XDG_DATA_HOME", "/home/someone/.custom-share")

        assert paths.app_data_dir() == Path("/home/someone/.custom-share/ToneSphere")

    def test_the_linux_path_defaults_to_local_share(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/someone")))

        assert paths.app_data_dir() == Path("/home/someone/.local/share/ToneSphere")

    def test_the_macos_path_is_application_support(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/Users/someone")))

        assert paths.app_data_dir() == Path("/Users/someone/Library/Application Support/ToneSphere")

    def test_asking_for_the_directory_does_not_create_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))

        assert not paths.app_data_dir().exists()

    def test_the_packaged_defaults_are_found_without_the_working_directory(self, tmp_path, monkeypatch):
        """
        `config/default_config.yaml` was only ever reachable as a relative path. The GUI is
        launched from wherever the user's shortcut points, which is rarely the checkout.
        """
        monkeypatch.chdir(tmp_path)

        assert paths.bundled_default_config() == PACKAGED_DEFAULTS

    def test_a_pyinstaller_bundle_is_searched_first(self, tmp_path, monkeypatch):
        """`tonesphere.spec` puts `config/` in the bundle, which unpacks to `sys._MEIPASS`."""
        bundled = tmp_path / "config" / "default_config.yaml"
        bundled.parent.mkdir()
        bundled.write_text("engine: {buffer_size: 64}\n")
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

        assert paths.bundled_default_config() == bundled

        config = ConfigManager(db_path=tmp_path / "settings.db").load_config()
        assert config["engine"]["buffer_size"] == 64
        assert config["engine"]["sample_rate"] == 48000, "an older bundle cannot remove a key"
