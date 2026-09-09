"""
Application settings, stored in SQLite under the per-user data directory.

Settings were a YAML file in the working directory. That cannot survive Store packaging —
MSIX mounts the install directory read-only — and it cannot survive the Qt app and
`main.py server` running at once, because two processes rewriting one YAML file means the
last writer wins and the other's changes vanish. SQLite in WAL mode handles both.

Values are stored JSON-encoded, one row per key, so a bool comes back a bool and an int
comes back an int. A settings store that answered `"True"` for `exclusive_mode` would be
its own small lie, and the caller would have to guess.
"""

import copy
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from tonesphere.utils.logger import get_logger
from tonesphere.utils.paths import bundled_default_config, settings_database_path

logger = get_logger(__name__)

LEGACY_CONFIG_NAME = "audio_engine_config.yaml"

# Both processes are expected to be open at once, so a writer will occasionally find the
# database locked. Five seconds is far longer than any write here takes and still short
# enough that a genuinely stuck lock surfaces as an error rather than a hang.
BUSY_TIMEOUT_SECONDS = 5.0

_MIGRATION_META_KEY = 'legacy_yaml_import'

_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    section TEXT NOT NULL,
    key     TEXT NOT NULL,
    value   TEXT NOT NULL,
    PRIMARY KEY (section, key)
);

CREATE TABLE IF NOT EXISTS accepted_documents (
    doc_id      TEXT PRIMARY KEY,
    sha256      TEXT NOT NULL,
    accepted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Mirrors config/default_config.yaml, which is the source of truth and is read whenever it
# can be found. This copy is for builds that do not carry it (an installed wheel has no
# `config/` directory beside the package), and must be kept in step with it.
#
# The two sets had drifted apart and were reconciled by asking, of every key, whether any
# code reads it. `engine.master_volume`, `virtual_devices.default_channels`,
# `virtual_devices.default_inputs`/`default_outputs`, `api.cors_enabled` and the `effects`,
# `network` and `performance` sections were read by nothing — a knob that changes nothing
# when you turn it is exactly the kind of claim this project removes — so they are gone
# from both. What is left is read by `core/engine_factory.py` (engine, virtual_devices),
# `api/server.py` (api) and `main.py`'s `setup_logging` (logging).
_DEFAULTS: dict[str, dict[str, Any]] = {
    'engine': {
        'sample_rate': 48000,
        'buffer_size': 128,
        'preferred_driver': 'auto',
        'exclusive_mode': True,
    },
    'virtual_devices': {
        'max_inputs': 10,
        'max_outputs': 10,
    },
    'api': {
        'host': '127.0.0.1',
        'port': 8080,
    },
    'logging': {
        'level': 'INFO',
        'enable_file_logging': False,
        'log_file': 'tonesphere.log',
        'log_dir': 'logs',
        'max_file_size_mb': 10,
        'backup_count': 5,
        'colored_console': True,
        'structured': False,
    },
}


class ConfigManager:
    """
    Application settings and recorded legal-document acceptances.

    Callers construct this with no arguments; `db_path` exists so tests can point it at a
    temporary file. The database is opened on first use rather than in the constructor,
    because `api/server.py` builds one at import time and importing a module should not
    create files.
    """

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else settings_database_path()
        self.config = self._load_default_config()
        self.storage_error: str | None = None
        self._schema_ready = False

    # --- Defaults ---

    def _load_default_config(self) -> dict:
        """The packaged defaults, or the in-code copy of them if this build has none."""
        defaults = copy.deepcopy(_DEFAULTS)

        path = bundled_default_config()
        if path is None:
            return defaults

        try:
            packaged = yaml.safe_load(path.read_text(encoding='utf-8'))
        except (OSError, yaml.YAMLError) as e:
            logger.error(f"Packaged defaults at {path} are unreadable ({e}); using in-code defaults")
            return defaults

        if not isinstance(packaged, dict):
            logger.error(f"Packaged defaults at {path} are not a mapping; using in-code defaults")
            return defaults

        # Merged over the in-code set rather than replacing it, so an edited or older
        # packaged file cannot remove a key the engine reads.
        for section, values in packaged.items():
            if isinstance(values, dict):
                defaults.setdefault(section, {}).update(values)

        return defaults

    # --- Storage ---

    def _open(self) -> sqlite3.Connection | None:
        """
        A connection with the schema in place, or None if the database is unusable.

        A failure is logged and recorded in `storage_error`; the caller falls back to
        defaults. Returning an empty config quietly would be indistinguishable from a
        fresh install, which is the one thing this must not look like.
        """
        connection = None
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.db_path, timeout=BUSY_TIMEOUT_SECONDS)

            if not self._schema_ready:
                # WAL is a property of the file, so this only does anything the first time;
                # it is what lets the GUI read while the API server writes.
                connection.execute("PRAGMA journal_mode=WAL")
                with connection:
                    connection.executescript(_SCHEMA)
                self._import_legacy_yaml(connection)
                self._schema_ready = True

            self.storage_error = None
            return connection
        except (sqlite3.Error, OSError) as e:
            if connection is not None:
                connection.close()
            self._report_failure(e)
            return None

    def _report_failure(self, error: Exception):
        message = f"{self.db_path}: {error}"

        # Only the first occurrence of a given failure is logged at error level: `get()`
        # may be called often, and a wall of identical lines buries the first one.
        if message != self.storage_error:
            logger.error(
                f"Settings database unusable ({message}). Running on default settings; "
                "changes made now will not be saved."
            )
        else:
            logger.debug(f"Settings database still unusable ({message})")

        self.storage_error = message

    def _import_legacy_yaml(self, connection: sqlite3.Connection):
        """
        Import `audio_engine_config.yaml` from the working directory, once ever.

        The outcome is recorded in `meta` whether or not a file was found, so a YAML left
        lying around cannot be re-imported later and overwrite settings changed since.
        """
        if connection.execute("SELECT 1 FROM meta WHERE key = ?", (_MIGRATION_META_KEY,)).fetchone():
            return

        legacy = Path.cwd() / LEGACY_CONFIG_NAME
        if not legacy.is_file():
            self._record_import(connection, "no legacy file found")
            return

        try:
            loaded = yaml.safe_load(legacy.read_text(encoding='utf-8'))
        except (OSError, yaml.YAMLError) as e:
            # Deliberately not recorded: nothing was imported, so a fixed file still can be.
            logger.error(f"Could not import {legacy} ({e}); it has been left in place")
            return

        if not isinstance(loaded, dict):
            logger.warning(f"{legacy} does not contain settings; nothing imported")
            self._record_import(connection, f"{legacy} was not a mapping")
            return

        rows = self._rows_for(loaded)
        with connection:
            connection.executemany(
                "INSERT OR REPLACE INTO settings (section, key, value) VALUES (?, ?, ?)", rows
            )
            connection.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                (_MIGRATION_META_KEY, f"imported {len(rows)} setting(s) from {legacy}"),
            )

        logger.info(f"Imported {len(rows)} setting(s) from {legacy} into {self.db_path}")

    def _record_import(self, connection: sqlite3.Connection, outcome: str):
        with connection:
            connection.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)", (_MIGRATION_META_KEY, outcome)
            )

    @staticmethod
    def _rows_for(config: dict) -> list[tuple[str, str, str]]:
        rows = []
        for section, values in config.items():
            if not isinstance(values, dict):
                logger.warning(
                    f"Not storing top-level setting '{section}': settings are section -> key -> value"
                )
                continue
            rows.extend((section, key, json.dumps(value)) for key, value in values.items())
        return rows

    # --- Settings ---

    def load_config(self) -> dict:
        """Stored values merged over the defaults. Returns the defaults if the store is unreadable."""
        connection = self._open()
        if connection is None:
            return self.config

        try:
            rows = connection.execute("SELECT section, key, value FROM settings").fetchall()
        except sqlite3.Error as e:
            self._report_failure(e)
            return self.config
        finally:
            connection.close()

        for section, key, value in rows:
            self.config.setdefault(section, {})[key] = json.loads(value)

        return self.config

    def save_config(self, config: dict | None = None) -> bool:
        """Write a whole config. Returns whether it was actually stored."""
        rows = self._rows_for(config if config is not None else self.config)

        connection = self._open()
        if connection is None:
            return False

        try:
            with connection:
                connection.executemany(
                    "INSERT OR REPLACE INTO settings (section, key, value) VALUES (?, ?, ?)", rows
                )
        except sqlite3.Error as e:
            self._report_failure(e)
            return False
        finally:
            connection.close()

        logger.info(f"Configuration saved ({len(rows)} setting(s))")
        return True

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """One setting: the stored value, else the default for it, else `default`."""
        connection = self._open()
        if connection is not None:
            try:
                row = connection.execute(
                    "SELECT value FROM settings WHERE section = ? AND key = ?", (section, key)
                ).fetchone()
                if row is not None:
                    return json.loads(row[0])
            except sqlite3.Error as e:
                self._report_failure(e)
            finally:
                connection.close()

        return self.config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> bool:
        """Store one setting. Returns whether it was actually stored."""
        connection = self._open()
        if connection is None:
            return False

        try:
            with connection:
                connection.execute(
                    "INSERT OR REPLACE INTO settings (section, key, value) VALUES (?, ?, ?)",
                    (section, key, json.dumps(value)),
                )
        except sqlite3.Error as e:
            self._report_failure(e)
            return False
        finally:
            connection.close()

        self.config.setdefault(section, {})[key] = value
        return True

    # --- Accepted legal documents ---

    def record_document_acceptance(self, doc_id: str, sha256: str) -> bool:
        """
        Record that the user accepted the document whose text hashes to `sha256`.

        The hash is stored rather than a bare "accepted", so a later revision of the
        document re-prompts instead of inheriting consent for text nobody was shown.
        Returns whether it was actually stored — consent that was not written down has
        not been obtained.
        """
        connection = self._open()
        if connection is None:
            return False

        try:
            with connection:
                connection.execute(
                    "INSERT INTO accepted_documents (doc_id, sha256, accepted_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(doc_id) DO UPDATE SET sha256 = excluded.sha256, "
                    "accepted_at = excluded.accepted_at",
                    (doc_id, sha256, datetime.now(UTC).isoformat()),
                )
        except sqlite3.Error as e:
            self._report_failure(e)
            return False
        finally:
            connection.close()

        return True

    def accepted_document_hash(self, doc_id: str) -> str | None:
        """The hash of the text accepted for `doc_id`, or None if there is no record of one."""
        connection = self._open()
        if connection is None:
            return None

        try:
            row = connection.execute(
                "SELECT sha256 FROM accepted_documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        except sqlite3.Error as e:
            self._report_failure(e)
            return None
        finally:
            connection.close()

        return row[0] if row else None
