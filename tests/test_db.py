"""Tests for :mod:`src.db` (bootstrap, schema and connection)."""

from contextlib import closing

import pytest

from src import db


@pytest.fixture(name="db_file")
def _db_file(tmp_path):
    """Return a path to a not-yet-created database file in a temp tree."""
    return tmp_path / "nested" / "cforge.db"


def _table_names(db_path) -> set:
    """Return the set of user table names in the database."""
    with closing(db.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {row["name"] for row in rows}


class TestEnsureDatabase:
    """Tests for :func:`db.ensure_database`."""

    def test_creates_file_and_parent_folder(self, db_file):
        """The database file and its missing parent folder are created."""
        assert not db_file.exists()
        db.ensure_database(db_file)
        assert db_file.exists()
        assert db_file.parent.is_dir()

    def test_creates_all_tables(self, db_file):
        """Every schema table is present after the first run."""
        db.ensure_database(db_file)
        expected = {
            "schema_version",
            "media",
            "media_quality",
            "media_file",
            "library",
            "dataset",
            "dataset_media",
            "caption_type",
            "caption",
            "caption_revision",
            "dataset_caption",
            "triggerword",
            "dataset_triggerword",
        }
        assert expected <= _table_names(db_file)

    def test_records_schema_version(self, db_file):
        """The current schema version is stored in a single row."""
        db.ensure_database(db_file)
        assert db.get_schema_version(db_file) == db.SCHEMA_VERSION

    def test_is_idempotent(self, db_file):
        """A second run leaves the schema and version untouched."""
        db.ensure_database(db_file)
        db.ensure_database(db_file)
        assert db.get_schema_version(db_file) == db.SCHEMA_VERSION
        with closing(db.connect(db_file)) as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS n FROM schema_version"
            ).fetchone()["n"]
        assert count == 1


class TestQualityColumnMigration:
    """Backfill of the old media quality columns into media_quality."""

    def _media_columns(self, db_path) -> set:
        """Return the column names of the media table."""
        with closing(db.connect(db_path)) as conn:
            return {
                row["name"] for row in conn.execute("PRAGMA table_info(media)")
            }

    def test_backfills_and_drops_old_quality_columns(self, db_file):
        """Old single-score columns move to media_quality, then vanish."""
        db.ensure_database(db_file)
        # Recreate the pre-migration shape: add the columns back and seed
        # media as an old development database would have.
        with closing(db.connect(db_file)) as conn:
            with conn:
                conn.execute("ALTER TABLE media ADD COLUMN quality_score REAL")
                conn.execute(
                    "ALTER TABLE media ADD COLUMN quality_metric TEXT"
                )
                conn.execute(
                    "INSERT INTO media (sha256, file_extension, "
                    "quality_score, quality_metric) VALUES "
                    "('a', 'png', 77.0, 'topiq_nr')"
                )
                conn.execute(
                    "INSERT INTO media (sha256, file_extension, "
                    "quality_score) VALUES ('b', 'png', 40.0)"
                )
                conn.execute(
                    "INSERT INTO media (sha256, file_extension) "
                    "VALUES ('c', 'png')"
                )

        db.ensure_database(db_file)  # triggers the migration

        columns = self._media_columns(db_file)
        assert "quality_score" not in columns
        assert "quality_metric" not in columns
        with closing(db.connect(db_file)) as conn:
            rows = conn.execute(
                "SELECT m.sha256 AS sha256, mq.metric_id AS metric_id, "
                "mq.score AS score FROM media_quality mq "
                "JOIN media m ON m.id = mq.media_id ORDER BY m.sha256"
            ).fetchall()
        # 'b' had a NULL metric -> backfilled as MUSIQ; 'c' was unscored.
        assert [
            (row["sha256"], row["metric_id"], row["score"]) for row in rows
        ] == [("a", "topiq_nr", 77.0), ("b", "musiq", 40.0)]

    def test_no_quality_rows_on_a_fresh_database(self, db_file):
        """A fresh database never had the columns, so nothing is backfilled."""
        db.ensure_database(db_file)
        db.ensure_database(db_file)
        with closing(db.connect(db_file)) as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS n FROM media_quality"
            ).fetchone()["n"]
        assert count == 0


class TestMemoryUri:
    """The database helpers accept shared-cache in-memory URIs.

    The test suite runs on such URIs (see ``tests/conftest.py``) so the
    SSD is never touched; these tests pin the plumbing itself.
    """

    _URI = "file:cf_test_db_uri?mode=memory&cache=shared"

    def test_ensure_database_on_memory_uri(self):
        """The schema is created in the in-memory database, no file made."""
        import sqlite3  # pylint: disable=import-outside-toplevel

        keeper = sqlite3.connect(self._URI, uri=True)
        try:
            db.ensure_database(self._URI)
            assert "dataset" in _table_names(self._URI)
            assert db.get_schema_version(self._URI) == db.SCHEMA_VERSION
        finally:
            keeper.close()

    def test_readonly_memory_connection_rejects_writes(self):
        """The read-only opener enforces ``query_only`` on memory URIs."""
        import sqlite3  # pylint: disable=import-outside-toplevel

        keeper = sqlite3.connect(self._URI, uri=True)
        try:
            db.ensure_database(self._URI)
            with closing(db.connect_readonly(self._URI)) as conn:
                with pytest.raises(sqlite3.OperationalError):
                    conn.execute("INSERT INTO dataset (name) VALUES ('x')")
        finally:
            keeper.close()


class TestSeedCaptionTypes:
    """Tests for :func:`db.seed_caption_types`."""

    def _types(self, db_path):
        """Return the caption type names present in the database."""
        with closing(db.connect(db_path)) as conn:
            rows = conn.execute(
                "SELECT name, file_extension FROM caption_type"
            ).fetchall()
        return {row["name"]: row["file_extension"] for row in rows}

    def test_inserts_each_extension(self, db_file):
        """Each extension becomes a caption type (name = file extension)."""
        db.ensure_database(db_file)
        db.seed_caption_types(["txt", "booru"], db_file)
        assert self._types(db_file) == {"txt": "txt", "booru": "booru"}

    def test_is_idempotent(self, db_file):
        """Re-seeding the same extensions does not duplicate rows."""
        db.ensure_database(db_file)
        db.seed_caption_types(["txt"], db_file)
        db.seed_caption_types(["txt"], db_file)
        assert self._types(db_file) == {"txt": "txt"}


class TestConnect:
    """Tests for :func:`db.connect`."""

    def test_foreign_keys_enforced(self, db_file):
        """Foreign-key enforcement is on, so a dangling link is rejected."""
        db.ensure_database(db_file)
        with closing(db.connect(db_file)) as conn:
            with pytest.raises(Exception):
                with conn:
                    conn.execute(
                        "INSERT INTO dataset_media "
                        "(dataset_id, media_id) VALUES (1, 1)"
                    )

    def test_rows_are_mappings(self, db_file):
        """Rows are ``sqlite3.Row`` so columns are reachable by name."""
        db.ensure_database(db_file)
        with closing(db.connect(db_file)) as conn:
            conn.execute("INSERT INTO dataset (name) VALUES ('demo')")
            row = conn.execute("SELECT name FROM dataset").fetchone()
        assert row["name"] == "demo"
