"""Tests for :mod:`src.db_backup` (database backup and restore)."""

from contextlib import closing

import pytest

from src import db
from src import db_backup
from src import sqlite_store as store


@pytest.fixture(name="backup_env")
def _backup_env(tmp_path, monkeypatch):
    """Point the live database and the backups dir at a temp tree."""
    db_path = tmp_path / "cforge.db"
    backup_dir = tmp_path / "backups"
    monkeypatch.setattr(db, "get_db_path", lambda: db_path)
    monkeypatch.setattr(db_backup, "DB_BACKUP_DIR", backup_dir)
    db.ensure_database()
    return db_path, backup_dir


def _dataset_names():
    """Return the dataset names in the live database."""
    with closing(db.connect()) as conn:
        rows = conn.execute(
            "SELECT name FROM dataset ORDER BY name"
        ).fetchall()
    return [row["name"] for row in rows]


class TestCreateBackup:
    """Tests for :func:`db_backup.create_backup`."""

    def test_writes_a_snapshot_file(self, backup_env):
        """A backup file is written into the backups directory."""
        store.create_dataset("alpha")
        path = db_backup.create_backup()
        assert path.exists()
        assert path.parent == backup_env[1]

    def test_snapshot_contains_committed_data(self, backup_env):
        """The snapshot holds the data committed before the backup."""
        store.create_dataset("alpha")
        path = db_backup.create_backup()
        with closing(db.connect(path)) as conn:
            names = [
                row["name"] for row in conn.execute("SELECT name FROM dataset")
            ]
        assert names == ["alpha"]


class TestListBackups:
    """Tests for :func:`db_backup.list_backups`."""

    def test_lists_newest_first(self, backup_env):
        """Backups are returned newest first by their timestamped name."""
        backup_dir = backup_env[1]
        backup_dir.mkdir(parents=True, exist_ok=True)
        (backup_dir / "cforge_20240101_000000.db").write_bytes(b"")
        (backup_dir / "cforge_20240102_000000.db").write_bytes(b"")
        names = [p.name for p in db_backup.list_backups()]
        assert names == [
            "cforge_20240102_000000.db",
            "cforge_20240101_000000.db",
        ]


class TestRestoreBackup:
    """Tests for :func:`db_backup.restore_backup`."""

    def test_restores_previous_state(self, backup_env):
        """Restoring reverts changes made after the backup was taken."""
        store.create_dataset("alpha")
        path = db_backup.create_backup()

        # Change the live database after the snapshot.
        store.create_dataset("beta")
        assert _dataset_names() == ["alpha", "beta"]

        db_backup.restore_backup(path.name)
        assert _dataset_names() == ["alpha"]

    def test_missing_backup_raises(self, backup_env):
        """Restoring an unknown file name raises."""
        with pytest.raises(FileNotFoundError):
            db_backup.restore_backup("nope.db")
