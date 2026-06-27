"""Backup and restore for the SQLite datasets database (System tab).

A backup is a consistent single-file snapshot taken with SQLite's online backup
API (so it captures committed WAL data, unlike a raw file copy). Restoring
copies a chosen snapshot back over the live database and drops its stale WAL
sidecars, so the next connection reads the restored data. Snapshots live in
``database/backups/`` (git-ignored).
"""

import shutil
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from src import db
from src.constants import DB_BACKUP_DIR

_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def _backup_dir() -> Path:
    """Return the backups directory, creating it if needed."""
    DB_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    return DB_BACKUP_DIR


def create_backup() -> Path:
    """Write a consistent snapshot of the live database; return its path."""
    stamp = datetime.now().strftime(_TIMESTAMP_FMT)
    dest = _backup_dir() / f"cforge_{stamp}.db"
    with closing(db.connect()) as source:
        with closing(sqlite3.connect(str(dest))) as target:
            source.backup(target)
    return dest


def list_backups() -> list[Path]:
    """Return the backup files, newest first."""
    return sorted(_backup_dir().glob("cforge_*.db"), reverse=True)


def restore_backup(filename: str) -> None:
    """Replace the live database with a backup file (by file name).

    ``filename`` is a name inside ``database/backups/``. Raises
    ``FileNotFoundError`` when no such backup exists.
    """
    source = _backup_dir() / filename
    if not source.is_file():
        raise FileNotFoundError(filename)
    live = db.get_db_path()
    shutil.copy2(source, live)
    # Drop the live WAL sidecars so the restored file is read as-is.
    for suffix in ("-wal", "-shm"):
        sidecar = live.with_name(live.name + suffix)
        if sidecar.exists():
            sidecar.unlink()
