"""Fixtures for feature (user-scenario) tests.

Each feature test runs against a fresh **in-memory** SQLite database, so the
SSD is never touched — the shared ``store_db`` fixture (see
``tests/conftest.py``) does the shared-cache URI plumbing; here it is simply
made ``autouse`` so every feature test gets it without asking.
"""

from contextlib import closing

import pytest

from src import db


@pytest.fixture(autouse=True)
def feature_db(store_db):
    """Give each feature test the shared in-memory database fixture."""
    return store_db


@pytest.fixture(name="query")
def _query():
    """Return a helper that runs a SELECT against the in-memory database."""

    def _run(sql, params=()):
        with closing(db.connect()) as conn:
            return conn.execute(sql, params).fetchall()

    return _run


@pytest.fixture(name="count")
def _count():
    """Return a helper that counts the rows of a table."""

    def _run(table):
        with closing(db.connect()) as conn:
            return conn.execute(
                f"SELECT COUNT(*) AS n FROM {table}"  # nosec - test table name
            ).fetchone()["n"]

    return _run


@pytest.fixture(name="make_media_file")
def _make_media_file(tmp_path):
    """Return a factory creating a tiny on-disk media file (for hashing).

    Each file gets unique content by default (so distinct files are distinct
    media now that media are de-duplicated by content). Pass an explicit
    ``data`` to force identical content across files (content-dedup tests).
    """
    created = {"n": 0}

    def _make(name="img.png", data=None):
        created["n"] += 1
        path = tmp_path / f"{created['n']}_{name}"
        if data is None:
            data = f"pixels-{created['n']}".encode()
        path.write_bytes(data)
        return str(path)

    return _make
