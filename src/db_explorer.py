"""SQLite browser backing the System view's developer tools.

A small database explorer for the bundled ``cforge.db``: it lists the tables
and runs read-only queries against a read-only connection (see
:func:`src.db.connect_readonly`). Only ``SELECT`` / ``PRAGMA`` / ``EXPLAIN`` /
``WITH`` statements are accepted as a clear-error guard on top of the
read-only connection.

Deleting a single row (through a writable connection) is the one destructive
escape hatch offered by the developer tools. Foreign keys are on, so a delete
cascades the same way the app itself would.
"""

import sqlite3
from contextlib import closing

from src import db

# Statement kinds the explorer accepts (read-only). The connection is
# read-only too, so this mainly gives a friendly message instead of a SQL
# error.
_READ_PREFIXES = ("select", "pragma", "explain", "with")


def list_tables() -> list[str]:
    """Return the database's table names, or [] if it cannot be read."""
    try:
        with closing(db.connect_readonly()) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' ORDER BY name"
            ).fetchall()
    except sqlite3.Error:
        return []
    return [row["name"] for row in rows]


def _is_read_only(sql: str) -> bool:
    """Return whether a statement is one of the allowed read-only kinds."""
    stripped = sql.lstrip().lower()
    return any(stripped.startswith(prefix) for prefix in _READ_PREFIXES)


def _execute(sql: str):
    """Run a read-only query; return ``(headers, data, error)``.

    ``error`` is ``None`` on success, otherwise a ready-to-show message with
    both ``headers`` and ``data`` empty.
    """
    try:
        with closing(db.connect_readonly()) as conn:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            headers = (
                [col[0] for col in cursor.description]
                if cursor.description
                else []
            )
    except sqlite3.Error as exc:
        return [], [], f"❌ {exc}"
    return headers, [list(row) for row in rows], None


def run_query(sql: str):
    """Run a guarded read-only query; return ``(headers, data, error)``.

    ``error`` is a friendly message (non-None) when the statement is not one
    of the allowed read-only kinds or SQLite rejects it.
    """
    sql = (sql or "").strip()
    if not sql:
        return [], [], "Enter a query."
    if not _is_read_only(sql):
        return (
            [],
            [],
            (
                "⚠️ Only read-only queries are allowed "
                "(SELECT / PRAGMA / EXPLAIN / WITH)."
            ),
        )
    return _execute(sql)


def delete_row(table: str, row_id) -> str:
    """Delete one row (by ``id``) from a table; return a status message.

    Guarded to real table names only; the delete runs on a writable
    connection with foreign keys on, so dependent rows cascade as in the live
    app.
    """
    if table not in list_tables():
        return f"❌ Unknown table {table!r}."
    try:
        with closing(db.connect()) as conn:
            with conn:
                cursor = conn.execute(
                    f'DELETE FROM "{table}" WHERE id = ?', (row_id,)
                )
    except sqlite3.Error as exc:
        return f"❌ {exc}"
    if cursor.rowcount:
        return f"🗑️ Deleted row id={row_id} from {table}."
    return f"⚠️ No row id={row_id} in {table}."
