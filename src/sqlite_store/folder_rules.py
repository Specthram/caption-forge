"""Repository for the Libraries "Subfolder mapping" rules.

Persists what the mapping wizard produces: a parent folder library's
auto-tag level, its per-sub-folder rules (:mod:`src.folder_rules`), and the
sub-library rows a ``sublib`` rule promotes a folder into. The pure
resolution of those rules (routing, effective tags) lives in
:mod:`src.folder_rules`; this module only reads and writes them.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from src import db
from src.folder_rules import AUTO_OFF
from src.sqlite_store.base import _query_all, _query_one, _write


def _rule_dict(row) -> dict:
    """Return a folder-rule row as a dict with ``tags``/``removed`` parsed."""
    return {
        "rel_path": row["rel_path"],
        "mode": row["mode"],
        "sub_library_id": row["sub_library_id"],
        "tags": json.loads(row["tags"]),
        "removed": json.loads(row["removed"]),
    }


def get_folder_mapping(library_id: int) -> dict:
    """Return a library's mapping: its auto-tag level and folder rules.

    Returns
    -------
    dict
        ``{"auto_tag_level": str, "rules": list of dict}`` — each rule a
        :func:`_rule_dict`. An unmapped library yields the default level and
        an empty rule list.
    """
    library = _query_one(
        "SELECT auto_tag_level FROM library WHERE id = ?", (library_id,)
    )
    level = library["auto_tag_level"] if library else AUTO_OFF
    rows = _query_all(
        "SELECT * FROM library_folder_rule WHERE library_id = ? "
        "ORDER BY rel_path",
        (library_id,),
    )
    return {
        "auto_tag_level": level,
        "rules": [_rule_dict(row) for row in rows],
    }


def folder_rules_map(library_id: int) -> tuple:
    """Return ``(auto_tag_level, {rel_path: rule})`` for the resolver.

    The shape :mod:`src.folder_rules` consumes: the level as a plain string
    and the rules keyed by ``rel_path``. An unmapped library yields an empty
    mapping, so the scanner treats every file as a plain ``keep``.
    """
    mapping = get_folder_mapping(library_id)
    return mapping["auto_tag_level"], {
        rule["rel_path"]: rule for rule in mapping["rules"]
    }


def set_library_auto_tag_level(library_id: int, level: str) -> None:
    """Set a library's auto-tag level ('0', '1' or 'all')."""
    _write(
        "UPDATE library SET auto_tag_level = ? WHERE id = ?",
        (str(level), library_id),
    )


def get_or_create_sub_library(
    parent_library_id: int, name: str, path, rel_path: str
) -> int:
    """Return the id of the sub-library for a folder, creating it if absent.

    A sub-library is a real ``folder`` library flagged with
    ``parent_library_id`` and its ``rel_path``. Keyed by its absolute
    ``path`` (``library.path`` is unique): re-applying an edited mapping
    reuses the existing row and refreshes its name/parent/rel_path rather
    than failing on the unique path. The name falls back to the folder's own
    name when empty.
    """
    path = str(Path(path))
    name = (name or "").strip() or Path(path).name or "library"
    existing = _query_one("SELECT id FROM library WHERE path = ?", (path,))
    if existing is not None:
        _write(
            "UPDATE library SET name = ?, kind = 'folder', "
            "parent_library_id = ?, rel_path = ? WHERE id = ?",
            (name, parent_library_id, rel_path, existing["id"]),
        )
        return existing["id"]
    return _write(
        "INSERT INTO library "
        "(name, path, kind, recursive, parent_library_id, rel_path) "
        "VALUES (?, ?, 'folder', 1, ?, ?)",
        (name, path, parent_library_id, rel_path),
    )


def list_sub_libraries(parent_library_id: int) -> list:
    """Return a parent library's sub-library rows, by name."""
    return _query_all(
        "SELECT * FROM library WHERE parent_library_id = ? ORDER BY name",
        (parent_library_id,),
    )


def library_mapping_stats(library_id: int) -> dict:
    """Return a library's subfolder-mapping summary for the sidebar card.

    Cheap ``COUNT`` queries (no disk walk): whether the library carries a
    mapping at all, how many folder rules it stores, how many of them skip
    their folder, and how many sub-libraries it owns. ``mapped`` is what the
    sidebar uses to render the group card instead of a flat row.
    """
    rules = _query_one(
        "SELECT COUNT(*) AS n, "
        "COUNT(CASE WHEN mode = 'exclude' THEN 1 END) AS skipped "
        "FROM library_folder_rule WHERE library_id = ?",
        (library_id,),
    )
    subs = _query_one(
        "SELECT COUNT(*) AS n FROM library WHERE parent_library_id = ?",
        (library_id,),
    )
    rule_count = rules["n"] if rules else 0
    sub_count = subs["n"] if subs else 0
    return {
        "mapped": rule_count > 0 or sub_count > 0,
        "rule_count": rule_count,
        "skipped_folders": (rules["skipped"] if rules else 0) or 0,
        "sub_count": sub_count,
    }


def prune_sub_libraries(parent_library_id: int, keep_ids) -> int:
    """Delete a parent's sub-libraries no longer referenced by a rule.

    Called after a mapping re-apply: a folder demoted from ``sublib`` back to
    ``keep`` leaves its sub-library unreferenced, so it is removed. The
    schema's ``ON DELETE SET NULL`` nulls the deleted library on its
    ``media_file`` rows (and any stale rule link); the re-scan that follows
    re-routes those files to the owning library (the parent, or a shallower
    sub-library). Returns how many sub-libraries were removed.
    """
    keep = {int(i) for i in keep_ids if i is not None}
    removed = 0
    for row in _query_all(
        "SELECT id FROM library WHERE parent_library_id = ?",
        (parent_library_id,),
    ):
        if row["id"] not in keep:
            _write("DELETE FROM library WHERE id = ?", (row["id"],))
            removed += 1
    return removed


def replace_folder_rules(library_id: int, entries) -> None:
    """Replace a library's whole folder-rule set in one transaction.

    Every existing rule of the library is dropped and the given ``entries``
    inserted, so the wizard persists the mapping as a single authoritative
    payload. Each entry is a dict ``{"rel_path", "mode", "sub_library_id",
    "tags", "removed"}``; ``tags``/``removed`` are lists of tag names, stored
    as JSON. Sub-library rows themselves are created by
    :func:`get_or_create_sub_library`, not here — an entry only references an
    already-created ``sub_library_id``.
    """
    rows = [
        (
            library_id,
            entry["rel_path"],
            entry.get("mode", "keep"),
            entry.get("sub_library_id"),
            json.dumps(list(entry.get("tags") or [])),
            json.dumps(list(entry.get("removed") or [])),
        )
        for entry in entries
    ]
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "DELETE FROM library_folder_rule WHERE library_id = ?",
                (library_id,),
            )
            conn.executemany(
                "INSERT INTO library_folder_rule "
                "(library_id, rel_path, mode, sub_library_id, tags, removed) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )


def _sub_abs_path(parent_path: str, rel_path: str) -> str:
    """Return the absolute path of a sub-folder from a parent + rel_path."""
    parts = [part for part in rel_path.split("/") if part]
    return str(Path(parent_path).joinpath(*parts))


def apply_folder_mapping(library_id: int, auto_tag_level: str, rules) -> dict:
    """Persist a whole subfolder mapping, creating its sub-libraries.

    The authoritative write the wizard's "Apply" performs: every ``sublib``
    rule gets (or reuses) a real sub-library row for its folder, the level and
    the full rule set are stored, and a summary is returned for the scan job's
    subtitle. Scanning is not done here — the caller queues it so the walk
    routes and tags the existing files against the freshly stored rules.

    Parameters
    ----------
    library_id : int
        The parent folder library the mapping belongs to.
    auto_tag_level : str
        The auto-tag level ('0', '1' or 'all').
    rules : iterable of dict
        One entry per mapped folder: ``{"rel_path", "mode", "tags",
        "removed", "sub_name"}`` (``sub_name`` only read for ``sublib``).

    Returns
    -------
    dict
        ``{"sub_libraries": int, "rules": int}`` — how many folders were
        promoted to sub-libraries and how many rules were stored.
    """
    parent = _query_one("SELECT path FROM library WHERE id = ?", (library_id,))
    parent_path = parent["path"] if parent else ""
    entries = []
    for rule in rules:
        entry = {
            "rel_path": rule["rel_path"],
            "mode": rule.get("mode", "keep"),
            "tags": rule.get("tags") or [],
            "removed": rule.get("removed") or [],
            "sub_library_id": None,
        }
        if entry["mode"] == "sublib":
            sub_path = _sub_abs_path(parent_path, rule["rel_path"])
            entry["sub_library_id"] = get_or_create_sub_library(
                library_id,
                rule.get("sub_name") or Path(sub_path).name,
                sub_path,
                rule["rel_path"],
            )
        entries.append(entry)
    set_library_auto_tag_level(library_id, auto_tag_level)
    replace_folder_rules(library_id, entries)
    # Drop sub-libraries whose folder was demoted back to keep/exclude; the
    # queued re-scan merges their media back into the owning library.
    keep_ids = [
        entry["sub_library_id"]
        for entry in entries
        if entry["mode"] == "sublib"
    ]
    prune_sub_libraries(library_id, keep_ids)
    return {
        "sub_libraries": sum(1 for e in entries if e["mode"] == "sublib"),
        "rules": len(entries),
    }


def get_parent_library(library_id: int):
    """Return the parent library row of a sub-library, or ``None``.

    ``None`` for a top-level library (no parent) or a missing id.
    """
    row = _query_one(
        "SELECT parent_library_id FROM library WHERE id = ?", (library_id,)
    )
    if row is None or row["parent_library_id"] is None:
        return None
    return _query_one(
        "SELECT * FROM library WHERE id = ?", (row["parent_library_id"],)
    )


def clear_orphan_rule_links(sub_library_id: int) -> None:
    """Null out any folder rule still pointing at a deleted sub-library.

    A safety net for the ``ON DELETE SET NULL`` the schema already applies to
    ``library_folder_rule.sub_library_id``: exposed so a caller deleting a
    sub-library outside a cascade can keep the parent's rules consistent.
    """
    try:
        _write(
            "UPDATE library_folder_rule SET sub_library_id = NULL "
            "WHERE sub_library_id = ?",
            (sub_library_id,),
        )
    except sqlite3.Error:
        pass
