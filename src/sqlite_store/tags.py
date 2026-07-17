"""Tag repository: categories, the catalogue and media tags.

The catalogue can hold a huge imported danbooru/e621 list, so
every listing here is bounded or paged; ``tag_category.position``
orders the categories (and a media's tags everywhere they are
joined).
"""

import sqlite3
from contextlib import closing

from src import db
from src.sqlite_store.base import chunked, _query_all, _query_one, _write


def create_tag_category(name: str, color: str = "#888888") -> int:
    """Create a tag category and return its id.

    Parameters
    ----------
    name : str
        The category name (unique).
    color : str, optional
        The category color as a hex string, used to render its tags.

    Returns
    -------
    int
        The new category's id.

    Raises
    ------
    ValueError
        If the name is empty or a category with this name already exists.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Category name must not be empty.")
    try:
        return _write(
            "INSERT INTO tag_category (name, color) VALUES (?, ?)",
            (name, color),
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"A category named {name!r} already exists.") from exc


def list_tag_categories() -> list:
    """Return all tag categories in display order (position, then name).

    Categories that were never reordered share ``position = 0`` and so keep
    the previous name-order until the user drags them (see
    :func:`reorder_tag_categories`).
    """
    return _query_all("SELECT * FROM tag_category ORDER BY position, name")


# The app-managed holding pen for auto-tags whose name is brand new (not
# reused from an existing category). It is created on demand and pruned
# when it falls empty (see :func:`prune_uncategorized_if_empty`).
UNCATEGORIZED_NAME = "Uncategorized"
_UNCATEGORIZED_COLOR = "#8a8f98"


def uncategorized_category_id() -> int:
    """Return the "Uncategorized" category id, or 0 when it does not exist."""
    row = _query_one(
        "SELECT id FROM tag_category WHERE name = ?", (UNCATEGORIZED_NAME,)
    )
    return int(row["id"]) if row else 0


def get_or_create_uncategorized_category() -> int:
    """Return the "Uncategorized" category id, creating it when missing.

    Where a WD14 auto-tag lands when its name matches no existing tag: a
    neutral holding pen the user can later recategorise from. It is removed
    again as soon as it holds no tag (see
    :func:`prune_uncategorized_if_empty`).
    """
    existing = uncategorized_category_id()
    if existing:
        return existing
    return _write(
        "INSERT INTO tag_category (name, color) VALUES (?, ?)",
        (UNCATEGORIZED_NAME, _UNCATEGORIZED_COLOR),
    )


def prune_uncategorized_if_empty() -> None:
    """Delete the "Uncategorized" category when it holds no tag anymore.

    Called after a tag leaves it (moved to a real category, or deleted), so
    the holding pen disappears the moment it is empty instead of lingering.
    """
    category_id = uncategorized_category_id()
    if not category_id:
        return
    row = _query_one(
        "SELECT COUNT(*) AS n FROM tag WHERE category_id = ?", (category_id,)
    )
    if row and row["n"] == 0:
        _write("DELETE FROM tag_category WHERE id = ?", (category_id,))


def reorder_tag_categories(ordered_ids) -> None:
    """Persist a new category display order from a list of ids.

    Each id is stamped with its index as ``position`` (0-based), so the given
    order becomes the order returned by :func:`list_tag_categories` and used to
    group a media's tags (see :func:`tags_for_media`). Unknown ids are ignored.

    Parameters
    ----------
    ordered_ids : iterable of int
        The category ids in the wanted top-to-bottom order.
    """
    with closing(db.connect()) as conn:
        with conn:
            for position, category_id in enumerate(ordered_ids):
                conn.execute(
                    "UPDATE tag_category SET position = ? WHERE id = ?",
                    (position, int(category_id)),
                )


def get_tag_category(category_id: int):
    """Return a tag category row by id, or None."""
    return _query_one(
        "SELECT * FROM tag_category WHERE id = ?", (category_id,)
    )


def update_tag_category(
    category_id: int, name: str = None, color: str = None
) -> None:
    """Update a category's name and/or color."""
    fields, params = [], []
    if name is not None:
        fields.append("name = ?")
        params.append(name.strip())
    if color is not None:
        fields.append("color = ?")
        params.append(color)
    if not fields:
        return
    params.append(category_id)
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                f"UPDATE tag_category SET {', '.join(fields)} WHERE id = ?",
                params,
            )


def delete_tag_category(category_id: int) -> None:
    """Delete a category, cascading to its tags and their media links."""
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "DELETE FROM tag_category WHERE id = ?", (category_id,)
            )


def get_or_create_tag(name: str, category_id: int) -> int:
    """Return the id of a tag in a category, creating it if absent.

    Raises
    ------
    ValueError
        If ``name`` is empty, or ``category_id`` no longer exists (e.g. the
        category was deleted from another tab while this one still held it
        selected) — reported instead of the raw ``IntegrityError`` its INSERT
        would otherwise raise on the ``category_id`` foreign key.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Tag name must not be empty.")
    if get_tag_category(category_id) is None:
        raise ValueError("That category no longer exists.")
    row = _query_one(
        "SELECT id FROM tag WHERE category_id = ? AND name = ?",
        (category_id, name),
    )
    if row is not None:
        return row["id"]
    return _write(
        "INSERT INTO tag (name, category_id) VALUES (?, ?)",
        (name, category_id),
    )


def get_or_create_tag_reuse(name: str) -> int:
    """Return an existing tag by name (any category), else create it.

    The auto-tagger's tag writer. The schema allows the same name under
    several categories (``UNIQUE (category_id, name)``), so writing a WD14
    suggestion straight through :func:`get_or_create_tag` would clone
    ``blue`` into a target category whenever ``blue`` already lives under
    another — splitting one concept across two rows. This reuses the
    existing tag wherever it is (its category is left untouched); a
    genuinely new name is created in the app-managed "Uncategorized"
    holding pen (see :func:`get_or_create_uncategorized_category`), which
    the user can recategorise from later.
    """
    existing = find_tag_by_name(name)
    if existing is not None:
        return existing["id"]
    return get_or_create_tag(name, get_or_create_uncategorized_category())


def list_tags(category_id: int = None) -> list:
    """Return tags (optionally of one category) with their category color.

    Unbounded: fetches every matching row. Safe for a single category shown
    in full (a handful to a few hundred tags), but an imported tag catalogue
    can run into the hundreds of thousands, so callers driving a dropdown or
    an on-screen list should use :func:`search_tags` or :func:`list_tags_page`
    instead, which cap how many rows come back.

    Returns
    -------
    list of sqlite3.Row
        Rows with the tag columns plus ``category_name`` and ``color``,
        ordered by category name then tag name.
    """
    query = (
        "SELECT t.*, c.name AS category_name, c.color AS color "
        "FROM tag t JOIN tag_category c ON c.id = t.category_id"
    )
    params = ()
    if category_id is not None:
        query += " WHERE t.category_id = ?"
        params = (category_id,)
    query += " ORDER BY c.name, t.name"
    return _query_all(query, params)


def _tag_relevance_order(query: str):
    """Return ``(order_by_sql, params)`` ranking tag names by match quality.

    Plain alphabetical order buries a good match deep in a large catalogue
    (e.g. searching "horse" would sort "drafthorse" and "laika horse" ahead
    of the exact tag "horse" merely because they're grouped by category
    first). This ranks an exact match first, then a name starting with the
    query, then one ending with it, then any other substring match; ties
    break on the shorter name, then alphabetically — so the closest matches
    to what was typed surface first regardless of category.
    """
    order_by = (
        "CASE WHEN LOWER(t.name) = LOWER(?) THEN 0 "
        "WHEN t.name LIKE ? THEN 1 "
        "WHEN t.name LIKE ? THEN 2 "
        "ELSE 3 END, LENGTH(t.name), t.name"
    )
    return order_by, [query, f"{query}%", f"%{query}"]


def _tag_search_relevance_order(query: str):
    """Return ``(order_by_sql, params)`` ranking search results for reuse.

    Backs :func:`search_tags` only (the live-search dropdowns used to find
    and apply a tag — see :mod:`src.media_grid`); :func:`list_tags_page`
    (the Tags tab's admin browser) keeps plain match relevance via
    :func:`_tag_relevance_order` instead, since browsing there is meant to
    stay alphabetical.

    Requires the caller's query to project a ``usage_count`` column (a
    ``LEFT JOIN media_tag ... GROUP BY t.id``, as in :func:`search_tags`).
    An exact name match still always wins outright; within every other
    match tier, a tag already applied to at least one media (``usage_count``
    > 0) is boosted ahead of one that has never been used, then ranked by
    how often it is used — the assumption being that a tag already in
    active use is more likely the one being searched for than an obscure,
    never-applied entry from a large imported catalogue.
    """
    order_by = (
        "CASE WHEN LOWER(t.name) = LOWER(?) THEN 0 "
        "WHEN t.name LIKE ? THEN 1 "
        "WHEN t.name LIKE ? THEN 2 "
        "ELSE 3 END, "
        "CASE WHEN usage_count > 0 THEN 0 ELSE 1 END, usage_count DESC, "
        "LENGTH(t.name), t.name"
    )
    return order_by, [query, f"{query}%", f"%{query}"]


# Default (no search text yet) order for search_tags: same used-tag boost as
# _tag_search_relevance_order, falling back to category then name instead of
# a match tier.
_TAG_SEARCH_BROWSE_ORDER = (
    "CASE WHEN usage_count > 0 THEN 0 ELSE 1 END, usage_count DESC, "
    "c.name, t.name"
)


def count_tags(category_id: int = None, query: str = "") -> int:
    """Return how many tags match a category and/or a name search."""
    clauses, params = [], []
    if category_id is not None:
        clauses.append("category_id = ?")
        params.append(category_id)
    if query:
        clauses.append("name LIKE ?")
        params.append(f"%{query}%")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    row = _query_one(f"SELECT COUNT(*) AS n FROM tag {where}", params)
    return row["n"] if row else 0


def list_tags_page(
    category_id: int, query: str = "", limit: int = 200, offset: int = 0
) -> list:
    """Return one page of a category's tags, optionally name-filtered.

    Bounded by ``limit``/``offset`` so a category with tens of thousands of
    tags (an imported danbooru/e621-style list can dump that many into one
    category) is paged through rather than rendered all at once. When
    searching, results are ranked by match quality (see
    :func:`_tag_relevance_order`) rather than plain alphabetical order.
    """
    clauses, params = ["t.category_id = ?"], [category_id]
    if query:
        clauses.append("t.name LIKE ?")
        params.append(f"%{query}%")
        order_by, order_params = _tag_relevance_order(query)
    else:
        order_by, order_params = "t.name", []
    where = " AND ".join(clauses)
    return _query_all(
        "SELECT t.*, c.name AS category_name, c.color AS color "
        "FROM tag t JOIN tag_category c ON c.id = t.category_id "
        f"WHERE {where} ORDER BY {order_by} LIMIT ? OFFSET ?",
        (*params, *order_params, limit, offset),
    )


_TAG_SEARCH_SELECT = (
    "SELECT t.*, c.name AS category_name, c.color AS color, "
    "COUNT(mt.media_id) AS usage_count "
    "FROM tag t "
    "JOIN tag_category c ON c.id = t.category_id "
    "LEFT JOIN media_tag mt ON mt.tag_id = t.id"
)


def search_tags(
    query: str = "", category_id: int = None, limit: int = 50, include_ids=None
) -> list:
    """Return up to ``limit`` tags matching ``query``, plus ``include_ids``.

    Backs the live-search dropdowns (see :mod:`src.media_grid`): a tag already
    selected must stay visible in its dropdown even once the search text no
    longer matches it (Gradio otherwise treats a selected value missing from
    ``choices`` as invalid), so any id in ``include_ids`` is fetched
    regardless of ``query`` and prepended to the result.

    Parameters
    ----------
    query : str, optional
        A substring to match against the tag name (empty matches everything,
        letting a dropdown show a default browse list).
    category_id : int, optional
        Restrict the search to one category.
    limit : int, optional
        The maximum number of *matched* rows (``include_ids`` rows are
        additional, not counted against this cap).
    include_ids : iterable of int, optional
        Tag ids to always include, matched or not.

    Returns
    -------
    list of sqlite3.Row
        Rows with the tag columns plus ``category_name``, ``color`` and
        ``usage_count`` (how many media currently carry the tag). Ranked by
        match quality when searching, with an extra boost for a tag already
        in use (see :func:`_tag_search_relevance_order`) — or, with no
        search text yet, by usage then category/name (see
        :data:`_TAG_SEARCH_BROWSE_ORDER`) — so e.g. searching "horse"
        surfaces the tag "horse" itself before "drafthorse", and an
        already-applied tag surfaces before a never-used one from the same
        tier.
    """
    include_ids = [int(i) for i in (include_ids or [])]
    clauses, params = [], []
    if category_id is not None:
        clauses.append("t.category_id = ?")
        params.append(category_id)
    if query:
        clauses.append("t.name LIKE ?")
        params.append(f"%{query}%")
        order_by, order_params = _tag_search_relevance_order(query)
    else:
        order_by, order_params = _TAG_SEARCH_BROWSE_ORDER, []
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = _query_all(
        f"{_TAG_SEARCH_SELECT} {where} GROUP BY t.id "
        f"ORDER BY {order_by} LIMIT ?",
        (*params, *order_params, limit),
    )
    seen = {row["id"] for row in rows}
    missing = [i for i in include_ids if i not in seen]
    if not missing:
        return rows
    placeholders = ", ".join("?" for _ in missing)
    extra = _query_all(
        f"{_TAG_SEARCH_SELECT} WHERE t.id IN ({placeholders}) GROUP BY t.id",
        missing,
    )
    return list(extra) + list(rows)


def bulk_create_tags(name_category_pairs) -> int:
    """Create many tags in a single transaction; return how many were new.

    Used by the CSV tag import (see :mod:`src.tags_gallery`), which can carry
    hundreds of thousands of rows: creating each with its own connection and
    transaction (like :func:`get_or_create_tag`) is prohibitively slow at that
    scale, so this batches every insert into one transaction and lets SQLite
    skip a name already used in its category (``INSERT OR IGNORE`` against the
    ``UNIQUE(category_id, name)`` constraint) instead of a SELECT per row.

    Parameters
    ----------
    name_category_pairs : iterable of tuple
        The ``(name, category_id)`` pairs to create.

    Raises
    ------
    ValueError
        If a ``category_id`` does not exist (its foreign key fails).
    """
    try:
        with closing(db.connect()) as conn:
            with conn:
                created = _insert_tag_pairs(conn, name_category_pairs)
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"Tag import failed: {exc}") from exc
    return created


def _insert_tag_pairs(conn, name_category_pairs) -> int:
    """Insert the pairs on an open connection; return how many were new."""
    created = 0
    for name, category_id in name_category_pairs:
        name = (name or "").strip()
        if not name:
            continue
        cursor = conn.execute(
            "INSERT OR IGNORE INTO tag (name, category_id) VALUES (?, ?)",
            (name, category_id),
        )
        if cursor.rowcount:
            created += 1
    return created


def get_tag(tag_id: int):
    """Return a tag row (with its category color) by id, or None."""
    return _query_one(
        "SELECT t.*, c.name AS category_name, c.color AS color "
        "FROM tag t JOIN tag_category c ON c.id = t.category_id "
        "WHERE t.id = ?",
        (tag_id,),
    )


def tag_usage_counts(tag_ids) -> dict:
    """Return ``{tag_id: media count}`` for many tags in one query.

    Backs the Tags-tab browser, which shows a per-tag usage count next to
    each chip. A tag attached to no media is absent from the result (the
    caller defaults it to 0), so the query only walks existing links.

    Parameters
    ----------
    tag_ids : iterable of int
        The tags to count (typically one page of a category).

    Returns
    -------
    dict
        ``{tag_id: count}``; tags with no media are omitted.
    """
    ids = [int(tag_id) for tag_id in tag_ids]
    if not ids:
        return {}
    placeholders = ", ".join("?" for _ in ids)
    return {
        row["tag_id"]: row["n"]
        for row in _query_all(
            "SELECT tag_id, COUNT(media_id) AS n FROM media_tag "
            f"WHERE tag_id IN ({placeholders}) GROUP BY tag_id",
            ids,
        )
    }


def existing_tag_names(names) -> list:
    """Return the subset of ``names`` that already exist as tags.

    A batch existence check (any category) for the subfolder-mapping wizard:
    it colours a folder's auto tag green when the name already exists and
    amber when applying will create it. One query for the whole set, so the
    wizard never fires a request per folder.
    """
    clean = []
    seen = set()
    for name in names or []:
        text = str(name).strip()
        if text and text not in seen:
            seen.add(text)
            clean.append(text)
    if not clean:
        return []
    placeholders = ", ".join("?" for _ in clean)
    rows = _query_all(
        f"SELECT DISTINCT name FROM tag WHERE name IN ({placeholders})", clean
    )
    return [row["name"] for row in rows]


def _normalized_tag(name) -> str:
    """Lowercase, trim and fold whitespace to underscores for matching.

    Mirrors :func:`src.framing.normalize_tag`, duplicated here so the low
    level store keeps no dependency on the higher-level ``framing`` module.
    """
    return "_".join(str(name or "").strip().lower().split())


def existing_normalized_tag_names(names) -> set:
    """Return the subset of ``names`` some tag carries, matched normalized.

    Like :func:`existing_tag_names` but case- and separator-insensitive: each
    input name and every tag name are lowercased with whitespace folded to
    underscores before matching, so a locked/excluded recipe tag resolves
    whether it was saved as ``Upper Body`` or ``upper_body``. Lets a caller
    tell a *deleted* tag apart from a live one. The match expression is not
    index-backed, so this scans the tag table once — meant for the handful of
    recipe tags it is asked about, never a hot per-item loop.

    Returns
    -------
    set of str
        The normalized names that exist (a subset of the normalized inputs).
    """
    wanted = {norm for norm in map(_normalized_tag, names or []) if norm}
    if not wanted:
        return set()
    placeholders = ", ".join("?" for _ in wanted)
    rows = _query_all(
        "SELECT DISTINCT lower(replace(trim(name), ' ', '_')) AS norm "
        f"FROM tag WHERE lower(replace(trim(name), ' ', '_')) IN "
        f"({placeholders})",
        tuple(wanted),
    )
    return {row["norm"] for row in rows}


def find_tag_by_name(name: str):
    """Return the first tag matching ``name`` exactly (any category), or None.

    Used by the auto-tagger to reuse an existing tag (e.g. from an imported
    danbooru/e621 catalogue) instead of creating a duplicate in its fallback
    category. When the same name exists in several categories, the
    lowest-id (oldest) one wins, deterministically.
    """
    name = (name or "").strip()
    if not name:
        return None
    return _query_one(
        "SELECT * FROM tag WHERE name = ? ORDER BY id LIMIT 1", (name,)
    )


def delete_tag(tag_id: int) -> None:
    """Delete a tag, cascading to its media links."""
    with closing(db.connect()) as conn:
        with conn:
            conn.execute("DELETE FROM tag WHERE id = ?", (tag_id,))
    prune_uncategorized_if_empty()


def _canonical_tag(rows, reserved_category_id):
    """Return the tag row to keep among same-name duplicates.

    ``rows`` are the same-name tags ordered oldest-first. The oldest copy that
    is *not* in ``reserved_category_id`` (the WD14 auto-tag holding pen) wins,
    so a tag keeps its real category; only when every copy sits in the
    reserved category does the oldest one there win.
    """
    outside = [
        row for row in rows if row["category_id"] != reserved_category_id
    ]
    return (outside or rows)[0]


def dedupe_tags(reserved_category_id=None) -> list[dict]:
    """Merge tags duplicated by name across categories into one canonical row.

    The one-off cleanup for tags the auto-tagger cloned into its fallback
    category before the reuse fix (see :func:`get_or_create_tag_reuse`): the
    same name ended up in both its real category and the WD14 holding pen.

    For each name held by more than one tag the canonical copy
    (:func:`_canonical_tag`) is kept and every other copy is merged into it —
    the media linked to a duplicate are re-pointed to the canonical tag (an
    existing link is left, never doubled), then the duplicate row is deleted
    (its ``media_tag`` / tag-grounding rows cascade away). Idempotent: a second
    run finds nothing left to merge.

    Parameters
    ----------
    reserved_category_id : int or None
        The WD14 auto-tag fallback category, whose copies are dropped in
        favour of the tag's real category. ``None`` keeps the oldest copy.

    Returns
    -------
    list of dict
        One ``{"name", "kept_tag_id", "kept_category_id", "merged"}`` per name
        that had duplicates (``merged`` is how many rows were removed).
    """
    report = []
    with closing(db.connect()) as conn:
        with conn:
            names = [
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM tag GROUP BY name HAVING COUNT(*) > 1"
                ).fetchall()
            ]
            for name in names:
                rows = conn.execute(
                    "SELECT id, category_id FROM tag "
                    "WHERE name = ? ORDER BY id",
                    (name,),
                ).fetchall()
                keep = _canonical_tag(rows, reserved_category_id)
                dups = [row["id"] for row in rows if row["id"] != keep["id"]]
                for dup in dups:
                    conn.execute(
                        "UPDATE OR IGNORE media_tag SET tag_id = ? "
                        "WHERE tag_id = ?",
                        (keep["id"], dup),
                    )
                    conn.execute("DELETE FROM tag WHERE id = ?", (dup,))
                report.append(
                    {
                        "name": name,
                        "kept_tag_id": keep["id"],
                        "kept_category_id": keep["category_id"],
                        "merged": len(dups),
                    }
                )
    return report


def move_tag(tag_id: int, category_id: int) -> int:
    """Move a tag to another category, merging on a name collision.

    The drag-and-drop of the Tags tab. A plain re-category unless the target
    already holds a tag with the same name (the ``UNIQUE (category_id, name)``
    the schema enforces): then the dragged tag is merged into that existing
    one — its media re-pointed, the dragged row deleted — instead of erroring.

    Parameters
    ----------
    tag_id : int
        The tag being moved.
    category_id : int
        The destination category.

    Returns
    -------
    int
        The surviving tag id: ``tag_id`` on a plain move, or the pre-existing
        target tag it merged into.

    Raises
    ------
    ValueError
        If the tag or the destination category no longer exists.
    """
    tag = get_tag(tag_id)
    if tag is None:
        raise ValueError("That tag no longer exists.")
    if get_tag_category(category_id) is None:
        raise ValueError("That category no longer exists.")
    if tag["category_id"] == category_id:
        return tag_id
    with closing(db.connect()) as conn:
        with conn:
            existing = conn.execute(
                "SELECT id FROM tag WHERE category_id = ? AND name = ?",
                (category_id, tag["name"]),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "UPDATE tag SET category_id = ? WHERE id = ?",
                    (category_id, tag_id),
                )
                kept = tag_id
            else:
                kept = existing["id"]
                conn.execute(
                    "UPDATE OR IGNORE media_tag SET tag_id = ? "
                    "WHERE tag_id = ?",
                    (kept, tag_id),
                )
                conn.execute("DELETE FROM tag WHERE id = ?", (tag_id,))
    # The tag left its old category, which may have been the holding pen.
    prune_uncategorized_if_empty()
    return kept


def rename_tag(tag_id: int, new_name: str) -> None:
    """Rename a tag in place.

    Raises
    ------
    ValueError
        If ``new_name`` is empty, the tag no longer exists, or another tag
        in the same category already uses that name.
    """
    new_name = (new_name or "").strip()
    if not new_name:
        raise ValueError("Tag name must not be empty.")
    if get_tag(tag_id) is None:
        raise ValueError("That tag no longer exists.")
    try:
        _write("UPDATE tag SET name = ? WHERE id = ?", (new_name, tag_id))
    except sqlite3.IntegrityError as exc:
        raise ValueError(
            f"A tag named {new_name!r} already exists in this category."
        ) from exc


def add_tag_to_media(media_id: int, tag_id: int, source: str = None) -> None:
    """Attach a tag to a media (idempotent), recording its provenance.

    ``source`` is ``'library'`` for a Libraries "Bulk tags" push and NULL
    for any genuine per-media tag (WD14 auto-tag or manual add). A NULL add
    promotes an existing ``'library'`` link to NULL — a media the tagger (or
    the user) touches individually then counts as tagged — while a
    ``'library'`` add never downgrades an existing per-media link.
    """
    with closing(db.connect()) as conn:
        with conn:
            if source == "library":
                conn.execute(
                    "INSERT OR IGNORE INTO media_tag "
                    "(media_id, tag_id, source) VALUES (?, ?, 'library')",
                    (media_id, tag_id),
                )
            else:
                conn.execute(
                    "INSERT INTO media_tag (media_id, tag_id, source) "
                    "VALUES (?, ?, NULL) ON CONFLICT (media_id, tag_id) "
                    "DO UPDATE SET source = NULL",
                    (media_id, tag_id),
                )


def remove_tag_from_media(media_id: int, tag_id: int) -> None:
    """Detach a tag from a media (the tag itself is kept)."""
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "DELETE FROM media_tag WHERE media_id = ? AND tag_id = ?",
                (media_id, tag_id),
            )


def add_tags_to_library(library_id, tag_ids) -> int:
    """Attach tags to every media in a library, or in every library.

    The Libraries tab's "add to all media" bulk action: each tag is linked
    to every media that has a file in the library (the same membership as
    :func:`count_media_in_library`). Existing links are left untouched and
    each media's tags stay individually editable afterwards — these are
    plain ``media_tag`` rows, not a persisted library setting.

    Parameters
    ----------
    library_id : int or None
        The library whose media receive the tags; ``None`` spans every media
        that belongs to any library (the "all libraries" variant).
    tag_ids : iterable of int
        The tags to attach.

    Returns
    -------
    int
        How many new media-tag links were created.
    """
    ids = [int(tag_id) for tag_id in tag_ids or []]
    if not ids:
        return 0
    scope = "" if library_id is None else " AND mf.library_id = ?"
    added = 0
    with closing(db.connect()) as conn:
        with conn:
            for tag_id in ids:
                params = (tag_id,)
                if library_id is not None:
                    params += (library_id,)
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO media_tag "
                    "(media_id, tag_id, source) "
                    "SELECT DISTINCT mf.media_id, ?, 'library' "
                    "FROM media_file mf "
                    "JOIN media m ON m.id = mf.media_id "
                    "WHERE m.deleted_at IS NULL" + scope,
                    params,
                )
                added += cursor.rowcount
    return added


def remove_tags_from_library(library_id, tag_ids) -> int:
    """Detach tags from every media in a library, or in every library.

    The "remove from all media" counterpart of :func:`add_tags_to_library`:
    each tag is unlinked from every media that has a file in the library
    (or, when ``library_id`` is ``None``, in any library). The tags
    themselves — and, in the per-library case, their links to media of other
    libraries — are kept.

    Parameters
    ----------
    library_id : int or None
        The library whose media lose the tags; ``None`` spans every media
        that belongs to any library (the "all libraries" variant).
    tag_ids : iterable of int
        The tags to detach.

    Returns
    -------
    int
        How many media-tag links were removed.
    """
    ids = [int(tag_id) for tag_id in tag_ids or []]
    if not ids:
        return 0
    placeholders = ", ".join("?" for _ in ids)
    scope = "" if library_id is None else " AND mf.library_id = ?"
    params = tuple(ids) if library_id is None else (*ids, library_id)
    with closing(db.connect()) as conn:
        with conn:
            cursor = conn.execute(
                f"DELETE FROM media_tag WHERE tag_id IN ({placeholders}) "
                "AND media_id IN ("
                "SELECT mf.media_id FROM media_file mf "
                "JOIN media m ON m.id = mf.media_id "
                "WHERE m.deleted_at IS NULL" + scope + ")",
                params,
            )
            return cursor.rowcount


def tags_for_media(media_id: int) -> list:
    """Return a media's tags with their category name and color.

    Returns
    -------
    list of sqlite3.Row
        Rows with ``id``, ``name``, ``category_id``, ``category_name`` and
        ``color``, ordered by category name then tag name.
    """
    return _query_all(
        "SELECT t.id AS id, t.name AS name, t.category_id AS category_id, "
        "c.name AS category_name, c.color AS color "
        "FROM media_tag mt "
        "JOIN tag t ON t.id = mt.tag_id "
        "JOIN tag_category c ON c.id = t.category_id "
        "WHERE mt.media_id = ? ORDER BY c.position, c.name, t.name",
        (media_id,),
    )


def tags_for_media_bulk(media_ids) -> dict:
    """Return the tags of many media in one query.

    Same rows and per-media order as :func:`tags_for_media`; one round-trip
    for a whole grid page instead of a query per card.

    Returns
    -------
    dict
        ``{media_id: [tag rows]}``; an untagged media is absent.
    """
    ids = [int(m) for m in media_ids]
    if not ids:
        return {}
    grouped: dict = {}
    # Chunk the ids: a whole-library pool (autobuild) can exceed SQLite's
    # bound-variable limit in one IN () list. Each media_id lands in a single
    # chunk, so its per-tag order is still the ORDER BY within that query.
    for chunk in chunked(ids):
        placeholders = ", ".join("?" for _ in chunk)
        for row in _query_all(
            "SELECT mt.media_id AS media_id, t.id AS id, t.name AS name, "
            "t.category_id AS category_id, c.name AS category_name, "
            "c.color AS color "
            "FROM media_tag mt "
            "JOIN tag t ON t.id = mt.tag_id "
            "JOIN tag_category c ON c.id = t.category_id "
            f"WHERE mt.media_id IN ({placeholders}) "
            "ORDER BY mt.media_id, c.position, c.name, t.name",
            chunk,
        ):
            grouped.setdefault(row["media_id"], []).append(row)
    return grouped


def media_tag_names(media_id: int) -> list:
    """Return a media's tag names in category display order.

    The order follows :func:`tags_for_media` (category position, then category
    name, then tag name) — the order used to build the deployable ``tags``
    caption (see :data:`src.storage.TAGS_TYPE`).
    """
    return [row["name"] for row in tags_for_media(media_id)]
