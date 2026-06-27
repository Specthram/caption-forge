"""Caption repository: types, revisions and dataset heads.

Implements the revision model described in the package
docstring: an append-only ``caption_revision`` chain per
``(media, caption_type)``, a mutable head, and per-dataset
``follow``/``pinned`` modes recorded in ``dataset_caption``.
"""

from contextlib import closing

from src import db
from src.sqlite_store.base import _query_all, _query_one, _write


def list_caption_types() -> list:
    """Return all caption types ordered by name."""
    return _query_all("SELECT * FROM caption_type ORDER BY name")


def get_or_create_caption_type(name: str, file_extension: str = None) -> int:
    """Return the id of a caption type, creating it if absent."""
    name = (name or "").strip()
    row = _query_one("SELECT id FROM caption_type WHERE name = ?", (name,))
    if row is not None:
        return row["id"]
    return _write(
        "INSERT INTO caption_type (name, file_extension) VALUES (?, ?)",
        (name, file_extension or name),
    )


def get_caption_type(caption_type_id: int):
    """Return a caption type row by id, or None."""
    return _query_one(
        "SELECT * FROM caption_type WHERE id = ?", (caption_type_id,)
    )


def get_or_create_caption(media_id: int, caption_type_id: int) -> int:
    """Return the caption id for a (media, type), creating it if absent."""
    row = _query_one(
        "SELECT id FROM caption WHERE media_id = ? AND caption_type_id = ?",
        (media_id, caption_type_id),
    )
    if row is not None:
        return row["id"]
    return _write(
        "INSERT INTO caption (media_id, caption_type_id) VALUES (?, ?)",
        (media_id, caption_type_id),
    )


def get_caption(media_id: int, caption_type_id: int):
    """Return the caption row for a (media, type), or None."""
    return _query_one(
        "SELECT * FROM caption WHERE media_id = ? AND caption_type_id = ?",
        (media_id, caption_type_id),
    )


def head_revision_id(caption_id: int):
    """Return a caption's head revision id, or None."""
    row = _query_one(
        "SELECT head_revision_id FROM caption WHERE id = ?", (caption_id,)
    )
    return row["head_revision_id"] if row else None


def list_revisions(caption_id: int) -> list:
    """Return a caption's revisions, newest first."""
    return _query_all(
        "SELECT * FROM caption_revision WHERE caption_id = ? "
        "ORDER BY id DESC",
        (caption_id,),
    )


def get_revision(revision_id: int):
    """Return a revision row by id, or None."""
    return _query_one(
        "SELECT * FROM caption_revision WHERE id = ?", (revision_id,)
    )


def _insert_revision(
    conn, caption_id: int, content: str, message, parent_revision_id
) -> int:
    """Insert a revision inside an open transaction; return its id."""
    return conn.execute(
        "INSERT INTO caption_revision "
        "(caption_id, parent_revision_id, content, message) "
        "VALUES (?, ?, ?, ?)",
        (caption_id, parent_revision_id, content, message),
    ).lastrowid


def get_dataset_caption(dataset_id: int, caption_id: int):
    """Return the dataset_caption row for a (dataset, caption), or None."""
    return _query_one(
        "SELECT * FROM dataset_caption "
        "WHERE dataset_id = ? AND caption_id = ?",
        (dataset_id, caption_id),
    )


def effective_revision_id(dataset_id: int, caption_id: int):
    """Return the revision a dataset effectively shows for a caption.

    ``pinned`` resolves to the dataset's frozen revision; ``follow`` (or no
    assignment row) resolves to the caption's head revision.
    """
    row = get_dataset_caption(dataset_id, caption_id)
    if row is not None and row["mode"] == "pinned" and row["revision_id"]:
        return row["revision_id"]
    return head_revision_id(caption_id)


def set_dataset_caption(
    dataset_id: int, caption_id: int, mode: str, revision_id=None
) -> None:
    """Create or update a dataset's assignment for a caption.

    Parameters
    ----------
    dataset_id : int
        The dataset.
    caption_id : int
        The caption.
    mode : str
        ``"follow"`` (track the head) or ``"pinned"`` (freeze a revision).
    revision_id : int, optional
        The revision to pin; required meaning only when ``mode`` is
        ``"pinned"``.
    """
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "INSERT INTO dataset_caption "
                "(dataset_id, caption_id, mode, revision_id) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT (dataset_id, caption_id) DO UPDATE SET "
                "mode = excluded.mode, revision_id = excluded.revision_id",
                (dataset_id, caption_id, mode, revision_id),
            )


def save_caption(
    dataset_id: int,
    media_id: int,
    caption_type_id: int,
    content: str,
    scope: str = "type",
    message: str = None,
) -> int:
    """Save caption text as a new revision and return its id.

    Parameters
    ----------
    dataset_id : int
        The dataset the edit is made from.
    media_id : int
        The captioned media.
    caption_type_id : int
        The caption type being edited.
    content : str
        The new caption text.
    scope : str, optional
        ``"type"`` advances the shared head (every ``follow`` dataset sees it);
        ``"dataset"`` branches a new revision and pins only this dataset.
    message : str, optional
        An optional revision message.

    Returns
    -------
    int
        The id of the created revision.
    """
    caption_id = get_or_create_caption(media_id, caption_type_id)
    with closing(db.connect()) as conn:
        with conn:
            cap = conn.execute(
                "SELECT head_revision_id FROM caption WHERE id = ?",
                (caption_id,),
            ).fetchone()
            head = cap["head_revision_id"]
            assignment = conn.execute(
                "SELECT id, mode, revision_id FROM dataset_caption "
                "WHERE dataset_id = ? AND caption_id = ?",
                (dataset_id, caption_id),
            ).fetchone()

            if scope == "dataset":
                if (
                    assignment is not None
                    and assignment["mode"] == "pinned"
                    and assignment["revision_id"]
                ):
                    parent = assignment["revision_id"]
                else:
                    parent = head
                revision = _insert_revision(
                    conn, caption_id, content, message, parent
                )
                if assignment is None:
                    conn.execute(
                        "INSERT INTO dataset_caption "
                        "(dataset_id, caption_id, mode, revision_id) "
                        "VALUES (?, ?, 'pinned', ?)",
                        (dataset_id, caption_id, revision),
                    )
                else:
                    conn.execute(
                        "UPDATE dataset_caption "
                        "SET mode = 'pinned', revision_id = ? WHERE id = ?",
                        (revision, assignment["id"]),
                    )
            else:
                revision = _insert_revision(
                    conn, caption_id, content, message, head
                )
                conn.execute(
                    "UPDATE caption SET head_revision_id = ? WHERE id = ?",
                    (revision, caption_id),
                )
                if assignment is None:
                    conn.execute(
                        "INSERT INTO dataset_caption "
                        "(dataset_id, caption_id, mode) "
                        "VALUES (?, ?, 'follow')",
                        (dataset_id, caption_id),
                    )
    return revision


def amend_caption(
    dataset_id: int,
    media_id: int,
    caption_type_id: int,
    content: str,
) -> int:
    """Overwrite the dataset's current revision in place; return its id.

    The write autosave uses: instead of appending a revision on every pause
    in typing (which floods the history), it rewrites the *effective*
    revision the dataset shows — its pinned revision when one is set, else
    the shared head. When the caption has no revision yet, the first one is
    created exactly like a ``type``-scope :func:`save_caption` (a head plus a
    ``follow`` assignment). An explicit Save still snapshots a new revision.
    """
    caption_id = get_or_create_caption(media_id, caption_type_id)
    with closing(db.connect()) as conn:
        with conn:
            cap = conn.execute(
                "SELECT head_revision_id FROM caption WHERE id = ?",
                (caption_id,),
            ).fetchone()
            head = cap["head_revision_id"]
            assignment = conn.execute(
                "SELECT mode, revision_id FROM dataset_caption "
                "WHERE dataset_id = ? AND caption_id = ?",
                (dataset_id, caption_id),
            ).fetchone()
            if (
                assignment is not None
                and assignment["mode"] == "pinned"
                and assignment["revision_id"]
            ):
                target = assignment["revision_id"]
            else:
                target = head
            if target is None:
                revision = _insert_revision(
                    conn, caption_id, content, None, None
                )
                conn.execute(
                    "UPDATE caption SET head_revision_id = ? WHERE id = ?",
                    (revision, caption_id),
                )
                if assignment is None:
                    conn.execute(
                        "INSERT INTO dataset_caption "
                        "(dataset_id, caption_id, mode) "
                        "VALUES (?, ?, 'follow')",
                        (dataset_id, caption_id),
                    )
                return revision
            conn.execute(
                "UPDATE caption_revision SET content = ? WHERE id = ?",
                (content, target),
            )
            return target


def read_caption(dataset_id: int, media_id: int, caption_type_id: int) -> str:
    """Return the caption text a dataset shows for a (media, type), or ""."""
    caption = get_caption(media_id, caption_type_id)
    if caption is None:
        return ""
    revision_id = effective_revision_id(dataset_id, caption["id"])
    if not revision_id:
        return ""
    revision = get_revision(revision_id)
    return revision["content"] if revision else ""


def dataset_captions_bulk(
    dataset_id: int, media_ids, caption_type_id: int
) -> dict:
    """Return the caption texts a dataset shows for many media, one query.

    Resolves each media's effective revision exactly like
    :func:`read_caption` (the dataset's pinned revision when one is set,
    else the caption's head) but for a whole page/dataset at once — the
    per-media form costs three queries per call.

    Returns
    -------
    dict
        ``{media_id: text}``; a media without a caption of this type (or
        with an empty resolution) is absent.
    """
    ids = [int(m) for m in media_ids]
    if not ids:
        return {}
    placeholders = ", ".join("?" for _ in ids)
    rows = _query_all(
        "SELECT c.media_id AS media_id, r.content AS content "
        "FROM caption c "
        "LEFT JOIN dataset_caption dc "
        "ON dc.caption_id = c.id AND dc.dataset_id = ? "
        "LEFT JOIN caption_revision r ON r.id = COALESCE("
        "CASE WHEN dc.mode = 'pinned' THEN dc.revision_id END, "
        "c.head_revision_id) "
        f"WHERE c.caption_type_id = ? AND c.media_id IN ({placeholders})",
        [dataset_id, caption_type_id] + ids,
    )
    return {
        row["media_id"]: row["content"]
        for row in rows
        if row["content"] is not None
    }


def captions_for_type_bulk(media_ids, caption_type_id: int) -> list:
    """Return the caption rows of many media for one type, one query."""
    ids = [int(m) for m in media_ids]
    if not ids:
        return []
    placeholders = ", ".join("?" for _ in ids)
    return _query_all(
        "SELECT * FROM caption "
        f"WHERE caption_type_id = ? AND media_id IN ({placeholders})",
        [caption_type_id] + ids,
    )


def revisions_bulk(caption_ids) -> dict:
    """Return the revisions of many captions (newest first), one query.

    Returns
    -------
    dict
        ``{caption_id: [revision rows, newest first]}``.
    """
    ids = [int(c) for c in caption_ids]
    if not ids:
        return {}
    placeholders = ", ".join("?" for _ in ids)
    grouped = {}
    for row in _query_all(
        "SELECT * FROM caption_revision "
        f"WHERE caption_id IN ({placeholders}) "
        "ORDER BY caption_id, id DESC",
        ids,
    ):
        grouped.setdefault(row["caption_id"], []).append(row)
    return grouped


def dataset_caption_bulk(dataset_id: int, caption_ids) -> dict:
    """Return a dataset's assignment rows for many captions, one query.

    Returns
    -------
    dict
        ``{caption_id: dataset_caption row}``; captions the dataset has no
        assignment for are absent.
    """
    ids = [int(c) for c in caption_ids]
    if not ids:
        return {}
    placeholders = ", ".join("?" for _ in ids)
    return {
        row["caption_id"]: row
        for row in _query_all(
            "SELECT * FROM dataset_caption "
            f"WHERE dataset_id = ? AND caption_id IN ({placeholders})",
            [dataset_id] + ids,
        )
    }


_UNUSED_CAPTION_WHERE = "media_id NOT IN (SELECT media_id FROM dataset_media)"


def unused_caption_count() -> int:
    """Return how many captions belong to a media in no dataset.

    A caption is identified by ``(media, caption_type)``. When its media is a
    member of no dataset it is shown and editable nowhere, so the caption (in
    every type) is unused.
    """
    row = _query_one(
        f"SELECT COUNT(*) AS n FROM caption WHERE {_UNUSED_CAPTION_WHERE}"
    )
    return row["n"] if row else 0


def delete_unused_captions() -> int:
    """Delete every caption of a media in no dataset; return the count removed.

    Removing the ``caption`` rows cascades (foreign keys on) to their revision
    history and any leftover dataset links. The media rows themselves are kept,
    so the media stay in the library and can be recaptioned later.
    """
    with closing(db.connect()) as conn:
        with conn:
            cursor = conn.execute(
                f"DELETE FROM caption WHERE {_UNUSED_CAPTION_WHERE}"
            )
            return cursor.rowcount


# A revision is "in use" when it is the head of its caption (the active,
# type-wide text) or when a dataset pins/points at it (dataset_caption
# records an explicit ``revision_id`` for a pinned caption). Every other
# revision is superseded history: kept only so the caption's timeline can be
# browsed, and safe to prune. NULLs are excluded so an ``IN`` never matches a
# NULL head/revision pointer.
_UNUSED_REVISION_WHERE = (
    "id NOT IN (SELECT head_revision_id FROM caption "
    "WHERE head_revision_id IS NOT NULL) "
    "AND id NOT IN (SELECT revision_id FROM dataset_caption "
    "WHERE revision_id IS NOT NULL)"
)


def unused_revision_count() -> int:
    """Return how many caption revisions are neither active nor pinned.

    Counts the superseded history: revisions that are not any caption's head
    and not referenced by a dataset. Pruning them (see
    :func:`prune_unused_revisions`) keeps the current caption of every media
    intact while dropping its edit trail.
    """
    row = _query_one(
        "SELECT COUNT(*) AS n FROM caption_revision "
        f"WHERE {_UNUSED_REVISION_WHERE}"
    )
    return row["n"] if row else 0


def prune_unused_revisions() -> int:
    """Delete every superseded caption revision; return how many were removed.

    Drops the revisions that are neither a caption's head nor pinned by a
    dataset. A revision's review, grounding and score children reference it
    ``ON DELETE CASCADE`` and vanish with it. ``parent_revision_id`` carries
    no cascade, so the surviving head could still point into the deleted set;
    those links are cleared to NULL first, both to unbreak the survivors and
    to let the doomed rows delete in any order without tripping the foreign
    key. The result is a caption whose head is untouched but whose ancestry
    chain is collapsed.
    """
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "UPDATE caption_revision SET parent_revision_id = NULL "
                "WHERE parent_revision_id IN ("
                "SELECT id FROM caption_revision "
                f"WHERE {_UNUSED_REVISION_WHERE})"
            )
            cursor = conn.execute(
                "DELETE FROM caption_revision "
                f"WHERE {_UNUSED_REVISION_WHERE}"
            )
            return cursor.rowcount
