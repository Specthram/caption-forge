"""Dataset repository: CRUD, media links and trigger words.

The ``dataset_media`` link carries the per-dataset ``hidden``
flag and ``repeats`` (the deploy copy count); trigger words are
per-dataset and prefixed to every deployed caption.
"""

import sqlite3
from contextlib import closing

from src import db
from src.sqlite_store.base import _query_all, _query_one, _write
from src.sqlite_store.media import _VIDEO_EXTS, _media_dicts


def create_dataset(name: str, description: str = "") -> int:
    """Create a dataset and return its id.

    Parameters
    ----------
    name : str
        The dataset name (unique).
    description : str, optional
        A free-text description.

    Returns
    -------
    int
        The new dataset's id.

    Raises
    ------
    ValueError
        If a dataset with this name already exists, or the name is empty.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Dataset name must not be empty.")
    try:
        return _write(
            "INSERT INTO dataset (name, description) VALUES (?, ?)",
            (name, description),
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"A dataset named {name!r} already exists.") from exc


def list_datasets() -> list:
    """Return all datasets ordered by name."""
    return _query_all("SELECT * FROM dataset ORDER BY name")


def get_dataset(dataset_id: int):
    """Return a dataset row by id, or None."""
    return _query_one("SELECT * FROM dataset WHERE id = ?", (dataset_id,))


def get_dataset_by_name(name: str):
    """Return a dataset row by name, or None."""
    return _query_one("SELECT * FROM dataset WHERE name = ?", (name,))


def update_dataset(
    dataset_id: int,
    name: str = None,
    description: str = None,
    deploy_name: str = None,
    deploy_resolution: int = None,
) -> None:
    """Update a dataset's fields (and its ``updated_at``).

    ``deploy_name`` is the deploy sub-folder name; an empty (whitespace-only)
    value is stored as NULL, meaning "use the dataset name".
    ``deploy_resolution`` is the deploy resize target in pixels (the shortest
    image side); ``0`` disables resizing (media deployed verbatim).
    """
    fields, params = [], []
    if name is not None:
        fields.append("name = ?")
        params.append(name.strip())
    if description is not None:
        fields.append("description = ?")
        params.append(description)
    if deploy_name is not None:
        fields.append("deploy_name = ?")
        params.append(deploy_name.strip() or None)
    if deploy_resolution is not None:
        fields.append("deploy_resolution = ?")
        params.append(max(0, int(deploy_resolution)))
    if not fields:
        return
    fields.append("updated_at = datetime('now')")
    params.append(dataset_id)
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                f"UPDATE dataset SET {', '.join(fields)} WHERE id = ?", params
            )


def delete_dataset(dataset_id: int) -> None:
    """Delete a dataset and its links (cascades to dataset_* rows)."""
    with closing(db.connect()) as conn:
        with conn:
            conn.execute("DELETE FROM dataset WHERE id = ?", (dataset_id,))


def add_media_to_dataset(
    dataset_id: int, media_id: int, hidden: bool = False
) -> int:
    """Link a media to a dataset (idempotent); return the link row id."""
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO dataset_media "
                "(dataset_id, media_id, hidden) VALUES (?, ?, ?)",
                (dataset_id, media_id, int(hidden)),
            )
            row = conn.execute(
                "SELECT id FROM dataset_media "
                "WHERE dataset_id = ? AND media_id = ?",
                (dataset_id, media_id),
            ).fetchone()
    return row["id"]


def remove_media_from_dataset(dataset_id: int, media_id: int) -> None:
    """Unlink a media from a dataset (its captions are left intact)."""
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "DELETE FROM dataset_media "
                "WHERE dataset_id = ? AND media_id = ?",
                (dataset_id, media_id),
            )


def add_media_ids_to_dataset(dataset_id: int, media_ids) -> int:
    """Link many media to a dataset in one transaction.

    Idempotent per media (``INSERT OR IGNORE``): already-linked media are
    left untouched. One transaction for the whole batch, so a bulk "Add all"
    over a large filtered library is a single write instead of one
    connection per media.

    Returns
    -------
    int
        How many media were actually (newly) linked.
    """
    with closing(db.connect()) as conn:
        with conn:
            before = conn.total_changes
            conn.executemany(
                "INSERT OR IGNORE INTO dataset_media (dataset_id, media_id) "
                "VALUES (?, ?)",
                [(dataset_id, int(media_id)) for media_id in media_ids],
            )
            return conn.total_changes - before


def remove_media_ids_from_dataset(dataset_id: int, media_ids) -> int:
    """Unlink many media from a dataset in one transaction.

    The mirror of :func:`add_media_ids_to_dataset` (captions are left
    intact, unknown ids are ignored).

    Returns
    -------
    int
        How many media were actually unlinked.
    """
    with closing(db.connect()) as conn:
        with conn:
            before = conn.total_changes
            conn.executemany(
                "DELETE FROM dataset_media "
                "WHERE dataset_id = ? AND media_id = ?",
                [(dataset_id, int(media_id)) for media_id in media_ids],
            )
            return conn.total_changes - before


def set_media_hidden(dataset_id: int, media_id: int, hidden: bool) -> None:
    """Set a media's ``hidden`` flag within a dataset (export exclusion)."""
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "UPDATE dataset_media SET hidden = ? "
                "WHERE dataset_id = ? AND media_id = ?",
                (int(hidden), dataset_id, media_id),
            )


def is_media_hidden(dataset_id: int, media_id: int) -> bool:
    """Return a media's ``hidden`` flag in a dataset (False if absent)."""
    row = _query_one(
        "SELECT hidden FROM dataset_media "
        "WHERE dataset_id = ? AND media_id = ?",
        (dataset_id, media_id),
    )
    return bool(row["hidden"]) if row is not None else False


def set_media_repeats(dataset_id: int, media_id: int, repeats: int) -> None:
    """Set how many copies of a media the dataset deploys (min 1)."""
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "UPDATE dataset_media SET repeats = ? "
                "WHERE dataset_id = ? AND media_id = ?",
                (max(1, int(repeats)), dataset_id, media_id),
            )


def get_media_repeats(dataset_id: int, media_id: int) -> int:
    """Return a media's deploy repeat count in a dataset (1 if absent)."""
    row = _query_one(
        "SELECT repeats FROM dataset_media "
        "WHERE dataset_id = ? AND media_id = ?",
        (dataset_id, media_id),
    )
    return max(1, row["repeats"]) if row is not None else 1


_DATASET_MEDIA_WHERE = (
    "SELECT m.id AS id, m.sha256 AS sha256, "
    "m.file_extension AS file_extension, m.width AS width, "
    "m.height AS height, m.parent_media_id AS parent_media_id, "
    "m.crop_rect AS crop_rect, m.crop_ratio AS crop_ratio, "
    "dm.hidden AS dm_hidden, dm.repeats AS dm_repeats "
    "FROM dataset_media dm JOIN media m ON m.id = dm.media_id "
    "WHERE dm.dataset_id = ? AND m.deleted_at IS NULL"
)
# A crop (see src.crops) owns no media_file row, so it sorts under its
# *parent's* file name and lands right after the image it frames instead of
# ahead of the whole grid.
_DATASET_MEDIA_ORDER = (
    " ORDER BY (SELECT MIN(mf.path) FROM media_file mf "
    "WHERE mf.media_id = COALESCE(m.parent_media_id, m.id)), "
    "m.parent_media_id IS NOT NULL, m.id"
)
_DATASET_MEDIA_SELECT = _DATASET_MEDIA_WHERE + _DATASET_MEDIA_ORDER


def _media_id_filter_sql(media_id_filter):
    """Return an ``AND media_id IN (...)`` fragment and its bind params.

    ``media_id_filter`` restricts a dataset listing to a subset of media
    (the Caption tab's review filter passes the flagged ids). ``None`` (or an
    empty collection) yields an empty fragment; callers short-circuit an
    empty filter to "match nothing" before reaching SQL.
    """
    if not media_id_filter:
        return "", []
    ids = [int(mid) for mid in media_id_filter]
    marks = ",".join("?" * len(ids))
    return f" AND dm.media_id IN ({marks})", ids


def _dataset_media_dicts(rows, quality_metric_selected=None) -> list:
    """Convert dataset-media rows to dicts, files batched, flags carried."""
    dicts = _media_dicts(
        rows,
        hidden_by_id={row["id"]: row["dm_hidden"] for row in rows},
        quality_metric_selected=quality_metric_selected,
    )
    for item, row in zip(dicts, rows):
        item["repeats"] = row["dm_repeats"]
    return dicts


def media_in_dataset(dataset_id: int, quality_metric_selected=None) -> list:
    """Return the media in a dataset as dicts (with their ``hidden`` flag).

    Materializes the whole dataset (a disk stat per media, files batched in
    one query); grid hot paths page with :func:`count_media_in_dataset` +
    :func:`media_in_dataset_page` instead.

    Parameters
    ----------
    dataset_id : int
        The dataset to list.
    quality_metric_selected : str, optional
        The metric each dict's quality badge should reflect (see
        :func:`src.sqlite_store.media._media_dicts`).

    Returns
    -------
    list of dict
        One dict per linked media (see :func:`_media_dict`) plus ``hidden``
        and ``repeats`` (the deploy repeat count), ordered by file name.
    """
    return _dataset_media_dicts(
        _query_all(_DATASET_MEDIA_SELECT, (dataset_id,)),
        quality_metric_selected=quality_metric_selected,
    )


def count_media_in_dataset(dataset_id: int, media_id_filter=None) -> int:
    """Return how many media a dataset links (no disk access).

    ``media_id_filter`` optionally restricts the count to a subset of media
    (the Caption tab's review filter); an empty collection means "none".
    """
    if media_id_filter is not None and not media_id_filter:
        return 0
    filter_sql, filter_params = _media_id_filter_sql(media_id_filter)
    row = _query_one(
        "SELECT COUNT(*) AS n "
        "FROM dataset_media dm JOIN media m ON m.id = dm.media_id "
        "WHERE dm.dataset_id = ? AND m.deleted_at IS NULL" + filter_sql,
        (dataset_id, *filter_params),
    )
    return row["n"] if row else 0


def media_in_dataset_page(
    dataset_id: int,
    offset: int,
    limit: int,
    quality_metric_selected=None,
    media_id_filter=None,
) -> list:
    """Return one page of a dataset's media as dicts (file-name order).

    Only the returned rows have their effective file resolved (a disk stat
    per media), so paging a large dataset never stats every file — unlike
    :func:`media_in_dataset`. Backs the Caption tab's gallery.

    ``quality_metric_selected`` selects which metric each card's quality
    badge reflects (see :func:`src.sqlite_store.media._media_dicts`).
    ``media_id_filter`` optionally restricts the page to a subset of media
    (the review filter); an empty collection yields no rows.
    """
    if media_id_filter is not None and not media_id_filter:
        return []
    filter_sql, filter_params = _media_id_filter_sql(media_id_filter)
    query = (
        _DATASET_MEDIA_WHERE
        + filter_sql
        + _DATASET_MEDIA_ORDER
        + " LIMIT ? OFFSET ?"
    )
    return _dataset_media_dicts(
        _query_all(
            query,
            (dataset_id, *filter_params, int(limit), int(offset)),
        ),
        quality_metric_selected=quality_metric_selected,
    )


def media_ids_in_dataset(dataset_id: int) -> set:
    """Return the set of media ids linked to a dataset."""
    rows = _query_all(
        "SELECT media_id FROM dataset_media WHERE dataset_id = ?",
        (dataset_id,),
    )
    return {row["media_id"] for row in rows}


def is_media_in_dataset(dataset_id: int, media_id: int) -> bool:
    """Return whether a media is linked to a dataset."""
    return (
        _query_one(
            "SELECT 1 FROM dataset_media "
            "WHERE dataset_id = ? AND media_id = ?",
            (dataset_id, media_id),
        )
        is not None
    )


def _dataset_pending(dataset_id: int, missing_sql: str, params) -> list:
    """Return a dataset's live, visible images missing some computation.

    Shared body of the two dataset-scoped "pending" readers the quality
    report runs on. Mirrors the library-scoped
    :func:`src.sqlite_store.media.media_pending_score` filters (live, not
    hidden, not discarded) and adds the dataset link plus its own
    ``dataset_media.hidden`` flag — a media hidden inside the dataset is
    never deployed, so it is never trained on and never evaluated. Videos
    are excluded: every visual analysis of the report is image-only.
    """
    placeholders = ", ".join("?" for _ in _VIDEO_EXTS)
    where = [
        "m.deleted_at IS NULL",
        "m.hidden = 0",
        "m.discarded_at IS NULL",
        f"lower(m.file_extension) NOT IN ({placeholders})",
        "dm.hidden = 0",
    ]
    query_params = [dataset_id, *_VIDEO_EXTS]
    if missing_sql:
        where.append(missing_sql)
        query_params.extend(params)
    rows = _query_all(
        "SELECT m.* FROM media m "
        "JOIN dataset_media dm ON dm.media_id = m.id "
        "WHERE dm.dataset_id = ? AND "
        + " AND ".join(where)
        + " ORDER BY m.id",
        query_params,
    )
    return _media_dicts(rows)


def dataset_media_pending_score(
    dataset_id: int, metric_id: str, force: bool = False
) -> list:
    """Return the dataset images a quality report must still score.

    Parameters
    ----------
    dataset_id : int
        The dataset being evaluated.
    metric_id : str
        The :data:`src.quality.QUALITY_METRICS` key to score with.
    force : bool, optional
        When true every image is returned (a full re-score); otherwise
        only those with no ``media_quality`` row for ``metric_id`` yet, so
        a repeat evaluation only computes what it is missing.

    Returns
    -------
    list of dict
        The matching media as :func:`_media_dict` dicts, oldest first.
    """
    if force:
        return _dataset_pending(dataset_id, "", ())
    return _dataset_pending(
        dataset_id,
        "m.id NOT IN (SELECT media_id FROM media_quality "
        "WHERE metric_id = ?)",
        (metric_id,),
    )


def dataset_media_pending_embedding(
    dataset_id: int, model_id: str, force: bool = False
) -> list:
    """Return the dataset images a quality report must still embed.

    Same shape and filters as :func:`dataset_media_pending_score`, for the
    DINOv2 stage of the run (see :data:`src.embeddings.MODEL_ID`).
    """
    if force:
        return _dataset_pending(dataset_id, "", ())
    return _dataset_pending(
        dataset_id,
        "m.id NOT IN (SELECT media_id FROM media_embedding "
        "WHERE model_id = ?)",
        (model_id,),
    )


def get_or_create_triggerword(name: str) -> int:
    """Return the id of a trigger word, creating it if absent."""
    name = (name or "").strip()
    row = _query_one("SELECT id FROM triggerword WHERE name = ?", (name,))
    if row is not None:
        return row["id"]
    return _write("INSERT INTO triggerword (name) VALUES (?)", (name,))


def add_triggerword_to_dataset(dataset_id: int, name: str) -> int:
    """Attach a trigger word (by name) to a dataset; return the link id."""
    triggerword_id = get_or_create_triggerword(name)
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO dataset_triggerword "
                "(dataset_id, triggerword_id) VALUES (?, ?)",
                (dataset_id, triggerword_id),
            )
            row = conn.execute(
                "SELECT id FROM dataset_triggerword "
                "WHERE dataset_id = ? AND triggerword_id = ?",
                (dataset_id, triggerword_id),
            ).fetchone()
    return row["id"]


def remove_triggerword_from_dataset(
    dataset_id: int, triggerword_id: int
) -> None:
    """Detach a trigger word from a dataset (the word itself is kept)."""
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "DELETE FROM dataset_triggerword "
                "WHERE dataset_id = ? AND triggerword_id = ?",
                (dataset_id, triggerword_id),
            )


def dataset_triggerwords(dataset_id: int) -> list:
    """Return a dataset's trigger words in attachment order.

    Returns
    -------
    list of sqlite3.Row
        Rows with ``triggerword_id`` and ``name``, oldest attachment first.
    """
    return _query_all(
        "SELECT t.id AS triggerword_id, t.name AS name "
        "FROM dataset_triggerword dt "
        "JOIN triggerword t ON t.id = dt.triggerword_id "
        "WHERE dt.dataset_id = ? ORDER BY dt.id",
        (dataset_id,),
    )


def triggerword_prefix(dataset_id: int) -> str:
    """Return the export prefix for a dataset's trigger words.

    Each word is emitted as ``"<name>. "`` in attachment order, so two words
    ``xbl1`` and ``xbl2`` yield ``"xbl1. xbl2. "``. The prefix is applied at
    export time only — it is never stored in a revision's content.
    """
    words = dataset_triggerwords(dataset_id)
    return "".join(f"{row['name']}. " for row in words)
