"""Media repository: identity, files, ingestion and the index.

A media is identified by the sha256 of its bytes and owns many
``media_file`` rows (same content through several paths); the
effective file is resolved live with fallback. Also holds the
index columns (dimensions, quality score) the Libraries tab's
Index action fills in.
"""

import os
from contextlib import closing
from pathlib import Path

from src import db
from src import quality
from src.blob_store import compute_sha256
from src.media import VIDEO_EXTENSIONS
from src.sqlite_store.base import chunked, _query_all, _query_one, _write

# Video file extensions without the leading dot, matching how
# ``file_extension`` is stored (see :func:`get_or_create_media`). Used to
# keep videos out of the lookalike detection set (see
# :func:`media_with_hashes`).
_VIDEO_EXTS = tuple(sorted(ext.lstrip(".") for ext in VIDEO_EXTENSIONS))


def get_or_create_media(sha256: str, file_extension: str = "") -> int:
    """Return the media id for a content hash, creating the row if absent.

    A media is identified by its ``sha256``: the same content reached through
    several files maps to one media. The files themselves are recorded
    separately (see :func:`add_media_file`).

    Parameters
    ----------
    sha256 : str
        The media's content hash.
    file_extension : str, optional
        The extension (no dot) recorded on first creation, used for deploy
        naming and image/video detection.

    Returns
    -------
    int
        The media id.
    """
    row = _query_one(
        "SELECT id FROM media WHERE sha256 = ? AND deleted_at IS NULL",
        (sha256,),
    )
    if row is not None:
        return row["id"]
    extension = str(file_extension).lstrip(".").lower()
    return _write(
        "INSERT INTO media (sha256, file_extension) VALUES (?, ?)",
        (sha256, extension),
    )


def add_media_file(media_id: int, path, library_id=None) -> None:
    """Record a file (by path) as belonging to a media (idempotent by path)."""
    _write(
        "INSERT OR IGNORE INTO media_file (media_id, library_id, path) "
        "VALUES (?, ?, ?)",
        (media_id, library_id, str(Path(path))),
    )


def _unhide_media(media_id: int) -> None:
    """Un-archive a media (its file was re-added to a library)."""
    _write(
        "UPDATE media SET hidden = 0 WHERE id = ? AND hidden = 1", (media_id,)
    )


def ingest_file(path, library_id=None):
    """Register a file: hash it, get/create its media, link the file.

    Re-adding a file (by path or by content) re-attaches it to the scanning
    library and un-archives its media, so a deleted-then-re-added library
    recovers the media's captions.

    Parameters
    ----------
    path : str or pathlib.Path
        The media file on disk.
    library_id : int or None, optional
        The library the file is discovered in (``None`` for a legacy/external
        reference).

    Returns
    -------
    tuple of (int, bool)
        ``(media_id, is_fresh)`` — ``is_fresh`` is true when the media is new
        content, was archived (its library had been deleted), or simply had
        no working file anywhere (every copy had gone missing from disk, with
        the library never formally deleted) and is now being re-added, so
        callers apply the library's default tags. An already active media
        (still reachable through some other file) returns false, keeping its
        own tags.
    """
    path_str = str(Path(path))
    known = _query_one(
        "SELECT m.id AS media_id, m.hidden AS hidden FROM media_file mf "
        "JOIN media m ON m.id = mf.media_id WHERE mf.path = ?",
        (path_str,),
    )
    if known is not None:
        media_id = known["media_id"]
        is_fresh = known["hidden"] or effective_file(media_id) is None
        _write(
            "UPDATE media_file SET library_id = ?, "
            "last_seen_at = datetime('now') WHERE path = ?",
            (library_id, path_str),
        )
        if known["hidden"]:
            _unhide_media(media_id)
        return media_id, bool(is_fresh)
    sha256 = compute_sha256(path)
    existing = _query_one(
        "SELECT id, hidden FROM media WHERE sha256 = ? AND deleted_at IS NULL",
        (sha256,),
    )
    if existing is None:
        media_id = get_or_create_media(sha256, Path(path).suffix)
        add_media_file(media_id, path_str, library_id)
        return media_id, True
    media_id = existing["id"]
    is_fresh = existing["hidden"] or effective_file(media_id) is None
    add_media_file(media_id, path_str, library_id)
    if existing["hidden"]:
        _unhide_media(media_id)
    return media_id, bool(is_fresh)


def media_files(media_id: int) -> list:
    """Return a media's file rows, oldest first by id."""
    return _query_all(
        "SELECT * FROM media_file WHERE media_id = ? ORDER BY id",
        (media_id,),
    )


def media_files_bulk(media_ids) -> dict:
    """Return the file rows of many media in one query.

    One round-trip instead of one :func:`media_files` query per media — the
    difference between an instant grid page and hundreds of per-card
    connections (see :func:`_media_dicts`).

    Returns
    -------
    dict
        ``{media_id: [file rows, oldest first]}``; a media with no files is
        absent from the dict.
    """
    ids = [int(m) for m in media_ids]
    if not ids:
        return {}
    grouped = {}
    for chunk in chunked(ids):
        placeholders = ", ".join("?" for _ in chunk)
        for row in _query_all(
            "SELECT * FROM media_file "
            f"WHERE media_id IN ({placeholders}) ORDER BY media_id, id",
            chunk,
        ):
            grouped.setdefault(row["media_id"], []).append(row)
    return grouped


def effective_file(media_id: int):
    """Return a media's effective file path on disk, or None when missing.

    The first file that still exists is returned (automatic fallback when the
    previous one was deleted outside the app); presence is probed live on each
    call. ``None`` means every file is gone — the media is *missing*.

    A *crop* (see :mod:`src.crops`) owns no ``media_file`` row: its pixels
    are the rectangle it records of its parent, rendered into a PNG cache the
    first time they are asked for. Resolving that cached file here is what
    makes every path-taking engine — thumbnails, quality, the tagger,
    grounding, the deploy copier — operate on the cropped pixels unchanged.
    """
    path = _first_existing(media_files(media_id))
    if path is not None:
        return path
    return _crop_effective_file(media_id)


def _crop_effective_file(media_id: int):
    """Return a crop's rendered PNG, or None when it is not a crop.

    Also returns None when the crop cannot be produced — an unreadable or
    vanished parent file makes the crop *missing*, exactly like a media whose
    every file disappeared.
    """
    row = _query_one(
        "SELECT sha256, parent_media_id, crop_rect FROM media WHERE id = ?",
        (media_id,),
    )
    if row is None or not row["parent_media_id"]:
        return None
    parent_path = _first_existing(media_files(row["parent_media_id"]))
    return _render_crop(row, parent_path)


def _render_crop(row, parent_path):
    """Return the cached PNG of a crop row given its parent's file, or None."""
    # pylint: disable=import-outside-toplevel  # Pillow-backed, and would
    # close an import cycle (crops resolves nothing from the repository).
    from src import crops

    rect = crops.rect_from_json(_row_get(row, "crop_rect"))
    if rect is None:
        return None
    rendered = crops.ensure_render(parent_path, row["sha256"], rect)
    return str(rendered) if rendered is not None else None


def _first_existing(file_rows):
    """Return the first file row path that still exists on disk, or None."""
    for row in file_rows or ():
        if os.path.exists(row["path"]):
            return row["path"]
    return None


def _basename_or_none(path):
    """Return a path's file name, or None when the path is None."""
    return os.path.basename(path) if path else None


def _row_get(row, key: str):
    """Return a sqlite3.Row column, or None when the query did not select it.

    A plain ``row[key]`` raises ``IndexError`` for a missing column, unlike a
    dict's ``.get`` — some callers (e.g. :func:`media_in_dataset`) select a
    narrow column list for performance, so :func:`_media_dict` reads the
    indexing columns defensively rather than requiring every caller's query
    to carry them.
    """
    return row[key] if key in row.keys() else None


def _display_name(row, eff, parent_name=None) -> str:
    """Return a media's display label.

    A crop is named after the image it frames — its own effective file is a
    cache entry named by a hash, which tells the user nothing.
    """
    if _row_get(row, "parent_media_id"):
        source = parent_name or f"media #{row['parent_media_id']}"
        return f"{source} · crop"
    if eff:
        return os.path.basename(eff)
    return f"media #{row['id']} (missing)"


def _media_dict(
    row, hidden: bool = False, eff=None, resolved=False, parent_name=None
) -> dict:
    """Return a media row augmented with its effective file and missing flag.

    The dict carries the media columns plus ``eff_path`` (the resolved file or
    None), ``missing`` and ``name`` (a display label), so the UI never needs to
    know about the underlying ``media_file`` rows.

    Parameters
    ----------
    row : sqlite3.Row
        The media row (or a narrow projection carrying at least ``id``,
        ``sha256`` and ``file_extension``).
    hidden : bool, optional
        The dataset-level hidden flag to carry.
    eff : str or None, optional
        A pre-resolved effective file (see :func:`_media_dicts`, which
        resolves a whole page in one query). Only honored with ``resolved``.
    resolved : bool, optional
        Whether ``eff`` was pre-resolved by the caller (None then means
        *missing*, not *unresolved*).
    """
    # pylint: disable=import-outside-toplevel  # See _render_crop.
    from src import crops

    parent_id = _row_get(row, "parent_media_id")
    if not resolved:
        eff = effective_file(row["id"])
        if parent_id and parent_name is None:
            parent_name = _basename_or_none(
                _first_existing(media_files(parent_id))
            )
    return {
        "id": row["id"],
        "sha256": row["sha256"],
        "file_extension": row["file_extension"],
        "hidden": bool(hidden),
        "favorite": bool(_row_get(row, "favorite")),
        "eff_path": eff,
        "missing": eff is None,
        "name": _display_name(row, eff, parent_name),
        "parent_media_id": parent_id,
        "crop_rect": crops.rect_from_json(_row_get(row, "crop_rect")),
        "crop_ratio": _row_get(row, "crop_ratio"),
        "width": _row_get(row, "width"),
        "height": _row_get(row, "height"),
        # The quality fields are filled by _media_dicts from the separate
        # media_quality table (the per-media scores no longer live on the
        # media row). ``quality_scores`` is the full {metric_id: score} map;
        # ``quality_score``/``quality_metric`` are the score and bands to
        # display for the selected metric (or the normalized average).
        "quality_scores": {},
        "quality_score": None,
        "quality_metric": None,
        "indexed_at": _row_get(row, "indexed_at"),
        # The model-free statistics of src.image_stats, or None when the
        # media was never analyzed (see set_media_stats).
        "stats": _stats_of(row),
    }


def _stats_of(row) -> dict | None:
    """Return a media row's ``image_stats`` triplet, or None when unfilled."""
    sharpness = _row_get(row, "sharpness")
    if sharpness is None:
        return None
    return {
        "sharpness": float(sharpness),
        "clipping": float(_row_get(row, "clipping") or 0.0),
        "cleanliness": float(_row_get(row, "cleanliness") or 0.0),
    }


def _quality_scores_bulk(media_ids) -> dict:
    """Return the ``{metric_id: score}`` map of many media in one query.

    Reads the :data:`media_quality` rows for every given media id at once
    (one round-trip instead of one per card), so a grid page resolves its
    quality badges without a per-media connection.

    Returns
    -------
    dict
        ``{media_id: {metric_id: score}}``; a media with no score is absent.
    """
    ids = [int(m) for m in media_ids]
    if not ids:
        return {}
    grouped = {}
    for chunk in chunked(ids):
        placeholders = ", ".join("?" for _ in chunk)
        for row in _query_all(
            "SELECT media_id, metric_id, score FROM media_quality "
            f"WHERE media_id IN ({placeholders})",
            chunk,
        ):
            grouped.setdefault(row["media_id"], {})[row["metric_id"]] = row[
                "score"
            ]
    return grouped


def _display_quality(scores: dict, selected: str):
    """Return the ``(score, metric_id)`` a grid badge should show.

    Given a media's ``{metric_id: score}`` map and the currently *selected*
    display metric, resolves what the corner quality badge reads:

    * a specific metric — the media's raw score for it (with that metric's
      own bands), or ``(None, metric)`` when the media has no score for it
      (no badge);
    * the :data:`src.quality.AVERAGE_METRIC_ID` pseudo-metric, or ``None``
      (no explicit selection) — the normalized average across every metric
      the media carries, or ``(None, "average")`` when it has none.

    Parameters
    ----------
    scores : dict
        The media's ``{metric_id: score}`` map (possibly empty).
    selected : str or None
        The selected display metric, ``AVERAGE_METRIC_ID`` or ``None``.
    """
    if selected and selected != quality.AVERAGE_METRIC_ID:
        return scores.get(selected), selected
    normalized = [
        value
        for value in (
            quality.normalize_score(metric_id, score)
            for metric_id, score in scores.items()
        )
        if value is not None
    ]
    if not normalized:
        return None, quality.AVERAGE_METRIC_ID
    return sum(normalized) / len(normalized), quality.AVERAGE_METRIC_ID


def _media_dicts(
    rows, hidden_by_id=None, quality_metric_selected=None
) -> list:
    """Return :func:`_media_dict` dicts for many rows, files batched.

    Resolves every row's effective file from ONE ``media_file`` query
    (:func:`media_files_bulk`) instead of one query per media — the per-row
    disk probes remain, but a page paint no longer opens a connection per
    card — and every row's quality scores from ONE ``media_quality`` query.

    Parameters
    ----------
    rows : list of sqlite3.Row
        The media rows to convert.
    hidden_by_id : dict, optional
        Per-media dataset ``hidden`` flags (``{media_id: bool}``); absent
        ids default to False.
    quality_metric_selected : str, optional
        The display metric each dict's ``quality_score``/``quality_metric``
        should reflect (a :data:`src.quality.QUALITY_METRICS` key, the
        :data:`src.quality.AVERAGE_METRIC_ID` pseudo-metric, or ``None`` for
        the average). ``quality_scores`` always carries the full per-metric
        map regardless.
    """
    hidden_by_id = hidden_by_id or {}
    ids = [row["id"] for row in rows]
    parent_ids = [
        row["parent_media_id"]
        for row in rows
        if _row_get(row, "parent_media_id")
    ]
    # Crops own no media_file row: their pixels come from their parent's, so
    # the parents ride along in the same batched query.
    files = media_files_bulk(ids + parent_ids)
    scores_by_id = _quality_scores_bulk(ids)
    dicts = []
    for row in rows:
        parent_id = _row_get(row, "parent_media_id")
        parent_path = (
            _first_existing(files.get(parent_id)) if parent_id else None
        )
        item = _media_dict(
            row,
            hidden=hidden_by_id.get(row["id"], False),
            eff=(
                _render_crop(row, parent_path)
                if parent_id
                else _first_existing(files.get(row["id"]))
            ),
            resolved=True,
            parent_name=_basename_or_none(parent_path),
        )
        scores = scores_by_id.get(row["id"], {})
        item["quality_scores"] = scores
        item["quality_score"], item["quality_metric"] = _display_quality(
            scores, quality_metric_selected
        )
        dicts.append(item)
    return dicts


def upsert_media_quality(media_id: int, metric_id: str, score) -> None:
    """Store (or replace) a media's quality score for one metric.

    Called once per media and metric by the Libraries tab's "Index" action:
    a media accumulates one :data:`media_quality` row per metric it was
    scored with, so switching the Settings metric adds a score instead of
    overwriting the previous one. ``INSERT OR REPLACE`` on the unique
    ``(media_id, metric_id)`` makes a re-index idempotent (the newest score
    wins). A ``None`` score is ignored — the scoring failed and there is
    nothing to record (the ``NOT NULL`` schema would reject it anyway).

    Parameters
    ----------
    media_id : int
        The media to score.
    metric_id : str
        The :data:`src.quality.QUALITY_METRICS` key the score was computed
        with (never the "average" pseudo-metric).
    score : float or None
        The raw score in the metric's native range.
    """
    if score is None:
        return
    _write(
        "INSERT OR REPLACE INTO media_quality "
        "(media_id, metric_id, score) VALUES (?, ?, ?)",
        (media_id, metric_id, float(score)),
    )


def available_quality_metrics() -> list:
    """Return the metric ids present in :data:`media_quality`, with counts.

    Backs the grids' "displayed quality" dropdown: only metrics some media
    was actually scored with are offered. The "average" pseudo-metric is
    never stored, so it never appears here — the UI adds it on top when two
    or more real metrics are present.

    Returns
    -------
    list of tuple
        ``(metric_id, count)`` pairs, most-used metric first.
    """
    return [
        (row["metric_id"], row["n"])
        for row in _query_all(
            "SELECT metric_id, COUNT(*) AS n FROM media_quality "
            "GROUP BY metric_id ORDER BY n DESC, metric_id"
        )
    ]


def get_media(media_id: int):
    """Return a media row by id, or None."""
    return _query_one("SELECT * FROM media WHERE id = ?", (media_id,))


def get_media_display(
    media_id: int, quality_metric_selected=None
) -> dict | None:
    """Return one media's UI dict (effective file, dims, quality), or None.

    The single-media form of :func:`_media_dicts`: the same
    :func:`_media_dict` shape with its effective file, ``missing`` flag,
    ``name``, dimensions and quality (``quality_scores`` plus the
    ``quality_score``/``quality_metric`` for the selected display metric).
    Used by callers that focus one media (e.g. a detail panel) without
    paging a whole grid.

    Parameters
    ----------
    media_id : int
        The media to resolve.
    quality_metric_selected : str, optional
        The display metric the ``quality_score``/``quality_metric`` fields
        should reflect (see :func:`_media_dicts`).

    Returns
    -------
    dict or None
        The media dict, or ``None`` when the id is unknown.
    """
    row = get_media(media_id)
    if row is None:
        return None
    dicts = _media_dicts(
        [row], quality_metric_selected=quality_metric_selected
    )
    return dicts[0] if dicts else None


def set_media_favorite(media_id: int, favorite: bool) -> None:
    """Set a media's favorite flag (the Medias grid heart)."""
    _write(
        "UPDATE media SET favorite = ? WHERE id = ?",
        (int(bool(favorite)), media_id),
    )


def toggle_media_favorite(media_id: int) -> bool:
    """Flip a media's favorite flag; return the new state.

    Backs the Medias grid's heart click. A media not found is a no-op that
    reports ``False``.
    """
    row = _query_one("SELECT favorite FROM media WHERE id = ?", (media_id,))
    if row is None:
        return False
    new_state = not bool(row["favorite"])
    _write(
        "UPDATE media SET favorite = ? WHERE id = ?",
        (int(new_state), media_id),
    )
    return new_state


def set_media_discarded(media_id: int) -> None:
    """Mark a media as discarded from the Lookalikes view (idempotent).

    Stamps ``discarded_at`` so the media drops out of every user listing
    (the grids, the index queue, :func:`media_with_hashes`) while its row,
    files, captions and tags stay intact. Crucially, unlike ``hidden``, a
    re-scan (:func:`ingest_file`) never clears it — the media stays set
    aside even when its source file is re-seen. Reversible with
    :func:`restore_media`. No file on disk is ever touched.
    """
    _write(
        "UPDATE media SET discarded_at = datetime('now') "
        "WHERE id = ? AND discarded_at IS NULL",
        (media_id,),
    )


def restore_media(media_id: int) -> None:
    """Un-discard a media, returning it to the listings (idempotent)."""
    _write("UPDATE media SET discarded_at = NULL WHERE id = ?", (media_id,))


def missing_media(library_id: int = None) -> list:
    """Return live media whose file is gone from disk (id + display name).

    A media is *missing* when every registered file has vanished from disk
    (:func:`effective_file` resolves to None) — the source was deleted
    outside the app. Backs the Libraries "missing files" cleanup, which
    offers to purge them (see :func:`purge_media`).

    Parameters
    ----------
    library_id : int, optional
        Restrict to one library's media; None (default) spans every library.
    """
    where, params = _live_scope(library_id)
    rows = _query_all(
        f"SELECT m.* FROM media m WHERE {where} ORDER BY m.id", params
    )
    return [
        {"id": item["id"], "name": item["name"]}
        for item in _media_dicts(rows)
        if item["missing"]
    ]


def purge_media(media_ids) -> int:
    """Hard-delete media rows by id; return how many were removed.

    Unlike discarding (a reversible flag), this drops the row for good; the
    ``media_file``, ``media_tag``, caption and score rows cascade away
    (every child references ``media`` ``ON DELETE CASCADE``). For purging
    media whose source file the user deleted from disk — never touches a
    file, only the database.
    """
    ids = [int(media_id) for media_id in media_ids or []]
    if not ids:
        return 0
    placeholders = ", ".join("?" for _ in ids)
    with closing(db.connect()) as conn:
        with conn:
            cursor = conn.execute(
                f"DELETE FROM media WHERE id IN ({placeholders})", ids
            )
            return cursor.rowcount


# An orphan media is a real (non-crop) media row that nothing references any
# more: it owns no ``media_file`` (every path was removed, not merely gone
# from disk) and belongs to no dataset. Such a row shows in no grid and
# deploys nowhere -- dead weight left by an unusual delete order. NOT the
# "missing on disk" media (those keep their file rows and stay in the
# library, purged from the Libraries "missing files" action) and NOT the far
# larger "in no dataset" set (in this app media live in libraries
# independently of datasets, so membership is no orphan signal).
_ORPHAN_MEDIA_WHERE = (
    "m.deleted_at IS NULL AND m.parent_media_id IS NULL "
    "AND NOT EXISTS (SELECT 1 FROM media_file f WHERE f.media_id = m.id) "
    "AND m.id NOT IN (SELECT media_id FROM dataset_media)"
)


def count_orphan_media() -> int:
    """Return how many media are referenced by no file and no dataset."""
    row = _query_one(
        f"SELECT COUNT(*) AS n FROM media m WHERE {_ORPHAN_MEDIA_WHERE}"
    )
    return row["n"] if row else 0


def orphan_media_ids() -> list:
    """Return the ids of the orphan media (see :func:`count_orphan_media`)."""
    rows = _query_all(
        f"SELECT m.id FROM media m WHERE {_ORPHAN_MEDIA_WHERE} ORDER BY m.id"
    )
    return [row["id"] for row in rows]


def media_sha256_bulk(media_ids) -> dict:
    """Return ``{media_id: sha256}`` for many media in one query.

    Lets the maintenance sweep rebuild the composite cache keys of the media
    that carry watermark zones (see :mod:`src.maintenance`) without a query
    per media. Unknown ids are simply absent from the result.
    """
    ids = [int(media_id) for media_id in media_ids or []]
    if not ids:
        return {}
    result: dict = {}
    for chunk in chunked(ids):
        placeholders = ", ".join("?" for _ in chunk)
        for row in _query_all(
            f"SELECT id, sha256 FROM media WHERE id IN ({placeholders})",
            chunk,
        ):
            result[row["id"]] = row["sha256"]
    return result


def set_media_lookalike_reviewed(media_id: int) -> None:
    """Mark a media's lookalike group as dismissed ("hide indefinitely").

    Stamps ``lookalike_reviewed_at`` so the Lookalikes view can hide a group
    the user chose to set aside without discarding any image. Unlike
    :func:`set_media_discarded`, the media stays live and hashed (it still
    appears in the grids and in :func:`media_with_hashes`), so it can still
    form a *new* group with a future import; only a group whose members are
    all reviewed is filtered out. Idempotent; reversible with
    :func:`clear_lookalike_reviewed`. No file on disk is ever touched.
    """
    _write(
        "UPDATE media SET lookalike_reviewed_at = datetime('now') "
        "WHERE id = ? AND lookalike_reviewed_at IS NULL",
        (media_id,),
    )


def lookalike_reviewed_ids() -> set:
    """Return the ids of every media whose lookalike group was dismissed.

    Backs the group filter in the Lookalikes router: a detected group is
    hidden when all of its members appear in this set.
    """
    rows = _query_all(
        "SELECT id FROM media WHERE lookalike_reviewed_at IS NOT NULL",
    )
    return {row["id"] for row in rows}


def clear_lookalike_reviewed() -> None:
    """Un-dismiss every lookalike group (the "Reset dismissed" action)."""
    _write(
        "UPDATE media SET lookalike_reviewed_at = NULL "
        "WHERE lookalike_reviewed_at IS NOT NULL",
    )


def list_discarded_media() -> list:
    """Return the discarded media as dicts, most recently discarded first.

    Backs the Lookalikes "Discarded media" restore section. Each dict is an
    :func:`_media_dict` augmented with ``discarded_at`` (the timestamp shown
    next to the thumbnail).
    """
    rows = _query_all(
        "SELECT m.* FROM media m "
        "WHERE m.deleted_at IS NULL AND m.discarded_at IS NOT NULL "
        "ORDER BY m.discarded_at DESC, m.id DESC",
    )
    dicts = _media_dicts(rows)
    when = {row["id"]: row["discarded_at"] for row in rows}
    for item in dicts:
        item["discarded_at"] = when[item["id"]]
    return dicts


def set_media_index(
    media_id: int,
    width,
    height,
    phash=None,
    dhash=None,
) -> None:
    """Persist a media's indexed dimensions and perceptual hashes.

    Called once per media by the Libraries tab's "Index" action; sets
    ``indexed_at`` to now, so :func:`media_pending_index` can tell an
    already-indexed media apart from one still pending. The quality score is
    stored separately, per metric, by :func:`upsert_media_quality` — a
    skipped or failed score therefore leaves any previously stored scores
    untouched.

    Parameters
    ----------
    media_id : int
        The media to update.
    width : int or None
        The probed pixel width.
    height : int or None
        The probed pixel height.
    phash : str or None, optional
        The media's DCT perceptual hash as a 16-char hex string (see
        :func:`src.perceptual_hash.compute_hashes`), the foundation of the
        "lookalike" near-duplicate detection. None when the file could not
        be hashed.
    dhash : str or None, optional
        The media's gradient perceptual hash — same format, source and
        role as ``phash``.
    """
    _write(
        "UPDATE media SET width = ?, height = ?, phash = ?, dhash = ?, "
        "indexed_at = datetime('now') WHERE id = ?",
        (width, height, phash, dhash, media_id),
    )


def set_media_stats(media_id: int, stats: dict | None) -> None:
    """Persist a media's model-free statistics (see :mod:`src.image_stats`).

    A None ``stats`` (a video, or an image OpenCV could not decode) is a
    no-op: the columns stay NULL, which every reader treats as "unknown",
    never as "flagged".

    Parameters
    ----------
    media_id : int
        The media to update.
    stats : dict or None
        The ``{"sharpness", "clipping", "cleanliness"}`` percentages
        returned by :func:`src.image_stats.analyze`.
    """
    if not stats:
        return
    _write(
        "UPDATE media SET sharpness = ?, clipping = ?, cleanliness = ? "
        "WHERE id = ?",
        (
            stats["sharpness"],
            stats["clipping"],
            stats["cleanliness"],
            media_id,
        ),
    )


def media_pending_index(
    library_id: int = None, force: bool = False, metric: str = None
) -> list:
    """Return the media an "Index" run should (re)compute a thumbnail for.

    Parameters
    ----------
    library_id : int, optional
        Restrict to one library's media (a file discovered there); None
        (default) spans every library — "Index all libraries".
    force : bool, optional
        When true, every matching media is returned (a full re-index);
        otherwise (the default, fast incremental pass) only media never
        indexed before (``indexed_at IS NULL``), media indexed before the
        perceptual hashes existed (``phash IS NULL``, so an existing
        database backfills its hashes on the next Index without a Force
        re-index or a quality recompute), images indexed before the
        :mod:`src.image_stats` columns existed (``sharpness IS NULL``,
        backfilled the same way) — plus, when ``metric`` is given,
        any media with no score yet for that metric, so switching the
        Settings-tab metric scores the library with it without a manual
        Force re-index (existing scores for other metrics are kept).
    metric : str, optional
        The currently configured quality metric (a
        :data:`src.quality.QUALITY_METRICS` key). Ignored when ``force`` is
        true.

    Returns
    -------
    list of dict
        The matching media as :func:`_media_dict` dicts, newest first.
    """
    where = [
        "m.deleted_at IS NULL",
        "m.hidden = 0",
        "m.discarded_at IS NULL",
        "m.parent_media_id IS NULL",
    ]
    params = []
    if library_id is not None:
        where.append(
            "m.id IN (SELECT media_id FROM media_file WHERE library_id = ?)"
        )
        params.append(library_id)
    if not force:
        conditions = ["m.indexed_at IS NULL", "m.phash IS NULL"]
        if metric is not None:
            conditions.append(
                "m.id NOT IN (SELECT media_id FROM media_quality "
                "WHERE metric_id = ?)"
            )
            params.append(metric)
        # Images indexed before the image_stats columns existed backfill
        # them on the next incremental Index, exactly like the hashes did.
        # Videos are never analyzed, so they must not read as pending.
        placeholders = ", ".join("?" for _ in _VIDEO_EXTS)
        conditions.append(
            "(m.sharpness IS NULL AND "
            f"lower(m.file_extension) NOT IN ({placeholders}))"
        )
        params.extend(_VIDEO_EXTS)
        where.append("(" + " OR ".join(conditions) + ")")
    rows = _query_all(
        "SELECT m.* FROM media m WHERE "
        + " AND ".join(where)
        + " ORDER BY m.id DESC",
        params,
    )
    return _media_dicts(rows)


def media_pending_score(
    metric_id: str, library_id: int = None, force: bool = False
) -> list:
    """Return the media a quality run should score with ``metric_id``.

    Quality scoring is a separate, on-demand Libraries action (see
    :func:`src.libraries_gallery.score_library_ui`), independent of the
    Index action that computes thumbnails/dimensions/hashes. It reads a
    media's *original* file, so it does not require the media to have been
    indexed first.

    Parameters
    ----------
    metric_id : str
        The :data:`src.quality.QUALITY_METRICS` key to score with.
    library_id : int, optional
        Restrict to one library's media; None (default) spans every library.
    force : bool, optional
        When true, every matching media is returned (a full re-score);
        otherwise (the default) only media with no ``media_quality`` row for
        ``metric_id`` yet, so a repeat run stays fast and chained metrics
        each score only what they are missing.

    Returns
    -------
    list of dict
        The matching media as :func:`_media_dict` dicts, newest first.
    """
    where = [
        "m.deleted_at IS NULL",
        "m.hidden = 0",
        "m.discarded_at IS NULL",
        "m.parent_media_id IS NULL",
    ]
    params = []
    if library_id is not None:
        where.append(
            "m.id IN (SELECT media_id FROM media_file WHERE library_id = ?)"
        )
        params.append(library_id)
    if not force:
        where.append(
            "m.id NOT IN (SELECT media_id FROM media_quality "
            "WHERE metric_id = ?)"
        )
        params.append(metric_id)
    rows = _query_all(
        "SELECT m.* FROM media m WHERE "
        + " AND ".join(where)
        + " ORDER BY m.id DESC",
        params,
    )
    return _media_dicts(rows)


def _live_scope(library_id: int = None, images_only: bool = False):
    """Return the ``(where, params)`` of the media an index step covers.

    The same projection every ``media_pending_*`` reader uses — live media
    only (never deleted, hidden, discarded or a virtual crop) — optionally
    narrowed to one library and to images (videos take no embedding and no
    auto-tag).

    Crops are excluded because the index describes files a library scan
    found: a crop has no file, no perceptual hash worth comparing (it would
    trivially match its parent) and is scored on demand from its dataset,
    not by the Libraries index steps.
    """
    where = [
        "m.deleted_at IS NULL",
        "m.hidden = 0",
        "m.discarded_at IS NULL",
        "m.parent_media_id IS NULL",
    ]
    params = []
    if images_only:
        placeholders = ", ".join("?" for _ in _VIDEO_EXTS)
        where.append(f"lower(m.file_extension) NOT IN ({placeholders})")
        params.extend(_VIDEO_EXTS)
    if library_id is not None:
        where.append(
            "m.id IN (SELECT media_id FROM media_file WHERE library_id = ?)"
        )
        params.append(library_id)
    return " AND ".join(where), params


def _count_live(extra: str, extra_params, library_id, images_only) -> int:
    """Count the live media matching one extra SQL predicate."""
    where, params = _live_scope(library_id, images_only)
    row = _query_one(
        f"SELECT COUNT(*) AS n FROM media m WHERE {where}{extra}",
        (*params, *extra_params),
    )
    return row["n"] if row else 0


def count_live_media(library_id: int = None, images_only: bool = False) -> int:
    """Return how many live media an index step has to cover.

    The denominator of the Index panel's ``done / total`` counters: the
    media a scan would look at, i.e. the ones the ``media_pending_*``
    readers select before their "already done" filter.
    """
    return _count_live("", (), library_id, images_only)


def count_media_scored(metric_ids, library_id: int = None) -> int:
    """Return how many live media carry a score for *every* given metric.

    Backs the "quality" index step's progress: a media counts as scored
    only once the whole chained metric set ran on it.

    Parameters
    ----------
    metric_ids : iterable of str
        The :data:`src.quality.QUALITY_METRICS` keys the step chains.
    library_id : int, optional
        Restrict to one library's media; None (default) spans every library.
    """
    metrics = [str(metric_id) for metric_id in metric_ids if metric_id]
    if not metrics:
        return 0
    placeholders = ", ".join("?" for _ in metrics)
    extra = (
        " AND (SELECT COUNT(DISTINCT q.metric_id) FROM media_quality q "
        f"WHERE q.media_id = m.id AND q.metric_id IN ({placeholders})) = ?"
    )
    return _count_live(extra, (*metrics, len(metrics)), library_id, False)


def count_media_with_embedding(model_id: str, library_id: int = None) -> int:
    """Return how many live images carry a vector for ``model_id``."""
    extra = (
        " AND m.id IN (SELECT media_id FROM media_embedding "
        "WHERE model_id = ?)"
    )
    return _count_live(extra, (model_id,), library_id, True)


def count_media_tagged(library_id: int = None) -> int:
    """Return how many live images carry at least one per-media tag.

    Backs the "Auto-tags" index step's progress. A media counts as tagged
    once it holds any tag whose ``media_tag.source`` is NULL — a WD14
    auto-tag (from the index or the Media tab) or a manual add — regardless
    of the tag's category. Tags pushed by the Libraries "Bulk tags" action
    (``source = 'library'``) do not count: a library-wide label is not a
    per-image tag.
    """
    extra = (
        " AND m.id IN (SELECT mt.media_id FROM media_tag mt "
        "WHERE mt.source IS NULL)"
    )
    return _count_live(extra, (), library_id, True)


def live_media_sha256(library_id: int = None) -> list:
    """Return the content hashes of the live media of a library (or all).

    The thumbnail cache is keyed by sha256 and holds no database row, so
    the "thumbs" step's progress is counted by intersecting these hashes
    with the cached files (see :func:`src.thumbnails.cached_sha256`).
    """
    where, params = _live_scope(library_id)
    rows = _query_all(
        f"SELECT m.sha256 AS sha256 FROM media m WHERE {where}", params
    )
    return [row["sha256"] for row in rows]


def media_pending_autotag(library_id: int = None, force: bool = False) -> list:
    """Return the live images the auto-tag step should run the tagger on.

    Same shape as :func:`media_pending_embedding`. Images only (a video's
    frame is not tagged), and — unless ``force`` — only media that carry no
    per-media tag yet (no ``media_tag`` row with ``source`` NULL); a media
    holding only Libraries "Bulk tags" (``source = 'library'``) still needs
    the tagger.

    Parameters
    ----------
    library_id : int, optional
        Restrict to one library's media; None (default) spans every library.
    force : bool, optional
        When true, every live image is returned (a full re-tag).
    """
    where, params = _live_scope(library_id, images_only=True)
    if not force:
        where += (
            " AND m.id NOT IN (SELECT mt.media_id FROM media_tag mt "
            "WHERE mt.source IS NULL)"
        )
    rows = _query_all(
        f"SELECT m.* FROM media m WHERE {where} ORDER BY m.id DESC", params
    )
    return _media_dicts(rows)


def delete_media_quality(metric_ids, library_id: int = None) -> int:
    """Delete stored quality scores for one or more metrics.

    Backs the Libraries tab's "delete a classification" action: drops every
    ``media_quality`` row for the given metrics, optionally scoped to one
    library's media. The media, their index columns and any other metric's
    scores are untouched — only the selected classifications go.

    Parameters
    ----------
    metric_ids : iterable of str
        The :data:`src.quality.QUALITY_METRICS` keys to clear.
    library_id : int, optional
        Restrict to one library's media; None (default) spans every library.

    Returns
    -------
    int
        How many ``media_quality`` rows were deleted.
    """
    metrics = [str(metric_id) for metric_id in metric_ids if metric_id]
    if not metrics:
        return 0
    placeholders = ", ".join("?" for _ in metrics)
    params = list(metrics)
    scope = ""
    if library_id is not None:
        scope = (
            " AND media_id IN "
            "(SELECT media_id FROM media_file WHERE library_id = ?)"
        )
        params.append(library_id)
    with closing(db.connect()) as conn:
        with conn:
            before = conn.total_changes
            conn.execute(
                f"DELETE FROM media_quality WHERE metric_id IN "
                f"({placeholders}){scope}",
                params,
            )
            return conn.total_changes - before


def media_with_hashes(quality_metric_selected=None) -> list:
    """Return every live, perceptually-hashed media for lookalike detection.

    A media qualifies once the Libraries tab's "Index" action has computed
    its perceptual hashes (``phash`` non NULL). Only active media are
    returned (``deleted_at IS NULL``, not ``hidden``, not discarded); the
    unique ``sha256`` index already guarantees no two live media are
    byte-for-byte identical, so any match the engine finds is a genuine
    *perceptual* near-duplicate. Discarded media are excluded so a resolved
    lookalike never resurfaces in a later detection run. Videos are excluded
    too: the side-by-side comparison the Lookalikes view offers is
    image-only for now (a single hashed frame is a poor duplicate signal
    for a clip).

    Parameters
    ----------
    quality_metric_selected : str, optional
        The metric each dict's ``quality_score``/``quality_metric`` should
        reflect for ranking and the badge (see :func:`_media_dicts`); the
        Lookalikes view passes the grids' selected display metric so a
        group's representative is chosen on the metric the user is looking
        at (falling back to the normalized average).

    Returns
    -------
    list of dict
        Each an :func:`_media_dict` (carrying ``id``, ``sha256``, the index
        columns, ``eff_path``/``missing``/``name``) augmented with the
        ``phash`` and ``dhash`` hex strings the engine compares (see
        :func:`src.lookalike.detect`), oldest first.
    """
    placeholders = ", ".join("?" for _ in _VIDEO_EXTS)
    rows = _query_all(
        "SELECT m.* FROM media m WHERE m.deleted_at IS NULL "
        "AND m.hidden = 0 AND m.discarded_at IS NULL "
        "AND m.parent_media_id IS NULL "
        "AND m.phash IS NOT NULL "
        f"AND lower(m.file_extension) NOT IN ({placeholders}) "
        "ORDER BY m.id",
        _VIDEO_EXTS,
    )
    dicts = _media_dicts(rows, quality_metric_selected=quality_metric_selected)
    hashes = {row["id"]: (row["phash"], row["dhash"]) for row in rows}
    for item in dicts:
        item["phash"], item["dhash"] = hashes[item["id"]]
    return dicts


def count_media_without_hash() -> int:
    """Return how many live media have no perceptual hash yet.

    Drives the Lookalikes panel's reminder to run "Index" first: a media
    scanned/uploaded but never indexed (``phash IS NULL``) cannot take part
    in detection.
    """
    row = _query_one(
        "SELECT COUNT(*) AS n FROM media WHERE deleted_at IS NULL "
        "AND hidden = 0 AND discarded_at IS NULL "
        "AND parent_media_id IS NULL AND phash IS NULL",
    )
    return row["n"]


def media_index_info(media_ids) -> dict:
    """Return the indexed dimensions and favorite flag of many media.

    The dataset-media projection (see
    ``src.sqlite_store.datasets._DATASET_MEDIA_WHERE``) is deliberately
    narrow — the caption grid needs neither column — so a caller that *does*
    need the index columns of exactly one dataset's media reads them here,
    in one query, rather than widening every grid page's SELECT. Backs the
    quality report's resolution floor and its diversity-map tooltips.

    Returns
    -------
    dict
        ``{media_id: {"width", "height", "favorite"}}``; a media that was
        never indexed carries ``None`` dimensions.
    """
    ids = [int(m) for m in media_ids]
    if not ids:
        return {}
    info = {}
    for chunk in chunked(ids):
        placeholders = ", ".join("?" for _ in chunk)
        for row in _query_all(
            "SELECT id, width, height, favorite FROM media "
            f"WHERE id IN ({placeholders})",
            chunk,
        ):
            info[row["id"]] = {
                "width": row["width"],
                "height": row["height"],
                "favorite": bool(row["favorite"]),
            }
    return info


def media_hashes(media_ids) -> dict:
    """Return the stored perceptual hashes of many media in one query.

    Backs the Datasets tab's quality evaluation, which needs the
    phash/dhash pairs of exactly one dataset's media (see
    :mod:`src.datasets_quality`) without materializing the whole hashed
    library like :func:`media_with_hashes` does.

    Returns
    -------
    dict
        ``{media_id: (phash, dhash)}`` hex strings; a media never
        indexed (or whose hashing failed) maps to ``(None, None)``.
    """
    ids = [int(m) for m in media_ids]
    if not ids:
        return {}
    hashes = {}
    for chunk in chunked(ids):
        placeholders = ", ".join("?" for _ in chunk)
        for row in _query_all(
            "SELECT id, phash, dhash FROM media "
            f"WHERE id IN ({placeholders})",
            chunk,
        ):
            hashes[row["id"]] = (row["phash"], row["dhash"])
    return hashes


def upsert_media_embedding(media_id: int, model_id: str, vector) -> None:
    """Store (or replace) a media's embedding vector for one model.

    Called once per media by the Libraries tab's "Embeddings" action;
    ``vector`` is the float32 BLOB built by
    :func:`src.embeddings.vector_to_blob`.
    """
    _write(
        "INSERT INTO media_embedding (media_id, model_id, vector) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT (media_id, model_id) DO UPDATE SET "
        "vector = excluded.vector, computed_at = datetime('now')",
        (media_id, model_id, vector),
    )


def media_embeddings(model_id: str, media_ids=None) -> dict:
    """Return the stored embedding BLOBs of many media in one query.

    Parameters
    ----------
    model_id : str
        The embedding model the vectors were computed with (see
        :data:`src.embeddings.MODEL_ID`).
    media_ids : iterable of int, optional
        Restrict to these media; None (default) returns every stored
        embedding for the model.

    Returns
    -------
    dict
        ``{media_id: vector BLOB}``; a media never embedded is absent.
    """
    base = "SELECT media_id, vector FROM media_embedding WHERE model_id = ?"
    if media_ids is None:
        return {
            row["media_id"]: row["vector"]
            for row in _query_all(base, [model_id])
        }
    ids = [int(m) for m in media_ids]
    if not ids:
        return {}
    vectors = {}
    for chunk in chunked(ids):
        placeholders = ", ".join("?" for _ in chunk)
        for row in _query_all(
            f"{base} AND media_id IN ({placeholders})",
            [model_id, *chunk],
        ):
            vectors[row["media_id"]] = row["vector"]
    return vectors


def media_pending_embedding(
    model_id: str, library_id: int = None, force: bool = False
) -> list:
    """Return the media an "Embeddings" run should vectorize.

    Same shape as :func:`media_pending_score`, for the Libraries tab's
    on-demand embedding action (see :mod:`src.libraries_embeddings`).
    Images only: videos are excluded like in :func:`media_with_hashes` —
    a single frame is a poor diversity signal for a clip.

    Parameters
    ----------
    model_id : str
        The embedding model to check coverage for.
    library_id : int, optional
        Restrict to one library's media; None (default) spans every
        library.
    force : bool, optional
        When true, every matching media is returned (a full recompute);
        otherwise (the default) only media with no stored vector yet.

    Returns
    -------
    list of dict
        The matching media as :func:`_media_dict` dicts, newest first.
    """
    placeholders = ", ".join("?" for _ in _VIDEO_EXTS)
    where = [
        "m.deleted_at IS NULL",
        "m.hidden = 0",
        "m.discarded_at IS NULL",
        "m.parent_media_id IS NULL",
        f"lower(m.file_extension) NOT IN ({placeholders})",
    ]
    params = list(_VIDEO_EXTS)
    if library_id is not None:
        where.append(
            "m.id IN (SELECT media_id FROM media_file WHERE library_id = ?)"
        )
        params.append(library_id)
    if not force:
        where.append(
            "m.id NOT IN (SELECT media_id FROM media_embedding "
            "WHERE model_id = ?)"
        )
        params.append(model_id)
    rows = _query_all(
        "SELECT m.* FROM media m WHERE "
        + " AND ".join(where)
        + " ORDER BY m.id DESC",
        params,
    )
    return _media_dicts(rows)


def count_media_without_embedding(model_id: str) -> int:
    """Return how many live image media have no embedding vector yet.

    Drives the Auto-build panel's reminder to run the Libraries tab's
    "Embeddings" action first: a media without a vector cannot take part
    in the diversity selection (it falls back to quality-only ranking).
    """
    placeholders = ", ".join("?" for _ in _VIDEO_EXTS)
    row = _query_one(
        "SELECT COUNT(*) AS n FROM media m WHERE m.deleted_at IS NULL "
        "AND m.hidden = 0 AND m.discarded_at IS NULL "
        "AND m.parent_media_id IS NULL "
        f"AND lower(m.file_extension) NOT IN ({placeholders}) "
        "AND m.id NOT IN (SELECT media_id FROM media_embedding "
        "WHERE model_id = ?)",
        (*_VIDEO_EXTS, model_id),
    )
    return row["n"]
