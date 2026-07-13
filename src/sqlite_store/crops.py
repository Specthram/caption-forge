"""Crop repository: the virtual media rows that alias a rectangle.

A crop is a ``media`` row with a ``parent_media_id``, a ``crop_rect`` and a
synthetic ``sha256`` derived from the two (see :mod:`src.crops`). It owns no
file: this module only creates, re-frames and deletes those rows, and keeps
the *derived* measurements consistent with the pixels they describe.

Re-framing a crop changes its pixels, so every score computed from them is
dropped in the same transaction: the quality scores, the embedding, the tag
groundings and tag score, and the caption groundings/scores of its revisions
(a caption's text is unchanged, but the image it was grounded against is
not). The captions and tags themselves survive — only what was *measured*
on the old rectangle is invalidated.
"""

from contextlib import closing

from src import crops
from src import db
from src.sqlite_store.base import _query_all, _query_one
from src.sqlite_store.media import effective_file

# The per-media tables holding a measurement made on the crop's pixels. Each
# is cleared when the rectangle moves (the numbers describe pixels that no
# longer exist); the captions and the tags themselves are kept.
_DERIVED_MEDIA_TABLES = (
    "media_quality",
    "media_embedding",
    "media_tag_grounding",
    "media_tag_score",
)

# Same, for the tables keyed on a caption *revision* of the crop.
_DERIVED_REVISION_TABLES = ("caption_grounding", "caption_score")


def is_crop(media_id: int) -> bool:
    """Return whether a media id is a virtual crop."""
    row = _query_one(
        "SELECT parent_media_id FROM media WHERE id = ?", (media_id,)
    )
    return bool(row and row["parent_media_id"])


def get_crop(media_id: int):
    """Return a crop's ``{id, parent_media_id, sha256, rect, ratio}``, or None.

    ``None`` is returned for an unknown id and for an ordinary media (one
    without a parent), so callers can use it as the "is this a crop" probe
    that also hands back the rectangle.
    """
    row = _query_one(
        "SELECT id, parent_media_id, sha256, crop_rect, crop_ratio, "
        "width, height FROM media WHERE id = ? AND deleted_at IS NULL",
        (media_id,),
    )
    if row is None or not row["parent_media_id"]:
        return None
    return _crop_dict(row)


def _crop_dict(row) -> dict:
    """Return one crop row as a dict with its rectangle already parsed."""
    return {
        "id": row["id"],
        "parent_media_id": row["parent_media_id"],
        "sha256": row["sha256"],
        "rect": crops.rect_from_json(row["crop_rect"]),
        "ratio": crops.normalize_ratio(row["crop_ratio"]),
        "width": row["width"],
        "height": row["height"],
    }


def all_crop_shas() -> set:
    """Return the content hashes of every live crop.

    The rendered-crop cache (:func:`src.crops.crop_cache_path`) keys each PNG
    by the crop's synthetic ``sha256``; this set is the "still referenced"
    side the maintenance sweep intersects against to spot cache files whose
    crop row is gone (see :mod:`src.maintenance`).
    """
    rows = _query_all(
        "SELECT sha256 FROM media "
        "WHERE parent_media_id IS NOT NULL AND deleted_at IS NULL"
    )
    return {row["sha256"] for row in rows}


def list_crops(parent_media_id: int) -> list:
    """Return every crop of a media, oldest first.

    Backs the Caption panel's "Crop · alias" list and the Datasets panel's
    "reusable crops of this image" list. Each dict carries the parsed
    rectangle, the rendered size and ``dataset_ids``: the datasets the crop
    is currently an entry of (empty when it was created and never placed).
    """
    rows = _query_all(
        "SELECT id, parent_media_id, sha256, crop_rect, crop_ratio, "
        "width, height FROM media "
        "WHERE parent_media_id = ? AND deleted_at IS NULL ORDER BY id",
        (parent_media_id,),
    )
    items = [_crop_dict(row) for row in rows]
    if not items:
        return []
    placeholders = ", ".join("?" for _ in items)
    memberships = _query_all(
        "SELECT media_id, dataset_id FROM dataset_media "
        f"WHERE media_id IN ({placeholders})",
        [item["id"] for item in items],
    )
    by_media = {}
    for row in memberships:
        by_media.setdefault(row["media_id"], []).append(row["dataset_id"])
    for item in items:
        item["dataset_ids"] = by_media.get(item["id"], [])
    return items


def _dataset_ids(media_id: int) -> list:
    """Return the datasets a media is an entry of."""
    return [
        row["dataset_id"]
        for row in _query_all(
            "SELECT dataset_id FROM dataset_media WHERE media_id = ?",
            (media_id,),
        )
    ]


def _crop_identity(parent_media_id: int, rect) -> tuple:
    """Return the ``(sha256, width, height)`` a crop of ``rect`` would carry.

    Raises
    ------
    ValueError
        When the parent is unknown or its file cannot be read (a crop of a
        missing image has no size and no pixels to frame).
    """
    parent = _query_one(
        "SELECT sha256, parent_media_id FROM media "
        "WHERE id = ? AND deleted_at IS NULL",
        (parent_media_id,),
    )
    if parent is None:
        raise ValueError(f"media #{parent_media_id} does not exist")
    if parent["parent_media_id"]:
        raise ValueError("a crop cannot be cropped again")
    source = effective_file(parent_media_id)
    if source is None:
        raise ValueError(f"media #{parent_media_id} has no file on disk")
    width, height = crops.crop_size(source, rect)
    return crops.crop_sha256(parent["sha256"], rect), width, height


def create_crop(parent_media_id: int, rect, ratio: str = "free") -> int:
    """Create (or reuse) the crop of ``parent_media_id`` framed by ``rect``.

    Two identical rectangles of the same parent are the same crop: the
    synthetic hash collides, and the existing row is returned instead of a
    duplicate. The pixels are rendered eagerly so the grid that shows the new
    card does not pay for it while painting.

    Parameters
    ----------
    parent_media_id : int
        The media to frame. Must be an ordinary media with a readable file.
    rect : dict
        The rectangle, in percentages of the source (clamped, see
        :func:`src.crops.normalize_rect`).
    ratio : str, optional
        The aspect ratio the overlay was locked to, restored on re-edit.

    Returns
    -------
    int
        The crop's media id.

    Raises
    ------
    ValueError
        When the parent does not exist, is itself a crop, or has no file.
    """
    box = crops.normalize_rect(rect)
    sha256, width, height = _crop_identity(parent_media_id, box)
    existing = _query_one(
        "SELECT id FROM media WHERE sha256 = ? AND deleted_at IS NULL",
        (sha256,),
    )
    if existing is not None:
        return existing["id"]
    with closing(db.connect()) as conn:
        with conn:
            cursor = conn.execute(
                "INSERT INTO media (sha256, file_extension, width, height, "
                "parent_media_id, crop_rect, crop_ratio) "
                "VALUES (?, 'png', ?, ?, ?, ?, ?)",
                (
                    sha256,
                    width,
                    height,
                    parent_media_id,
                    crops.rect_to_json(box),
                    crops.normalize_ratio(ratio),
                ),
            )
            media_id = cursor.lastrowid
    effective_file(media_id)
    return media_id


def update_crop(media_id: int, rect, ratio: str = None) -> dict:
    """Re-frame an existing crop; return its refreshed dict.

    The rectangle drives the crop's identity, so moving it re-hashes the row
    and invalidates everything measured on the old pixels (see the module
    docstring). The stale PNG is dropped from the cache and the new one is
    rendered eagerly. Re-framing to a rectangle another crop of the same
    parent already occupies is refused: the two would be the same media.

    Raises
    ------
    ValueError
        When ``media_id`` is not a crop, its parent has no file, or the new
        rectangle already belongs to a sibling crop.
    """
    crop = get_crop(media_id)
    if crop is None:
        raise ValueError(f"media #{media_id} is not a crop")
    box = crops.normalize_rect(rect)
    sha256, width, height = _crop_identity(crop["parent_media_id"], box)
    new_ratio = crops.normalize_ratio(
        ratio if ratio is not None else crop["ratio"]
    )
    if sha256 != crop["sha256"]:
        clash = _query_one(
            "SELECT id FROM media WHERE sha256 = ? AND deleted_at IS NULL",
            (sha256,),
        )
        if clash is not None:
            raise ValueError(
                f"an identical crop already exists (media #{clash['id']})"
            )
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "UPDATE media SET sha256 = ?, width = ?, height = ?, "
                "crop_rect = ?, crop_ratio = ?, "
                "updated_at = datetime('now') WHERE id = ?",
                (
                    sha256,
                    width,
                    height,
                    crops.rect_to_json(box),
                    new_ratio,
                    media_id,
                ),
            )
            if sha256 != crop["sha256"]:
                _drop_derived(conn, media_id)
    if sha256 != crop["sha256"]:
        crops.delete_render(crop["sha256"])
    effective_file(media_id)
    return get_crop(media_id)


def _drop_derived(conn, media_id: int) -> None:
    """Delete every measurement made on a crop's former pixels."""
    for table in _DERIVED_MEDIA_TABLES:
        conn.execute(f"DELETE FROM {table} WHERE media_id = ?", (media_id,))
    revisions = (
        "SELECT r.id FROM caption_revision r "
        "JOIN caption c ON c.id = r.caption_id WHERE c.media_id = ?"
    )
    for table in _DERIVED_REVISION_TABLES:
        conn.execute(
            f"DELETE FROM {table} WHERE revision_id IN ({revisions})",
            (media_id,),
        )


def place_crop(dataset_id: int, crop_id: int, mode: str = "replace") -> dict:
    """Add a crop to a dataset, either replacing its parent or beside it.

    ``"replace"`` swaps the parent's entry for the crop, inheriting its
    ``hidden`` flag and ``repeats`` count — the dataset keeps the same number
    of samples, framed differently. It requires the parent to *be* an entry:
    with nothing to replace, the crop would later hand the dataset a media it
    never held (see :func:`delete_crop`). ``"beside"`` keeps the parent's
    entry and adds the crop as a second sample of the same image.

    Both are idempotent: a crop already linked to the dataset is left alone.

    Returns
    -------
    dict
        ``{"crop_id": int, "replaced": bool}`` — ``replaced`` tells whether
        the parent's entry was actually removed.

    Raises
    ------
    ValueError
        When ``crop_id`` is not a crop, or ``"replace"`` is asked of a
        dataset the parent does not belong to.
    """
    crop = get_crop(crop_id)
    if crop is None:
        raise ValueError(f"media #{crop_id} is not a crop")
    parent_id = crop["parent_media_id"]
    replace = mode != "beside"
    if dataset_id in _dataset_ids(crop_id):
        return {"crop_id": crop_id, "replaced": False}
    with closing(db.connect()) as conn:
        with conn:
            parent = conn.execute(
                "SELECT hidden, repeats FROM dataset_media "
                "WHERE dataset_id = ? AND media_id = ?",
                (dataset_id, parent_id),
            ).fetchone()
            if replace and parent is None:
                raise ValueError(
                    f"media #{parent_id} is not in dataset #{dataset_id}"
                )
            hidden = parent["hidden"] if replace else 0
            repeats = parent["repeats"] if replace else 1
            conn.execute(
                "INSERT OR IGNORE INTO dataset_media "
                "(dataset_id, media_id, hidden, repeats) VALUES (?, ?, ?, ?)",
                (dataset_id, crop_id, hidden, repeats),
            )
            if replace:
                conn.execute(
                    "DELETE FROM dataset_media "
                    "WHERE dataset_id = ? AND media_id = ?",
                    (dataset_id, parent_id),
                )
    return {"crop_id": crop_id, "replaced": replace}


def delete_crop(media_id: int) -> dict:
    """Delete a crop, restoring its parent in the datasets it stood in.

    Removing a crop must not silently shrink a dataset: wherever the crop
    replaced its parent (the parent is not an entry of that dataset), the
    parent takes its place again. Where both were present side by side, only
    the crop goes. The cached PNG is dropped; the parent's own pixels, tags
    and captions are untouched.

    Returns
    -------
    dict
        ``{"deleted": bool, "restored": [dataset_id, ...]}``.
    """
    crop = get_crop(media_id)
    if crop is None:
        return {"deleted": False, "restored": []}
    parent_id = crop["parent_media_id"]
    restored = []
    with closing(db.connect()) as conn:
        with conn:
            rows = conn.execute(
                "SELECT dataset_id FROM dataset_media WHERE media_id = ?",
                (media_id,),
            ).fetchall()
            for row in rows:
                dataset_id = row["dataset_id"]
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO dataset_media "
                    "(dataset_id, media_id) VALUES (?, ?)",
                    (dataset_id, parent_id),
                )
                if cursor.rowcount:
                    restored.append(dataset_id)
            conn.execute("DELETE FROM media WHERE id = ?", (media_id,))
    crops.delete_render(crop["sha256"])
    return {"deleted": True, "restored": restored}
