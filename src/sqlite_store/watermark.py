"""Watermark-zone repository: the boxes and patches that erase filigranes.

A watermark zone is a rectangle of a media (fractions 0-1 of the source) plus
the FLUX.2 klein edit patch that virtually erases it (see
:mod:`src.watermark`). This module only stores and reads those rows; the
pixels — detection, editing and composition — live in the engine modules. The
source file is never touched, so nothing here writes to disk outside the
database — except the flatten flags below, which record that a media's patches
were *baked* into its source file on explicit demand (the pixels are still
written by :mod:`src.watermark`, never here).

Watermark Lab v2: a zone is either ``detected`` (found, no patch) or
``patched`` (erased) — no "review", no media-level "excluded". A media's
aggregate is ``detected`` if any zone is still un-patched, else ``patched``;
no zone at all means the media is neutral/clean.
"""

from src.sqlite_store.base import chunked, _query_all, _query_one, _write

# Per-zone lifecycle (v2): just found vs erased.
STATUS_DETECTED = "detected"
STATUS_PATCHED = "patched"


def _clamp_box(box) -> tuple:
    """Return a ``(x, y, w, h)`` clamped to a valid fraction rectangle.

    Each side is kept within ``[0, 1]`` and given a minimum extent so a zone
    dragged to nothing stays renderable; the box never leaves the image.
    """

    def num(key, default):
        try:
            return float((box or {}).get(key, default))
        except (TypeError, ValueError):
            return default

    width = min(1.0, max(0.005, num("w", 0.1)))
    height = min(1.0, max(0.005, num("h", 0.1)))
    left = min(max(0.0, num("x", 0.0)), 1.0 - width)
    top = min(max(0.0, num("y", 0.0)), 1.0 - height)
    return left, top, width, height


def _zone_dict(row) -> dict:
    """Return a zone row as an API-friendly dict (box nested, patch flat)."""
    return {
        "id": row["id"],
        "media_id": row["media_id"],
        "box": {"x": row["x"], "y": row["y"], "w": row["w"], "h": row["h"]},
        "status": row["status"],
        "model": row["model"],
        "seed": row["seed"],
        "score": row["score"],
        "patch_sha": row["patch_sha"],
        "detector": row["detector"],
        "query": row["query"],
        "dilate_px": row["dilate_px"],
        "prompt": row["prompt"],
        "edit_ms": row["edit_ms"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_zones(media_id: int) -> list:
    """Return a media's watermark zones as dicts, oldest first."""
    rows = _query_all(
        "SELECT * FROM watermark_zone WHERE media_id = ? ORDER BY id",
        (media_id,),
    )
    return [_zone_dict(row) for row in rows]


def get_zone(zone_id: int):
    """Return one zone dict by id, or None."""
    row = _query_one("SELECT * FROM watermark_zone WHERE id = ?", (zone_id,))
    return _zone_dict(row) if row is not None else None


def zones_bulk(media_ids) -> dict:
    """Return ``{media_id: [zone dicts]}`` for many media in one query.

    Backs the grid badges and the Lab inventory: one round-trip instead of a
    per-card query. A media with no zone is absent from the dict.
    """
    ids = [int(m) for m in media_ids]
    if not ids:
        return {}
    grouped: dict = {}
    for chunk in chunked(ids):
        placeholders = ", ".join("?" for _ in chunk)
        for row in _query_all(
            "SELECT * FROM watermark_zone "
            f"WHERE media_id IN ({placeholders}) ORDER BY media_id, id",
            chunk,
        ):
            grouped.setdefault(row["media_id"], []).append(_zone_dict(row))
    return grouped


def create_zone(
    media_id: int,
    box,
    detector: str = "manual",
    score=None,
    status: str = STATUS_DETECTED,
    dilate_px: int = 8,
    query=None,
) -> int:
    """Insert a watermark zone; return its id.

    ``box`` is a ``{"x", "y", "w", "h"}`` mapping in fractions of the source
    image (clamped). ``query`` is the OWLv2 term that matched (NULL for YOLO
    or manual). A fresh zone carries no patch — that is added later by
    :func:`set_zone_patch` once inpainting runs.
    """
    left, top, width, height = _clamp_box(box)
    return _write(
        "INSERT INTO watermark_zone "
        "(media_id, x, y, w, h, status, score, detector, query, dilate_px) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            media_id,
            left,
            top,
            width,
            height,
            status,
            score,
            detector,
            str(query).strip() if query else None,
            int(dilate_px),
        ),
    )


def update_zone_box(zone_id: int, box, dilate_px=None) -> None:
    """Move/resize a zone's box (fractions of the source).

    Only the geometry changes here; the caller regenerates the patch after,
    because the old patch no longer covers the new rectangle.
    """
    left, top, width, height = _clamp_box(box)
    if dilate_px is None:
        _write(
            "UPDATE watermark_zone SET x = ?, y = ?, w = ?, h = ?, "
            "updated_at = datetime('now') WHERE id = ?",
            (left, top, width, height, zone_id),
        )
        return
    _write(
        "UPDATE watermark_zone SET x = ?, y = ?, w = ?, h = ?, "
        "dilate_px = ?, updated_at = datetime('now') WHERE id = ?",
        (left, top, width, height, int(dilate_px), zone_id),
    )


def set_zone_patch(
    zone_id: int,
    model: str,
    patch_sha: str,
    seed=None,
    dilate_px=None,
    prompt=None,
    edit_ms=None,
    status: str = STATUS_PATCHED,
) -> None:
    """Record a freshly generated patch on a zone (marks it patched)."""
    _write(
        "UPDATE watermark_zone SET model = ?, patch_sha = ?, seed = ?, "
        "dilate_px = COALESCE(?, dilate_px), prompt = ?, edit_ms = ?, "
        "status = ?, updated_at = datetime('now') WHERE id = ?",
        (
            model,
            patch_sha,
            seed,
            None if dilate_px is None else int(dilate_px),
            prompt,
            edit_ms,
            status,
            zone_id,
        ),
    )


def set_zone_prompt(zone_id: int, prompt) -> None:
    """Set (or clear) a zone's per-zone edit instruction override.

    ``None``/empty stores NULL so the zone falls back to the global prompt on
    the next regeneration; the patch itself is untouched until then.
    """
    text = str(prompt).strip() if prompt is not None else ""
    _write(
        "UPDATE watermark_zone SET prompt = ?, updated_at = datetime('now') "
        "WHERE id = ?",
        (text or None, zone_id),
    )


def set_zone_status(zone_id: int, status: str) -> None:
    """Set a zone's status (e.g. re-detect a patched zone)."""
    _write(
        "UPDATE watermark_zone SET status = ?, updated_at = datetime('now') "
        "WHERE id = ?",
        (status, zone_id),
    )


def clear_zone_patch(zone_id: int) -> None:
    """Drop a zone's patch, returning it to the ``detected`` state.

    The watermark becomes visible again (the composite stops pasting it); the
    zone box and its detection metadata are kept so the user can re-generate
    without redrawing.
    """
    _write(
        "UPDATE watermark_zone SET status = ?, model = NULL, "
        "patch_sha = NULL, prompt = NULL, edit_ms = NULL, "
        "updated_at = datetime('now') WHERE id = ?",
        (STATUS_DETECTED, zone_id),
    )


def delete_zone(zone_id: int):
    """Delete a zone; return its ``media_id`` (or None when it did not exist).

    The caller recomposes the media's display image afterwards.
    """
    row = _query_one(
        "SELECT media_id FROM watermark_zone WHERE id = ?", (zone_id,)
    )
    if row is None:
        return None
    _write("DELETE FROM watermark_zone WHERE id = ?", (zone_id,))
    return row["media_id"]


def delete_media_zones(media_id: int) -> list:
    """Delete every zone of a media; return the deleted zone ids.

    Backs "Dismiss detections": the media becomes neutral/clean again (no
    trace, re-scannable). The caller drops each zone's cached patch PNG.
    """
    ids = [
        row["id"]
        for row in _query_all(
            "SELECT id FROM watermark_zone WHERE media_id = ?", (media_id,)
        )
    ]
    if ids:
        _write("DELETE FROM watermark_zone WHERE media_id = ?", (media_id,))
    return ids


# --- Flatten to disk (media-level) ---


def is_flattened(media_id: int) -> bool:
    """Return whether a media's patches were baked into its source file."""
    row = _query_one(
        "SELECT wm_flattened FROM media WHERE id = ?", (media_id,)
    )
    return bool(row and row["wm_flattened"])


def mark_flattened(media_id: int, original_sha: str, new_sha: str) -> None:
    """Record a media as flattened, swapping in the baked file's content sha.

    ``original_sha`` (the pre-flatten identity) is stashed in ``wm_flat_sha``
    so :func:`unmark_flattened` can restore it; ``new_sha`` becomes the
    media's live content hash so a re-scan of the baked file matches this row
    instead of ingesting a duplicate.
    """
    _write(
        "UPDATE media SET wm_flattened = 1, wm_flat_sha = ?, sha256 = ?, "
        "updated_at = datetime('now') WHERE id = ?",
        (original_sha, new_sha, media_id),
    )


def unmark_flattened(media_id: int):
    """Clear a media's flatten flag, restoring its pre-flatten content sha.

    Returns the stashed original sha (or None), which the caller uses to
    restore the source bytes from the backup.
    """
    row = _query_one("SELECT wm_flat_sha FROM media WHERE id = ?", (media_id,))
    original = row["wm_flat_sha"] if row else None
    _write(
        "UPDATE media SET wm_flattened = 0, wm_flat_sha = NULL, "
        "sha256 = COALESCE(?, sha256), updated_at = datetime('now') "
        "WHERE id = ?",
        (original, media_id),
    )
    return original


def flattened_media_ids(media_ids) -> set:
    """Return which of the given media are flattened, in one query per chunk.

    Backs the inventory page badge without a per-card ``is_flattened`` read.
    """
    ids = [int(m) for m in media_ids]
    out: set = set()
    for chunk in chunked(ids):
        placeholders = ", ".join("?" for _ in chunk)
        for row in _query_all(
            "SELECT id FROM media WHERE wm_flattened = 1 "
            f"AND id IN ({placeholders})",
            chunk,
        ):
            out.add(row["id"])
    return out


def aggregate_status(zones):
    """Return a media's aggregate watermark status, or None when untouched.

    ``detected`` if any zone still lacks a patch, else ``patched`` when the
    media carries zones; ``None`` (neutral/clean) when it has none.
    """
    if not zones:
        return None
    if any(zone["status"] == STATUS_DETECTED for zone in zones):
        return STATUS_DETECTED
    return STATUS_PATCHED


def media_ids_with_zones() -> list:
    """Return the ids of every media carrying at least one watermark zone."""
    rows = _query_all(
        "SELECT DISTINCT media_id FROM watermark_zone ORDER BY media_id"
    )
    return [row["media_id"] for row in rows]


def media_ids_with_detected_zones() -> list:
    """Return the ids of media that still carry a ``detected`` zone.

    Backs "Patch" over a whole tab: each media awaiting an erase.
    """
    rows = _query_all(
        "SELECT DISTINCT media_id FROM watermark_zone WHERE status = ? "
        "ORDER BY media_id",
        (STATUS_DETECTED,),
    )
    return [row["media_id"] for row in rows]


def all_zone_ids() -> set:
    """Return the ids of every watermark zone.

    The patch cache (:func:`src.wm_compose.patch_path`) names each PNG
    ``<zone_id>.png``; this set is the "still referenced" side the
    maintenance sweep intersects against to spot patch files whose zone row
    is gone (see :mod:`src.maintenance`).
    """
    rows = _query_all("SELECT id FROM watermark_zone")
    return {row["id"] for row in rows}
