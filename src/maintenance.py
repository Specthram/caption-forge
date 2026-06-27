"""Database and cache maintenance sweeps for the System view.

The "Database cleanup" block purges four kinds of orphaned data, each behind a
two-step confirmation. This module owns the semantics; the router is a thin
adapter. Categories, adapted to CF's library-centric schema (see
:mod:`src.db`):

* **orphan media** -- media belongs to a *library*, not a dataset, so "media
  in no dataset" is most of the library and must never be purged. A safe
  orphan is referenced by no file row *and* no dataset — a dead row from an
  odd delete order, invisible in every grid. Rare, so usually zero.
* **unused caption versions** -- superseded revisions of the append-only
  history: every revision neither a caption's head nor pinned by a dataset.
* **orphan patches** -- watermark/crop *cache files* with no live row. The
  rows themselves can't orphan (they cascade on delete), so only stray PNGs
  the delete path missed: patches, composites, rendered crops.
* **thumbnail cache** -- the whole thumbnail cache, rebuilt lazily.

Each category exposes a ``*_report`` (count + reclaimable bytes) and a
``*_purge``. The DB categories run ``VACUUM`` afterwards to shrink the file.
"""

import logging
import os
from contextlib import closing
from pathlib import Path

from src import crops, db, thumbnails, wm_compose
from src import sqlite_store as store

logger = logging.getLogger(__name__)

# The cleanup categories, in the order the System view lists them.
CATEGORIES = ("media", "captions", "patches", "thumbs")


def _safe_size(path: Path) -> int:
    """Return a file's size in bytes, or 0 when it cannot be stat-ed."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _unlink(path: Path) -> bool:
    """Delete a file, returning whether it went away (a no-op when absent)."""
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("maintenance: could not delete %s: %s", path, exc)
        return False


def _tree_size(root: Path) -> tuple:
    """Return ``(file_count, total_bytes)`` under a directory tree."""
    count = 0
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            count += 1
            total += _safe_size(Path(dirpath) / name)
    return count, total


def _empty_tree(root: Path) -> tuple:
    """Delete every file under a tree; return ``(removed, freed_bytes)``.

    Files are removed but the root and its shard sub-folders are left in
    place (an empty directory costs nothing and saves the next write a
    ``mkdir``).
    """
    removed = 0
    freed = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = Path(dirpath) / name
            size = _safe_size(path)
            if _unlink(path):
                removed += 1
                freed += size
    return removed, freed


# -- orphan media -----------------------------------------------------------


def orphan_media_report() -> dict:
    """Return ``{count, bytes}`` for the orphan media rows (no disk use)."""
    return {"count": store.count_orphan_media(), "bytes": 0}


def purge_orphan_media() -> dict:
    """Hard-delete the orphan media rows; return ``{purged, bytes}``."""
    purged = store.purge_media(store.orphan_media_ids())
    return {"purged": purged, "bytes": 0}


# -- unused caption versions ------------------------------------------------


def unused_caption_report() -> dict:
    """Return ``{count, bytes}`` for the superseded caption revisions."""
    return {"count": store.unused_revision_count(), "bytes": 0}


def purge_unused_captions() -> dict:
    """Prune the superseded caption revisions; return ``{purged, bytes}``."""
    return {"purged": store.prune_unused_revisions(), "bytes": 0}


# -- orphan patch / composite / crop cache files ----------------------------


def _live_composed_shas() -> set:
    """Return the cache keys of every composite a live media still needs.

    A composite is keyed by source hash folded with its patched zones, so a
    patch regen or moved box dangles the old file. Rebuilding the current keys
    of zone-carrying media is the "still referenced" side of the sweep.
    """
    ids = store.media_ids_with_zones()
    if not ids:
        return set()
    shas = store.media_sha256_bulk(ids)
    zones = store.zones_bulk(ids)
    live = set()
    for media_id in ids:
        sha = shas.get(media_id)
        if not sha:
            continue
        key = wm_compose.composed_sha(sha, zones.get(media_id, []))
        if key is not None:
            live.add(key)
    return live


def _orphan_patch_files() -> list:
    """Return the patch/composite/crop cache files with no live owner."""
    orphans = []

    zone_ids = store.all_zone_ids()
    for path in wm_compose.get_patches_dir().glob("*.png"):
        try:
            zone_id = int(path.stem)
        except ValueError:
            continue
        if zone_id not in zone_ids:
            orphans.append(path)

    crop_shas = store.all_crop_shas()
    for path in crops.get_crops_dir().glob("*/*.png"):
        if path.stem not in crop_shas:
            orphans.append(path)

    live_composed = _live_composed_shas()
    for path in wm_compose.get_composed_dir().glob("*/*.png"):
        if path.stem not in live_composed:
            orphans.append(path)

    return orphans


def orphan_patch_report() -> dict:
    """Return ``{count, bytes}`` for the orphan patch/crop cache files."""
    files = _orphan_patch_files()
    return {"count": len(files), "bytes": sum(_safe_size(p) for p in files)}


def purge_orphan_patches() -> dict:
    """Delete the orphan patch/crop cache files; return ``{purged, bytes}``."""
    removed = 0
    freed = 0
    for path in _orphan_patch_files():
        size = _safe_size(path)
        if _unlink(path):
            removed += 1
            freed += size
    return {"purged": removed, "bytes": freed}


# -- thumbnail cache --------------------------------------------------------


def thumbnail_report() -> dict:
    """Return ``{count, bytes}`` for the generated-thumbnail cache."""
    count, total = _tree_size(thumbnails.get_thumbnails_dir())
    return {"count": count, "bytes": total}


def purge_thumbnail_cache() -> dict:
    """Empty the thumbnail cache; return ``{purged, bytes}``."""
    removed, freed = _empty_tree(thumbnails.get_thumbnails_dir())
    return {"purged": removed, "bytes": freed}


# -- orchestration ----------------------------------------------------------

_REPORTS = {
    "media": orphan_media_report,
    "captions": unused_caption_report,
    "patches": orphan_patch_report,
    "thumbs": thumbnail_report,
}

_PURGES = {
    "media": purge_orphan_media,
    "captions": purge_unused_captions,
    "patches": purge_orphan_patches,
    "thumbs": purge_thumbnail_cache,
}

# Only the categories that delete database rows benefit from a VACUUM; the
# file-cache sweeps free their space on unlink.
_VACUUM_AFTER = {"media", "captions"}


def vacuum() -> None:
    """Rewrite the database file to reclaim the space freed by a purge."""
    with closing(db.connect()) as conn:
        conn.execute("VACUUM")


def cleanup_report() -> dict:
    """Return ``{category: {count, bytes}}`` for every cleanup category."""
    return {category: report() for category, report in _REPORTS.items()}


def run_cleanup(category: str) -> dict:
    """Purge one :data:`CATEGORIES` category.

    Returns ``{purged, bytes, vacuumed}`` — rows/files removed, bytes
    reclaimed, and whether the DB was compacted. Raises ``ValueError`` on an
    unknown category.
    """
    purge = _PURGES.get(category)
    if purge is None:
        raise ValueError(f"unknown cleanup category: {category!r}")
    result = purge()
    vacuumed = category in _VACUUM_AFTER
    if vacuumed:
        vacuum()
    return {**result, "vacuumed": vacuumed}
