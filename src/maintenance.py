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
from src.constants import WATERMARK_BACKUPS_DIR

logger = logging.getLogger(__name__)

# The cleanup categories, in the order the System view lists them.
CATEGORIES = (
    "media",
    "captions",
    "dataset_captions",
    "claims",
    "quality",
    "embeddings",
    "index",
    "crops",
    "patches",
    "wm_backups",
    "thumbs",
)


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
    return {
        "count": store.unused_revision_count(),
        "bytes": store.unused_revision_bytes(),
    }


def purge_unused_captions() -> dict:
    """Prune the superseded caption revisions; return ``{purged, bytes}``."""
    freed = store.unused_revision_bytes()
    return {"purged": store.prune_unused_revisions(), "bytes": freed}


# -- captions of media in no dataset ------------------------------------------


def unlinked_caption_report() -> dict:
    """Return ``{count, bytes}`` for captions of media in no dataset."""
    return store.unlinked_caption_report()


def purge_unlinked_captions() -> dict:
    """Delete those caption histories; return ``{purged, bytes}``."""
    freed = store.unlinked_caption_report()["bytes"]
    return {"purged": store.purge_unlinked_captions(), "bytes": freed}


# -- caption grounding (claims) history ---------------------------------------


def claims_report() -> dict:
    """Return ``{count, bytes}`` for the stored caption-claim history."""
    return store.grounding_history_report()


def purge_claims() -> dict:
    """Clear the whole grounding history; return ``{purged, bytes}``."""
    freed = store.grounding_history_report()["bytes"]
    return {"purged": store.purge_grounding_history(), "bytes": freed}


# -- index data (quality / embeddings / dims+hashes) --------------------------


def quality_report() -> dict:
    """Return ``{count, bytes}`` for the stored quality scores."""
    return store.quality_scores_report()


def purge_quality() -> dict:
    """Delete every quality score; return ``{purged, bytes}``."""
    freed = store.quality_scores_report()["bytes"]
    return {"purged": store.purge_quality_scores(), "bytes": freed}


def embeddings_report() -> dict:
    """Return ``{count, bytes}`` for the stored embedding vectors."""
    return store.embeddings_report()


def purge_embeddings() -> dict:
    """Delete every embedding vector; return ``{purged, bytes}``."""
    freed = store.embeddings_report()["bytes"]
    return {"purged": store.purge_embeddings(), "bytes": freed}


def media_index_report() -> dict:
    """Return ``{count, bytes}`` for the per-media index columns."""
    return store.media_index_report()


def purge_media_index() -> dict:
    """Reset the per-media index columns; return ``{purged, bytes}``."""
    freed = store.media_index_report()["bytes"]
    return {"purged": store.purge_media_index(), "bytes": freed}


# -- rendered-crop cache ------------------------------------------------------


def crops_cache_report() -> dict:
    """Return ``{count, bytes}`` for the rendered-crop cache."""
    count, total = _tree_size(crops.get_crops_dir())
    return {"count": count, "bytes": total}


def purge_crops_cache() -> dict:
    """Empty the rendered-crop cache; return ``{purged, bytes}``.

    Crops are virtual (a rectangle over the parent media), so the PNGs are
    re-materialized on demand the next time something reads them.
    """
    removed, freed = _empty_tree(crops.get_crops_dir())
    return {"purged": removed, "bytes": freed}


# -- watermark backups --------------------------------------------------------


def wm_backup_report() -> dict:
    """Return ``{count, bytes}`` for the pre-patch original backups."""
    count, total = _tree_size(WATERMARK_BACKUPS_DIR)
    return {"count": count, "bytes": total}


def purge_wm_backups() -> dict:
    """Empty the watermark backups; return ``{purged, bytes}``.

    Loses the "restore original" option of already-patched media; the
    patched pixels themselves are untouched.
    """
    removed, freed = _empty_tree(WATERMARK_BACKUPS_DIR)
    return {"purged": removed, "bytes": freed}


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
    "dataset_captions": unlinked_caption_report,
    "claims": claims_report,
    "quality": quality_report,
    "embeddings": embeddings_report,
    "index": media_index_report,
    "crops": crops_cache_report,
    "patches": orphan_patch_report,
    "wm_backups": wm_backup_report,
    "thumbs": thumbnail_report,
}

_PURGES = {
    "media": purge_orphan_media,
    "captions": purge_unused_captions,
    "dataset_captions": purge_unlinked_captions,
    "claims": purge_claims,
    "quality": purge_quality,
    "embeddings": purge_embeddings,
    "index": purge_media_index,
    "crops": purge_crops_cache,
    "patches": purge_orphan_patches,
    "wm_backups": purge_wm_backups,
    "thumbs": purge_thumbnail_cache,
}

# Only the categories that delete database rows benefit from a VACUUM; the
# file-cache sweeps free their space on unlink.
_VACUUM_AFTER = {
    "media",
    "captions",
    "dataset_captions",
    "claims",
    "quality",
    "embeddings",
    "index",
}


def vacuum() -> None:
    """Rewrite the database file to reclaim the space freed by a purge."""
    with closing(db.connect()) as conn:
        conn.execute("VACUUM")


def cleanup_report() -> dict:
    """Return ``{category: {count, bytes}}`` for every cleanup category."""
    return {category: report() for category, report in _REPORTS.items()}


def category_report(category: str) -> dict:
    """Return one category's ``{count, bytes}`` (the per-row loader).

    The System view fetches each category independently so a slow scan
    (patch orphans, big cache trees) never blocks the cheap ones. Raises
    ``ValueError`` on an unknown category.
    """
    report = _REPORTS.get(category)
    if report is None:
        raise ValueError(f"unknown cleanup category: {category!r}")
    return report()


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
