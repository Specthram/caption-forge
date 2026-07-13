"""Watermark Lab orchestration: detection, patching, flatten and tag cleanup.

The glue between the repository (:mod:`src.sqlite_store.watermark`), the model
engines (:mod:`src.watermark_detect`, :mod:`src.watermark_flux`) and the
non-destructive composition (:mod:`src.wm_compose`). The batch *scan* is driven
by the job runner (one model loaded at a time — the detector, then FLUX.2
klein); this module owns the per-media steps each phase calls and the review
actions (regenerate, revert, dismiss, tag cleanup) the UI triggers directly.

By default nothing here writes to a media's source file: a patch is a PNG in
the cache, composed over the original only when the media is shown or deployed.
The one exception is :func:`flatten_media` — an explicit, reversible action
that bakes the composite into the source file (a backup is kept).
"""

import hashlib
import logging
import random
import shutil
import time
from pathlib import Path

from src import sqlite_store as store
from src import watermark_flux, wm_compose
from src.constants import DEFAULT_WATERMARK_TAGS, WATERMARK_BACKUPS_DIR

logger = logging.getLogger(__name__)

# Tags stripped from a media once its every watermark zone is patched. Editable
# and persisted in Settings; this is the factory list.
DEFAULT_TAGS_TO_REMOVE = tuple(DEFAULT_WATERMARK_TAGS)

# How a baked source is re-encoded, by extension. Unknown formats fall back to
# PNG (lossless) so a flatten never silently degrades an image.
_FLATTEN_SAVE = {
    ".jpg": ("JPEG", {"quality": 95}),
    ".jpeg": ("JPEG", {"quality": 95}),
    ".png": ("PNG", {}),
    ".webp": ("WEBP", {"quality": 95}),
    ".bmp": ("BMP", {}),
}


def _normalize_tag(name: str) -> str:
    """Return a tag name folded for case- and ``_``/space-insensitive match."""
    return " ".join(str(name).lower().replace("_", " ").split())


# --- Detection helpers (a phase loads the model; these run per media) ---


def create_zones(media_id: int, detections, dilate_px: int) -> list:
    """Create watermark zones from a media's detections; return their ids.

    ``detections`` is a list of ``{"box", "score", "detector", "query"}``
    dicts. A media that already carries zones is left untouched (a re-scan
    never duplicates or clobbers reviewed work) — the caller filters those out.
    """
    ids = []
    for item in detections:
        ids.append(
            store.create_zone(
                media_id,
                item["box"],
                detector=item.get("detector", "manual"),
                score=item.get("score"),
                dilate_px=dilate_px,
                query=item.get("query"),
            )
        )
    return ids


# --- Patch generation ---


def _effective_prompt(zone, prefs, prompt):
    """Return the ``(stored, used)`` edit prompt for a zone.

    ``stored`` is the per-zone override kept on the row (``None`` when the zone
    rides the global instruction); ``used`` is what FLUX actually receives —
    the override, else the zone's existing override, else the global prompt.
    """
    if prompt is not None:
        stored = prompt.strip() or None
    else:
        stored = zone.get("prompt")
    used = (
        stored or prefs.get("prompt") or "remove any watermark, logo or brand"
    )
    return stored, used


def generate_patch(zone_id: int, prefs, prompt=None, seed=None):
    """Run a FLUX edit for a zone and store the patch; return the fresh zone.

    The patch lands under ``patches/<zone_id>.png`` and the zone flips to
    ``patched``. The ``prefs`` carry the engine choice, resolution and global
    prompt; ``prompt`` (when given) overrides the instruction for this zone.
    The source file is never touched. Raises ``ValueError`` when the zone is
    unknown or its media has no readable file.
    """
    zone = store.get_zone(zone_id)
    if zone is None:
        raise ValueError(f"watermark zone #{zone_id} does not exist")
    source = store.effective_file(zone["media_id"])
    if source is None:
        raise ValueError("media has no file on disk")
    dilate = max(0, int(zone["dilate_px"]))
    stored_prompt, used_prompt = _effective_prompt(zone, prefs, prompt)
    started = time.monotonic()
    patch_image = watermark_flux.edit(
        source,
        zone["box"],
        dilate_px=dilate,
        prompt=used_prompt,
        seed=seed,
        max_res=prefs.get("max_res", 1024),
        res_side=prefs.get("res_side", "long"),
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    patch_sha = wm_compose.write_patch(zone_id, patch_image)
    store.set_zone_patch(
        zone_id,
        watermark_flux.model_label(prefs),
        patch_sha,
        seed=seed,
        dilate_px=dilate,
        prompt=stored_prompt,
        edit_ms=elapsed_ms,
        status=store.STATUS_PATCHED,
    )
    return store.get_zone(zone_id)


def regenerate_zone(zone_id: int, prefs, prompt=None, seed=None):
    """Regenerate a zone's patch (new random seed by default); return it."""
    zone = store.get_zone(zone_id)
    if zone is None:
        raise ValueError(f"watermark zone #{zone_id} does not exist")
    if seed is None:
        seed = random.randint(1, 1_000_000)
    return generate_patch(zone_id, prefs, prompt=prompt, seed=seed)


def remove_patch(zone_id: int) -> None:
    """Drop a zone's patch (watermark visible again) and its cached PNG."""
    store.clear_zone_patch(zone_id)
    wm_compose.delete_patch(zone_id)


def restore_media_patches(media_id: int) -> int:
    """Drop every patch of a media, restoring the untouched original.

    Removes the PNG and clears the patch of each zone (the detection boxes
    stay, so the media returns to Watermarked, re-patchable). The composite
    falls back to the source file. Returns how many zones were reset.
    """
    zones = store.list_zones(media_id)
    for zone in zones:
        remove_patch(zone["id"])
    return len(zones)


def delete_zone(zone_id: int):
    """Delete a zone and its patch; return the media id (or None)."""
    wm_compose.delete_patch(zone_id)
    return store.delete_zone(zone_id)


def dismiss_media(media_id: int) -> int:
    """Delete every zone of a media (and its patches); return how many.

    "Dismiss detections": the media becomes neutral/clean, leaving no trace —
    a later scan detects it afresh (no blacklist).
    """
    ids = store.delete_media_zones(media_id)
    for zone_id in ids:
        wm_compose.delete_patch(zone_id)
    return len(ids)


# --- Flatten to disk (explicit, reversible) ---


def _backup_path(media_id: int, filename: str) -> Path:
    """Return where a media's pre-flatten original is stashed."""
    return WATERMARK_BACKUPS_DIR / str(media_id) / filename


def flatten_media(media_id: int) -> bool:
    """Bake a media's patches into its source file; return whether it ran.

    The composite (original + every patched zone) is written over the source
    file so it becomes a single ordinary image with no watermark left to
    detect. The pre-flatten original is copied to a backup and the media's
    content sha is swapped to the baked file's, so a re-scan matches this row
    rather than ingesting a duplicate. Reversible via :func:`unflatten_media`.
    Raises ``ValueError`` when the media has no file or nothing patched.
    """
    if store.is_flattened(media_id):
        return False
    source = store.effective_file(media_id)
    if source is None:
        raise ValueError("media has no file on disk")
    zones = store.list_zones(media_id)
    if not any(wm_compose.has_patch(zone) for zone in zones):
        raise ValueError("media has no patch to flatten")
    media = store.get_media(media_id)
    src_path = Path(source)
    backup = _backup_path(media_id, src_path.name)
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, backup)
    image = wm_compose.compose_image(source, zones)
    fmt, opts = _FLATTEN_SAVE.get(src_path.suffix.lower(), ("PNG", {}))
    image.save(src_path, format=fmt, **opts)
    new_sha = hashlib.sha256(src_path.read_bytes()).hexdigest()
    store.mark_flattened(media_id, media["sha256"], new_sha)
    return True


def unflatten_media(media_id: int) -> bool:
    """Undo a flatten: restore the original bytes and content sha; return ran.

    The backup taken by :func:`flatten_media` is copied back over the source
    file and the media's pre-flatten sha is restored, so its virtual patches
    (still on disk) compose again exactly as before.
    """
    if not store.is_flattened(media_id):
        return False
    source = store.effective_file(media_id)
    store.unmark_flattened(media_id)
    if source is not None:
        src_path = Path(source)
        backup = _backup_path(media_id, src_path.name)
        if backup.is_file():
            shutil.copy2(backup, src_path)
    return True


# --- Tag cleanup ---


def maybe_cleanup_tags(media_id: int, tags_to_remove=DEFAULT_TAGS_TO_REMOVE):
    """Strip the configured tags once every zone of a media is patched.

    Only fires when the media has zones and none is still a bare ``detected``
    (every watermark carries a patch). Matching is case- and ``_``/space-
    insensitive. Returns the list of tag names actually removed.
    """
    zones = store.list_zones(media_id)
    if not zones or any(
        zone["status"] == store.STATUS_DETECTED for zone in zones
    ):
        return []
    wanted = {_normalize_tag(name) for name in tags_to_remove}
    removed = []
    for tag in store.tags_for_media(media_id):
        if _normalize_tag(tag["name"]) in wanted:
            store.remove_tag_from_media(media_id, tag["id"])
            removed.append(tag["name"])
    return removed


# --- Status / composition ---


def display_source(media_id: int):
    """Return ``(path, composed_sha)`` for showing/deploying a media.

    ``path`` is the composited image when the media has patched zones, else
    its plain effective file; ``composed_sha`` is the composite cache key (the
    thumbnail cache key too) or ``None`` when nothing is composed. A flattened
    media already carries the patches in its source file, so it serves that
    file directly. The source file itself is never returned modified (bar a
    deliberate flatten).
    """
    eff = store.effective_file(media_id)
    if eff is None:
        return None, None
    if store.is_flattened(media_id):
        return eff, None
    zones = store.list_zones(media_id)
    if not zones:
        return eff, None
    media = store.get_media(media_id)
    composed = wm_compose.ensure_composed(eff, media["sha256"], zones)
    if composed is None:
        return eff, None
    return str(composed), wm_compose.composed_sha(media["sha256"], zones)


def media_status(media_id: int) -> dict:
    """Return a media's aggregate watermark status and zone details.

    Backs the badges, the Media side panel encart and the Lab review panel.
    """
    zones = store.list_zones(media_id)
    scores = [zone["score"] for zone in zones if zone["score"] is not None]
    models = sorted({zone["model"] for zone in zones if zone["model"]})
    detectors = sorted(
        {zone["detector"] for zone in zones if zone["detector"]}
    )
    return {
        "media_id": media_id,
        "status": store.aggregate_status(zones),
        "flattened": store.is_flattened(media_id),
        "zones": zones,
        "zone_count": len(zones),
        "score_min": min(scores) if scores else None,
        "models": models,
        "detectors": detectors,
    }
