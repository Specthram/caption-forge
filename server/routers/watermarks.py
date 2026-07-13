"""Watermark Lab routes (v2): decoupled scan / patch / flatten, tab inventory.

Detection and editing are separate actions on a self-contained screen: a scan
locates boxes (OWLv2 the default), a patch runs FLUX.2 klein over detected
zones, scan+patch chains both. Every model action is a job the single-worker
queue serialises; the pure-database actions (dismiss, revert, zone prompt) are
plain synchronous REST. The inventory pages three tabs — Medias, Watermarked,
Patched — with the Media page's own tag/favorite/sort filters.

Nothing here writes a media's source file except the explicit, reversible
``flatten`` (a backup is kept); a patch is a cached PNG composed over the
original at display and deploy time (see :mod:`src.wm_compose`).
"""

import random

from fastapi import APIRouter, HTTPException, Query

from server.jobs import manager
from server.runners import watermark as runner
from server.schemas import (
    WatermarkCreateZoneBody,
    WatermarkRegenBody,
    WatermarkSelectionBody,
    WatermarkZoneBody,
)
from src import settings, watermark
from src import sqlite_store as store
from src import watermark_detect as detect
from src import watermark_flux as flux
from src.constants import OWLV2_MODELS
from src.media import is_video_file

router = APIRouter(prefix="/api/watermarks", tags=["watermarks"])

# Sort keys the inventory accepts, mirroring the Media grid.
_SORTS = {
    "date_desc": store.SORT_DATE_DESC,
    "date_asc": store.SORT_DATE_ASC,
    "quality_desc": store.SORT_QUALITY_DESC,
    "quality_asc": store.SORT_QUALITY_ASC,
    "dimension_desc": store.SORT_DIMENSION_DESC,
    "dimension_asc": store.SORT_DIMENSION_ASC,
}

# Tab -> watermark-membership constraint for the library query.
_WM_TAB = {"media": None, "watermarked": "watermarked", "patched": "patched"}


def _cleanup_tags():
    """Return the configured tags to strip on full patch, or None when off."""
    prefs = settings.get_watermark_prefs()
    return prefs["tags_to_remove"] if prefs["tag_cleanup"] else None


def _media_payload(media_id: int) -> dict:
    """Return one media's watermark status with its name and thumb url."""
    payload = watermark.media_status(media_id)
    row = store.get_media(media_id)
    display = store.get_media_display(media_id)
    payload["name"] = display["name"] if display else f"media #{media_id}"
    payload["thumb"] = f"/api/media/{media_id}/thumb"
    payload["is_video"] = bool(
        row and is_video_file(f"x.{row['file_extension']}")
    )
    return payload


def _resolve_ids(body: WatermarkSelectionBody) -> list:
    """Return the media ids a batch acts on (explicit list, or all filtered).

    ``select_all`` resolves the whole filtered result of the body's tab across
    every page (the check-all box), so a batch scan/patch is not limited to the
    visible page.
    """
    if not body.select_all:
        return [int(mid) for mid in body.media_ids]
    return store.library_media_ids(
        tag_ids=body.tag_ids or None,
        match=body.match,
        exclude_tag_ids=body.exclude_tag_ids or None,
        favorites_only=body.favorites_only,
        wm_tab=_WM_TAB.get(body.tab),
    )


# --- Config -----------------------------------------------------------------


@router.get("/config")
def watermark_config() -> dict:
    """Return the rail preferences and the model availability the Lab needs."""
    prefs = settings.get_watermark_prefs()
    return {
        "prefs": prefs,
        "media_total": store.count_live_media(images_only=True),
        "owlv2_model_id": detect.OWLV2_MODEL_ID,
        "owlv2_models": [
            {"id": model_id, "label": label}
            for model_id, label in OWLV2_MODELS
        ],
        "yolo_available": detect.is_yolo_available(),
        "yolo_models": detect.list_yolo_models(),
        "yolo_models_dir": str(detect.yolo_models_dir()),
        "flux_available": flux.is_available(prefs),
        "flux_repo": flux.resolve_repo(
            prefs["model"], prefs["precision"], prefs["kv"]
        ),
        "flux_label": flux.model_label(prefs),
        "encoder_repo": flux.encoder_repo(prefs["model"]),
    }


@router.patch("/config")
def update_watermark_config(prefs: dict) -> dict:
    """Persist a partial rail-preferences update; return the merged prefs."""
    return {"prefs": settings.set_watermark_prefs(prefs)}


# --- Inventory --------------------------------------------------------------


def _inventory_item(media: dict, zones: list, flattened: bool) -> dict:
    """Return one inventory card: media fields + its watermark state."""
    scores = [zone["score"] for zone in zones if zone["score"] is not None]
    detectors = sorted({z["detector"] for z in zones if z["detector"]})
    models = sorted({z["model"] for z in zones if z["model"]})
    return {
        "media_id": media["id"],
        "key": str(media["id"]),
        "name": media["name"],
        "thumb": f"/api/media/{media['id']}/thumb",
        "width": media["width"],
        "height": media["height"],
        "quality": media.get("quality_score"),
        "favorite": bool(media.get("favorite")),
        "is_video": is_video_file(f"x.{media['file_extension']}"),
        "status": store.aggregate_status(zones),
        "flattened": flattened,
        "zone_count": len(zones),
        "score_min": min(scores) if scores else None,
        "detectors": detectors,
        "models": models,
        "zones": zones,
    }


@router.get("")
def watermark_inventory(
    tab: str = "media",
    tag_ids: list[int] = Query(default=[]),
    exclude_tag_ids: list[int] = Query(default=[]),
    match: str = "any",
    favorites_only: bool = False,
    sort: str = "date_desc",
    offset: int = 0,
    limit: int = 60,
) -> dict:
    """Return one filtered page of a Lab tab, with per-tab counters.

    The counters are the same tag/favorite filter counted under each tab's
    watermark membership, so the tab labels track the current filter.
    """
    filt = {
        "tag_ids": tag_ids or None,
        "match": match,
        "exclude_tag_ids": exclude_tag_ids or None,
        "favorites_only": favorites_only,
    }
    counts = {
        "media": store.count_library_media(**filt),
        "watermarked": store.count_library_media(**filt, wm_tab="watermarked"),
        "patched": store.count_library_media(**filt, wm_tab="patched"),
    }
    page = store.library_media_page(
        tag_ids or None,
        match,
        _SORTS.get(sort, store.SORT_DATE_DESC),
        offset,
        limit,
        favorites_only=favorites_only,
        exclude_tag_ids=exclude_tag_ids or None,
        wm_tab=_WM_TAB.get(tab),
    )
    ids = [item["id"] for item in page]
    zones_by = store.zones_bulk(ids)
    flat = store.flattened_media_ids(ids)
    items = [
        _inventory_item(item, zones_by.get(item["id"], []), item["id"] in flat)
        for item in page
    ]
    return {
        "items": items,
        "counts": counts,
        "total": counts.get(tab, counts["media"]),
        "tab": tab,
    }


@router.get("/{media_id}")
def watermark_media(media_id: int) -> dict:
    """Return one media's watermark zones and aggregate status."""
    if store.get_media(media_id) is None:
        raise HTTPException(status_code=404, detail="media not found")
    return _media_payload(media_id)


# --- Scan / patch jobs ------------------------------------------------------


@router.post("/scan")
def watermark_scan(body: WatermarkSelectionBody) -> dict:
    """Enqueue a detect-only scan over the selection; return job id."""
    ids = _resolve_ids(body)
    job = manager.submit(
        "watermark",
        f"Watermark scan · {len(ids)} media",
        runner.scan_body(ids),
    )
    return {"job_id": job.id}


@router.post("/scan_and_patch")
def watermark_scan_and_patch(body: WatermarkSelectionBody) -> dict:
    """Enqueue a detect-then-patch run over the selection; return job id."""
    ids = _resolve_ids(body)
    job = manager.submit(
        "watermark",
        f"Watermark scan & patch · {len(ids)} media",
        runner.scan_and_patch_body(ids),
    )
    return {"job_id": job.id}


@router.post("/patch")
def watermark_patch(body: WatermarkSelectionBody) -> dict:
    """Enqueue a FLUX patch of the detected zones of the selection."""
    ids = _resolve_ids(body)
    prefs = settings.get_watermark_prefs()
    job = manager.submit(
        "watermark",
        f"Watermark patch · {len(ids)} media · FLUX.2 "
        f"{flux.model_label(prefs)}",
        runner.patch_body(ids, prefs, _cleanup_tags()),
    )
    return {"job_id": job.id}


@router.post("/{media_id}/patch")
def patch_media(media_id: int) -> dict:
    """Queue a FLUX patch for one media's detected zones; return job id."""
    if store.get_media(media_id) is None:
        raise HTTPException(status_code=404, detail="media not found")
    prefs = settings.get_watermark_prefs()
    job = manager.submit(
        "watermark",
        f"Patch media {media_id} · FLUX.2 {flux.model_label(prefs)}",
        runner.patch_media_body(media_id, prefs, _cleanup_tags()),
    )
    return {"job_id": job.id}


def _patch_zone(zone_id: int, prompt=None, seed=None) -> dict:
    """Queue a FLUX edit for a zone; return its ``job_id`` and media id."""
    zone = store.get_zone(zone_id)
    if zone is None:
        raise HTTPException(status_code=404, detail="zone not found")
    prefs = settings.get_watermark_prefs()
    job = manager.submit(
        "watermark",
        f"Patch zone {zone_id} · FLUX.2 {flux.model_label(prefs)}",
        runner.patch_zone_body(zone_id, prefs, prompt, seed, _cleanup_tags()),
    )
    return {"job_id": job.id, "media_id": zone["media_id"]}


@router.post("/{zone_id}/regenerate")
def regenerate_zone(zone_id: int, body: WatermarkRegenBody) -> dict:
    """Regenerate a zone's patch (new seed by default, optional new prompt)."""
    zone = store.get_zone(zone_id)
    if zone is None:
        raise HTTPException(status_code=404, detail="zone not found")
    seed = body.seed if body.seed is not None else random.randint(1, 1_000_000)
    return _patch_zone(zone_id, body.prompt, seed)


@router.patch("/{zone_id}")
def edit_zone(zone_id: int, body: WatermarkZoneBody) -> dict:
    """Move a zone's box (auto-regen), set its prompt, or revert its patch.

    A moved box queues a regeneration. A ``prompt`` persists the per-zone edit
    instruction. A ``detected`` status reverts the patch (watermark visible
    again).
    """
    zone = store.get_zone(zone_id)
    if zone is None:
        raise HTTPException(status_code=404, detail="zone not found")
    media_id = zone["media_id"]
    if body.box is not None:
        store.update_zone_box(zone_id, body.box.model_dump())
        return _patch_zone(zone_id, body.prompt)
    if body.prompt is not None:
        store.set_zone_prompt(zone_id, body.prompt)
        return {"media": _media_payload(media_id)}
    if body.status == store.STATUS_DETECTED:
        watermark.remove_patch(zone_id)
    return {"media": _media_payload(media_id)}


@router.post("/{media_id}/zones")
def add_zone(media_id: int, body: WatermarkCreateZoneBody) -> dict:
    """Add a manual watermark zone (no patch yet — the user generates one)."""
    if store.get_media(media_id) is None:
        raise HTTPException(status_code=404, detail="media not found")
    zone_id = store.create_zone(
        media_id,
        body.box.model_dump(),
        detector="manual",
        dilate_px=settings.get_watermark_prefs()["dilate_px"],
    )
    return {"zone_id": zone_id, "media": _media_payload(media_id)}


@router.delete("/{zone_id}")
def delete_zone(zone_id: int) -> dict:
    """Delete a zone and its patch; return the media's refreshed status."""
    media_id = watermark.delete_zone(zone_id)
    if media_id is None:
        raise HTTPException(status_code=404, detail="zone not found")
    return {"media": _media_payload(media_id)}


# --- Dismiss / revert / flatten ---------------------------------------------


@router.post("/dismiss")
def dismiss_selection(body: WatermarkSelectionBody) -> dict:
    """Delete every zone of each selected media; return how many were reset."""
    dismissed = sum(watermark.dismiss_media(mid) for mid in _resolve_ids(body))
    return {"dismissed": dismissed}


@router.delete("/media/{media_id}")
def dismiss_media(media_id: int) -> dict:
    """Delete every zone of one media (back to clean); refresh status."""
    if store.get_media(media_id) is None:
        raise HTTPException(status_code=404, detail="media not found")
    watermark.dismiss_media(media_id)
    return {"media": _media_payload(media_id)}


def _revert_media(media_id: int) -> None:
    """Send a media back to Watermarked: unflatten (if baked), drop patches."""
    if store.is_flattened(media_id):
        watermark.unflatten_media(media_id)
    watermark.restore_media_patches(media_id)


@router.post("/revert")
def revert_selection(body: WatermarkSelectionBody) -> dict:
    """Revert every selected media's patches (back to Watermarked)."""
    ids = _resolve_ids(body)
    for media_id in ids:
        _revert_media(media_id)
    return {"reverted": len(ids)}


@router.post("/{media_id}/revert")
def revert_media(media_id: int) -> dict:
    """Revert one media's patches (unflatten if baked); refresh status."""
    if store.get_media(media_id) is None:
        raise HTTPException(status_code=404, detail="media not found")
    _revert_media(media_id)
    return {"media": _media_payload(media_id)}


@router.post("/flatten")
def flatten_selection(body: WatermarkSelectionBody) -> dict:
    """Enqueue a flatten (bake patches into the source file) for the set."""
    ids = _resolve_ids(body)
    job = manager.submit(
        "watermark",
        f"Watermark flatten · {len(ids)} media",
        runner.flatten_body(ids),
    )
    return {"job_id": job.id}


@router.post("/{media_id}/flatten")
def flatten_media(media_id: int) -> dict:
    """Enqueue a flatten for one media; return job id."""
    if store.get_media(media_id) is None:
        raise HTTPException(status_code=404, detail="media not found")
    job = manager.submit(
        "watermark",
        f"Watermark flatten · media {media_id}",
        runner.flatten_body([media_id]),
    )
    return {"job_id": job.id}
