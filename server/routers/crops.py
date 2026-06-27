"""Crop routes: the virtual, non-destructive crop aliases of a media.

A crop creates no file (see :mod:`src.crops`): it is a rectangle of a parent
media, living as a dataset entry and rendered on the fly. These routes are
all light and synchronous — creating a crop only writes a row and renders one
PNG into the cache. Re-running quality, auto-tagging or grounding *on* a crop
goes through the existing jobs: a crop is a media id like any other.
"""

from fastapi import APIRouter, HTTPException

from server.schemas import CropCreateBody, CropPlaceBody, CropUpdateBody
from src import crops as crop_engine
from src import sqlite_store as store
from src.media import is_video_file

router = APIRouter(prefix="/api/crops", tags=["crops"])


def crop_card(item: dict) -> dict | None:
    """Return a media dict's crop descriptor, or None on an ordinary media.

    A crop's thumbnail already *is* the cropped pixels (its effective file is
    the rendered PNG), so a card needs the rectangle only to badge itself and
    to reopen the overlay on the right frame.
    """
    parent_id = item.get("parent_media_id")
    if not parent_id:
        return None
    return {
        "parent_media_id": parent_id,
        "rect": item.get("crop_rect"),
        "ratio": item.get("crop_ratio") or "free",
        "width": item.get("width"),
        "height": item.get("height"),
    }


def _crop_payload(crop: dict) -> dict:
    """Return one crop as the front-end shape (rect, ratio, rendered size)."""
    return {
        "id": crop["id"],
        "parent_media_id": crop["parent_media_id"],
        "rect": crop["rect"],
        "ratio": crop["ratio"],
        "width": crop["width"],
        "height": crop["height"],
        "thumb": f"/api/media/{crop['id']}/thumb",
        "dataset_ids": crop.get("dataset_ids", []),
    }


@router.get("/source/{media_id}")
def crop_source(media_id: int) -> dict:
    """Return what the crop overlay needs to frame a media.

    The source dimensions come from the index when it ran, and are probed
    from the file otherwise — the overlay cannot draw a rectangle in
    percentages of an unknown size.
    """
    media = store.get_media_display(media_id)
    if media is None:
        raise HTTPException(status_code=404, detail="media not found")
    return _source_payload(media, media_id)


def _source_payload(media: dict, media_id: int) -> dict:
    """Return the overlay's source descriptor for a resolved media dict."""
    if media["parent_media_id"]:
        raise HTTPException(status_code=400, detail="a crop cannot be cropped")
    if media["missing"]:
        raise HTTPException(status_code=404, detail="file missing on disk")
    if is_video_file(f"x.{media['file_extension']}"):
        raise HTTPException(status_code=400, detail="videos cannot be cropped")
    width, height = media["width"], media["height"]
    if not width or not height:
        width, height = crop_engine.source_size(media["eff_path"])
    return {
        "media_id": media_id,
        "name": media["name"],
        "width": width,
        "height": height,
        "file": f"/api/media/{media_id}/file",
    }


@router.get("")
def list_crops(media_id: int, dataset_id: int | None = None) -> dict:
    """Return the crops of a media's *source*, marked against a dataset.

    Pass an ordinary media and these are its own crops; pass a crop and these
    are its siblings — the panels offer the same list either way, since a crop
    is always re-framed from the image it aliases. ``in_dataset`` marks the
    crops already standing in ``dataset_id``.
    """
    crop = store.get_crop(media_id)
    source_id = crop["parent_media_id"] if crop else media_id
    payloads = []
    for row in store.list_crops(source_id):
        payload = _crop_payload(row)
        payload["in_dataset"] = dataset_id in row["dataset_ids"]
        payloads.append(payload)
    return {"crops": payloads}


@router.post("")
def create_crop(body: CropCreateBody) -> dict:
    """Create a crop of a media and place it in a dataset.

    An identical rectangle of the same parent resolves to the crop that
    already exists rather than a duplicate, so the route is safe to replay.
    """
    try:
        crop_id = store.create_crop(
            body.media_id, body.rect.model_dump(), body.ratio
        )
        placed = store.place_crop(body.dataset_id, crop_id, body.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    crop = store.get_crop(crop_id)
    crop["dataset_ids"] = [body.dataset_id]
    return {**_crop_payload(crop), **placed}


@router.post("/{crop_id}")
def update_crop(crop_id: int, body: CropUpdateBody) -> dict:
    """Re-frame a crop; its quality, grounding and scores are invalidated."""
    try:
        crop = store.update_crop(crop_id, body.rect.model_dump(), body.ratio)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _crop_payload(crop)


@router.post("/{crop_id}/place")
def place_crop(crop_id: int, body: CropPlaceBody) -> dict:
    """Add an existing crop to a dataset (replacing its parent, or beside)."""
    try:
        return store.place_crop(body.dataset_id, crop_id, body.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{crop_id}")
def delete_crop(crop_id: int) -> dict:
    """Delete a crop; its parent takes its place in the datasets it was in."""
    result = store.delete_crop(crop_id)
    if not result["deleted"]:
        raise HTTPException(status_code=404, detail="crop not found")
    return result
