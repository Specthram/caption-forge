"""Media library routes: filtered grid, detail panel, favorite toggle."""

from fastapi import APIRouter, HTTPException, Query

from server.routers import crops
from src import caption_score, quality, settings, storage
from src import sqlite_store as store
from src.media import is_video_file

router = APIRouter(prefix="/api/medias", tags=["medias"])

_SORTS = {
    "date_desc": store.SORT_DATE_DESC,
    "date_asc": store.SORT_DATE_ASC,
    "quality_desc": store.SORT_QUALITY_DESC,
    "quality_asc": store.SORT_QUALITY_ASC,
    "dimension_desc": store.SORT_DIMENSION_DESC,
    "dimension_asc": store.SORT_DIMENSION_ASC,
}


@router.get("/grid")
def grid(
    offset: int = 0,
    limit: int = 60,
    tag_ids: list[int] = Query(default=[]),
    exclude_tag_ids: list[int] = Query(default=[]),
    match: str = "any",
    favorites_only: bool = False,
    sort: str = "date_desc",
    quality_metric: str | None = None,
) -> dict:
    """Return one filtered page of the library as media cards (with tags)."""
    total = store.count_library_media(
        tag_ids=tag_ids or None,
        match=match,
        favorites_only=favorites_only,
        exclude_tag_ids=exclude_tag_ids or None,
    )
    page = store.library_media_page(
        tag_ids or None,
        match,
        _SORTS.get(sort, store.SORT_DATE_DESC),
        offset,
        limit,
        quality_metric_selected=quality_metric,
        favorites_only=favorites_only,
        exclude_tag_ids=exclude_tag_ids or None,
    )
    ids = [item["id"] for item in page]
    tags = store.tags_for_media_bulk(ids)
    wm = _watermark_status_bulk(ids)
    return {
        "total": total,
        "items": [
            _card(item, tags.get(item["id"], []), wm.get(item["id"]))
            for item in page
        ],
    }


def _watermark_status_bulk(media_ids: list[int]) -> dict:
    """Return ``{media_id: aggregate watermark status}`` for a grid page.

    One ``watermark_zone`` read for the whole page, so the "visible
    everywhere" badge never costs a query per card.
    """
    zones_by = store.zones_bulk(media_ids)
    status = {}
    for media_id in media_ids:
        value = store.aggregate_status(zones_by.get(media_id, []))
        if value is not None:
            status[media_id] = value
    return status


def _card(item: dict, tag_rows: list, wm_status=None) -> dict:
    """Assemble a media card with its first tags and overflow count."""
    return {
        "key": str(item["id"]),
        "name": item["name"],
        "thumb": f"/api/media/{item['id']}/thumb",
        "quality": quality.normalize_score(
            item["quality_metric"], item["quality_score"]
        ),
        "width": item["width"],
        "height": item["height"],
        "favorite": bool(item.get("favorite")),
        "is_video": is_video_file(f"x.{item['file_extension']}"),
        "wm_status": wm_status,
        "tags": [
            {"name": row["name"], "color": row["color"]}
            for row in tag_rows[:3]
        ],
        "tag_count": len(tag_rows),
    }


@router.post("/{media_id}/favorite")
def toggle_favorite(media_id: int) -> dict:
    """Flip a media's favorite flag; return the new state."""
    return {"favorite": store.toggle_media_favorite(media_id)}


@router.get("/{media_id}")
def detail(media_id: int, quality_metric: str | None = None) -> dict:
    """Return the Media-tab detail payload for one media."""
    media = store.get_media_display(
        media_id, quality_metric_selected=quality_metric
    )
    if media is None:
        raise HTTPException(status_code=404, detail="media not found")
    return _detail_payload(media, media_id)


def _detail_payload(media: dict, media_id: int) -> dict:
    """Build the media detail from a resolved media dict."""
    files = store.media_files(media_id)
    return {
        "key": str(media_id),
        "name": media["name"],
        "is_video": is_video_file(f"x.{media['file_extension']}"),
        "favorite": media["favorite"],
        "crop": crops.crop_card(media),
        "file": f"/api/media/{media_id}/file",
        "thumb": f"/api/media/{media_id}/thumb",
        "meta": {
            "width": media["width"],
            "height": media["height"],
            "sha256": media["sha256"],
            "files": len(files),
            "effective": media["eff_path"],
        },
        "quality_scores": {
            metric: quality.normalize_score(metric, score)
            for metric, score in media["quality_scores"].items()
        },
        "datasets": [
            dataset["name"]
            for dataset in store.list_datasets()
            if store.is_media_in_dataset(dataset["id"], media_id)
        ],
        "tags": _tags_by_category(media_id),
        "tag_score": _tag_score_lines(str(media_id)),
        "captions": _captions(media_id),
    }


def _tag_score_lines(key: str) -> list[dict]:
    """Return one Tags-Score line per encoder for the Media detail panel.

    A line is stale when its stored checkpoint no longer matches the
    configured one, or when the media's tags have changed since it was scored
    (the stored text no longer equals the current comma-joined tags).
    """
    stored = storage.media_tag_scores(key)
    scores, current_text = stored["scores"], stored["text"]
    model_ids = {
        "siglip2": settings.get_grounding_model_id(),
        "clip": settings.get_caption_score_clip_id(),
        "blip": settings.get_caption_score_blip_id(),
    }
    lines = []
    for kind in caption_score.KINDS:
        entry = scores.get(kind)
        stale = bool(entry) and (
            entry["model_id"] != model_ids[kind]
            or entry["scored_text"] != current_text
        )
        lines.append(
            {
                "kind": kind,
                "label": caption_score.LABELS[kind],
                "model_id": model_ids[kind],
                "score": entry["score"] if entry else None,
                "stale": stale,
            }
        )
    return lines


def _tags_by_category(media_id: int) -> list[dict]:
    """Return the media's tags grouped by category (display order)."""
    groups: dict[str, dict] = {}
    for tag in store.tags_for_media(media_id):
        group = groups.setdefault(
            tag["category_name"],
            {
                "category": tag["category_name"],
                "color": tag["color"],
                "tags": [],
            },
        )
        group["tags"].append({"id": tag["id"], "name": tag["name"]})
    return list(groups.values())


def _captions(media_id: int) -> list[dict]:
    """Return a one-line head preview per caption type the media carries."""
    result = []
    for caption_type in storage.caption_types():
        if caption_type == storage.TAGS_TYPE:
            continue
        type_id = store.get_or_create_caption_type(caption_type)
        caption = store.get_caption(media_id, type_id)
        if caption is None or not caption["head_revision_id"]:
            continue
        revision = store.get_revision(caption["head_revision_id"])
        text = (revision["content"] or "").strip() if revision else ""
        if text:
            result.append({"type": caption_type, "preview": text})
    return result
