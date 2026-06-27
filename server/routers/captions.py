"""Caption workspace routes: the dataset grid, media detail and edits.

Wraps the gradio-free :mod:`src.storage` facade (the Caption-tab vocabulary
over :mod:`src.sqlite_store`). Grids page in the database — every list uses
the ``count`` + ``page`` + ``*_bulk`` readers, never a whole-set scan.
"""

import os

from fastapi import APIRouter, HTTPException

from server.jobs import manager
from server.routers import crops
from server.runners import generate as generate_runner
from server.schemas import (
    GenerateBody,
    HiddenBody,
    RepeatsBody,
    SaveCaptionBody,
    SelectRevisionBody,
    TagRefBody,
)
from src import caption_score, deploy, quality, settings, storage
from src import sqlite_store as store
from src.media import is_video_file

router = APIRouter(prefix="/api/captions", tags=["captions"])


@router.get("/types")
def caption_types() -> dict:
    """Return the caption types to choose between (incl. virtual tags)."""
    return {"types": storage.caption_types()}


def _is_video(media: dict) -> bool:
    """Return whether a media dict is a video by its file extension."""
    return is_video_file(f"x.{media['file_extension']}")


def _review_label(review) -> tuple[str, list]:
    """Return the card review badge label and its issue codes.

    ``ok`` for a clean integrity review, ``warn`` for a flagged one,
    ``none`` when the caption was never reviewed.
    """
    if review is None:
        return "none", []
    if review["status"] == "ok":
        return "ok", []
    return "warn", [issue["code"] for issue in review["issues"]]


def _grounding_summary(grounding, threshold: float, model_id: str):
    """Return the card's grounding tiles, or None when never grounded.

    ``coverage`` is the share of the *caption* its validated claims account
    for — not image area. The image-area coverage of the modal needs the
    heat maps, which are recomputed on demand and never stored.
    """
    if grounding is None:
        return None
    claims = grounding["claims"]
    active = [claim for claim in claims if not claim["rejected"]]
    validated = [claim for claim in active if claim["score"] >= threshold]
    total = len(active)
    return {
        "validated": len(validated),
        "flagged": total - len(validated),
        "total": total,
        "coverage": round(100 * len(validated) / total) if total else 0,
        # Scores from another checkpoint are not comparable to this one's;
        # the UI says so rather than quietly presenting them as current.
        "stale": grounding["model_id"] != model_id,
    }


def _caption_score_lines(scores: dict) -> list[dict]:
    """Return one score line per encoder for the detail panel.

    Merges the stored scores with the configured checkpoints: every encoder
    gets a line (``score`` None when never run), and a line whose stored
    checkpoint no longer matches the configured one is flagged ``stale`` so
    the UI can offer a re-run instead of trusting an incomparable number.
    """
    model_ids = {
        "siglip2": settings.get_grounding_model_id(),
        "clip": settings.get_caption_score_clip_id(),
        "blip": settings.get_caption_score_blip_id(),
    }
    lines = []
    for kind in caption_score.KINDS:
        stored = scores.get(kind)
        lines.append(
            {
                "kind": kind,
                "label": caption_score.LABELS[kind],
                "model_id": model_ids[kind],
                "score": stored["score"] if stored else None,
                "stale": bool(stored)
                and stored["model_id"] != model_ids[kind],
            }
        )
    return lines


@router.get("/grid")
def grid(
    dataset_id: int,
    caption_type: str,
    review_filter: str = "all",
    offset: int = 0,
    limit: int = 40,
    quality_metric: str | None = None,
) -> dict:
    """Return one paged screen of a dataset's media as caption cards."""
    media_filter = storage.flagged_media_ids(
        dataset_id, caption_type, review_filter
    )
    total = storage.count_media(dataset_id, media_id_filter=media_filter)
    page = storage.list_media_page(
        dataset_id,
        offset,
        limit,
        quality_metric_selected=quality_metric,
        media_id_filter=media_filter,
    )
    keys = [item["key"] for item in page]
    captions = storage.read_captions_bulk(dataset_id, keys, caption_type)
    reviews = storage.reviews_bulk(dataset_id, keys, caption_type)
    groundings = storage.groundings_bulk(dataset_id, keys, caption_type)
    revisions = storage.revision_options_bulk(dataset_id, keys, caption_type)
    threshold = settings.get_grounding_threshold_caption()
    model_id = settings.get_grounding_model_id()
    items = [
        _card(
            item,
            caption_type,
            captions,
            reviews,
            _grounding_summary(
                groundings.get(item["key"]), threshold, model_id
            ),
            revisions,
        )
        for item in page
    ]
    return {"total": total, "items": items}


def _card(item, caption_type, captions, reviews, grounding, revisions):
    """Assemble one caption-grid card dict from the bulk-read maps."""
    key = item["key"]
    label, codes = _review_label(reviews.get(key))
    choices, value = revisions.get(key, ([], None))
    return {
        "key": key,
        "name": item["name"],
        "is_video": item["is_video"],
        "missing": item["missing"],
        "hidden": item["hidden"],
        "repeats": item["repeats"],
        "thumb": f"/api/media/{key}/thumb",
        "quality": quality.normalize_score(
            item["quality_metric"], item["quality_score"]
        ),
        "caption": captions.get(key, ""),
        "review": label,
        "review_issues": codes,
        "grounding": grounding,
        "ext": caption_type,
        "revisions": max(0, len(choices) - 1),
        "revision_pinned": value not in (None, storage.FOLLOW),
        "crop": crops.crop_card(item),
    }


@router.get("/media/{key}")
def media_detail(
    key: str,
    dataset_id: int,
    caption_type: str,
    quality_metric: str | None = None,
) -> dict:
    """Return the full detail payload for the focused media."""
    media_id = int(key)
    media = store.get_media_display(
        media_id, quality_metric_selected=quality_metric
    )
    if media is None:
        raise HTTPException(status_code=404, detail="media not found")
    return _detail_payload(media, key, dataset_id, caption_type)


def _detail_payload(
    media: dict, key: str, dataset_id: int, caption_type: str
) -> dict:
    """Assemble the media-detail response from a resolved media dict."""
    text = storage.read_caption(dataset_id, key, caption_type)
    choices, value = storage.revision_options(dataset_id, key, caption_type)
    reviews = storage.reviews_bulk(dataset_id, [key], caption_type)
    groundings = storage.groundings_bulk(dataset_id, [key], caption_type)
    label, codes = _review_label(reviews.get(key))
    grounding = _grounding_summary(
        groundings.get(key),
        settings.get_grounding_threshold_caption(),
        settings.get_grounding_model_id(),
    )
    hidden = storage.is_hidden(dataset_id, key)
    repeats = storage.media_repeats(dataset_id, key)
    scores = storage.caption_scores(dataset_id, key, caption_type)
    return {
        "key": key,
        "name": media["name"],
        "is_video": _is_video(media),
        "missing": media["missing"],
        "favorite": media["favorite"],
        "hidden": hidden,
        "repeats": repeats,
        "file": f"/api/media/{key}/file",
        "thumb": f"/api/media/{key}/thumb",
        "caption": text,
        "char_count": len(text),
        "word_count": len(text.split()),
        "revisions": [
            {"label": rev_label, "value": rev_value}
            for rev_label, rev_value in choices
        ],
        "revision_value": value,
        "tags": _detail_tags(int(key)),
        "meta": _detail_meta(media),
        "quality": quality.normalize_score(
            media["quality_metric"], media["quality_score"]
        ),
        "quality_metric": media["quality_metric"],
        "quality_scores": {
            metric: quality.normalize_score(metric, score)
            for metric, score in media["quality_scores"].items()
        },
        "crop": crops.crop_card(media),
        "review": label,
        "review_issues": codes,
        "grounding": grounding,
        "caption_score": _caption_score_lines(scores),
        "deploy": deploy.image_status(
            dataset_id,
            key,
            caption_type,
            hidden,
            media["sha256"],
            deploy.deployed_ext(
                media["file_extension"],
                _is_video(media),
                deploy.dataset_deploy_resolution(dataset_id),
            ),
            repeats,
        ),
    }


def _detail_tags(media_id: int) -> list[dict]:
    """Return a media's tags as category-coloured chips."""
    return [
        {
            "id": tag["id"],
            "name": tag["name"],
            "category": tag["category_name"],
            "color": tag["color"],
        }
        for tag in store.tags_for_media(media_id)
    ]


def _detail_meta(media: dict) -> dict:
    """Return the media's dimensions, size, hash and dataset count."""
    source = media["eff_path"]
    size = os.path.getsize(source) if source and os.path.exists(source) else 0
    datasets = sum(
        1
        for dataset in store.list_datasets()
        if store.is_media_in_dataset(dataset["id"], media["id"])
    )
    return {
        "width": media["width"],
        "height": media["height"],
        "size_bytes": size,
        "sha256": media["sha256"],
        "datasets": datasets,
    }


@router.post("/media/{key}/caption")
def save_caption(
    key: str, dataset_id: int, caption_type: str, body: SaveCaptionBody
) -> dict:
    """Save a caption and return the fresh revision dropdown.

    ``amend`` (autosave) overwrites the current revision in place; otherwise
    a new revision is appended.
    """
    storage.write_caption(
        dataset_id,
        key,
        caption_type,
        body.content,
        scope=body.scope,
        amend=body.amend,
    )
    choices, value = storage.revision_options(dataset_id, key, caption_type)
    return {
        "revisions": [{"label": lbl, "value": val} for lbl, val in choices],
        "revision_value": value,
    }


@router.post("/media/{key}/revision")
def select_revision(
    key: str, dataset_id: int, caption_type: str, body: SelectRevisionBody
) -> dict:
    """Pin (or follow) a caption revision; return the resulting text."""
    selection = (
        storage.FOLLOW if body.revision_id is None else body.revision_id
    )
    text = storage.select_revision(dataset_id, key, caption_type, selection)
    return {"caption": text}


@router.post("/media/{key}/repeats")
def set_repeats(key: str, dataset_id: int, body: RepeatsBody) -> dict:
    """Set a media's deploy repeat count within the dataset."""
    storage.set_media_repeats(dataset_id, key, body.repeats)
    return {"repeats": storage.media_repeats(dataset_id, key)}


@router.post("/media/{key}/hidden")
def set_hidden(key: str, dataset_id: int, body: HiddenBody) -> dict:
    """Hide or unhide a media within the dataset."""
    storage.set_hidden(dataset_id, key, body.hidden)
    return {"hidden": storage.is_hidden(dataset_id, key)}


@router.post("/media/{key}/tags/add")
def add_tag(key: str, body: TagRefBody) -> dict:
    """Attach a tag to a media (by id, or created from name + category)."""
    tag_id = body.tag_id
    if tag_id is None:
        if body.name is None or body.category_id is None:
            raise HTTPException(status_code=400, detail="tag id or name")
        tag_id = store.get_or_create_tag(body.name, body.category_id)
    store.add_tag_to_media(int(key), tag_id)
    return {"tags": _detail_tags(int(key))}


@router.post("/media/{key}/tags/remove")
def remove_tag(key: str, body: TagRefBody) -> dict:
    """Detach a tag from a media."""
    if body.tag_id is None:
        raise HTTPException(status_code=400, detail="tag id required")
    store.remove_tag_from_media(int(key), body.tag_id)
    return {"tags": _detail_tags(int(key))}


@router.delete("/media/{key}")
def remove_from_dataset(key: str, dataset_id: int, caption_type: str) -> dict:
    """Unlink a media from the dataset (file and captions are kept)."""
    storage.delete_media(dataset_id, key, [caption_type])
    return {"ok": True}


@router.post("/generate")
def generate(body: GenerateBody) -> dict:
    """Enqueue the Generate-all captions job; return its job id.

    Requires a VLM already loaded (the single-worker queue serialises this
    with any load/unload job). Each media is captioned, saved as a new
    revision, then optionally reviewed (``review_after`` / ``full_auto``).
    """
    name = f"Generate {body.caption_type} · dataset {body.dataset_id}"
    job = manager.submit("generate", name, generate_runner.generate_body(body))
    return {"job_id": job.id}
