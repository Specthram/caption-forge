"""Reference-free caption-score routes: the catalogue and the scoring job.

Reading a caption's stored scores is not a route of its own — they ride the
media-detail payload (``captions.media_detail``), exactly like grounding.
Only the two GPU-bound and config surfaces live here: the encoder catalogue
the Settings tab renders, and the job that scores one caption.
"""

from fastapi import APIRouter

from server.jobs import manager
from server.runners import caption_score as runner
from server.schemas import (
    CaptionScoreBody,
    CaptionScoreDatasetBody,
    TagScoreBody,
)
from src import caption_score, settings, storage

router = APIRouter(prefix="/api/caption-score", tags=["caption-score"])


def _configured_model_ids() -> dict:
    """Return the checkpoint in effect for each encoder family."""
    return {
        "siglip2": settings.get_grounding_model_id(),
        "clip": settings.get_caption_score_clip_id(),
        "blip": settings.get_caption_score_blip_id(),
    }


def _mean(values) -> float | None:
    """Return the mean of a list, or None when it is empty."""
    values = list(values)
    return round(sum(values) / len(values), 1) if values else None


def _build_report(rows: list[dict], model_ids: dict) -> dict:
    """Aggregate per-media caption scores into the dataset report.

    ``mean`` is a media's average across the encoders that scored it; the
    media list is sorted worst-first so the captions dragging the dataset
    down sit at the top. A per-media score left by a checkpoint no longer
    configured is flagged ``stale`` rather than folded into the average.
    """
    scored, per_kind = [], {kind: [] for kind in caption_score.KINDS}
    for row in rows:
        fresh = {
            kind: value
            for kind, value in row["scores"].items()
            if row["model_ids"].get(kind) == model_ids.get(kind)
        }
        if not fresh:
            continue
        for kind, value in fresh.items():
            per_kind[kind].append(value)
        scored.append(
            {
                "key": row["key"],
                "name": row["name"],
                "scores": row["scores"],
                "stale": {
                    kind: row["model_ids"].get(kind) != model_ids.get(kind)
                    for kind in row["scores"]
                },
                "mean": _mean(fresh.values()),
            }
        )
    scored.sort(key=lambda item: item["mean"])
    return {
        "kinds": [
            {
                "kind": kind,
                "label": caption_score.LABELS[kind],
                "model_id": model_ids[kind],
            }
            for kind in caption_score.KINDS
        ],
        "averages": {kind: _mean(per_kind[kind]) for kind in per_kind},
        "overall": _mean([item["mean"] for item in scored]),
        "scored_media": len(scored),
        "total_media": len(rows),
        "media": scored,
    }


@router.get("/config")
def caption_score_config() -> dict:
    """Return the encoder catalogue and the checkpoints in effect."""
    return {
        "catalogue": caption_score.CATALOGUE,
        "labels": caption_score.LABELS,
        "kinds": list(caption_score.KINDS),
        "model_ids": {
            "siglip2": settings.get_grounding_model_id(),
            "clip": settings.get_caption_score_clip_id(),
            "blip": settings.get_caption_score_blip_id(),
        },
    }


@router.get("/report")
def caption_score_report(dataset_id: int, caption_type: str) -> dict:
    """Return the dataset caption-score report (aggregated stored scores)."""
    rows = storage.caption_score_rows(dataset_id, caption_type)
    return _build_report(rows, _configured_model_ids())


@router.post("/caption")
def score_caption(body: CaptionScoreBody) -> dict:
    """Enqueue the reference-free scoring of one caption; return its job id."""
    job = manager.submit(
        "caption-score",
        f"Score caption {body.key}",
        runner.score_caption_body(
            body.dataset_id, body.key, body.caption_type
        ),
    )
    return {"job_id": job.id}


@router.post("/dataset")
def score_dataset(body: CaptionScoreDatasetBody) -> dict:
    """Enqueue reference-free scoring of a whole dataset; return its job id."""
    job = manager.submit(
        "caption-score",
        f"Score captions · dataset {body.dataset_id}",
        runner.score_dataset_body(body.dataset_id, body.caption_type),
    )
    return {"job_id": job.id}


@router.post("/tags")
def score_tags(body: TagScoreBody) -> dict:
    """Enqueue reference-free scoring of one media's tags; return its job."""
    job = manager.submit(
        "caption-score",
        f"Score tags {body.key}",
        runner.score_tags_body(body.key),
    )
    return {"job_id": job.id}
