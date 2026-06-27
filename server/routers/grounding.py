"""SigLIP grounding routes: scores (jobs), heat maps (jobs), user verdicts.

Anything that touches the GPU is a job returning a ``job_id`` — the
single-worker queue serialises SigLIP against every other model. Reading
back a stored run, marking a claim non-validated or dropping a hallucinated
tag are plain synchronous REST: no model, no queue.
"""

from fastapi import APIRouter

from server.jobs import manager
from server.runners import grounding as runner
from server.schemas import (
    GroundDatasetBody,
    GroundTagsBody,
    GroundTargetBody,
    RejectClaimBody,
    RemoveTagBody,
)
from src import settings, siglip_grounding, storage

router = APIRouter(prefix="/api/grounding", tags=["grounding"])


@router.get("/config")
def grounding_config() -> dict:
    """Return what the modal needs to render before any job has run."""
    return {
        "model_id": settings.get_grounding_model_id(),
        "threshold_caption": settings.get_grounding_threshold_caption(),
        "threshold_tags": settings.get_grounding_threshold_tags(),
        "tag_prompt": siglip_grounding.TAG_PROMPT,
        # The default claim-decomposition model, so the Caption card knows it
        # can ground even with no VLM loaded (the job auto-loads this one).
        "claim_model": settings.get_grounding_claim_model(),
    }


@router.get("/caption")
def read_caption_grounding(
    dataset_id: int, key: str, caption_type: str
) -> dict:
    """Return the stored grounding of a caption (no GPU, no job)."""
    return {
        "grounding": storage.caption_grounding(dataset_id, key, caption_type),
        "model_id": settings.get_grounding_model_id(),
        "threshold": settings.get_grounding_threshold_caption(),
    }


@router.get("/tags")
def read_tag_grounding(key: str) -> dict:
    """Return a media's tags with their stored SigLIP scores."""
    model_id = settings.get_grounding_model_id()
    return {
        "tags": storage.tag_grounding(key, model_id),
        "model_id": model_id,
        "threshold": settings.get_grounding_threshold_tags(),
    }


@router.post("/caption")
def ground_caption(body: GroundTargetBody) -> dict:
    """Enqueue the grounding of one caption; return its job id.

    Requires a VLM already loaded — it decomposes the caption into claims
    before SigLIP scores them.
    """
    job = manager.submit(
        "grounding",
        f"Ground caption {body.key}",
        runner.ground_caption_body(
            body.dataset_id, body.key, body.caption_type
        ),
    )
    return {"job_id": job.id}


@router.post("/dataset")
def ground_dataset(body: GroundDatasetBody) -> dict:
    """Enqueue the grounding of a whole dataset's captions."""
    name = f"Ground {body.caption_type} · dataset {body.dataset_id}"
    job = manager.submit(
        "grounding",
        name,
        runner.ground_dataset_body(
            body.dataset_id, body.caption_type, body.media_ids
        ),
    )
    return {"job_id": job.id}


@router.post("/tags")
def ground_tags(body: GroundTagsBody) -> dict:
    """Enqueue the tag grounding of one or many media (no LLM involved)."""
    count = len(body.media_ids)
    name = f"Ground tags · {count} media" if count > 1 else "Ground tags"
    job = manager.submit(
        "grounding", name, runner.ground_tags_body(body.media_ids)
    )
    return {"job_id": job.id}


@router.post("/caption/heat")
def caption_heat(body: GroundTargetBody) -> dict:
    """Enqueue the heat-map rebuild for a grounded caption."""
    job = manager.submit(
        "grounding",
        f"Heatmap {body.key}",
        runner.caption_heat_body(body.dataset_id, body.key, body.caption_type),
    )
    return {"job_id": job.id}


@router.post("/tags/heat")
def tag_heat(body: GroundTagsBody) -> dict:
    """Enqueue the heat-map rebuild for one media's tags."""
    key = body.media_ids[0]
    job = manager.submit(
        "grounding", f"Heatmap {key}", runner.tag_heat_body(key)
    )
    return {"job_id": job.id}


@router.post("/claim/reject")
def reject_claim(body: RejectClaimBody) -> dict:
    """Mark (or restore) a claim the user judged unsupported by the image."""
    storage.reject_claim(body.claim_id, body.rejected)
    return {"ok": True}


@router.post("/tags/remove")
def remove_tag(body: RemoveTagBody) -> dict:
    """Detach a hallucinated tag from a media and drop its scores."""
    storage.remove_grounded_tag(body.key, body.tag_id)
    model_id = settings.get_grounding_model_id()
    return {"tags": storage.tag_grounding(body.key, model_id)}
