"""Auto-build routes: Studio config, live preview, creation, suggestions."""

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from server.runners import autobuild as engine
from server.schemas import (
    AutobuildApplyUpgradesBody,
    AutobuildCreateBody,
    AutobuildNeighborsBody,
    AutobuildPreviewBody,
)
from src import embeddings, quality
from src import sqlite_store as store

router = APIRouter(prefix="/api/autobuild", tags=["autobuild"])


@router.get("/config")
def autobuild_config() -> dict:
    """Return the Studio's framing presets, metrics and index coverage."""
    return {
        "framing_presets": [
            {"key": key, "label": entry["label"]}
            for key, entry in engine.FRAMING_PRESETS.items()
        ],
        "metrics": [{"id": quality.AVERAGE_METRIC_ID, "label": "Average"}]
        + [
            {"id": metric_id, "label": label}
            for label, metric_id in quality.metric_choices()
        ],
        "libraries": [
            {"id": row["id"], "name": row["name"]}
            for row in store.list_libraries()
        ],
        "unhashed": store.count_media_without_hash(),
        "unembedded": store.count_media_without_embedding(embeddings.MODEL_ID),
    }


@router.post("/preview")
def preview(body: AutobuildPreviewBody) -> dict:
    """Run the Studio selection; return the live payload, nothing persisted."""
    return engine.run_preview(body)


@router.post("/preview-stream")
def preview_stream(body: AutobuildPreviewBody) -> StreamingResponse:
    """Stream the Studio selection as NDJSON: stage events, then result.

    Each line is one JSON event — a ``{"stage", "label", "index",
    "total"}`` progress marker for the recompute overlay, or the final
    ``{"result": payload}``. Nothing is persisted.
    """

    def lines():
        for event in engine.run_preview_events(body):
            yield json.dumps(event) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@router.get("/suggest-tags")
def suggest_tags(q: str = "") -> dict:
    """Return the WD14 tags the best semantic matches of a query share."""
    return engine.suggested_tags(q)


@router.post("/neighbors")
def neighbors(body: AutobuildNeighborsBody) -> dict:
    """Return the DINOv2-nearest candidates of a pick (the swap strip)."""
    return engine.neighbors(body)


@router.post("/release")
def release() -> dict:
    """Unload the SigLIP checkpoint the semantic search kept resident."""
    engine.release_model()
    return {"ok": True}


@router.get("/upgrades-summary")
def upgrades_summary() -> dict:
    """Return the upgrade count of each living dataset (the rail banner)."""
    return engine.upgrades_summary()


@router.get("/upgrades/{dataset_id}")
def upgrades(dataset_id: int) -> dict:
    """Replay a dataset's living recipe; return the proposed swaps."""
    return engine.compute_upgrades(dataset_id)


@router.post("/upgrades/{dataset_id}/apply")
def apply_upgrades(dataset_id: int, body: AutobuildApplyUpgradesBody) -> dict:
    """Apply the living-dataset upgrades (out unlinked, in linked)."""
    return engine.apply_upgrades(
        dataset_id, [swap.model_dump() for swap in body.swaps]
    )


@router.post("/create")
def create(body: AutobuildCreateBody) -> dict:
    """Create a dataset from a Studio selection; store its recipe."""
    if not body.selection:
        raise HTTPException(status_code=400, detail="empty selection")
    try:
        dataset_id = engine.create(body.name, body.selection, body.recipe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": dataset_id}
