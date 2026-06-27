"""Dataset routes: CRUD, media links, trigger words, composer, report."""

from fastapi import APIRouter, HTTPException, Query

from server.jobs import manager
from server.routers import crops
from server.runners import dataset_report as report_runner
from server.schemas import (
    ComposePreviewBody,
    DatasetCreateBody,
    DatasetReportBody,
    DatasetUpdateBody,
    IssueResolutionBody,
    MediaIdsBody,
    TriggerwordBody,
)
from src import config, dataset_compose, dataset_quality, embeddings
from src import index_steps, quality, settings, siglip_grounding
from src import sqlite_store as store
from src.media import is_video_file

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _media_card(item: dict) -> dict:
    """Turn a repository media dict into a compact grid card."""
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
        "crop": crops.crop_card(item),
    }


@router.get("")
def list_datasets() -> dict:
    """Return every dataset with its media count and deploy name."""
    return {
        "datasets": [
            {
                "id": row["id"],
                "name": row["name"],
                "count": store.count_media_in_dataset(row["id"]),
                "deploy_name": row["deploy_name"] or "",
                "deploy_resolution": row["deploy_resolution"],
            }
            for row in store.list_datasets()
        ]
    }


@router.post("")
def create_dataset(body: DatasetCreateBody) -> dict:
    """Create a dataset; return its id."""
    try:
        dataset_id = store.create_dataset(body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": dataset_id}


@router.post("/{dataset_id}")
def update_dataset(dataset_id: int, body: DatasetUpdateBody) -> dict:
    """Rename a dataset and/or its deploy folder / resize resolution."""
    store.update_dataset(
        dataset_id,
        name=body.name,
        deploy_name=body.deploy_name,
        deploy_resolution=body.deploy_resolution,
    )
    return {"ok": True}


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int) -> dict:
    """Delete a dataset (media and captions are kept)."""
    store.delete_dataset(dataset_id)
    return {"ok": True}


@router.get("/{dataset_id}/media")
def dataset_media(
    dataset_id: int,
    offset: int = 0,
    limit: int = 60,
    quality_metric: str | None = None,
) -> dict:
    """Return one page of a dataset's media as cards."""
    total = store.count_media_in_dataset(dataset_id)
    page = store.media_in_dataset_page(
        dataset_id, offset, limit, quality_metric_selected=quality_metric
    )
    return {"total": total, "items": [_media_card(item) for item in page]}


@router.post("/{dataset_id}/media")
def add_media(dataset_id: int, body: MediaIdsBody) -> dict:
    """Link media to the dataset (never copied)."""
    added = store.add_media_ids_to_dataset(dataset_id, body.media_ids)
    return {"added": added}


@router.delete("/{dataset_id}/media")
def remove_media(dataset_id: int, body: MediaIdsBody) -> dict:
    """Unlink media from the dataset."""
    removed = store.remove_media_ids_from_dataset(dataset_id, body.media_ids)
    return {"removed": removed}


def _vectors_of(model_id: str, media_ids) -> dict:
    """Return ``{media_id: numpy vector}`` for one embedding model."""
    return {
        media_id: embeddings.blob_to_vector(blob)
        for media_id, blob in store.media_embeddings(
            model_id, media_ids
        ).items()
    }


def _corpus_of(dataset_id: int, filters) -> tuple:
    """Return the composer's ``(corpus, pool)`` for one dataset.

    The pool is the *tag-scoped* candidate set: the tag filter runs in
    SQL (it is a membership test the database answers best), everything
    else runs in :mod:`src.dataset_compose` over the whole pool, because a
    diversity gain, a near-duplicate link or an empty zone only exists
    relative to the other candidates — no single page can answer them.
    """
    members = store.media_in_dataset(dataset_id)
    pool = store.library_media_set(
        tag_ids=filters["tag_ids"] or None,
        match=filters["match"],
        exclude_tag_ids=filters["exclude_tag_ids"] or None,
        not_in_dataset_id=dataset_id,
    )
    ids = [item["id"] for item in members + pool]
    corpus = dataset_compose.build_corpus(
        members,
        pool,
        _vectors_of(embeddings.MODEL_ID, ids),
        store.media_hashes(ids),
    )
    return corpus, pool


def _semantic_relevance(query: str, media_ids) -> dict | None:
    """Return ``{media_id: cosine}`` of a typed query, or None.

    None when the query is empty or when no image carries a vector for the
    configured SigLIP checkpoint — the front-end then greys the search box
    and the filter is skipped rather than emptying the grid.
    """
    if not query.strip():
        return None
    model_id = settings.get_grounding_model_id()
    vectors = _vectors_of(model_id, media_ids)
    if not vectors:
        return None
    siglip_grounding.load_model(
        settings.get_grounding_model_size(),
        settings.get_grounding_resolution(),
    )
    return dataset_compose.semantic_relevance(
        siglip_grounding.embed_text(query.strip()), vectors
    )


def _semantic_available() -> bool:
    """Return whether the library carries SigLIP vectors at all."""
    model_id = settings.get_grounding_model_id()
    return store.count_media_with_embedding(model_id) > 0


def _candidate_card(row: dict, metric: str) -> dict:
    """Return one annotated candidate as a grid card."""
    card = _media_card(row["item"])
    card["quality"] = row["score"]
    card["score"] = row["score"]
    card["gain"] = row["gain"]
    card["near_dup"] = row["near_dup"]
    card["in_gap"] = row["in_gap"]
    card["xy"] = row["xy"]
    card["metric"] = metric
    return card


def _metric_choices() -> list:
    """Return the quality metrics the composer's select offers."""
    chips = [
        {"id": metric_id, "label": dataset_quality.scorer_label(metric_id)}
        for metric_id in quality.QUALITY_METRICS
        if not metric_id.endswith(("_8bit", "_4bit"))
    ]
    chips.append({"id": quality.AVERAGE_METRIC_ID, "label": "Average"})
    return chips


@router.get("/{dataset_id}/candidates")
def candidates(  # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    dataset_id: int,
    offset: int = 0,
    limit: int = 60,
    tag_ids: list[int] = Query(default=[]),
    exclude_tag_ids: list[int] = Query(default=[]),
    match: str = "any",
    favorites_only: bool = False,
    metric: str = quality.DEFAULT_METRIC,
    min_score: float = 0.0,
    min_side: int = 0,
    exclude_blur: bool = False,
    exclude_noise: bool = False,
    media_type: str = dataset_compose.MEDIA_TYPE_ANY,
    hide_near_dups: bool = False,
    gaps_only: bool = False,
    similar_to_selection: bool = False,
    sort: str = dataset_compose.SORT_QUALITY,
    semantic_q: str = "",
    selected_ids: list[int] = Query(default=[]),
) -> dict:
    """Return the composer's candidates: library media not in the dataset.

    Beyond the page itself, the payload carries what the composer draws
    around it: how many candidates fall in an empty zone of the corpus,
    the map point of every filtered candidate (not just the page) and the
    static choices of the left rail.
    """
    corpus, pool = _corpus_of(
        dataset_id,
        {
            "tag_ids": tag_ids,
            "exclude_tag_ids": exclude_tag_ids,
            "match": match,
        },
    )
    pool_ids = [item["id"] for item in pool]
    filters = dataset_compose.Filters(
        metric=metric,
        min_score=min_score,
        min_side=min_side,
        exclude_blur=exclude_blur,
        exclude_noise=exclude_noise,
        favorites_only=favorites_only,
        media_type=media_type,
        hide_near_dups=hide_near_dups,
        gaps_only=gaps_only,
        similar_to_selection=similar_to_selection,
        sort=sort,
    )
    rows = dataset_compose.candidates(
        corpus,
        filters,
        set(selected_ids),
        _semantic_relevance(semantic_q, pool_ids),
    )
    page = rows[offset : offset + limit]
    return {
        "total": len(rows),
        "pool": len(pool),
        "items": [_candidate_card(row, metric) for row in page],
        "pool_points": [row["xy"] for row in rows if row["xy"]],
        "gap_count": dataset_compose.gap_count(corpus),
        "semantic_available": _semantic_available(),
        "metrics": _metric_choices(),
        "libraries": [row["name"] for row in store.list_libraries()],
    }


@router.post("/{dataset_id}/compose/preview")
def compose_preview(dataset_id: int, body: ComposePreviewBody) -> dict:
    """Return the live composition panel for a candidate selection."""
    corpus, _pool = _corpus_of(
        dataset_id,
        {"tag_ids": [], "exclude_tag_ids": [], "match": "any"},
    )
    picked = set(body.selected_media_ids)
    ids = [item["id"] for item in corpus.dataset] + sorted(picked)
    autobuild = config.load_autobuild_config()
    tags = store.tags_for_media_bulk(ids)
    return dataset_compose.preview(
        corpus,
        picked,
        body.metric,
        {
            media_id: [row["name"] for row in tags.get(media_id, [])]
            for media_id in ids
        },
        autobuild.get("framing_buckets") or {},
        dataset_quality.recommended_size_range(
            config.load_dataset_quality_config(), body.target_type
        ),
    )


@router.post("/compose/release")
def compose_release() -> dict:
    """Unload the SigLIP checkpoint the semantic search kept resident.

    The search encodes a query per keystroke, so the checkpoint stays
    loaded while the composer is open; closing it frees the VRAM before
    the next model job needs it.
    """
    siglip_grounding.unload_model()
    return {"ok": True}


@router.get("/{dataset_id}/triggerwords")
def list_triggerwords(dataset_id: int) -> dict:
    """Return the dataset's trigger words in attachment order."""
    return {
        "triggerwords": [
            {"id": row["triggerword_id"], "name": row["name"]}
            for row in store.dataset_triggerwords(dataset_id)
        ]
    }


@router.post("/{dataset_id}/triggerwords")
def add_triggerword(dataset_id: int, body: TriggerwordBody) -> dict:
    """Attach a trigger word to the dataset."""
    store.add_triggerword_to_dataset(dataset_id, body.name)
    return {"ok": True}


@router.delete("/{dataset_id}/triggerwords/{triggerword_id}")
def remove_triggerword(dataset_id: int, triggerword_id: int) -> dict:
    """Detach a trigger word from the dataset."""
    store.remove_triggerword_from_dataset(dataset_id, triggerword_id)
    return {"ok": True}


def _scorer_catalogue() -> list:
    """Return the toggle chips of the Quality report toolbar.

    A chip whose index step is off on this machine (see
    :func:`src.index_steps.scorer_step`) is reported unavailable: the
    front-end greys it out and the run drops it, since the scores or the
    vectors it needs are never computed here.
    """
    machine = settings.get_index_steps()
    defaults = set(
        config.load_dataset_quality_config().get("default_scorers")
        or dataset_quality.DEFAULT_SCORERS
    )
    chips = [
        {
            "id": metric_id,
            "label": dataset_quality.scorer_label(metric_id),
            "kind": "iqa",
            "default": metric_id in defaults,
        }
        for metric_id in quality.QUALITY_METRICS
        if not metric_id.endswith(("_8bit", "_4bit"))
    ]
    embedding = dataset_quality.EMBEDDING_SCORER
    chips.append(
        {
            "id": embedding,
            "label": dataset_quality.scorer_label(embedding),
            "kind": "embedding",
            "default": embedding in defaults,
        }
    )
    for chip in chips:
        step = index_steps.scorer_step(chip["id"])
        chip["available"] = bool(machine.get(step, False))
        chip["default"] = chip["default"] and chip["available"]
    return chips


def _available_scorers(requested) -> list:
    """Drop the scorers whose index step is disabled on this machine."""
    machine = settings.get_index_steps()
    return [
        scorer
        for scorer in requested
        if machine.get(index_steps.scorer_step(scorer), False)
    ]


@router.get("/{dataset_id}/report")
def dataset_report(dataset_id: int) -> dict:
    """Return the dataset's last quality report and its resolutions."""
    stored = store.get_dataset_report(dataset_id)
    return {
        "scorer_catalogue": _scorer_catalogue(),
        "resolutions": store.dataset_resolutions(dataset_id),
        "report": stored["report"] if stored else None,
        "created_at": stored["created_at"] if stored else None,
        "duration_s": stored["duration_s"] if stored else 0.0,
        "scorers": stored["scorers"] if stored else [],
        "caption_type": stored["caption_type"] if stored else "",
    }


@router.post("/{dataset_id}/report")
def run_dataset_report(dataset_id: int, body: DatasetReportBody) -> dict:
    """Enqueue an evaluation of the dataset; return its job id."""
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="unknown dataset")
    scorers = body.scorers or list(dataset_quality.DEFAULT_SCORERS) + [
        dataset_quality.EMBEDDING_SCORER
    ]
    scorers = _available_scorers(scorers)
    spec = report_runner.RunSpec(
        dataset_id=dataset_id,
        scorers=tuple(scorers),
        caption_type=body.caption_type,
        target_type=body.target_type,
        force=body.force,
    )
    job = manager.submit(
        "dataset-report",
        f"Quality report · {dataset['name']}",
        report_runner.report_body(spec),
    )
    return {"job_id": job.id}


@router.delete("/{dataset_id}/report")
def clear_dataset_report(dataset_id: int) -> dict:
    """Drop the dataset's stored report (the tab returns to "never run")."""
    store.delete_dataset_report(dataset_id)
    return {"ok": True}


@router.post("/{dataset_id}/report/issues/{issue_key}")
def resolve_issue(
    dataset_id: int, issue_key: str, body: IssueResolutionBody
) -> dict:
    """Record how a flagged-media finding was handled."""
    try:
        store.set_issue_resolution(
            dataset_id,
            issue_key,
            body.resolution,
            fingerprint=body.fingerprint,
            note=body.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@router.delete("/{dataset_id}/report/issues/{issue_key}")
def unresolve_issue(dataset_id: int, issue_key: str) -> dict:
    """Reopen a finding the user had marked as handled."""
    store.clear_issue_resolution(dataset_id, issue_key)
    return {"ok": True}
