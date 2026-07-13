"""Library routes: sources CRUD, scan/index/quality/embeddings, near-dupes."""

from fastapi import APIRouter, HTTPException

from server.jobs import manager
from server.runners import library as runner
from server.schemas import (
    BulkTagsBody,
    FolderRulesBody,
    IndexBody,
    LibraryCreateBody,
    LibraryPathBody,
    LookalikeBody,
    MediaIdsBody,
    MergeBody,
    RecursiveBody,
)
from src import embeddings, fs_browse, index_steps, lookalike, quality
from src import settings
from src import sqlite_store as store
from src import tagger, thumbnails
from src.media import is_video_file

router = APIRouter(prefix="/api/libraries", tags=["libraries"])


def _card(item: dict) -> dict:
    """Compact library-media card."""
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
    }


@router.get("")
def list_libraries() -> dict:
    """Return every library source with its media count."""
    internal = store.get_internal_library()
    internal_id = internal["id"] if internal else None
    return {
        "libraries": [
            {
                "id": row["id"],
                "name": row["name"],
                "path": row["path"],
                "recursive": bool(row["recursive"]),
                "internal": row["id"] == internal_id,
                "count": store.count_media_in_library(row["id"]),
                "parent_library_id": row["parent_library_id"],
                "rel_path": row["rel_path"],
                **store.library_mapping_stats(row["id"]),
            }
            for row in store.list_libraries()
        ]
    }


@router.post("")
def create_library(body: LibraryCreateBody) -> dict:
    """Register a folder library."""
    try:
        library_id = store.create_library(
            body.name, body.path, recursive=body.recursive
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": library_id}


@router.delete("/{library_id}")
def delete_library(library_id: int) -> dict:
    """Delete a folder library (orphaned media are archived)."""
    store.delete_library(library_id)
    return {"ok": True}


@router.post("/merge")
def merge_libraries(body: MergeBody) -> dict:
    """Merge folder libraries into a destination."""
    try:
        return store.merge_libraries(body.source_ids, body.dest_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{library_id}/missing")
def library_missing(library_id: int) -> dict:
    """Return the library's media whose source file is gone from disk."""
    return {"media": store.missing_media(library_id)}


@router.post("/purge-media")
def purge_media(body: MediaIdsBody) -> dict:
    """Hard-delete media from the app (for files deleted off disk)."""
    return {"removed": store.purge_media(body.media_ids)}


@router.post("/{library_id}/recursive")
def set_recursive(library_id: int, body: RecursiveBody) -> dict:
    """Toggle a library's recursive-scan flag."""
    store.set_library_recursive(library_id, body.recursive)
    return {"ok": True}


@router.post("/{library_id}/path")
def set_path(library_id: int, body: LibraryPathBody) -> dict:
    """Repoint a folder library at a new source folder."""
    try:
        store.set_library_path(library_id, body.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@router.post("/scan-all")
def scan_all_libraries() -> dict:
    """Enqueue a rescan of every library (one job); return its job id."""
    job = manager.submit(
        "scan-all", "Rescan all libraries", runner.scan_all_body()
    )
    return {"job_id": job.id}


@router.post("/reindex-all")
def reindex_all_libraries() -> dict:
    """Enqueue a full rescan + re-index of every library; return job id.

    One chained job: rescan every folder, then run the whole enabled index
    chain (thumbnails, quality, embeddings, WD14 auto-tags) so newly found
    media are indexed and tagged in the same pass.
    """
    job = manager.submit(
        "reindex-all",
        "Rescan + re-index all",
        runner.full_reindex_body(),
    )
    return {"job_id": job.id}


@router.post("/{library_id}/scan")
def scan_library(library_id: int) -> dict:
    """Enqueue a rescan of one library; return its job id."""
    job = manager.submit(
        "scan", f"Scan library {library_id}", runner.scan_body(library_id)
    )
    return {"job_id": job.id}


@router.get("/folder-tree")
def folder_tree(path: str) -> dict:
    """Return a folder's sub-folder tree with media counts for the wizard."""
    try:
        return fs_browse.folder_tree(path)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{library_id}/folder-rules")
def get_folder_rules(library_id: int) -> dict:
    """Return a library's subfolder mapping (auto-tag level + folder rules)."""
    return store.get_folder_mapping(library_id)


@router.put("/{library_id}/folder-rules")
def put_folder_rules(library_id: int, body: FolderRulesBody) -> dict:
    """Persist a subfolder mapping, then queue a scan that applies it.

    Creates the sub-libraries the ``sublib`` rules promote, stores the whole
    mapping, and enqueues one scan job that re-resolves every file against it
    (routing + tags). Returns the job id and how many libraries/rules resulted.
    """
    library = store.get_library(library_id)
    if library is None:
        raise HTTPException(status_code=404, detail="Library not found.")
    store.set_library_name(library_id, body.name or "")
    name = (body.name or "").strip() or library["name"]
    rules = [rule.model_dump() for rule in body.rules]
    summary = store.apply_folder_mapping(
        library_id, body.auto_tag_level, rules
    )
    job = manager.submit(
        "scan",
        f"Scan & map — {name}",
        runner.scan_body(library_id),
        sub=(
            f"{summary['sub_libraries']} libraries · "
            f"{summary['rules']} tag rules"
        ),
    )
    return {"job_id": job.id, **summary}


@router.get("/coverage")
def coverage() -> dict:
    """Return index/embedding gaps and the metrics already present."""
    return {
        "unhashed": store.count_media_without_hash(),
        "unembedded": store.count_media_without_embedding(embeddings.MODEL_ID),
        "metrics_present": [
            {"id": metric_id, "count": count}
            for metric_id, count in store.available_quality_metrics()
        ],
    }


@router.get("/quality-metrics")
def quality_metrics() -> dict:
    """Return the selectable quality metrics."""
    return {
        "metrics": [
            {"id": metric_id, "label": label}
            for label, metric_id in quality.metric_choices()
        ]
    }


@router.get("/{library_id}/media")
def library_media(
    library_id: int,
    offset: int = 0,
    limit: int = 60,
    quality_metric: str | None = None,
) -> dict:
    """Return one page of a library's media."""
    total = store.count_media_in_library(library_id)
    page = store.media_in_library_page(
        library_id, offset, limit, quality_metric_selected=quality_metric
    )
    return {"total": total, "items": [_card(item) for item in page]}


def _step_counts(library_id, cached: set, metrics) -> dict:
    """Return ``{step_key: {done, total}}`` for one library (None = all)."""
    media = store.count_live_media(library_id)
    images = store.count_live_media(library_id, images_only=True)
    thumbed = sum(
        1 for sha in store.live_media_sha256(library_id) if sha in cached
    )
    return {
        index_steps.THUMBS: {"done": thumbed, "total": media},
        index_steps.QUALITY: {
            "done": store.count_media_scored(metrics, library_id),
            "total": media,
        },
        index_steps.EMBED: {
            "done": store.count_media_with_embedding(
                embeddings.MODEL_ID, library_id
            ),
            "total": images,
        },
        index_steps.SIGLIP: {
            "done": store.count_media_with_embedding(
                settings.get_grounding_model_id(), library_id
            ),
            "total": images,
        },
        index_steps.WD14: {
            "done": store.count_media_tagged(library_id),
            "total": images,
        },
    }


def _step_models(step_key: str, metrics: list) -> str | None:
    """Return the models a step actually runs, or None to keep the default.

    Overrides the static catalogue text so the panel shows the *configured*
    models, not a factory default: the quality step lists the IQA metrics
    selected in Settings (or a "none" note), the semantic-search step the
    SigLIP checkpoint its vectors are keyed by, the auto-tags step the WD14
    checkpoint chosen there.
    """
    if step_key == index_steps.QUALITY:
        labels = [
            quality.QUALITY_METRICS[metric].label.split(" (")[0]
            for metric in metrics
            if metric in quality.QUALITY_METRICS
        ]
        return " · ".join(labels) if labels else "no metric selected"
    if step_key == index_steps.SIGLIP:
        model_id = settings.get_grounding_model_id()
        return model_id.rsplit("/", maxsplit=1)[-1]
    if step_key == index_steps.WD14:
        source = settings.get_autotag_source()
        if source == tagger.LOCAL_SOURCE:
            return "Local tagger"
        return tagger.KNOWN_TAGGERS.get(source, "WD tagger")
    return None


@router.get("/index-status")
def index_status() -> dict:
    """Return the index coverage of every library, step by step.

    Backs the Libraries "Index" panels: the step catalogue with this
    machine's toggles, the per-library ``done / total`` of each step, and
    the same counts across every library (computed once, not summed — a
    media reachable from two libraries must not count twice).
    """
    enabled = settings.get_index_steps()
    metrics = settings.get_index_quality_metrics()
    cached = thumbnails.cached_sha256()
    steps = []
    for step in index_steps.STEPS:
        entry = {**step, "enabled": enabled.get(step["key"], False)}
        models = _step_models(step["key"], metrics)
        if models is not None:
            entry["models"] = models
        steps.append(entry)
    return {
        "steps": steps,
        "totals": _step_counts(None, cached, metrics),
        "libraries": [
            {
                "id": row["id"],
                "name": row["name"],
                "steps": _step_counts(row["id"], cached, metrics),
            }
            for row in store.list_libraries()
        ],
    }


@router.post("/index")
def index(body: IndexBody) -> dict:
    """Enqueue an Index run (one chained job); return its job id."""
    enabled = settings.get_index_steps()
    plan = index_steps.normalize_steps(body.steps, enabled)
    scope = "all libraries"
    if body.library_id is not None:
        library = store.get_library(body.library_id)
        scope = library["name"] if library else str(body.library_id)
    shorts = [
        step["short"] for step in index_steps.STEPS if step["key"] in plan
    ]
    job = manager.submit(
        "index",
        "Index",
        runner.index_body(body.library_id, body.steps, body.force),
        sub=f"{scope} · {' + '.join(shorts) or 'geometry'}",
    )
    return {"job_id": job.id}


@router.post("/bulk-tags")
def bulk_tags(body: BulkTagsBody) -> dict:
    """Add and/or remove tags on every media of a library (or all)."""
    added = store.add_tags_to_library(body.library_id, body.add_tag_ids)
    removed = store.remove_tags_from_library(
        body.library_id, body.remove_tag_ids
    )
    return {"added": added, "removed": removed}


def _active_lookalike_groups(similarity: int):
    """Detect near-duplicate groups, dropping the ones fully dismissed.

    A group whose every member was "hidden indefinitely"
    (``lookalike_reviewed_at`` set, tracked by
    :func:`store.lookalike_reviewed_ids`) is filtered out so it never
    resurfaces; a group still mixing at least one un-dismissed member stays.
    """
    result = lookalike.detect(store.media_with_hashes(), similarity)
    reviewed = store.lookalike_reviewed_ids()
    groups = [
        group
        for group in result.groups
        if not all(member.media_id in reviewed for member in group.media)
    ]
    return result.hashed_count, groups


@router.post("/lookalike/detect")
def lookalike_detect(body: LookalikeBody) -> dict:
    """Detect near-duplicate groups over every hashed media (read-only)."""
    hashed_count, groups = _active_lookalike_groups(body.similarity)
    return {
        "hashed_count": hashed_count,
        "groups": [
            {
                "members": [
                    {
                        "key": str(member.media_id),
                        "name": member.name,
                        "thumb": f"/api/media/{member.media_id}/thumb",
                        "quality": quality.normalize_score(
                            member.quality_metric, member.quality_score
                        ),
                        "is_best": index == 0,
                    }
                    for index, member in enumerate(group.media)
                ]
            }
            for group in groups
        ],
    }


@router.post("/lookalike/keep-best")
def lookalike_keep_best(body: LookalikeBody) -> dict:
    """Set aside every non-best member of each near-duplicate group."""
    _, groups = _active_lookalike_groups(body.similarity)
    discarded = 0
    for group in groups:
        for member in group.media[1:]:
            store.set_media_discarded(member.media_id)
            discarded += 1
    return {"discarded": discarded}


@router.post("/lookalike/discard")
def lookalike_discard(body: MediaIdsBody) -> dict:
    """Discard the chosen media (validate a group: drop the unkept)."""
    for media_id in body.media_ids:
        store.set_media_discarded(media_id)
    return {"discarded": len(body.media_ids)}


@router.post("/lookalike/dismiss")
def lookalike_dismiss(body: MediaIdsBody) -> dict:
    """Hide a near-duplicate group indefinitely (no image discarded)."""
    for media_id in body.media_ids:
        store.set_media_lookalike_reviewed(media_id)
    return {"dismissed": len(body.media_ids)}


@router.post("/lookalike/reset-dismissed")
def lookalike_reset_dismissed() -> dict:
    """Un-hide every dismissed near-duplicate group."""
    store.clear_lookalike_reviewed()
    return {"ok": True}
