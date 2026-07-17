"""Storage operations backing the Caption gallery.

Datasets, media, captions and their revisions live in the SQLite database
(see :mod:`src.db`); this module wraps :mod:`src.sqlite_store` with the
Caption-tab vocabulary: a *dataset ref* (the tab's current-dataset state,
the dataset id as an int or string) and an opaque per-media *key* (the
media's database id as a string). It also owns the virtual "tags" caption
type, which is threaded through every caption operation below.
"""

from src import caption_claims, caption_judge, caption_review, caption_score
from src import siglip_grounding
from src import sqlite_store as store
from src.media import is_video_file
from src.settings import (
    get_caption_extensions,
    get_caption_score_blip_id,
    get_caption_score_clip_id,
    get_grounding_model_id,
    get_grounding_model_size,
    get_grounding_resolution,
    get_grounding_threshold_caption,
)

# Sentinel dropdown value meaning "track the caption's head revision".
FOLLOW = "follow"
FOLLOW_LABEL = "● Follow (latest)"

# Virtual caption type (Caption tab only): its text is the media's gallery
# tags (``media_tag``) in category order, comma-separated. It is never a real
# ``caption_type`` row and has no revision history — it always reflects the
# live tags, edited through the card's tag multiselect (like Medias).
TAGS_TYPE = "tags"

# Caption-review filter values (the Caption tab's "review" dropdown). ALL
# shows every media; TO_REVIEW keeps only the ones the integrity heuristics
# flagged; UNGROUNDED keeps the ones whose SigLIP grounding left at least one
# claim under the validation threshold. The map below turns a filter into the
# review statuses that count as flagged.
REVIEW_FILTER_ALL = "all"
REVIEW_FILTER_TO_REVIEW = "to_review"
REVIEW_FILTER_UNGROUNDED = "ungrounded"
_REVIEW_FILTER_STATUSES = {
    REVIEW_FILTER_TO_REVIEW: ("integrity",),
}


def _dataset_id(dataset_ref):
    """Return the dataset id from a ref, or None when unset."""
    if dataset_ref in {None, ""}:
        return None
    try:
        return int(dataset_ref)
    except (TypeError, ValueError):
        return None


def has_dataset(dataset_ref) -> bool:
    """Return whether ``dataset_ref`` points to an existing dataset."""
    dataset_id = _dataset_id(dataset_ref)
    return dataset_id is not None and store.get_dataset(dataset_id) is not None


def sqlite_dataset_id(dataset_ref):
    """Return the dataset id for a ref, or None when unset/invalid."""
    return _dataset_id(dataset_ref)


def caption_types() -> list[str]:
    """Return the caption types to choose between.

    The ``caption_type`` names from the database (the caption extensions are
    seeded as types at launch) plus the virtual :data:`TAGS_TYPE`. Falls back
    to the configured extensions when the database has no types yet.
    """
    names = [row["name"] for row in store.list_caption_types()]
    return (names or list(get_caption_extensions())) + [TAGS_TYPE]


def media_tags_text(key: str) -> str:
    """Return a media's gallery tags as a comma-separated string.

    The tags are ordered by category (see
    :func:`src.sqlite_store.media_tag_names`); this is the text the virtual
    :data:`TAGS_TYPE` caption resolves to (and what deploy writes to ``.txt``).
    """
    media_id = _media_id(key)
    if media_id is None:
        return ""
    return ", ".join(store.media_tag_names(media_id))


def _media_id(key):
    """Return the integer media id encoded in an opaque ``key``."""
    try:
        return int(key)
    except (TypeError, ValueError):
        return None


def _media_item(row) -> dict:
    """Return one gallery media dict from a repository media dict."""
    return {
        "key": str(row["id"]),
        "path": row["eff_path"],
        "is_video": is_video_file(f"x.{row['file_extension']}"),
        "hidden": row["hidden"],
        "sha256": row["sha256"],
        "file_extension": row["file_extension"],
        "missing": row["missing"],
        "repeats": row["repeats"],
        "name": row["name"],
        "width": row["width"],
        "height": row["height"],
        # Virtual crop alias (see src.crops); None on an ordinary media.
        "parent_media_id": row["parent_media_id"],
        "crop_rect": row["crop_rect"],
        "crop_ratio": row["crop_ratio"],
        "quality_score": row["quality_score"],
        "quality_metric": row["quality_metric"],
    }


def list_media(dataset_ref, quality_metric_selected=None) -> list[dict]:
    """Return a dataset's media as dicts.

    Each dict has ``"key"``, ``"path"``, ``"is_video"``, ``"hidden"``,
    ``"sha256"``, ``"missing"``, ``"repeats"``, ``"name"``,
    ``"quality_score"`` and ``"quality_metric"`` keys. The list is empty
    when no dataset is selected. ``key`` is the media's database id
    (its stable identity), ``path`` is its effective file on disk (the first
    file that still exists, or None when the media is *missing*), ``sha256``
    the content hash used for deploy naming and ``repeats`` the deploy copy
    count. ``quality_score``/``quality_metric`` reflect the
    ``quality_metric_selected`` display metric (see
    :func:`src.sqlite_store.media._media_dicts`).

    Materializes the whole dataset (one disk stat per media); the Caption
    gallery pages with :func:`count_media` + :func:`list_media_page`.
    """
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return []
    return [
        _media_item(row)
        for row in store.media_in_dataset(
            dataset_id, quality_metric_selected=quality_metric_selected
        )
    ]


def count_media(dataset_ref, media_id_filter=None) -> int:
    """Return how many media the dataset holds (no disk access).

    ``media_id_filter`` optionally restricts the count to a subset of media
    (the review filter); an empty collection means "none".
    """
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return 0
    return store.count_media_in_dataset(
        dataset_id, media_id_filter=media_id_filter
    )


def list_media_page(
    dataset_ref,
    offset: int,
    limit: int,
    quality_metric_selected=None,
    media_id_filter=None,
) -> list[dict]:
    """Return one page of a dataset's media as gallery dicts.

    Same dicts as :func:`list_media`, but only the page's rows are fetched
    and resolved from disk (see
    :func:`src.sqlite_store.media_in_dataset_page`).
    ``quality_metric_selected`` selects the metric each card's badge shows.
    ``media_id_filter`` optionally restricts the page to a subset of media
    (the review filter); an empty collection yields no rows.
    """
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return []
    return [
        _media_item(row)
        for row in store.media_in_dataset_page(
            dataset_id,
            offset,
            limit,
            quality_metric_selected=quality_metric_selected,
            media_id_filter=media_id_filter,
        )
    ]


def flagged_media_ids(dataset_ref, caption_type: str, review_filter: str):
    """Return the media ids matching a review filter, or None for "all".

    ``None`` means "no restriction" (the ALL filter, the virtual tags type,
    or no dataset). Otherwise a set of flagged media ids is returned, which
    the gallery threads into the paged listing as ``media_id_filter``.
    """
    if review_filter in {None, REVIEW_FILTER_ALL} or caption_type == TAGS_TYPE:
        return None
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return None
    type_id = store.get_or_create_caption_type(caption_type)
    if review_filter == REVIEW_FILTER_UNGROUNDED:
        return store.low_grounding_media_ids(
            dataset_id,
            type_id,
            get_grounding_model_id(),
            get_grounding_threshold_caption(),
        )
    statuses = _REVIEW_FILTER_STATUSES.get(review_filter)
    if statuses is None:
        return None
    return store.flagged_media_ids(dataset_id, type_id, statuses)


def effective_revision_id(dataset_ref, key: str, caption_type: str):
    """Return the revision a dataset shows for a media + type, or None.

    ``None`` for the virtual :data:`TAGS_TYPE` (no revisions), when no
    caption exists yet, or when no dataset is selected.
    """
    if caption_type == TAGS_TYPE or not key:
        return None
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return None
    type_id = store.get_or_create_caption_type(caption_type)
    caption = store.get_caption(int(key), type_id)
    if caption is None:
        return None
    return store.effective_revision_id(dataset_id, caption["id"])


def _effective_rev_by_media(dataset_id, media_ids, type_id) -> dict:
    """Return ``{media_id: effective revision id}`` for a page of media.

    The dataset's pinned revision when one is set, else the caption head —
    the same resolution as :func:`read_caption`, in three queries for the
    whole page. Media with no caption are simply absent from the result.
    """
    captions = store.captions_for_type_bulk(media_ids, type_id)
    caption_by_media = {row["media_id"]: row for row in captions}
    pins = store.dataset_caption_bulk(
        dataset_id, [row["id"] for row in captions]
    )
    rev_by_media = {}
    for media_id, caption in caption_by_media.items():
        assignment = pins.get(caption["id"])
        if (
            assignment is not None
            and assignment["mode"] == "pinned"
            and assignment["revision_id"]
        ):
            rev_by_media[media_id] = assignment["revision_id"]
        else:
            rev_by_media[media_id] = caption["head_revision_id"]
    return rev_by_media


def reviews_bulk(dataset_ref, keys, caption_type: str) -> dict:
    """Return ``{key: review dict or None}`` for many media at once.

    Resolves each media's effective revision for the type (dataset pin else
    head) and reads their reviews in one query. Every given key gets an
    entry; a media with no caption or no review maps to ``None``. Always
    ``None`` for the virtual :data:`TAGS_TYPE` (excluded from review).
    """
    keys = [str(key) for key in keys]
    if caption_type == TAGS_TYPE:
        return {key: None for key in keys}
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return {key: None for key in keys}
    type_id = store.get_or_create_caption_type(caption_type)
    rev_by_media = _effective_rev_by_media(
        dataset_id, [int(key) for key in keys if key], type_id
    )
    reviews = store.reviews_bulk([rid for rid in rev_by_media.values() if rid])
    result = {}
    for key in keys:
        rev_id = rev_by_media.get(int(key)) if key else None
        result[key] = reviews.get(rev_id) if rev_id else None
    return result


def groundings_bulk(dataset_ref, keys, caption_type: str) -> dict:
    """Return ``{key: grounding dict or None}`` for many media at once.

    Same effective-revision resolution as :func:`reviews_bulk`; the grounding
    carries its ``claims`` list and the ``model_id`` that scored them.
    ``None`` for the tags type or a media with no caption / no grounding run.
    """
    keys = [str(key) for key in keys]
    if caption_type == TAGS_TYPE:
        return {key: None for key in keys}
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return {key: None for key in keys}
    type_id = store.get_or_create_caption_type(caption_type)
    rev_by_media = _effective_rev_by_media(
        dataset_id, [int(key) for key in keys if key], type_id
    )
    groundings = store.caption_groundings_bulk(
        [rid for rid in rev_by_media.values() if rid]
    )
    result = {}
    for key in keys:
        rev_id = rev_by_media.get(int(key)) if key else None
        result[key] = groundings.get(rev_id) if rev_id else None
    return result


# --- SigLIP grounding ---
# Two paths, one engine. A caption is first decomposed by the loaded VLM
# (:mod:`src.caption_claims`) and each claim scored; a media's tags skip the
# LLM entirely and go straight through the fixed pre-prompt. Both assume the
# SigLIP checkpoint is already loaded — the caller owns the load/unload pair
# so a batch run pays for the weights once (see the runners).


def _groundable_image(dataset_ref, key: str) -> str | None:
    """Return the image path to ground, or None when there is nothing to.

    Grounding is images-only: SigLIP scores a picture, and a video's first
    frame would score its captions against a moment, not the clip.
    """
    path = media_path(dataset_ref, key)
    if not path or is_video_file(path):
        return None
    return path


def caption_grounding(dataset_ref, key: str, caption_type: str):
    """Return the stored grounding of a media's caption, or None."""
    revision_id = effective_revision_id(dataset_ref, key, caption_type)
    if revision_id is None:
        return None
    return store.get_caption_grounding(revision_id)


def extract_caption_claims(dataset_ref, key: str, caption_type: str) -> dict:
    """Decompose a media's caption into claims — the LLM half, alone.

    Nothing is scored and nothing is stored. It is split from
    :func:`score_caption_claims` because the two halves need *different*
    models: a batch run drains every caption through the loaded VLM first,
    then loads SigLIP once for the whole set instead of per media.

    Returns
    -------
    dict
        ``{"status": "ok", "path": ..., "revision_id": ..., "claims": [...]}``
        or ``{"status": "skipped", "reason": ...}`` for a video, a missing
        file, or a caption with no revision / no text / no claim.
    """
    revision_id = effective_revision_id(dataset_ref, key, caption_type)
    if revision_id is None:
        return {"status": "skipped", "reason": "no caption"}
    path = _groundable_image(dataset_ref, key)
    if path is None:
        return {"status": "skipped", "reason": "video or missing file"}
    text = read_caption(dataset_ref, key, caption_type)
    if not text.strip():
        return {"status": "skipped", "reason": "empty caption"}
    claims = caption_claims.extract_claims(path, text)
    if not claims:
        store.delete_caption_grounding(revision_id)
        return {"status": "skipped", "reason": "no claim extracted"}
    return {
        "status": "ok",
        "path": path,
        "revision_id": revision_id,
        "claims": claims,
    }


def score_caption_claims(
    path: str, revision_id: int, claims, model_id: str
) -> dict:
    """Score an :func:`extract_caption_claims` batch and persist the run.

    The SigLIP half. Takes the image path and revision the extraction
    already resolved, so a batch never re-walks the database.
    """
    scored = siglip_grounding.ground_image(
        path, [claim["text"] for claim in claims], with_heat=False
    )
    for claim, result in zip(claims, scored):
        claim["score"] = result["score"]
    store.upsert_caption_grounding(revision_id, model_id, claims)
    return {"status": "ok", "claims": claims, "model_id": model_id}


def ground_caption(
    dataset_ref, key: str, caption_type: str, model_id: str
) -> dict:
    """Decompose a caption into claims, score them, and persist the run.

    The single-media path: both models are already loaded (the VLM in its
    slot, SigLIP by the caller). Returns the ``score_caption_claims``
    verdict, or the extraction's ``"skipped"`` one.
    """
    extraction = extract_caption_claims(dataset_ref, key, caption_type)
    if extraction["status"] != "ok":
        return extraction
    return score_caption_claims(
        extraction["path"],
        extraction["revision_id"],
        extraction["claims"],
        model_id,
    )


# --- Reference-free caption score ---
# The zero-reference companion to grounding (:mod:`src.caption_score`): no
# LLM, no claims — every configured encoder scores the whole caption against
# the image. Persisted per revision, so an edited caption drops its scores.


def caption_score_specs() -> list[dict]:
    """Return the encoder specs to score a caption with, read from Settings.

    SigLIP2 reuses the grounding checkpoint (one SigLIP for both features);
    CLIP and BLIP carry their own configured size. See
    :func:`src.caption_score.score_caption`.
    """
    size = get_grounding_model_size()
    resolution = get_grounding_resolution()
    return [
        {
            "kind": "siglip2",
            "label": caption_score.LABELS["siglip2"],
            "model_id": siglip_grounding.repo_id(size, resolution),
            "size": size,
            "resolution": resolution,
        },
        {
            "kind": "clip",
            "label": caption_score.LABELS["clip"],
            "model_id": get_caption_score_clip_id(),
        },
        {
            "kind": "blip",
            "label": caption_score.LABELS["blip"],
            "model_id": get_caption_score_blip_id(),
        },
    ]


def caption_score_target(dataset_ref, key: str, caption_type: str) -> dict:
    """Resolve the image, revision and text to score, or a skip reason.

    Returns
    -------
    dict
        ``{"status": "ok", "path", "revision_id", "caption"}`` or
        ``{"status": "skipped", "reason": ...}`` for a video, a missing file
        or a caption with no revision / no text. An empty caption also drops
        any stale stored scores.
    """
    revision_id = effective_revision_id(dataset_ref, key, caption_type)
    if revision_id is None:
        return {"status": "skipped", "reason": "no caption"}
    path = _groundable_image(dataset_ref, key)
    if path is None:
        return {"status": "skipped", "reason": "video or missing file"}
    text = read_caption(dataset_ref, key, caption_type)
    if not text.strip():
        store.delete_caption_scores(revision_id)
        return {"status": "skipped", "reason": "empty caption"}
    return {
        "status": "ok",
        "path": path,
        "revision_id": revision_id,
        "caption": text,
    }


def score_caption(path: str, revision_id: int, caption: str, progress=None):
    """Score a caption with every configured encoder and persist each line.

    Each encoder loads and frees its own weights in turn (see the engine); a
    family that fails to load is skipped, not fatal. Only successful lines are
    persisted, so a re-run refreshes what it can and leaves the rest.
    """
    specs = caption_score_specs()
    results = caption_score.score_caption(path, caption, specs, progress)
    scored = 0
    for result in results:
        if result["score"] is None:
            continue
        store.upsert_caption_score(
            revision_id, result["kind"], result["model_id"], result["score"]
        )
        scored += 1
    return {"status": "ok", "results": results, "scored": scored}


def caption_scores(dataset_ref, key: str, caption_type: str) -> dict:
    """Return ``{model_kind: {model_id, score}}`` stored for a caption."""
    revision_id = effective_revision_id(dataset_ref, key, caption_type)
    if revision_id is None:
        return {}
    return store.get_caption_scores(revision_id)


def score_dataset_captions(dataset_ref, caption_type: str, progress=None):
    """Score every scorable caption of a dataset, loading each model once.

    The batch path behind the Datasets → Caption score tab. Resolves the
    scorable media (skipping videos, missing files and empty captions), then
    hands them to :func:`src.caption_score.score_dataset`, which runs each
    encoder once over the whole set. Persists every score and returns a count.
    """
    items = []
    for media in list_media(dataset_ref):
        target = caption_score_target(dataset_ref, media["key"], caption_type)
        if target["status"] == "ok":
            items.append(
                {
                    "revision_id": target["revision_id"],
                    "path": target["path"],
                    "caption": target["caption"],
                }
            )
    specs = caption_score_specs()
    scores = caption_score.score_dataset(items, specs, progress)
    model_id_of = {spec["kind"]: spec["model_id"] for spec in specs}
    scored = 0
    for revision_id, by_kind in scores.items():
        for kind, score in by_kind.items():
            store.upsert_caption_score(
                revision_id, kind, model_id_of[kind], score
            )
            scored += 1
    return {"status": "ok", "media": len(items), "scored": scored}


def caption_score_rows(dataset_ref, caption_type: str) -> list[dict]:
    """Return per-media caption scores for a dataset (report input).

    One row per non-video media: its key, name and the stored score of each
    encoder (absent kinds simply missing from ``scores``). Aggregation —
    averages, ranking, stale flags — is the router's job, on top of this.
    """
    media = [item for item in list_media(dataset_ref) if not item["is_video"]]
    rev_of = {
        item["key"]: effective_revision_id(
            dataset_ref, item["key"], caption_type
        )
        for item in media
    }
    bulk = store.caption_scores_bulk([rev for rev in rev_of.values() if rev])
    rows = []
    for item in media:
        stored = bulk.get(rev_of[item["key"]], {})
        rows.append(
            {
                "key": item["key"],
                "name": item["name"],
                "scores": {
                    kind: value["score"] for kind, value in stored.items()
                },
                "model_ids": {
                    kind: value["model_id"] for kind, value in stored.items()
                },
            }
        )
    return rows


# --- Reference-free tag score (Media tab "Tags Score") ---
# The same three encoders as the caption score, but pointed at a media's
# *tags* joined into one comma-separated text. Keyed on the media (tags have
# no revision); the scored text is stored so a later tag edit marks the line
# stale instead of showing a number for tags that no longer exist.


def media_tag_score_target(key: str) -> dict:
    """Resolve the image and comma-joined tag text to score, or a skip.

    Returns
    -------
    dict
        ``{"status": "ok", "media_id", "path", "text"}`` or
        ``{"status": "skipped", "reason": ...}`` for a video, a missing file
        or a media with no tags (whose stale stored scores are dropped).
    """
    media_id = _media_id(key)
    if media_id is None:
        return {"status": "skipped", "reason": "no media"}
    path = store.effective_file(media_id)
    if not path or is_video_file(path):
        return {"status": "skipped", "reason": "video or missing file"}
    text = media_tags_text(key)
    if not text.strip():
        store.delete_media_tag_scores(media_id)
        return {"status": "skipped", "reason": "no tags"}
    return {"status": "ok", "media_id": media_id, "path": path, "text": text}


def score_media_tags(key: str, progress=None):
    """Score a media's tags (as one comma-joined text) and persist each line.

    Reuses the caption-score engine on the tag string. Each encoder's score
    is stored alongside the exact text it scored, so the reader can flag a
    line stale once the tags change.
    """
    target = media_tag_score_target(key)
    if target["status"] != "ok":
        return target
    specs = caption_score_specs()
    results = caption_score.score_caption(
        target["path"], target["text"], specs, progress
    )
    scored = 0
    for result in results:
        if result["score"] is None:
            continue
        store.upsert_media_tag_score(
            target["media_id"],
            result["kind"],
            result["model_id"],
            result["score"],
            target["text"],
        )
        scored += 1
    return {"status": "ok", "results": results, "scored": scored}


def media_tag_scores(key: str) -> dict:
    """Return the stored tag scores plus the media's current tag text.

    ``{"scores": {model_kind: {model_id, score, scored_text}}, "text": ...}``
    — the router compares ``scored_text`` to ``text`` to flag stale lines.
    """
    media_id = _media_id(key)
    if media_id is None:
        return {"scores": {}, "text": ""}
    return {
        "scores": store.get_media_tag_scores(media_id),
        "text": media_tags_text(key),
    }


def caption_heatmaps(dataset_ref, key: str, caption_type: str) -> list[dict]:
    """Return a caption's stored claims, each with a fresh heat grid.

    Heat maps are never persisted — one image forward pass rebuilds every
    claim's map at once, which is cheaper than invalidating stored grids
    whenever the checkpoint or the resolution changes.
    """
    grounding = caption_grounding(dataset_ref, key, caption_type)
    path = _groundable_image(dataset_ref, key)
    if grounding is None or path is None:
        return []
    claims = grounding["claims"]
    scored = siglip_grounding.ground_image(
        path, [claim["text"] for claim in claims], with_heat=True
    )
    for claim, result in zip(claims, scored):
        claim["heat"] = result["heat"]
        claim["side"] = result["side"]
    return claims


def reject_claim(claim_id: int, rejected: bool) -> None:
    """Mark (or restore) a grounded claim the user judged unsupported."""
    store.set_claim_rejected(int(claim_id), bool(rejected))


def tag_grounding(key: str, model_id: str) -> list[dict]:
    """Return a media's tags with their stored SigLIP score, or None each.

    Every attached tag is listed, in gallery order, so an ungrounded tag is
    visible as such instead of vanishing from the panel.
    """
    media_id = _media_id(key)
    if media_id is None:
        return []
    scores = store.tag_grounding_for_media(media_id, model_id)
    return [
        {
            "id": tag["id"],
            "name": tag["name"],
            "category": tag["category_name"],
            "color": tag["color"],
            "score": scores.get(tag["id"]),
        }
        for tag in store.tags_for_media(media_id)
    ]


def ground_tags(key: str, model_id: str) -> dict:
    """Score every tag of a media against its image, and persist the run.

    No LLM: each tag is injected into :data:`src.siglip_grounding.TAG_PROMPT`
    and scored on its own. Returns the same shape as :func:`tag_grounding`,
    or a ``"skipped"`` verdict for a video / missing file / untagged media.
    """
    media_id = _media_id(key)
    if media_id is None:
        return {"status": "skipped", "reason": "no media"}
    path = _groundable_image(None, key)
    if path is None:
        return {"status": "skipped", "reason": "video or missing file"}
    tags = store.tags_for_media(media_id)
    if not tags:
        return {"status": "skipped", "reason": "no tag"}
    scored = siglip_grounding.ground_image(
        path,
        [siglip_grounding.tag_prompt(tag["name"]) for tag in tags],
        with_heat=False,
    )
    for tag, result in zip(tags, scored):
        store.upsert_tag_grounding(
            media_id, tag["id"], model_id, result["score"]
        )
    return {"status": "ok", "tags": tag_grounding(key, model_id)}


def tag_heatmaps(key: str, model_id: str) -> list[dict]:
    """Return a media's grounded tags, each with a fresh heat grid."""
    media_id = _media_id(key)
    path = _groundable_image(None, key)
    if media_id is None or path is None:
        return []
    tags = tag_grounding(key, model_id)
    scored = siglip_grounding.ground_image(
        path,
        [siglip_grounding.tag_prompt(tag["name"]) for tag in tags],
        with_heat=True,
    )
    for tag, result in zip(tags, scored):
        tag["heat"] = result["heat"]
        tag["side"] = result["side"]
        # A tag the batch never scored still earns a live score here — the
        # forward pass produced it anyway.
        if tag["score"] is None:
            tag["score"] = result["score"]
    return tags


def remove_grounded_tag(key: str, tag_id: int) -> None:
    """Detach a hallucinated tag from a media and drop its scores."""
    media_id = _media_id(key)
    if media_id is None:
        return
    store.remove_tag_from_media(media_id, int(tag_id))
    store.delete_tag_grounding(media_id, int(tag_id))


def run_integrity_review(dataset_ref, key: str, caption_type: str):
    """Run the integrity heuristics on a media's caption and store the verdict.

    Resolves the caption's effective revision, checks the text with
    :mod:`src.caption_review`, upserts the review row and returns
    ``(status, issues)``. Returns ``(None, [])`` when there is nothing to
    review (the tags type, or no caption revision yet).
    """
    revision_id = effective_revision_id(dataset_ref, key, caption_type)
    if revision_id is None:
        return None, []
    text = read_caption(dataset_ref, key, caption_type)
    verdict = caption_review.review_integrity(text)
    store.upsert_review(revision_id, verdict["status"], verdict["issues"])
    return verdict["status"], verdict["issues"]


# --- Rule-based caption review (the Caption tab's Review sub-tab) ---

# Presets seeded the first time a dataset's rules are read. The trigger-word
# check is deterministic (safe); the two vision rules need the judge to see
# the image, and "nothing omitted" ships off by default (noisy on terse
# captions). ``{word}`` is filled from the dataset's first trigger word.
_BUILTIN_RULES = (
    (
        'The caption must contain the trigger word "{word}".',
        "det",
        False,
        True,
    ),
    (
        "The clothing and colors described in the caption must match what "
        "is visible in the image.",
        "vlm",
        True,
        True,
    ),
    (
        "Nothing clearly visible and important in the image is omitted "
        "from the caption.",
        "vlm",
        True,
        False,
    ),
)


def ensure_review_rules(dataset_ref) -> None:
    """Seed the builtin review presets for a dataset that has no rules yet."""
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None or store.has_rules(dataset_id):
        return
    words = [row["name"] for row in store.dataset_triggerwords(dataset_id)]
    word = words[0] if words else "trigger"
    for text, kind, needs_image, enabled in _BUILTIN_RULES:
        store.create_rule(
            dataset_id,
            text.format(word=word),
            kind,
            needs_image=needs_image,
            enabled=enabled,
            builtin=True,
        )


def review_rules(dataset_ref) -> list:
    """Return a dataset's review rules, seeding the presets on first read."""
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return []
    ensure_review_rules(dataset_ref)
    return store.list_rules(dataset_id)


def create_review_rule(dataset_ref, text: str, needs_image: bool) -> dict:
    """Add a custom rule; its kind is vision when it needs the image."""
    dataset_id = _dataset_id(dataset_ref)
    kind = store.KIND_VLM if needs_image else store.KIND_TEXT
    rule_id = store.create_rule(
        dataset_id, text, kind, needs_image=needs_image, builtin=False
    )
    return store.get_rule(rule_id)


def review_counts(dataset_ref) -> dict:
    """Return the ``{pending, accepted, rejected}`` finding counts."""
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return {"pending": 0, "accepted": 0, "rejected": 0}
    return store.findings_counts(dataset_id)


def _enrich_finding(finding: dict) -> dict:
    """Return a finding with its caption type, media key and stale flag.

    ``stale`` is true when the media's *current* caption no longer matches
    the ``caption_before`` the diff was computed against (a manual edit landed
    meanwhile), so the front can skip a proposal that no longer applies.
    """
    type_row = store.get_caption_type(finding["caption_type_id"])
    caption_type = type_row["name"] if type_row else ""
    key = str(finding["media_id"])
    current = read_caption(finding["dataset_id"], key, caption_type)
    data = dict(finding)
    data["caption_type"] = caption_type
    data["key"] = key
    data["stale"] = (
        current.strip() != (finding["caption_before"] or "").strip()
    )
    return data


def review_findings(dataset_ref, status: str = None) -> list:
    """Return the enriched review queue for a dataset (filtered by status)."""
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return []
    return [
        _enrich_finding(finding)
        for finding in store.list_findings(dataset_id, status)
    ]


def open_review_run(
    dataset_ref, judge_model: str, scope: str, total: int
) -> int:
    """Open a review run for a dataset and return its id."""
    return store.create_run(
        _dataset_id(dataset_ref), judge_model, scope, total
    )


def close_review_run(run_id: int, findings_count: int) -> None:
    """Mark a review run finished with its final finding count."""
    store.finish_run(run_id, findings_count)


def reset_review_queue(dataset_ref, media_id: int = None) -> None:
    """Clear a dataset's findings (all, or just one media's for a re-run)."""
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return
    if media_id is None:
        store.clear_dataset_findings(dataset_id)
    else:
        store.clear_media_findings(dataset_id, media_id)


def review_targets(dataset_ref, caption_type: str, media_ids) -> list:
    """Return per-media review inputs (path + current caption) for a run.

    Skips media with no caption revision yet (nothing to review) and, for a
    vision pass, media with no file on disk.
    """
    targets = []
    for key in media_ids:
        key = str(key)
        text = read_caption(dataset_ref, key, caption_type)
        if not text.strip():
            continue
        targets.append(
            {
                "key": key,
                "media_id": int(key),
                "path": media_path(dataset_ref, key),
                "caption": text,
                "is_video": is_video_file(media_path(dataset_ref, key) or ""),
            }
        )
    return targets


def record_review_finding(
    run_id: int,
    media_id: int,
    caption_type: str,
    note: str,
    caption_before: str,
    caption_after: str,
    rule_id: int = None,
    rule_kind: str = "",
) -> int:
    """Persist one pending finding produced by a run; return its id."""
    type_id = store.get_or_create_caption_type(caption_type)
    return store.add_finding(
        run_id,
        media_id,
        type_id,
        note,
        caption_before,
        caption_after,
        rule_id=rule_id,
        rule_kind=rule_kind,
    )


def _apply_accept(finding: dict, caption: str = None) -> None:
    """Write the accepted caption as a new revision for a finding's media.

    The fix is merged into the *live* caption (never written verbatim) so a
    second accept keeps the first; an explicit ``caption`` wins as-is.
    """
    type_row = store.get_caption_type(finding["caption_type_id"])
    caption_type = type_row["name"] if type_row else ""
    if caption is not None:
        final = caption
    else:
        current = read_caption(
            finding["dataset_id"], str(finding["media_id"]), caption_type
        )
        final = caption_judge.apply_fix(
            finding["caption_before"], current, finding["caption_after"]
        )
    write_caption(
        finding["dataset_id"], str(finding["media_id"]), caption_type, final
    )
    store.decide_finding(
        finding["id"], store.STATUS_ACCEPTED, applied_caption=final
    )
    # Rebase the media's other pending findings onto the new caption: their
    # "original" shows this accept applied, their proposal the fix on top.
    for sibling in store.pending_for_media(
        finding["dataset_id"],
        finding["media_id"],
        finding["caption_type_id"],
    ):
        rebased = caption_judge.apply_fix(
            sibling["caption_before"], final, sibling["caption_after"]
        )
        store.rebase_finding(sibling["id"], final, rebased)


def decide_review_finding(
    finding_id: int, action: str, caption: str = None
) -> dict:
    """Accept (``caption`` = inline-edit override) or reject one finding."""
    finding = store.get_finding(finding_id)
    if finding is None:
        return {}
    if action == "accept":
        _apply_accept(finding, caption)
    else:
        store.decide_finding(finding_id, store.STATUS_REJECTED)
    return store.get_finding(finding_id)


def undo_review_finding(finding_id: int) -> dict:
    """Undo a decision: restore the original caption and reopen the finding."""
    finding = store.get_finding(finding_id)
    if finding is None:
        return {}
    if finding["status"] == store.STATUS_ACCEPTED:
        type_row = store.get_caption_type(finding["caption_type_id"])
        caption_type = type_row["name"] if type_row else ""
        write_caption(
            finding["dataset_id"],
            str(finding["media_id"]),
            caption_type,
            finding["caption_before"],
        )
    store.reopen_finding(finding_id)
    return store.get_finding(finding_id)


def accept_safe_fixes(dataset_ref) -> int:
    """Accept all pending safe findings (det + integrity); return count."""
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return 0
    findings = store.safe_pending(dataset_id)
    for finding in findings:
        _apply_accept(finding)
    return len(findings)


def accept_rule_fixes(dataset_ref, rule_id: int) -> int:
    """Accept every pending finding of one rule; return the count."""
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return 0
    findings = store.pending_for_rule(dataset_id, rule_id)
    for finding in findings:
        _apply_accept(finding)
    return len(findings)


def reject_all_findings(dataset_ref) -> int:
    """Reject every pending finding of a dataset; return the count."""
    dataset_id = _dataset_id(dataset_ref)
    return 0 if dataset_id is None else store.reject_all_pending(dataset_id)


def clear_review_history(dataset_ref) -> int:
    """Delete the dataset's decided findings (history); return the count."""
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return 0
    return store.clear_decided_findings(dataset_id)


def media_path(dataset_ref, key: str) -> str | None:
    """Return the effective media file path for a key, or None when unset.

    This is the media's first file that still exists on disk (None when
    every file is gone). ``dataset_ref`` is unused (the key alone identifies
    the media) but kept so every storage operation shares one signature.
    """
    # pylint: disable=unused-argument
    media_id = _media_id(key)
    return store.effective_file(media_id) if media_id else None


def read_caption(dataset_ref, key: str, caption_type: str) -> str:
    """Return the caption text a dataset shows for a media + type, or ""."""
    if caption_type == TAGS_TYPE:
        return media_tags_text(key)
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None or not key:
        return ""
    media_id = int(key)
    type_id = store.get_or_create_caption_type(caption_type)
    return store.read_caption(dataset_id, media_id, type_id)


def write_caption(
    dataset_ref,
    key: str,
    caption_type: str,
    text: str,
    scope: str = "type",
    amend: bool = False,
) -> None:
    """Persist caption text for a media + type.

    ``scope`` chooses type-wide vs dataset-only (see
    :func:`src.sqlite_store.save_caption`). ``amend`` (autosave) overwrites
    the dataset's current revision in place instead of appending a new one,
    so a pause in typing never floods the history; its stale review,
    grounding and score children (all derived from the old text) are dropped.
    An explicit Save (``amend`` false) snapshots a fresh revision as before.
    The virtual :data:`TAGS_TYPE` is edited through the card's tag
    multiselect (which writes ``media_tag`` directly), not the caption box,
    so a write for it is a no-op here.
    """
    if caption_type == TAGS_TYPE:
        return
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None or not key:
        return
    media_id = int(key)
    type_id = store.get_or_create_caption_type(caption_type)
    if amend:
        revision_id = store.amend_caption(dataset_id, media_id, type_id, text)
        store.delete_caption_scores(revision_id)
        store.delete_caption_grounding(revision_id)
        store.delete_review(revision_id)
        return
    store.save_caption(dataset_id, media_id, type_id, text, scope=scope)


def present_types(dataset_ref, key: str, types: list[str]) -> list[str]:
    """Return the ``types`` that already have a caption for a media.

    The virtual :data:`TAGS_TYPE` counts as present when the media carries any
    gallery tag.
    """
    return [
        caption_type
        for caption_type in types
        if read_caption(dataset_ref, key, caption_type).strip()
    ]


def _revision_label(revision) -> str:
    """Return a short dropdown label for a revision row."""
    snippet = (revision["content"] or "").strip().replace("\n", " ")
    if len(snippet) > 40:
        snippet = snippet[:40] + "…"
    return (
        f"#{revision['id']}: {snippet}"
        if snippet
        else (f"#{revision['id']}: (empty)")
    )


def revision_options(dataset_ref, key: str, caption_type: str):
    """Return ``(choices, value)`` for a media's revision dropdown.

    ``choices`` is a leading "Follow (latest)" option then one entry per
    revision (newest first); ``value`` is the dataset's current selection
    (:data:`FOLLOW` or a revision id). Both are empty for the virtual
    :data:`TAGS_TYPE` (which has no revision history) or when no media/dataset
    is selected.
    """
    if caption_type == TAGS_TYPE or not key:
        return [], None
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return [], None
    media_id = int(key)
    type_id = store.get_or_create_caption_type(caption_type)
    caption = store.get_caption(media_id, type_id)
    if caption is None:
        return [(FOLLOW_LABEL, FOLLOW)], FOLLOW
    choices = [(FOLLOW_LABEL, FOLLOW)]
    for revision in store.list_revisions(caption["id"]):
        choices.append((_revision_label(revision), revision["id"]))
    assignment = store.get_dataset_caption(dataset_id, caption["id"])
    if (
        assignment is not None
        and assignment["mode"] == "pinned"
        and assignment["revision_id"]
    ):
        return choices, assignment["revision_id"]
    return choices, FOLLOW


def read_captions_bulk(dataset_ref, keys, caption_type: str) -> dict:
    """Return the caption texts for many media at once.

    Same per-media resolution as :func:`read_caption` (pinned revision else
    head; the virtual :data:`TAGS_TYPE` joins the media's gallery tags), in
    one query per call instead of several per media.

    Returns
    -------
    dict
        ``{key: text}`` with an entry ("" included) for every given key.
    """
    keys = [key for key in keys if key]
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None or not keys:
        return {str(key): "" for key in keys}
    media_ids = [int(key) for key in keys]
    if caption_type == TAGS_TYPE:
        tags = store.tags_for_media_bulk(media_ids)
        return {
            str(media_id): ", ".join(
                row["name"] for row in tags.get(media_id, [])
            )
            for media_id in media_ids
        }
    type_id = store.get_or_create_caption_type(caption_type)
    texts = store.dataset_captions_bulk(dataset_id, media_ids, type_id)
    return {str(media_id): texts.get(media_id, "") for media_id in media_ids}


def present_types_bulk(dataset_ref, keys, types) -> dict:
    """Return the caption types already present for many media at once.

    The bulk form of :func:`present_types`: one query per type instead of
    one per (media, type).

    Returns
    -------
    dict
        ``{key: [present types, in ``types`` order]}``.
    """
    present = {str(key): [] for key in keys}
    for caption_type in types:
        for key, text in read_captions_bulk(
            dataset_ref, keys, caption_type
        ).items():
            if text.strip():
                present[key].append(caption_type)
    return present


def revision_options_bulk(dataset_ref, keys, caption_type: str) -> dict:
    """Return the revision-dropdown ``(choices, value)`` for many media.

    The bulk form of :func:`revision_options`: three queries for a whole
    page instead of three per card.

    Returns
    -------
    dict
        ``{key: (choices, value)}`` for every given key.
    """
    keys = [str(key) for key in keys]
    if caption_type == TAGS_TYPE:
        return {key: ([], None) for key in keys}
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return {key: ([], None) for key in keys}
    media_ids = [int(key) for key in keys if key]
    type_id = store.get_or_create_caption_type(caption_type)
    captions = store.captions_for_type_bulk(media_ids, type_id)
    caption_by_media = {row["media_id"]: row for row in captions}
    caption_ids = [row["id"] for row in captions]
    revisions = store.revisions_bulk(caption_ids)
    pins = store.dataset_caption_bulk(dataset_id, caption_ids)

    options = {}
    for key in keys:
        if not key:
            options[key] = ([], None)
            continue
        caption = caption_by_media.get(int(key))
        if caption is None:
            options[key] = ([(FOLLOW_LABEL, FOLLOW)], FOLLOW)
            continue
        choices = [(FOLLOW_LABEL, FOLLOW)]
        for revision in revisions.get(caption["id"], []):
            choices.append((_revision_label(revision), revision["id"]))
        assignment = pins.get(caption["id"])
        if (
            assignment is not None
            and assignment["mode"] == "pinned"
            and assignment["revision_id"]
        ):
            options[key] = (choices, assignment["revision_id"])
        else:
            options[key] = (choices, FOLLOW)
    return options


def select_revision(dataset_ref, key: str, caption_type: str, selection):
    """Apply a revision-dropdown selection; return the caption text to show.

    :data:`FOLLOW` puts the dataset back on the head; any other value pins it
    to that revision.
    """
    if not key:
        return read_caption(dataset_ref, key, caption_type)
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return ""
    media_id = int(key)
    type_id = store.get_or_create_caption_type(caption_type)
    caption = store.get_caption(media_id, type_id)
    if caption is None:
        return ""
    if selection == FOLLOW or selection is None:
        store.set_dataset_caption(dataset_id, caption["id"], "follow")
    else:
        store.set_dataset_caption(
            dataset_id, caption["id"], "pinned", int(selection)
        )
    return read_caption(dataset_ref, key, caption_type)


def triggerword_prefix(dataset_ref) -> str:
    """Return the dataset's trigger-word deploy prefix ("" when unset)."""
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return ""
    return store.triggerword_prefix(dataset_id)


def media_tags(key: str) -> list[str]:
    """Return a media's library tag names.

    The tags are the ones managed on the Tags/Medias tabs, ordered by category
    then tag name; they live at the media (library) level, not per dataset.
    """
    if not key:
        return []
    media_id = int(key)
    return [row["name"] for row in store.tags_for_media(media_id)]


def is_hidden(dataset_ref, key: str) -> bool:
    """Return whether a media is hidden in the dataset."""
    if not key:
        return False
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return False
    media_id = int(key)
    return store.is_media_hidden(dataset_id, media_id)


def set_hidden(dataset_ref, key: str, hidden: bool) -> None:
    """Set a media's hidden flag in the dataset (no-op when unselected)."""
    if not key:
        return
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return
    media_id = int(key)
    store.set_media_hidden(dataset_id, media_id, hidden)


def media_repeats(dataset_ref, key: str) -> int:
    """Return a media's deploy repeat count (1 when unselected)."""
    if not key:
        return 1
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return 1
    return store.get_media_repeats(dataset_id, int(key))


def set_media_repeats(dataset_ref, key: str, repeats: int) -> None:
    """Set a media's deploy repeat count (no-op when unselected)."""
    if not key:
        return
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return
    store.set_media_repeats(dataset_id, int(key), repeats)


def dataset_deploy_name(dataset_ref) -> str:
    """Return the dataset's deploy folder name ("" when unset/unavailable).

    An empty string means "use the dataset name" (see
    :func:`src.deploy.dataset_deploy_dir`).
    """
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return ""
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        return ""
    return dataset["deploy_name"] or ""


def set_dataset_deploy_name(dataset_ref, name: str) -> None:
    """Persist the dataset's deploy folder name (no-op when unselected)."""
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None:
        return
    store.update_dataset(dataset_id, deploy_name=name or "")


def delete_media(dataset_ref, key: str, types: list[str]) -> None:
    """Remove a media from the dataset.

    The media is only unlinked from the dataset — its file and captions are
    kept. ``types`` is unused (kept so the operation signature stays stable
    for the gallery callers).
    """
    # pylint: disable=unused-argument
    dataset_id = _dataset_id(dataset_ref)
    if dataset_id is None or not key:
        return
    media_id = int(key)
    store.remove_media_from_dataset(dataset_id, media_id)
