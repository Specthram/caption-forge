"""Gradio-free auto-build: the Studio's live preview, creation, upgrades.

Wraps the pure engines around the request. The Studio proposes a whole
training set from a library and explains every pick: the selection score,
the coverage map, the clusters and the uncovered zones all come from
:mod:`src.autobuild_studio`, which itself composes :mod:`src.dataset_compose`
(the corpus geometry), :mod:`src.siglip_grounding` (the semantic subject
signal) and :mod:`src.embeddings` (the DINOv2 vectors). No model is loaded
except SigLIP for the semantic query — kept resident between keystrokes and
released with :func:`release_model`, exactly like the composer — so the
preview runs synchronously in the request.
"""

import json
import re

from src import (
    autobuild_studio,
    config,
    dataset_compose,
    dataset_quality,
    embeddings,
    framing,
    settings,
    siglip_grounding,
)
from src import sqlite_store as store
from src.autobuild_studio import Recipe
from src.media import is_video_file

# The framing presets the Studio's segmented control offers, each a set of
# ratios over the config's framing buckets. "free" ignores framing entirely.
FRAMING_PRESETS = {
    "balanced": {
        "label": "Balanced",
        "ratios": {
            "face": 30,
            "upper_body": 40,
            "full_body": 20,
            "body_part": 10,
        },
    },
    "portraits": {
        "label": "Portraits",
        "ratios": {"face": 55, "upper_body": 35, "full_body": 10},
    },
    "wide": {
        "label": "Wide",
        "ratios": {"full_body": 45, "upper_body": 35, "face": 20},
    },
    "free": {"label": "Free", "ratios": {"any": 100}},
}

# How many best-matching media a suggested-tag query aggregates over.
_SUGGEST_TOP_K = 40

# The live preview's coarse stages, in order. Surfaced to the front-end so
# the recompute overlay names the step in flight (its progress bar) rather
# than spinning blindly — the heavy work of each runs after its event.
_PREVIEW_STAGES = (
    ("pool", "Loading the pool"),
    ("vectors", "Vectors & geometry"),
    ("semantic", "SigLIP relevance"),
    ("select", "Selecting picks"),
    ("assemble", "Assembling"),
)


def _ratios_of(preset: str) -> dict:
    """Return the framing ratios of a preset (falls back to balanced)."""
    entry = FRAMING_PRESETS.get(preset) or FRAMING_PRESETS["balanced"]
    return dict(entry["ratios"])


def _recipe_of(params, buckets: dict) -> Recipe:
    """Build the engine :class:`Recipe` from a request body."""
    return Recipe(
        media_type=params.media_type,
        semantic_q=params.semantic_q or "",
        locked_tags=tuple(params.locked_tags or ()),
        exclude_tags=tuple(params.exclude_tags or ()),
        seed_media_ids=tuple(params.seed_media_ids or ()),
        library_ids=tuple(params.library_ids or ()),
        size=int(params.size),
        metric=params.metric or dataset_compose.quality.DEFAULT_METRIC,
        min_score=float(params.min_score),
        exclude_blur=bool(params.exclude_blur),
        framing_preset=params.framing_preset or "balanced",
        ratios=_ratios_of(params.framing_preset or "balanced"),
        buckets=buckets,
        live=bool(params.live),
        dropped=tuple(params.dropped or ()),
        forced=tuple(params.forced or ()),
        kept=tuple(params.kept or ()),
        rebal=bool(params.rebal),
    )


def _library_scope(recipe: Recipe) -> set | None:
    """Return the media ids the recipe's libraries allow, or None for all."""
    if not recipe.library_ids:
        return None
    allowed: set = set()
    for library_id in recipe.library_ids:
        allowed.update(
            item["id"] for item in store.media_in_library(library_id)
        )
    return allowed


def _pool_of(recipe: Recipe) -> list:
    """Return the pool media dicts for a recipe, type- and scope-filtered.

    Every indexed library media of the chosen type and selected libraries
    with a file on disk: images carry hashes and DINOv2 vectors (the map,
    the diversity), videos are ranked on quality and framing alone. The
    excluded-tag filter runs later (it needs the tags).
    """
    media = store.library_media_set(quality_metric_selected=recipe.metric)
    wants_video = recipe.media_type == dataset_compose.MEDIA_TYPE_VIDEO
    scope = _library_scope(recipe)
    return [
        item
        for item in media
        if not item["missing"]
        and is_video_file(f"x.{item['file_extension']}") == wants_video
        and (scope is None or item["id"] in scope)
    ]


def _drop_excluded_tags(recipe: Recipe, pool: list, tags: dict) -> list:
    """Return the pool with every media carrying an excluded tag removed."""
    if not recipe.exclude_tags:
        return pool
    excluded = {framing.normalize_tag(name) for name in recipe.exclude_tags}
    return [
        item
        for item in pool
        if not (
            excluded
            & {
                framing.normalize_tag(row["name"])
                for row in tags.get(item["id"], [])
            }
        )
    ]


def _keep_required_tags(recipe: Recipe, pool: list, tags: dict) -> list:
    """Return the pool keeping only media that carry every locked tag.

    Locked tags are a hard include filter (the user's "must have these"):
    a media missing any locked tag is cut before scoring, so the selection
    and its coverage map are drawn over the matching media alone.
    """
    if not recipe.locked_tags:
        return pool
    required = {framing.normalize_tag(name) for name in recipe.locked_tags}
    return [
        item
        for item in pool
        if required
        <= {
            framing.normalize_tag(row["name"])
            for row in tags.get(item["id"], [])
        }
    ]


def _dinov2_vectors(ids) -> dict:
    """Return ``{id: numpy vector}`` DINOv2 embeddings for the pool."""
    return {
        media_id: embeddings.blob_to_vector(blob)
        for media_id, blob in store.media_embeddings(
            embeddings.MODEL_ID, ids
        ).items()
    }


def _semantic_relevance(query: str, ids) -> dict | None:
    """Return ``{id: cosine}`` of a typed query, or None when unavailable."""
    if not query.strip():
        return None
    model_id = settings.get_grounding_model_id()
    vectors = {
        media_id: embeddings.blob_to_vector(blob)
        for media_id, blob in store.media_embeddings(model_id, ids).items()
    }
    if not vectors:
        return None
    siglip_grounding.load_model(
        settings.get_grounding_model_size(),
        settings.get_grounding_resolution(),
    )
    return dataset_compose.semantic_relevance(
        siglip_grounding.embed_text(query.strip()), vectors
    )


def _select_stages(recipe: Recipe):
    """Run pool → corpus → prepare → select, yielding each stage key.

    A generator form of :func:`_select_for`: it yields the key of a stage
    *before* running its work (so a caller can surface the step in flight)
    and returns ``(corpus, candidates, picks, meta, tags, pref)`` on
    exhaustion (``StopIteration.value``); ``pref`` is the ``(before, after)``
    tag pre-filter count.
    """
    yield "pool"
    pool = _pool_of(recipe)
    tags = store.tags_for_media_bulk([item["id"] for item in pool])
    pref_before = len(pool)
    pool = _drop_excluded_tags(recipe, pool, tags)
    pool = _keep_required_tags(recipe, pool, tags)
    pref = (pref_before, len(pool))
    ids = [item["id"] for item in pool]
    yield "vectors"
    vectors = _dinov2_vectors(ids)
    corpus = dataset_compose.build_corpus(
        [], pool, vectors, store.media_hashes(ids)
    )
    yield "semantic"
    relevance = _semantic_relevance(recipe.semantic_q, ids)
    yield "select"
    candidates = autobuild_studio.prepare(
        recipe, pool, tags, corpus.vectors, relevance
    )
    picks, meta = autobuild_studio.select(corpus, candidates, recipe)
    return corpus, candidates, picks, meta, tags, pref


def _select_for(recipe: Recipe) -> tuple:
    """Run the pool → corpus → prepare → select pipeline for a recipe.

    The shared core of the live preview and the living-dataset upgrades:
    returns ``(corpus, candidates, picks, meta, tags, pref)``. Drives
    :func:`_select_stages` to completion, ignoring the stage keys.
    """
    stages = _select_stages(recipe)
    outcome: tuple = ()
    try:
        while True:
            next(stages)
    except StopIteration as done:
        outcome = done.value
    return outcome


def run_preview(params) -> dict:
    """Run the Studio selection and return its full live payload."""
    autobuild = config.load_autobuild_config()
    buckets = autobuild.get("framing_buckets") or {}
    recipe = _recipe_of(params, buckets)
    if recipe.size <= 0:
        return _empty_preview()
    corpus, candidates, picks, meta, tags, pref = _select_for(recipe)
    return _assemble(recipe, corpus, candidates, picks, meta, tags, pref)


def run_preview_events(params):
    """Yield the Studio selection as staged progress events, then result.

    Drives :func:`_select_stages` and emits one ``{"stage", "label",
    "index", "total"}`` event per step (the recompute overlay's progress
    bar), then a final ``{"result": payload}``. Each event announces the
    step about to run, so the client can name the work in flight.
    """
    autobuild = config.load_autobuild_config()
    buckets = autobuild.get("framing_buckets") or {}
    recipe = _recipe_of(params, buckets)
    total = len(_PREVIEW_STAGES)
    index_of = {key: i for i, (key, _label) in enumerate(_PREVIEW_STAGES)}
    label_of = dict(_PREVIEW_STAGES)

    def event(key: str) -> dict:
        return {
            "stage": key,
            "label": label_of[key],
            "index": index_of[key],
            "total": total,
        }

    if recipe.size <= 0:
        yield {"result": _empty_preview()}
        return
    corpus, candidates, picks, meta, tags, pref = yield from _emit_stages(
        recipe, event
    )
    yield event("assemble")
    yield {
        "result": _assemble(
            recipe, corpus, candidates, picks, meta, tags, pref
        )
    }


def _emit_stages(recipe: Recipe, event) -> tuple:
    """Yield each select stage as an event; return the select outcome.

    ``event`` maps a stage key to its wire event. Used with ``yield from``
    so the caller receives the ``(corpus, candidates, picks, meta, tags)``
    return value while the stage events stream through.
    """
    stages = _select_stages(recipe)
    outcome: tuple = ()
    try:
        while True:
            yield event(next(stages))
    except StopIteration as done:
        outcome = done.value
    return outcome


def _assemble(recipe, corpus, candidates, picks, meta, tags, pref):
    """Assemble the preview payload from the selection outcome."""
    by_id = {cand.id: cand for cand in candidates}
    pick_cands = [by_id[mid] for mid in picks if mid in by_id]
    cluster = autobuild_studio.cluster_of(corpus, pick_cands, recipe)
    panel = _panel(recipe, corpus, picks, tags)
    picked_cards = [
        _pick_card(by_id[mid], meta, recipe, corpus, cluster)
        for mid in picks
        if mid in by_id
    ]
    eligible = [cand for cand in candidates if cand.eligible]
    matched = (
        sum(1 for cand in eligible if cand.subject >= 0)
        if not autobuild_studio.subject_active(recipe)
        else sum(
            1
            for cand in candidates
            if cand.subject >= autobuild_studio.ELIGIBILITY_GATE
        )
    )
    return {
        "picks": picked_cards,
        "eligible": len(eligible),
        "pool_size": len(candidates),
        "matched": matched,
        "requested": recipe.size,
        "shortfall": max(0, recipe.size - len(picks)),
        "pref_before": pref[0],
        "pref_after": pref[1],
        "clusters": autobuild_studio.clusters(corpus, pick_cands, recipe),
        "zones": autobuild_studio.uncovered_zones(corpus, candidates, picks),
        "map": _map(corpus, candidates, picks, recipe, cluster),
        "dominant_tag": _dominant_tag(picked_cards),
        "semantic_available": _semantic_available(),
        **panel,
    }


def _panel(recipe, corpus, picks, tags):
    """Return the right-panel composition (grade, pillars, framing…).

    Reuses :func:`src.dataset_compose.preview` over the empty base dataset
    and the picks; the map and the zones are recomputed by the Studio
    engine (they are relative to the picks, not to an empty dataset) and
    replace the composer's here.
    """
    names = {
        media_id: [row["name"] for row in tags.get(media_id, [])]
        for media_id in [item["id"] for item in corpus.pool]
    }
    preview = dataset_compose.preview(
        corpus,
        set(picks),
        recipe.metric,
        names,
        recipe.buckets,
        dataset_quality.recommended_size_range(
            config.load_dataset_quality_config(), "character"
        ),
    )
    return {
        "grade": preview["grade"],
        "score": preview["score"],
        "pillars": preview["pillars"],
        "framing": preview["framing"],
        "size": preview["size"],
        "advice": preview["advice"],
    }


def _pick_card(cand, meta, recipe, corpus, cluster) -> dict:
    """Return one picked media as a Studio card."""
    return {
        "media_id": cand.id,
        "name": cand.name,
        "is_video": cand.is_video,
        "favorite": cand.favorite,
        "width": cand.width,
        "height": cand.height,
        "quality": cand.quality,
        "subject": round(cand.subject, 3),
        "gain": round(meta.get(cand.id, {}).get("gain", 0.0), 3),
        "bucket": cand.bucket,
        "cluster": cluster.get(cand.id),
        "reasons": autobuild_studio.reasons_for(cand, meta, recipe),
        "flag": autobuild_studio.flag_for(cand, meta, recipe),
        "xy": _round_point(corpus.xy.get(cand.id)),
    }


def _map(corpus, candidates, picks, recipe, cluster) -> dict:
    """Return the coverage map's role-tagged points and zones.

    ``cluster`` maps a pick id to its sub-family index, sent per point so
    the front-end can colour picks by cluster when a highlight is active.
    """
    picked = set(picks)
    seeds = set(recipe.seed_media_ids)
    points = []
    for cand in candidates:
        point = corpus.xy.get(cand.id)
        if point is None:
            continue
        if cand.id in seeds:
            role = "seed"
        elif cand.id in picked:
            role = "pick"
        else:
            role = "candidate"
        points.append(
            {
                "id": cand.id,
                "name": cand.name,
                "xy": _round_point(point),
                "role": role,
                "cluster": cluster.get(cand.id),
            }
        )
    return {
        "width": dataset_compose.MAP_WIDTH,
        "height": dataset_compose.MAP_HEIGHT,
        "points": points,
        "zones": autobuild_studio.uncovered_zones(corpus, candidates, picks),
    }


def _dominant_tag(picks) -> dict | None:
    """Return the tag most picks share, with its share — the trigger hint."""
    counts: dict = {}
    for card in picks:
        for chip in card["reasons"]:
            if chip["icon"] == "#":
                counts[chip["label"]] = counts.get(chip["label"], 0) + 1
    if not counts:
        return None
    name = max(counts, key=lambda key: counts[key])
    if not picks:
        return None
    return {"name": name, "share": round(100.0 * counts[name] / len(picks))}


def _round_point(point):
    """Return a map point rounded for the wire, or None."""
    if point is None:
        return None
    return [round(point[0], 2), round(point[1], 2)]


def _empty_preview() -> dict:
    """Return the zero-state payload (an empty or size-0 recipe)."""
    return {
        "picks": [],
        "eligible": 0,
        "pool_size": 0,
        "matched": 0,
        "requested": 0,
        "shortfall": 0,
        "pref_before": 0,
        "pref_after": 0,
        "clusters": [],
        "zones": [],
        "map": {
            "width": dataset_compose.MAP_WIDTH,
            "height": dataset_compose.MAP_HEIGHT,
            "points": [],
            "zones": [],
        },
        "dominant_tag": None,
        "semantic_available": _semantic_available(),
        "grade": "—",
        "score": None,
        "pillars": {},
        "framing": [],
        "size": {},
        "advice": [],
    }


def _semantic_available() -> bool:
    """Return whether the library carries SigLIP vectors at all."""
    return (
        store.count_media_with_embedding(settings.get_grounding_model_id()) > 0
    )


def suggested_tags(query: str) -> dict:
    """Return the WD14 tags the best semantic matches of a query share.

    The query is embedded with SigLIP, ranked against the library's SigLIP
    vectors, and the WD14 tags of the top matches are counted: a tag on
    most matches is what "red hair on a beach" tends to be tagged as. The
    front-end shows them as clickable chips that become locked subject tags.
    """
    query = (query or "").strip()
    if not query:
        return {"tags": []}
    model_id = settings.get_grounding_model_id()
    blobs = store.media_embeddings(model_id)
    if not blobs:
        return {"tags": []}
    vectors = {
        media_id: embeddings.blob_to_vector(blob)
        for media_id, blob in blobs.items()
    }
    siglip_grounding.load_model(
        settings.get_grounding_model_size(),
        settings.get_grounding_resolution(),
    )
    relevance = dataset_compose.semantic_relevance(
        siglip_grounding.embed_text(query), vectors
    )
    ranked = sorted(relevance, key=lambda mid: -relevance[mid])
    top = ranked[:_SUGGEST_TOP_K]
    if not top:
        return {"tags": []}
    tags = store.tags_for_media_bulk(top)
    counts: dict = {}
    for media_id in top:
        for row in tags.get(media_id, []):
            counts[row["name"]] = counts.get(row["name"], 0) + 1
    ranked_tags = sorted(counts, key=lambda name: (-counts[name], name))
    return {
        "tags": [
            {"name": name, "pct": round(100.0 * counts[name] / len(top))}
            for name in ranked_tags[:12]
        ]
    }


def neighbors(params) -> dict:
    """Return replacement candidates for a pick: visual neighbours or a search.

    The ``⇄`` swap strip. With no query it returns the 3 DINOv2-closest
    candidates that are not themselves picks (``exclude_ids``). With a ``q``
    it instead re-ranks the eligible candidates by WD14 tag and file-name
    match (top 5, cosine distance as the tie-break), so the user can steer
    the swap toward a specific variation. Images only.
    """
    metric = params.metric or dataset_compose.quality.DEFAULT_METRIC
    recipe = Recipe(media_type=params.media_type, metric=metric)
    pool = _pool_of(recipe)
    ids = [item["id"] for item in pool]
    vectors = _dinov2_vectors(ids)
    target = vectors.get(params.media_id)
    if target is None:
        return {"neighbors": []}
    unit = target / (float((target @ target) ** 0.5) or 1.0)
    exclude = set(params.exclude_ids or ()) | {params.media_id}
    by_id = {item["id"]: item for item in pool}
    cosine_of = {}
    for media_id, vector in vectors.items():
        if media_id in exclude:
            continue
        other = vector / (float((vector @ vector) ** 0.5) or 1.0)
        cosine_of[media_id] = float(unit @ other)
    query = (getattr(params, "q", "") or "").strip().lower()
    if query:
        ranked = _neighbor_search(query, by_id, cosine_of)
    else:
        ranked = [
            (media_id, "visual neighbour")
            for media_id in sorted(cosine_of, key=lambda mid: -cosine_of[mid])[
                :3
            ]
        ]
    return {
        "neighbors": [
            {
                "media_id": media_id,
                "name": by_id[media_id]["name"],
                "cosine": round(cosine_of.get(media_id, 0.0), 3),
                "quality": dataset_compose.metric_score(
                    by_id[media_id], metric
                ),
                "why": why,
            }
            for media_id, why in ranked
        ]
    }


def _neighbor_search(query: str, by_id: dict, cosine_of: dict) -> list:
    """Return ``(id, why)`` of the top tag/name matches for a swap query.

    Tokenised query; a candidate scores 2 per WD14 tag a token is a
    substring of, plus 1 for a file-name hit, cosine distance breaking ties.
    Only candidates scoring above zero survive; top 5.
    """
    tokens = [tok for tok in re.split(r"[^a-z0-9_]+", query) if tok]
    if not tokens:
        return []
    tags = store.tags_for_media_bulk(list(cosine_of))
    scored = []
    for media_id in cosine_of:
        names = [row["name"].lower() for row in tags.get(media_id, [])]
        matched = [tok for tok in tokens if any(tok in name for name in names)]
        name_hit = any(
            tok in by_id[media_id]["name"].lower() for tok in tokens
        )
        score = len(matched) * 2 + (1 if name_hit else 0)
        if score <= 0:
            continue
        why = (
            " ".join(f"#{tok}" for tok in matched) if matched else "name match"
        )
        scored.append((score, cosine_of[media_id], media_id, why))
    scored.sort(key=lambda row: (-row[0], -row[1]))
    return [(media_id, why) for _score, _cos, media_id, why in scored[:5]]


def release_model() -> None:
    """Unload the SigLIP checkpoint the semantic search kept resident."""
    siglip_grounding.unload_model()


def _recipe_from_dict(data: dict) -> Recipe:
    """Rebuild an engine :class:`Recipe` from a stored recipe blob."""
    preset = data.get("framing_preset") or "balanced"
    return Recipe(
        media_type=data.get("media_type", "img"),
        semantic_q=data.get("semantic_q", "") or "",
        locked_tags=tuple(data.get("locked_tags") or ()),
        exclude_tags=tuple(data.get("exclude_tags") or ()),
        seed_media_ids=tuple(data.get("seed_media_ids") or ()),
        library_ids=tuple(data.get("library_ids") or ()),
        size=int(data.get("size", 50)),
        metric=data.get("metric") or dataset_compose.quality.DEFAULT_METRIC,
        min_score=float(data.get("min_score", 60.0)),
        exclude_blur=bool(data.get("exclude_blur", True)),
        framing_preset=preset,
        ratios=_ratios_of(preset),
        buckets=config.load_autobuild_config().get("framing_buckets") or {},
        live=bool(data.get("live", True)),
        dropped=tuple(data.get("dropped") or ()),
        forced=tuple(data.get("forced") or ()),
        kept=tuple(data.get("kept") or ()),
        rebal=bool(data.get("rebal", False)),
    )


def _upgrade_reason(cand, gain: float, out_quality) -> str:
    """Return why an incoming candidate beats the member it replaces."""
    if any(cand.signals.get(key) for key in ("semantic", "seed")):
        return "stronger subject match"
    if cand.quality is not None and (
        out_quality is None or cand.quality > out_quality
    ):
        return "higher quality"
    if gain >= 0.5:
        return "adds visual variety"
    return "the recipe now prefers this pick"


def compute_upgrades(dataset_id: int) -> dict:
    """Replay a dataset's living recipe and return the proposed swaps.

    An upgrade is a candidate the recipe would now pick that is not in the
    dataset, paired with the member the recipe would now drop (weakest
    first). Nothing is applied — the caller confirms.
    """
    stored = store.get_autobuild_recipe(dataset_id)
    if stored is None or not stored["live"]:
        return {"dataset_id": dataset_id, "upgrades": []}
    recipe = _recipe_from_dict(stored["recipe"])
    _corpus, candidates, picks, meta, _tags, _pref = _select_for(recipe)
    by_id = {cand.id: cand for cand in candidates}
    members = store.media_in_dataset(dataset_id)
    member_ids = {item["id"] for item in members}
    member_quality = {
        item["id"]: dataset_compose.metric_score(item, recipe.metric)
        for item in members
    }
    member_name = {item["id"]: item["name"] for item in members}
    incoming = [mid for mid in picks if mid not in member_ids]
    outgoing = sorted(
        (mid for mid in member_ids if mid not in set(picks)),
        key=lambda mid: (
            member_quality.get(mid) is not None,
            member_quality.get(mid) or 0.0,
            mid,
        ),
    )
    upgrades = []
    for in_id, out_id in zip(incoming, outgoing):
        pick = by_id.get(in_id)
        if pick is None:
            continue
        gain = meta.get(in_id, {}).get("gain", 0.0)
        upgrades.append(
            {
                "out_media_id": out_id,
                "out_name": member_name.get(out_id, ""),
                "out_quality": member_quality.get(out_id),
                "in_media_id": in_id,
                "in_name": pick.name,
                "in_quality": pick.quality,
                "reason": _upgrade_reason(
                    pick, gain, member_quality.get(out_id)
                ),
                "gain": round(gain, 3),
            }
        )
    return {"dataset_id": dataset_id, "upgrades": upgrades}


def apply_upgrades(dataset_id: int, swaps) -> dict:
    """Apply the ``out → in`` swaps to a dataset; return how many landed.

    The outgoing media are only unlinked from the dataset — they stay in
    the library.
    """
    applied = 0
    for swap in swaps:
        out_id = int(swap["out_media_id"])
        in_id = int(swap["in_media_id"])
        store.remove_media_ids_from_dataset(dataset_id, [out_id])
        store.add_media_ids_to_dataset(dataset_id, [in_id])
        applied += 1
    return {"applied": applied}


def upgrades_summary() -> dict:
    """Return the upgrade count of every living dataset (banner source)."""
    summary = []
    for row in store.live_autobuild_recipes():
        result = compute_upgrades(row["dataset_id"])
        count = len(result["upgrades"])
        if count:
            summary.append({"dataset_id": row["dataset_id"], "count": count})
    return {"datasets": summary}


def create(name: str, selection: list[int], recipe: dict | None) -> int:
    """Create a dataset from a selection and store its Studio recipe.

    Raises
    ------
    ValueError
        On an empty/duplicate name (propagated from the store).
    """
    dataset_id = store.create_dataset(name)
    store.add_media_ids_to_dataset(dataset_id, selection)
    if recipe is not None:
        store.save_autobuild_recipe(
            dataset_id, json.dumps(recipe), bool(recipe.get("live", True))
        )
    return dataset_id
