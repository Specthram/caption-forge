"""The dataset composer: rank library media against a dataset in progress.

The "add media" picker of the Datasets tab is a composer, not a file list:
it answers *which* image is worth adding to a LoRA set, and *why*. Nothing
here computes a new signal — every number comes from an engine the app
already runs at index time:

* the IQA scores of :mod:`src.quality` (the per-candidate badge and the
  quality pillar),
* the model-free blur / noise statistics of :mod:`src.image_stats` (the
  two exclusion checkboxes),
* the DINOv2 vectors of :mod:`src.embeddings` (the diversity gain, the
  near-duplicate alert, the 2-D coverage map and its empty zones),
* the perceptual hashes of :mod:`src.lookalike` (the stronger "probable
  duplicate" alert a cosine alone would miss on a re-encode),
* the SigLIP 2 vectors of :mod:`src.siglip_grounding` (the semantic
  search),
* the framing buckets of :mod:`src.framing` and the grade bands, pillar
  weights and thresholds of :mod:`src.dataset_quality`.

The module is pure: the caller reads the media, the vectors and the tags
from the repository and passes them in. Every geometry (the map layout,
the empty zones) is derived from the *whole* corpus — the dataset plus
every candidate, filters excluded — so a filter change never moves a dot.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from src import dataset_quality, embedding_map, framing, image_stats
from src import lookalike, quality
from src.media import is_video_file

# The coverage map's viewBox, in the units the front-end draws with.
MAP_WIDTH = 100.0
MAP_HEIGHT = 74.0

# The map is diced into this grid to find the corpus' empty zones: a cell
# holding candidates but no dataset media is a region of the visual space
# the dataset does not cover yet.
GRID_COLUMNS = 6
GRID_ROWS = 4

# An empty zone is drawn as a circle of this fraction of the cell's short
# side — big enough to read, small enough not to swallow its neighbours.
ZONE_RADIUS_FACTOR = 0.5

# A candidate whose cosine similarity to a selected media reaches this is
# "similar to the selection" (the suggestion button). Below the
# near-duplicate threshold: the button gathers variations, not clones.
SIMILAR_COSINE = 0.80

# The diversity gain saturates at this cosine distance from everything
# already in the set — the same ceiling the spread score calibrates on.
GAIN_CEILING = dataset_quality.SPREAD_CEILING

# A semantic query keeps this fraction of the pool, best matches first.
# An absolute SigLIP cosine is not comparable across queries — "a woman on
# a beach" and "hands" do not peak at the same value — but a rank is.
SEMANTIC_TOP_FRACTION = 0.15

# Each near-duplicate inside the selection costs the hygiene pillar this
# many points.
HYGIENE_PENALTY = 22.0

# A framing bucket under this share of the projected dataset is called
# out as under-represented.
FRAMING_FLOOR = 10.0

# Headroom of the size bar above the recommended maximum, so a set sitting
# exactly on the ceiling does not paint a full bar.
SIZE_HEADROOM = 10

# How many advice cards the panel shows at most.
MAX_ADVICE = 3

MEDIA_TYPE_ANY = "all"
MEDIA_TYPE_IMAGE = "img"
MEDIA_TYPE_VIDEO = "vid"

SORT_QUALITY = "quality"
SORT_GAIN = "gain"
SORT_NAME = "name"


@dataclass(frozen=True)
class Filters:  # pylint: disable=too-many-instance-attributes
    """The left rail and toolbar of the composer, as one value.

    Attributes
    ----------
    metric : str
        The quality metric the badges show and the floor applies to (a
        :data:`src.quality.QUALITY_METRICS` key, or the "average"
        pseudo-metric).
    min_score : float
        The 0-100 floor a scored candidate must reach. A candidate that
        was never scored with the metric is never dropped by it — a
        library indexed without the quality step would otherwise show an
        empty grid.
    min_side : int
        The long-side floor in pixels; 0 keeps every resolution.
    exclude_blur, exclude_noise : bool
        Drop the candidates :func:`src.image_stats.is_flagged` reads as
        blurry / noisy. A media never analyzed is never dropped.
    favorites_only : bool
        Keep only the hearted media.
    media_type : str
        :data:`MEDIA_TYPE_ANY`, :data:`MEDIA_TYPE_IMAGE` or
        :data:`MEDIA_TYPE_VIDEO`.
    hide_near_dups : bool
        Drop the candidates that are near-duplicates of the dataset or of
        the current selection.
    gaps_only : bool
        Keep only the candidates sitting in an empty zone of the corpus.
    similar_to_selection : bool
        Keep only the candidates visually close to the selection.
    sort : str
        :data:`SORT_QUALITY`, :data:`SORT_GAIN` or :data:`SORT_NAME`.
    """

    metric: str = quality.DEFAULT_METRIC
    min_score: float = 0.0
    min_side: int = 0
    exclude_blur: bool = False
    exclude_noise: bool = False
    favorites_only: bool = False
    media_type: str = MEDIA_TYPE_ANY
    hide_near_dups: bool = False
    gaps_only: bool = False
    similar_to_selection: bool = False
    sort: str = SORT_QUALITY


@dataclass(frozen=True)
class Zone:
    """An empty region of the corpus, in map units."""

    x: float
    y: float
    r: float


@dataclass(frozen=True)
class Corpus:  # pylint: disable=too-many-instance-attributes
    """Everything the composer derives from the dataset and its candidates.

    Built once per request from the *unfiltered* pool, so the map layout,
    the empty zones and the near-duplicate links are stable while the user
    plays with the filters.

    Attributes
    ----------
    dataset : list of dict
        The media already linked to the dataset.
    pool : list of dict
        Every library media not in the dataset (the candidates).
    vectors : dict
        ``{media_id: unit numpy vector}`` — DINOv2, images only.
    names : dict
        ``{media_id: display name}`` over both sets.
    xy : dict
        ``{media_id: (x, y)}`` in the map viewBox; a media without a
        vector is absent.
    zones : tuple of Zone
        The empty zones of the corpus.
    hash_twins : dict
        ``{media_id: set of media ids}`` sharing a perceptual-hash group.
    depth_vectors : dict
        ``{media_id: unit numpy vector}`` — Depth-Anything V2 composition
        signatures, images only. Empty when the depth index step has not run;
        the Proximity fusion then falls back to DINOv2 alone.
    """

    dataset: list
    pool: list
    vectors: dict
    names: dict
    xy: dict
    zones: tuple
    hash_twins: dict
    depth_vectors: dict = field(default_factory=dict)


def metric_score(item: dict, metric: str) -> float | None:
    """Return a media's 0-100 quality for one metric, or None.

    Parameters
    ----------
    item : dict
        A media dict carrying the ``quality_scores`` map.
    metric : str
        A :data:`src.quality.QUALITY_METRICS` key, or
        :data:`src.quality.AVERAGE_METRIC_ID` for the mean of every stored
        metric.

    Returns
    -------
    float or None
        None when the media carries no usable score for the metric.
    """
    scores = item.get("quality_scores") or {}
    if metric == quality.AVERAGE_METRIC_ID:
        values = [
            value
            for value in (
                quality.normalize_score(key, raw)
                for key, raw in scores.items()
            )
            if value is not None
        ]
        return sum(values) / len(values) if values else None
    if metric not in scores:
        return None
    return quality.normalize_score(metric, scores[metric])


def _unit(vector) -> np.ndarray:
    """Return one vector L2-normalized as ``float64``."""
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    return array if norm == 0 else array / norm


def _layout(ids, vectors: dict) -> dict:
    """Return ``{media_id: (x, y)}`` — a PCA projection in map units."""
    embedded = [media_id for media_id in ids if media_id in vectors]
    if len(embedded) < 2:
        return {
            media_id: (MAP_WIDTH / 2.0, MAP_HEIGHT / 2.0)
            for media_id in embedded
        }
    stacked = np.stack([vectors[media_id] for media_id in embedded])
    coords = embedding_map.project(stacked)
    return {
        media_id: (float(x) * MAP_WIDTH, float(y) * MAP_HEIGHT)
        for media_id, (x, y) in zip(embedded, coords)
    }


def _cell_of(point) -> tuple:
    """Return the grid cell ``(column, row)`` a map point falls in."""
    column = min(GRID_COLUMNS - 1, int(point[0] / MAP_WIDTH * GRID_COLUMNS))
    row = min(GRID_ROWS - 1, int(point[1] / MAP_HEIGHT * GRID_ROWS))
    return column, row


def empty_zones(dataset_points, pool_points) -> tuple:
    """Return the grid cells holding candidates but no dataset media.

    Parameters
    ----------
    dataset_points, pool_points : iterable of tuple
        The ``(x, y)`` map points of the dataset and of the candidates.

    Returns
    -------
    tuple of Zone
        One circle per uncovered cell, in reading order.
    """
    covered = {_cell_of(point) for point in dataset_points}
    populated = {_cell_of(point) for point in pool_points}
    width = MAP_WIDTH / GRID_COLUMNS
    height = MAP_HEIGHT / GRID_ROWS
    radius = min(width, height) * ZONE_RADIUS_FACTOR
    return tuple(
        Zone(
            x=(column + 0.5) * width,
            y=(row + 0.5) * height,
            r=radius,
        )
        for column, row in sorted(populated - covered)
    )


def _hash_twins(items) -> dict:
    """Return ``{media_id: {media_id}}`` of the perceptual-hash groups."""
    groups = lookalike.detect(items).groups
    twins: dict = {}
    for group in groups:
        members = {media.media_id for media in group.media}
        for media_id in members:
            twins[media_id] = members - {media_id}
    return twins


def build_corpus(dataset, pool, vectors, hashes, depth_vectors=None) -> Corpus:
    """Derive the stable geometry of one composer session.

    Parameters
    ----------
    dataset : list of dict
        The dataset's media dicts.
    pool : list of dict
        Every candidate media dict (no filter applied).
    vectors : dict
        ``{media_id: numpy vector}`` DINOv2 embeddings of both sets.
    hashes : dict
        ``{media_id: (phash, dhash)}`` of both sets (see
        :func:`src.sqlite_store.media_hashes`).
    depth_vectors : dict, optional
        ``{media_id: numpy vector}`` Depth-Anything V2 composition signatures
        of both sets. ``None`` (or empty) leaves the corpus without a depth
        signal — the Proximity fusion then falls back to DINOv2 alone.

    Returns
    -------
    Corpus
        The map layout, the empty zones, the hash groups and the depth
        signatures.
    """
    units = {mid: _unit(vector) for mid, vector in vectors.items()}
    depth_units = {
        mid: _unit(vector) for mid, vector in (depth_vectors or {}).items()
    }
    names = {item["id"]: item["name"] for item in list(dataset) + list(pool)}
    ids = [item["id"] for item in list(dataset) + list(pool)]
    xy = _layout(ids, units)
    dataset_ids = [item["id"] for item in dataset]
    zones = empty_zones(
        [xy[media_id] for media_id in dataset_ids if media_id in xy],
        [xy[item["id"]] for item in pool if item["id"] in xy],
    )
    hashable = [
        {**item, "phash": pair[0], "dhash": pair[1]}
        for item, pair in (
            (item, hashes.get(item["id"]) or (None, None))
            for item in list(dataset) + list(pool)
        )
    ]
    return Corpus(
        dataset=list(dataset),
        pool=list(pool),
        vectors=units,
        names=names,
        xy=xy,
        zones=zones,
        hash_twins=_hash_twins(hashable),
        depth_vectors=depth_units,
    )


def _reference_ids(corpus: Corpus, picked) -> list:
    """Return the ids a candidate is compared against (dataset + picked)."""
    return [item["id"] for item in corpus.dataset] + list(picked)


def _cosines(corpus: Corpus, media_id: int, reference_ids) -> dict:
    """Return ``{reference_id: cosine}`` of one candidate, embedded only."""
    vector = corpus.vectors.get(media_id)
    if vector is None:
        return {}
    return {
        other: float(vector @ corpus.vectors[other])
        for other in reference_ids
        if other != media_id and other in corpus.vectors
    }


def near_duplicate(
    corpus: Corpus, media_id: int, reference_ids
) -> dict | None:
    """Return the near-duplicate alert of one candidate, or None.

    A shared perceptual-hash group wins over a cosine match: a re-encoded
    copy is a *probable duplicate*, not merely a close neighbour.

    Returns
    -------
    dict or None
        ``{"media_id", "name", "similarity", "cosine", "kind"}`` where
        ``kind`` is ``"hash"`` or ``"cosine"``.
    """
    references = set(reference_ids) - {media_id}
    cosines = _cosines(corpus, media_id, references)
    twins = corpus.hash_twins.get(media_id, set()) & references
    if twins:
        twin = sorted(twins)[0]
        return {
            "media_id": twin,
            "name": corpus.names.get(twin, ""),
            "similarity": lookalike.DEFAULT_SIMILARITY,
            "cosine": round(cosines.get(twin, 0.0), 3),
            "kind": "hash",
        }
    if not cosines:
        return None
    best = max(cosines, key=cosines.get)
    if cosines[best] < dataset_quality.NEAR_DUP_COSINE:
        return None
    return {
        "media_id": best,
        "name": corpus.names.get(best, ""),
        "similarity": int(round(cosines[best] * 100)),
        "cosine": round(cosines[best], 3),
        "kind": "cosine",
    }


def diversity_gain(corpus: Corpus, media_id: int, reference_ids) -> float:
    """Return how much visual ground a candidate adds, 0-1.

    The cosine distance to the closest media already in the set, mapped
    linearly onto :data:`GAIN_CEILING`. A media without a vector, or a set
    with nothing to compare against, scores a neutral 0.
    """
    cosines = _cosines(corpus, media_id, set(reference_ids) - {media_id})
    if not cosines:
        return 0.0
    distance = 1.0 - max(cosines.values())
    return max(0.0, min(1.0, distance / GAIN_CEILING))


def in_empty_zone(corpus: Corpus, media_id: int) -> bool:
    """Return whether a candidate sits in one of the corpus' empty zones."""
    point = corpus.xy.get(media_id)
    if point is None:
        return False
    return any(
        (point[0] - zone.x) ** 2 + (point[1] - zone.y) ** 2 <= zone.r**2
        for zone in corpus.zones
    )


def _similar_to_selection(corpus: Corpus, media_id: int, picked) -> bool:
    """Return whether a candidate is visually close to the selection."""
    cosines = _cosines(corpus, media_id, set(picked) - {media_id})
    return bool(cosines) and max(cosines.values()) >= SIMILAR_COSINE


def semantic_relevance(query_vector, image_vectors: dict) -> dict:
    """Return ``{media_id: cosine}`` of a text query against the images."""
    if query_vector is None or not image_vectors:
        return {}
    unit_query = _unit(query_vector)
    return {
        media_id: float(_unit(vector) @ unit_query)
        for media_id, vector in image_vectors.items()
    }


def _semantic_keep(relevance: dict) -> set:
    """Return the best-matching media ids of a semantic query."""
    if not relevance:
        return set()
    keep = math.ceil(len(relevance) * SEMANTIC_TOP_FRACTION)
    ranked = sorted(relevance, key=lambda mid: -relevance[mid])
    return set(ranked[: max(1, keep)])


def _is_video(item: dict) -> bool:
    """Return whether a media dict is a video."""
    return is_video_file(f"x.{item['file_extension']}")


def _passes_resolution(item: dict, min_side: int) -> bool:
    """Return whether a media's long side clears the floor (0 = any)."""
    if not min_side:
        return True
    side = max(item.get("width") or 0, item.get("height") or 0)
    return not side or side >= min_side


def _passes_stats(item: dict, filters: Filters) -> bool:
    """Return whether a media clears the blur / noise floors.

    A media the Index never analyzed carries no statistics; it is unknown,
    not flagged, and the checkboxes leave it alone.
    """
    stats = item.get("stats")
    if not stats:
        return True
    if filters.exclude_blur and stats["sharpness"] < image_stats.BLUR_FLOOR:
        return False
    return not (
        filters.exclude_noise
        and stats["cleanliness"] < image_stats.NOISE_FLOOR
    )


def _passes_source(item: dict, filters: Filters) -> bool:
    """Return whether a media matches the favorites / type filters."""
    if filters.favorites_only and not item.get("favorite"):
        return False
    if filters.media_type == MEDIA_TYPE_ANY:
        return True
    wants_video = filters.media_type == MEDIA_TYPE_VIDEO
    return wants_video == _is_video(item)


def _passes_static(item: dict, filters: Filters, score) -> bool:
    """Return whether a candidate passes the filters that need no corpus.

    An unscored media (no row for the selected metric) is never dropped by
    the score floor: a library indexed without the quality step would
    otherwise show an empty grid.
    """
    if score is not None and score < filters.min_score:
        return False
    return (
        _passes_resolution(item, filters.min_side)
        and _passes_stats(item, filters)
        and _passes_source(item, filters)
    )


def _sort_key(filters: Filters, row: dict):
    """Return the ordering key of an annotated candidate row."""
    if filters.sort == SORT_GAIN:
        return (-row["gain"], row["name"].lower())
    if filters.sort == SORT_NAME:
        return (row["name"].lower(),)
    score = row["score"]
    return (-(score if score is not None else -1.0), row["name"].lower())


def candidates(
    corpus: Corpus,
    filters: Filters,
    picked,
    relevance: dict | None = None,
) -> list:
    """Return the annotated, filtered and sorted candidates.

    A selected candidate always survives the filters: hiding an image the
    user just picked, because a slider moved, would silently drop it from
    the composition.

    Parameters
    ----------
    corpus : Corpus
        The session geometry (see :func:`build_corpus`).
    filters : Filters
        The left rail and toolbar state.
    picked : set of int
        The media ids currently selected.
    relevance : dict, optional
        ``{media_id: cosine}`` of a semantic query; None when the search
        box is empty (or no SigLIP vector is stored).

    Returns
    -------
    list of dict
        One row per surviving candidate, in display order, each carrying
        the media dict under ``"item"`` plus the annotations of
        :func:`annotate`.
    """
    picked = set(picked)
    references = _reference_ids(corpus, picked)
    semantic = _semantic_keep(relevance) if relevance else None
    rows = []
    for item in corpus.pool:
        media_id = item["id"]
        selected = media_id in picked
        score = metric_score(item, filters.metric)
        near = near_duplicate(corpus, media_id, references)
        gap = in_empty_zone(corpus, media_id)
        if not selected:
            if not _passes_static(item, filters, score):
                continue
            if semantic is not None and media_id not in semantic:
                continue
            if filters.hide_near_dups and near:
                continue
            if filters.gaps_only and not gap:
                continue
            if filters.similar_to_selection and not _similar_to_selection(
                corpus, media_id, picked
            ):
                continue
        point = corpus.xy.get(media_id)
        rows.append(
            {
                "item": item,
                "id": media_id,
                "name": item["name"],
                "score": score,
                "gain": diversity_gain(corpus, media_id, references),
                "near_dup": near,
                "in_gap": gap,
                "xy": _round_point(point),
            }
        )
    rows.sort(key=lambda row: _sort_key(filters, row))
    return rows


def gap_count(corpus: Corpus) -> int:
    """Return how many candidates sit in an empty zone of the corpus."""
    return sum(1 for item in corpus.pool if in_empty_zone(corpus, item["id"]))


# --- live composition preview ---------------------------------------------


def _mean_quality(items, metric: str) -> float | None:
    """Return the mean 0-100 quality of the scored media of a set."""
    values = [
        score
        for score in (metric_score(item, metric) for item in items)
        if score is not None
    ]
    return sum(values) / len(values) if values else None


def _spread(corpus: Corpus, ids) -> float | None:
    """Return the 0-100 embedding spread of a set of media ids."""
    vectors = [corpus.vectors[i] for i in ids if i in corpus.vectors]
    if len(vectors) < 2:
        return None
    similarity = embedding_map.cosine_matrix(np.stack(vectors))
    distance = embedding_map.mean_pairwise_distance(similarity)
    return dataset_quality.spread_score(distance)


def duplicate_alerts(corpus: Corpus, picked) -> list:
    """Return one alert per selected media that duplicates the set."""
    picked = set(picked)
    references = _reference_ids(corpus, picked)
    alerts = []
    for media_id in sorted(picked):
        near = near_duplicate(corpus, media_id, references)
        if near:
            alerts.append(
                {
                    "media_id": media_id,
                    "name": corpus.names.get(media_id, ""),
                    "near_dup": near,
                }
            )
    return alerts


def _pillars(corpus: Corpus, picked, metric: str) -> dict:
    """Return the projected quality / diversity / hygiene pillars."""
    picked = set(picked)
    items = corpus.dataset + [
        item for item in corpus.pool if item["id"] in picked
    ]
    ids = [item["id"] for item in items]
    dups = len(duplicate_alerts(corpus, picked))
    return {
        "quality": _mean_quality(items, metric),
        "diversity": _spread(corpus, ids),
        "hygiene": max(0.0, 100.0 - HYGIENE_PENALTY * dups),
        "duplicates": dups,
    }


def _overall(pillars: dict) -> float | None:
    """Return the weighted 0-100 score over the scorable pillars."""
    weights = dataset_quality.DEFAULT_WEIGHTS
    scored = [
        (pillars[key], weight)
        for key, weight in weights.items()
        if pillars.get(key) is not None
    ]
    total = sum(weight for _, weight in scored)
    if not total:
        return None
    return sum(value * weight for value, weight in scored) / total


def framing_distribution(corpus: Corpus, picked, tags_by_id, buckets) -> list:
    """Return the per-bucket framing split of the projected dataset.

    Each row separates what the dataset already holds from what the
    selection would add, which is exactly how the panel stacks its bars.

    Returns
    -------
    list of dict
        ``{"bucket", "base", "added", "total", "share", "under"}``.
    """
    picked = set(picked)
    added_items = [item for item in corpus.pool if item["id"] in picked]
    counts: dict = {}
    for items, key in ((corpus.dataset, "base"), (added_items, "added")):
        for item in items:
            bucket = framing.classify(tags_by_id.get(item["id"], ()), buckets)
            counts.setdefault(bucket, {"base": 0, "added": 0})[key] += 1
    total = len(corpus.dataset) + len(added_items)
    rows = []
    for bucket in list(buckets) + [framing.UNKNOWN_BUCKET]:
        entry = counts.get(bucket)
        if not entry:
            continue
        count = entry["base"] + entry["added"]
        share = 100.0 * count / total if total else 0.0
        rows.append(
            {
                "bucket": bucket,
                "base": entry["base"],
                "added": entry["added"],
                "total": count,
                "share": share,
                "under": share < FRAMING_FLOOR,
            }
        )
    return rows


def _advice_rows(context: dict) -> list:
    """Return the advice cards, in priority order (see the handoff)."""
    rows = []
    if context["duplicates"]:
        rows.append(
            {
                "tone": "danger",
                "text": (
                    f"{context['duplicates']} selected near-duplicate(s) — "
                    "keep the best of each pair."
                ),
            }
        )
    if context["low_quality"]:
        floor = int(dataset_quality.LOW_QUALITY_FLOOR)
        rows.append(
            {
                "tone": "warn",
                "text": (
                    f"{context['low_quality']} selected image(s) under "
                    f"{floor} in quality — the projected score suffers."
                ),
            }
        )
    if context["under_framing"]:
        joined = ", ".join(context["under_framing"])
        rows.append(
            {
                "tone": "warn",
                "text": (
                    f"Under-represented framing: {joined}. Filter by tag "
                    "to fill it in."
                ),
            }
        )
    if context["gaps"] and context["picked"] < 4:
        rows.append(
            {
                "tone": "info",
                "text": (
                    f"{context['gaps']} candidates cover empty zones of the "
                    'corpus — the "Fill the gaps" button.'
                ),
            }
        )
    if not rows:
        rows.append(
            {"tone": "ok", "text": "Balanced selection — nothing flagged."}
        )
    return rows[:MAX_ADVICE]


def _round_point(point):
    """Return a map point rounded for the wire, or None."""
    if point is None:
        return None
    return [round(point[0], 2), round(point[1], 2)]


def _delta(score, base_score):
    """Return the projected score's move against the current dataset."""
    if score is None or base_score is None:
        return None
    return score - base_score


def _map_points(corpus: Corpus, ids) -> list:
    """Return the rounded map points of the media that carry one."""
    return [
        [round(corpus.xy[i][0], 2), round(corpus.xy[i][1], 2)]
        for i in ids
        if i in corpus.xy
    ]


def preview(
    corpus: Corpus,
    picked,
    metric: str,
    tags_by_id: dict,
    buckets: dict,
    size_range: tuple = dataset_quality.DEFAULT_SIZE_RANGE,
) -> dict:
    """Return the live composition panel of one selection.

    Every number is recomputed from scratch over ``dataset + picked``, and
    the same numbers over ``dataset`` alone give the delta the grade card
    shows.

    Parameters
    ----------
    corpus : Corpus
        The session geometry (see :func:`build_corpus`).
    picked : iterable of int
        The selected media ids.
    metric : str
        The quality metric of the badges and of the quality pillar.
    tags_by_id : dict
        ``{media_id: [tag names]}`` for the framing classification.
    buckets : dict
        The ``framing_buckets`` section of the auto-build config.
    size_range : tuple, optional
        The recommended ``(min, max)`` dataset size.

    Returns
    -------
    dict
        The payload of ``POST /datasets/{id}/compose/preview``.
    """
    picked = set(picked)
    projected = _pillars(corpus, picked, metric)
    base = _pillars(corpus, set(), metric)
    score = _overall(projected)
    base_score = _overall(base)
    dataset_ids = [item["id"] for item in corpus.dataset]
    low_quality = sum(
        1
        for item in corpus.pool
        if item["id"] in picked
        and (metric_score(item, metric) or 0.0)
        < dataset_quality.LOW_QUALITY_FLOOR
    )
    framing_rows = framing_distribution(corpus, picked, tags_by_id, buckets)
    total = len(corpus.dataset) + len(picked)
    return {
        "score": score,
        "base_score": base_score,
        "delta": _delta(score, base_score),
        "grade": dataset_quality.grade_of(score),
        "pillars": {
            "quality": projected["quality"],
            "diversity": projected["diversity"],
            "hygiene": projected["hygiene"],
            "duplicates": projected["duplicates"],
        },
        "size": {
            "base": len(corpus.dataset),
            "picked": len(picked),
            "total": total,
            "min": size_range[0],
            "max": size_range[1],
            "percent": min(
                100.0, 100.0 * total / (size_range[1] + SIZE_HEADROOM)
            ),
            "over": total > size_range[1],
        },
        "framing": framing_rows,
        "dup_alerts": duplicate_alerts(corpus, picked),
        "map": {
            "dataset": _map_points(corpus, dataset_ids),
            "selected": _map_points(corpus, sorted(picked)),
            "zones": [
                {"x": round(z.x, 2), "y": round(z.y, 2), "r": round(z.r, 2)}
                for z in corpus.zones
            ],
            "width": MAP_WIDTH,
            "height": MAP_HEIGHT,
        },
        "advice": _advice_rows(
            {
                "duplicates": projected["duplicates"],
                "low_quality": low_quality,
                "under_framing": [
                    row["bucket"] for row in framing_rows if row["under"]
                ],
                "gaps": gap_count(corpus),
                "picked": len(picked),
            }
        ),
    }
