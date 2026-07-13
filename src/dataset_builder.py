"""Automatic dataset selection engine (the "Auto-build" feature).

Given a pool of candidate media, picks the ``size`` most useful ones for
training: framing ratios per target type (face / upper body / full body /
body parts — see :mod:`src.framing`), a minimum quality bar, guaranteed
favorites, perceptual near-duplicate removal (one survivor per
:mod:`src.lookalike` group) and, inside each framing bucket, a greedy
farthest-point selection over the DINOv2 embeddings (see
:mod:`src.embeddings`) so the chosen pictures are as visually varied as
possible.

Pure functions over plain media dicts — the Datasets tab's behavior
module (:mod:`src.dataset_autobuild`) fetches the pool, tags and vectors
from the repository and renders the result. Every tie is broken
deterministically (quality first, then lowest media id), so the same
pool and parameters always select the same media.
"""

from dataclasses import dataclass, field

import numpy as np

from src import framing
from src import lookalike
from src import quality
from src.embeddings import blob_to_vector

# Default composite weights: how much a pick rewards being far from the
# already-selected pictures versus scoring high on quality. Overridden by
# the ``selection`` section of the auto-build config.
DIVERSITY_WEIGHT = 0.7
QUALITY_WEIGHT = 0.3


@dataclass(frozen=True)
class BucketReport:
    """The per-bucket outcome of a selection run.

    ``requested`` quota; ``selected`` picked; ``favorites`` guaranteed
    favorites among them; ``candidates`` eligible; ``without_embedding``
    eligible candidates with no vector (compete on quality alone).
    """

    requested: int
    selected: int
    favorites: int
    candidates: int
    without_embedding: int


@dataclass
class BuildResult:  # pylint: disable=too-many-instance-attributes
    """The outcome of a selection run.

    ``media_ids`` selected, pick order; ``buckets`` ``{ratio key:
    BucketReport}``; ``pool_size`` candidates entering selection;
    ``quality_dropped`` below the quality bar; ``unscored_dropped`` no score
    with the bar active; ``duplicates_dropped`` near-duplicates (best of each
    lookalike group survives); ``unknown_count`` tags matched no bucket
    (``any`` only); ``redistributed`` extra picks when a bucket underfilled;
    ``shortfall`` how far short of the target size.
    """

    media_ids: list = field(default_factory=list)
    buckets: dict = field(default_factory=dict)
    pool_size: int = 0
    quality_dropped: int = 0
    unscored_dropped: int = 0
    duplicates_dropped: int = 0
    unknown_count: int = 0
    redistributed: int = 0
    shortfall: int = 0


def _area(item: dict) -> int:
    """Return a media dict's pixel area, or 0 when a dimension is missing."""
    return int(item.get("width") or 0) * int(item.get("height") or 0)


def _keeper_key(item: dict, quality_by_id: dict):
    """Return the "best of a lookalike group" descending sort key.

    A favorite always survives its group; ties fall to the normalized
    quality, then the pixel area, then the lowest media id.
    """
    score = quality_by_id.get(item["id"])
    return (
        bool(item.get("favorite")),
        score is not None,
        score if score is not None else 0.0,
        _area(item),
        -item["id"],
    )


def _dedup(media, quality_by_id: dict, similarity: float):
    """Drop every lookalike-group member but the best; return survivors.

    Runs :func:`src.lookalike.detect` (unhashed media never group) and keeps
    one per group (:func:`_keeper_key`). Returns ``(survivors in pool order,
    dropped count)``.
    """
    result = lookalike.detect(media, similarity)
    by_id = {item["id"]: item for item in media}
    dropped = set()
    for group in result.groups:
        members = [by_id[entry.media_id] for entry in group.media]
        members.sort(key=lambda m: _keeper_key(m, quality_by_id), reverse=True)
        dropped.update(item["id"] for item in members[1:])
    survivors = [item for item in media if item["id"] not in dropped]
    return survivors, len(dropped)


def prepare_candidates(
    media,
    tags_by_id: dict,
    vector_blobs: dict,
    buckets: dict,
    min_quality: float = 0.0,
    similarity: float = lookalike.DEFAULT_SIMILARITY,
):
    """Turn pool media dicts into selection candidates.

    Applies the quality bar, removes near-duplicates and classifies each
    survivor's framing bucket. Caller fetches the pool with hashes + resolved
    display quality (see :func:`src.sqlite_store.media_with_hashes`).

    ``media`` carry ``id``/``favorite``/``phash``/``dhash``/``width``/
    ``height``/``quality_score``/``quality_metric``; ``tags_by_id`` is
    ``{media_id: [tag rows]}``; ``vector_blobs`` ``{media_id: BLOB}``;
    ``buckets`` the ``framing_buckets`` config. ``min_quality`` is a normalized
    0-100 bar (0 disables; when active an unscored media is dropped too);
    ``similarity`` the near-duplicate threshold. Returns ``(candidate dicts,
    partially-filled BuildResult)``.
    """
    stats = BuildResult()
    quality_by_id = {
        item["id"]: quality.normalize_score(
            item.get("quality_metric"), item.get("quality_score")
        )
        for item in media
    }
    kept = []
    for item in media:
        score = quality_by_id[item["id"]]
        if min_quality > 0:
            if score is None:
                stats.unscored_dropped += 1
                continue
            if score < min_quality:
                stats.quality_dropped += 1
                continue
        kept.append(item)
    kept, stats.duplicates_dropped = _dedup(kept, quality_by_id, similarity)
    candidates = []
    for item in kept:
        names = [row["name"] for row in tags_by_id.get(item["id"], ())]
        bucket = framing.classify(names, buckets)
        if bucket == framing.UNKNOWN_BUCKET:
            stats.unknown_count += 1
        blob = vector_blobs.get(item["id"])
        candidates.append(
            {
                "id": item["id"],
                "favorite": bool(item.get("favorite")),
                "quality": quality_by_id[item["id"]],
                "bucket": bucket,
                "vector": blob_to_vector(blob) if blob else None,
            }
        )
    stats.pool_size = len(candidates)
    return candidates, stats


def _quotas(size: int, ratios: dict) -> dict:
    """Split ``size`` into per-bucket quotas honoring the ratio weights.

    Only positive weights count. Rounding leftovers go to the largest
    fractional parts (ties to declaration order), so the quotas always
    sum to ``size`` (or to the ratio count when weights allow).
    """
    active = {k: float(v) for k, v in ratios.items() if float(v or 0) > 0}
    total = sum(active.values())
    if not active or total <= 0 or size <= 0:
        return {}
    shares = {k: size * v / total for k, v in active.items()}
    quotas = {k: int(share) for k, share in shares.items()}
    leftover = size - sum(quotas.values())
    by_fraction = sorted(
        active,
        key=lambda k: shares[k] - quotas[k],
        reverse=True,
    )
    for key in by_fraction[:leftover]:
        quotas[key] += 1
    return quotas


class _GreedyPicker:
    """Incremental farthest-point picker over a fixed candidate list.

    Holds the candidates' vectors as one matrix and a running
    "distance to the nearest already-picked media" per candidate, updated
    with one matrix-vector product per pick — so a run stays linear in
    picks instead of quadratic.
    """

    def __init__(self, candidates, diversity_weight, quality_weight):
        self._candidates = candidates
        self._dw = float(diversity_weight)
        self._qw = float(quality_weight)
        dim = next(
            (len(c["vector"]) for c in candidates if c["vector"] is not None),
            1,
        )
        self._vectors = np.zeros((len(candidates), dim), dtype=np.float32)
        self._has_vector = np.zeros(len(candidates), dtype=bool)
        for row, candidate in enumerate(candidates):
            if candidate["vector"] is not None:
                self._vectors[row] = candidate["vector"]
                self._has_vector[row] = True
        # Before anything is picked a vector-bearing candidate is
        # "maximally far" (1.0); a vectorless one contributes nothing.
        self._min_div = np.where(self._has_vector, 1.0, 0.0)
        self._taken = np.zeros(len(candidates), dtype=bool)

    def taken(self, row: int) -> bool:
        """Return whether a candidate row has already been picked."""
        return bool(self._taken[row])

    def take(self, row: int) -> None:
        """Mark a row picked and fold its vector into the distances."""
        self._taken[row] = True
        if self._has_vector[row]:
            self._update_distances(self._vectors[row])

    def _update_distances(self, vector) -> None:
        """Shrink each candidate's nearest-picked distance for a new pick.

        Cosine distance of unit vectors, rescaled to ``[0, 1]``:
        ``(1 - dot) / 2``. Vectorless candidates stay at zero.
        """
        distance = (1.0 - self._vectors @ vector) / 2.0
        self._min_div = np.minimum(
            self._min_div, np.where(self._has_vector, distance, 0.0)
        )

    def pick(self, eligible_rows) -> int | None:
        """Pick the best remaining candidate among ``eligible_rows``.

        The composite score is ``diversity_weight * distance +
        quality_weight * quality/100``; ties fall to the higher quality,
        then the lower media id. Returns the picked row, or None when
        nothing is eligible.
        """
        best_row = None
        best_key = None
        for row in eligible_rows:
            if self._taken[row]:
                continue
            candidate = self._candidates[row]
            grade = candidate["quality"]
            score = self._dw * float(self._min_div[row]) + self._qw * (
                (grade or 0.0) / 100.0
            )
            key = (score, grade is not None, grade or 0.0, -candidate["id"])
            if best_key is None or key > best_key:
                best_key = key
                best_row = row
        if best_row is None:
            return None
        self._taken[best_row] = True
        if self._has_vector[best_row]:
            self._update_distances(self._vectors[best_row])
        return best_row


def _bucket_rows(candidates, key: str) -> list:
    """Return the candidate rows eligible for one ratio key."""
    if key == framing.ANY_BUCKET:
        return list(range(len(candidates)))
    return [
        row
        for row, candidate in enumerate(candidates)
        if candidate["bucket"] == key
    ]


def _take_favorites(candidates, rows, quota: int, picker) -> list:
    """Return the guaranteed favorite rows of a bucket (quota-capped).

    Favorites are picked before the diversity loop, best quality first
    (an unscored favorite sorts last, ties to the lowest id).
    """
    favorite_rows = [
        row
        for row in rows
        if candidates[row]["favorite"] and not picker.taken(row)
    ]
    favorite_rows.sort(
        key=lambda row: (
            candidates[row]["quality"] is not None,
            candidates[row]["quality"] or 0.0,
            -candidates[row]["id"],
        ),
        reverse=True,
    )
    return favorite_rows[:quota]


def select(
    candidates,
    size: int,
    ratios: dict,
    diversity_weight: float = DIVERSITY_WEIGHT,
    quality_weight: float = QUALITY_WEIGHT,
    stats: BuildResult = None,
) -> BuildResult:
    """Pick ``size`` media from prepared candidates, honoring the ratios.

    Per ratio key (declaration order): guaranteed favorites first (capped at
    the bucket quota), then greedy farthest-point picks on the embeddings —
    each maximizes ``diversity_weight * distance to nearest selected +
    quality_weight * quality`` (distance measured against the whole selection,
    across buckets, so two buckets never hoard near-identical pictures). An
    underfilled bucket redistributes its slots over every ratio-eligible
    candidate.

    ``ratios`` is ``{bucket id or "any": weight}`` (relative); ``stats`` the
    preparation counters to fill (fresh result otherwise). Returns the
    :class:`BuildResult`.
    """
    result = stats if stats is not None else BuildResult()
    result.media_ids = []
    result.buckets = {}
    quotas = _quotas(int(size), ratios or {})
    picker = _GreedyPicker(candidates, diversity_weight, quality_weight)
    for key, quota in quotas.items():
        rows = _bucket_rows(candidates, key)
        eligible = [row for row in rows if not picker.taken(row)]
        vectorless = sum(
            1 for row in rows if candidates[row]["vector"] is None
        )
        picked_rows = []
        for row in _take_favorites(candidates, eligible, quota, picker):
            picker.take(row)
            picked_rows.append(row)
        while len(picked_rows) < quota:
            row = picker.pick(rows)
            if row is None:
                break
            picked_rows.append(row)
        result.media_ids.extend(candidates[row]["id"] for row in picked_rows)
        result.buckets[key] = BucketReport(
            requested=quota,
            selected=len(picked_rows),
            favorites=sum(
                1 for row in picked_rows if candidates[row]["favorite"]
            ),
            candidates=len(rows),
            without_embedding=vectorless,
        )
    # Redistribute unfilled quota over every ratio-eligible candidate.
    eligible_rows = sorted(
        {row for key in quotas for row in _bucket_rows(candidates, key)}
    )
    while len(result.media_ids) < int(size):
        row = picker.pick(eligible_rows)
        if row is None:
            break
        result.media_ids.append(candidates[row]["id"])
        result.redistributed += 1
    result.shortfall = max(0, int(size) - len(result.media_ids))
    return result


def build(
    media,
    tags_by_id: dict,
    vector_blobs: dict,
    size: int,
    ratios: dict,
    buckets: dict,
    min_quality: float = 0.0,
    similarity: float = lookalike.DEFAULT_SIMILARITY,
    diversity_weight: float = DIVERSITY_WEIGHT,
    quality_weight: float = QUALITY_WEIGHT,
) -> BuildResult:
    """Prepare the pool and run the selection in one call.

    See :func:`prepare_candidates` and :func:`select` for the parameters
    and the report's semantics.
    """
    candidates, stats = prepare_candidates(
        media,
        tags_by_id,
        vector_blobs,
        buckets,
        min_quality=min_quality,
        similarity=similarity,
    )
    return select(
        candidates,
        size,
        ratios,
        diversity_weight=diversity_weight,
        quality_weight=quality_weight,
        stats=stats,
    )
