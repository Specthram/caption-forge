"""Flagged-media detection for the dataset quality report.

Turns the raw signals a report run gathers — IQA scores, DINOv2
similarities and clusters, perceptual hashes, caption text — into the
actionable list the "Flagged media" card shows: one :class:`Issue` per
finding, each carrying the media it points at, a one-line reason, the
metric that triggered it and the payload its inspector needs.

Four kinds of finding:

* ``near_dup`` — two images the embeddings (or the perceptual hashes) call
  the same picture; one of them is redundant training signal.
* ``low_quality`` — an image whose mean IQA score sits under the floor.
* ``outlier`` — an image whose nearest neighbors are all far: off-concept.
* ``caption`` — a caption the integrity heuristics call degenerate.

Every issue owns a stable ``key`` (so a resolution survives a re-run) and
a ``fingerprint`` of the underlying measurement (so a resolution is
dropped once that measurement changes). Pure functions over plain dicts:
no database, no model, no I/O.
"""

import hashlib
from dataclasses import dataclass, field

from src import caption_review
from src import lookalike

# Sort weight of each kind: near-duplicates waste the most training steps,
# outliers the least. The per-issue bonus (severity of the measurement)
# is added on top, so a catastrophic low-quality image can still outrank a
# borderline near-duplicate.
_BASE_IMPACT = {
    "near_dup": 90.0,
    "low_quality": 70.0,
    "caption": 55.0,
    "outlier": 40.0,
}

# How many nearest neighbors the outlier inspector shows.
_NEIGHBORS = 3

# Weight of the learned IQA score against the raw sharpness when picking
# the best image of a near-duplicate pair.
_QUALITY_WEIGHT = 0.7
_SHARPNESS_WEIGHT = 0.3

# Caps keeping the report readable whatever the dataset size.
MAX_ISSUES_PER_KIND = 40


@dataclass(frozen=True)
class Issue:  # pylint: disable=too-many-instance-attributes
    """One actionable finding of a dataset quality report.

    Attributes
    ----------
    key : str
        Stable identifier of the finding (``"dup:3:7"``, ``"lowq:12"``…);
        a stored resolution is keyed on it.
    kind : str
        ``near_dup`` | ``low_quality`` | ``outlier`` | ``caption``.
    media_ids : tuple of int
        The media the finding points at (one, or two for a pair).
    names : tuple of str
        Their display names, aligned with ``media_ids`` — the row's mono
        label, so the front-end needs no second lookup.
    reason : str
        The one-line explanation shown on the collapsed row.
    metric : str
        The measurement, formatted for the mono column (``"sim 0.97"``).
    value : float
        The raw measurement behind ``metric`` (for colouring/sorting).
    impact : float
        The sort weight; the list is ordered by it, descending.
    fingerprint : str
        A digest of the measurement — a resolution is cleared when it
        changes (the finding is no longer the same one).
    detail : dict
        The kind-specific payload the expanded inspector renders.
    """

    key: str
    kind: str
    media_ids: tuple
    reason: str
    metric: str
    value: float
    impact: float
    fingerprint: str
    names: tuple = ()
    detail: dict = field(default_factory=dict)


def _short_digest(text: str) -> str:
    """Return a short, stable digest of a text (fingerprints)."""
    return hashlib.sha1(
        text.encode("utf-8", "replace"), usedforsecurity=False
    ).hexdigest()[:12]


@dataclass(frozen=True)
class IssueContext:  # pylint: disable=too-many-instance-attributes
    """Everything the detectors read, already gathered by the caller.

    Attributes
    ----------
    images : tuple of dict
        The dataset's image dicts (``id``, ``name``, ``width``,
        ``height``, ``phash``/``dhash``…).
    quality_by_id : dict
        ``{media_id: mean normalized 0-100 quality}`` (None when unscored).
    scores_by_id : dict
        ``{media_id: {metric_id: normalized score}}`` for the inspectors.
    stats_by_id : dict
        ``{media_id: src.image_stats.analyze(...) dict}``.
    captions_by_id : dict
        ``{media_id: caption text}`` for the chosen caption type.
    map_result : src.embedding_map.MapResult or None
        The clustered projection; None when fewer than two embeddings.
    near_dup_cosine : float
        DINOv2 cosine similarity above which a pair is a near-duplicate.
    low_quality_floor : float
        Normalized quality below which an image is flagged.
    index : dict
        The images keyed by media id; filled from ``images``.
    """

    images: tuple = ()
    quality_by_id: dict = field(default_factory=dict)
    scores_by_id: dict = field(default_factory=dict)
    stats_by_id: dict = field(default_factory=dict)
    captions_by_id: dict = field(default_factory=dict)
    map_result: object = None
    near_dup_cosine: float = 0.92
    low_quality_floor: float = 70.0
    index: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Index the images by media id (set once, frozen afterwards)."""
        object.__setattr__(
            self, "index", {item["id"]: item for item in self.images}
        )


def _area(item: dict) -> int:
    """Return an image's pixel area (0 when it was never indexed)."""
    return (item.get("width") or 0) * (item.get("height") or 0)


def _pick_score(media_id: int, context: IssueContext) -> tuple:
    """Return the sort key ranking a near-duplicate candidate."""
    stats = context.stats_by_id.get(media_id) or {}
    quality = context.quality_by_id.get(media_id)
    composite = _QUALITY_WEIGHT * (quality or 0.0)
    composite += _SHARPNESS_WEIGHT * stats.get("sharpness", 0.0)
    item = context.index.get(media_id, {})
    return (composite, _area(item), -media_id)


def _pair_metrics(media_id: int, context: IssueContext) -> dict:
    """Return the per-image metric block of a near-duplicate inspector."""
    item = context.index.get(media_id, {})
    stats = context.stats_by_id.get(media_id) or {}
    return {
        "id": media_id,
        "name": item.get("name", ""),
        "width": item.get("width"),
        "height": item.get("height"),
        "quality": context.quality_by_id.get(media_id),
        "scores": context.scores_by_id.get(media_id, {}),
        "sharpness": stats.get("sharpness"),
        "clipping": stats.get("clipping"),
        "cleanliness": stats.get("cleanliness"),
    }


def _dup_issue(left: int, right: int, sim: float, context, source) -> Issue:
    """Build the near-duplicate issue of one pair (best image picked)."""
    ranked = sorted([left, right], key=lambda mid: _pick_score(mid, context))
    loser, best = ranked
    names = context.index
    key_ids = sorted((left, right))
    return Issue(
        key=f"dup:{key_ids[0]}:{key_ids[1]}",
        kind="near_dup",
        media_ids=(best, loser),
        names=(names[best]["name"], names[loser]["name"]),
        reason=(
            f"{names[best]['name']} and {names[loser]['name']} are the "
            f"same picture to the model — one of them is wasted steps."
        ),
        metric=f"sim {sim:.2f}",
        value=float(sim),
        impact=_BASE_IMPACT["near_dup"]
        + max(0.0, (sim - context.near_dup_cosine) * 100.0),
        fingerprint=f"{sim:.2f}",
        detail={
            "similarity": float(sim),
            "threshold": context.near_dup_cosine,
            "source": source,
            "best_id": best,
            "loser_id": loser,
            "metrics": [
                _pair_metrics(best, context),
                _pair_metrics(loser, context),
            ],
        },
    )


def _embedding_pairs(context: IssueContext) -> list:
    """Return the ``(similarity, left, right)`` pairs over the threshold."""
    result = context.map_result
    if result is None:
        return []
    pairs = []
    ids = result.ids
    matrix = result.similarity
    for i, left in enumerate(ids):
        for j in range(i + 1, len(ids)):
            sim = float(matrix[i, j])
            if sim >= context.near_dup_cosine:
                pairs.append((sim, left, ids[j]))
    return pairs


def _hash_pairs(context: IssueContext, similarity: float) -> list:
    """Return the near-duplicate pairs the perceptual hashes agree on.

    Covers the media DINOv2 never embedded (a run with the DINOv2 scorer
    off, or an image whose embedding failed): the pixel-level detector of
    :mod:`src.lookalike` still catches a re-encoded copy. Each member's
    similarity is measured against its group's representative, so a group
    of three yields two pairs, both anchored on the best image.
    """
    hashed = [
        item
        for item in context.images
        if item.get("phash") and item.get("dhash")
    ]
    if len(hashed) < 2:
        return []
    pairs = []
    for group in lookalike.detect(hashed, similarity=similarity).groups:
        best = group.media[0].media_id
        for member in group.media[1:]:
            pairs.append((member.similarity / 100.0, best, member.media_id))
    return pairs


def near_dup_issues(context: IssueContext, hash_similarity: float) -> list:
    """Return the near-duplicate issues, strongest similarity first."""
    seen: dict = {}
    for sim, left, right in _embedding_pairs(context):
        seen[tuple(sorted((left, right)))] = (sim, "DINOv2")
    for sim, left, right in _hash_pairs(context, hash_similarity):
        seen.setdefault(tuple(sorted((left, right))), (sim, "perceptual"))
    known = context.index
    issues = [
        _dup_issue(left, right, sim, context, source)
        for (left, right), (sim, source) in seen.items()
        if left in known and right in known
    ]
    issues.sort(key=lambda issue: -issue.value)
    return issues[:MAX_ISSUES_PER_KIND]


def low_quality_issues(context: IssueContext, mean: float | None) -> list:
    """Return one issue per image scoring under the quality floor."""
    floor = context.low_quality_floor
    issues = []
    for item in context.images:
        score = context.quality_by_id.get(item["id"])
        if score is None or score >= floor:
            continue
        stats = context.stats_by_id.get(item["id"]) or {}
        share = 100.0 / len(context.images) if context.images else 0.0
        issues.append(
            Issue(
                key=f"lowq:{item['id']}",
                kind="low_quality",
                media_ids=(item["id"],),
                names=(item["name"],),
                reason=(
                    f"{item['name']} scores {score:.0f} — under the "
                    f"{floor:.0f} floor; it drags the whole set down."
                ),
                metric=f"{score:.0f}",
                value=float(score),
                impact=_BASE_IMPACT["low_quality"] + (floor - score) / 2.0,
                fingerprint=f"{score:.0f}",
                detail={
                    "quality": score,
                    "mean": mean,
                    "floor": floor,
                    "scores": context.scores_by_id.get(item["id"], {}),
                    "sharpness": stats.get("sharpness"),
                    "clipping": stats.get("clipping"),
                    "cleanliness": stats.get("cleanliness"),
                    "gradient_share": share,
                },
            )
        )
    issues.sort(key=lambda issue: issue.value)
    return issues[:MAX_ISSUES_PER_KIND]


def _neighbors(media_id: int, context: IssueContext) -> list:
    """Return the ``_NEIGHBORS`` closest media of an outlier."""
    result = context.map_result
    index = result.ids.index(media_id)
    row = result.similarity[index]
    known = context.index
    ranked = sorted(
        (
            (float(row[other]), result.ids[other])
            for other in range(len(result.ids))
            if other != index
        ),
        reverse=True,
    )
    return [
        {
            "id": other_id,
            "name": known[other_id]["name"],
            "distance": 1.0 - sim,
        }
        for sim, other_id in ranked[:_NEIGHBORS]
        if other_id in known
    ]


def outlier_issues(context: IssueContext) -> list:
    """Return one issue per media whose nearest neighbors are all far."""
    result = context.map_result
    if result is None or not result.outliers:
        return []
    known = context.index
    distances = dict(zip(result.ids, result.outlier_score))
    issues = []
    for media_id in result.outliers:
        if media_id not in known:
            continue
        distance = float(distances[media_id])
        excess = distance - result.outlier_threshold
        issues.append(
            Issue(
                key=f"out:{media_id}",
                kind="outlier",
                media_ids=(media_id,),
                names=(known[media_id]["name"],),
                reason=(
                    f"{known[media_id]['name']} has no close neighbor — "
                    "it may be off-concept for this LoRA."
                ),
                metric=f"d {distance:.2f}",
                value=distance,
                impact=_BASE_IMPACT["outlier"] + min(20.0, excess * 20.0),
                fingerprint=f"{distance:.2f}",
                detail={
                    "distance": distance,
                    "threshold": result.outlier_threshold,
                    "neighbors": _neighbors(media_id, context),
                },
            )
        )
    issues.sort(key=lambda issue: -issue.value)
    return issues[:MAX_ISSUES_PER_KIND]


def caption_issues(context: IssueContext) -> list:
    """Return one issue per degenerate caption.

    A missing caption is *not* an issue here — the hygiene pillar already
    counts the coverage gap and the recommendations tell the user to fill
    it. What is flagged is a caption that exists and is broken: a looping
    n-gram, a truncated sentence, mojibake, leftover reasoning markers.
    """
    issues = []
    for item in context.images:
        text = (context.captions_by_id.get(item["id"]) or "").strip()
        if not text:
            continue
        codes = [
            issue
            for issue in caption_review.check_integrity(text)
            if issue["code"] != "empty"
        ]
        if not codes:
            continue
        loop = caption_review.repeated_phrase(text)
        issues.append(
            Issue(
                key=f"cap:{item['id']}",
                kind="caption",
                media_ids=(item["id"],),
                names=(item["name"],),
                reason=(
                    f"{item['name']}: {codes[0]['detail']}"
                    if codes
                    else item["name"]
                ),
                metric=codes[0]["code"],
                value=float(len(codes)),
                impact=_BASE_IMPACT["caption"] + 5.0 * len(codes),
                fingerprint=_short_digest(text),
                detail={
                    "text": text,
                    "codes": codes,
                    "phrase": loop[0] if loop else None,
                },
            )
        )
    issues.sort(key=lambda issue: -issue.impact)
    return issues[:MAX_ISSUES_PER_KIND]


def detect(
    context: IssueContext, hash_similarity: float, mean: float | None
) -> tuple:
    """Run every detector and return the issues sorted by impact."""
    issues = (
        near_dup_issues(context, hash_similarity)
        + low_quality_issues(context, mean)
        + outlier_issues(context)
        + caption_issues(context)
    )
    issues.sort(key=lambda issue: (-issue.impact, issue.key))
    return tuple(issues)
