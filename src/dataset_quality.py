"""Dataset quality evaluation engine (the Datasets → Quality report tab).

Answers one question about a dataset — *is this good enough to train a
LoRA?* — with a single 0-100 grade, three explainable pillars and the list
of media to act on:

* **Image quality** (weight 0.35) — the mean of the enabled IQA scorers
  (see :mod:`src.quality`) over the dataset's images, plus the blur/noise
  flags of :mod:`src.image_stats`.
* **Diversity** (weight 0.30) — the DINOv2 spread of :mod:`src.embeddings`
  (mean pairwise cosine distance, clusters, near-duplicate pairs and
  neighbor-less outliers, see :mod:`src.embedding_map`), tempered by the
  perceptual redundancy :mod:`src.lookalike` finds in the stored hashes.
* **Composition** (weight 0.15) — the Depth-Anything V2 spread of
  :mod:`src.depth_embeddings` (the style-invariant framing/pose variety):
  distinct framing clusters, the depth-signature spread and the re-skin
  pairs (same framing, different style) DINOv2 alone rates as far apart.
  Diagnostic-only when the depth index step has not run — it then carries no
  score and the remaining weights renormalize (see :func:`overall_score`).
* **Hygiene** (weight 0.20) — structural readiness: caption coverage, the
  resolution floor, degenerate captions and exact duplicates.

The flagged media themselves live in :mod:`src.dataset_issues`; the
scatter plot in :mod:`src.embedding_map`. This module assembles the three
of them into a :class:`QualityReport` and serialises it.

Pure functions over plain media dicts: the caller (the API's report
runner) fetches media, vectors, hashes and captions from the repository,
computes whatever is missing, and persists the result. No Gradio, no model
import, no database here.
"""

from dataclasses import asdict, dataclass, field

from src import dataset_issues
from src import embedding_map
from src import framing
from src import image_stats
from src import lookalike
from src import quality

# Pillar weights of the overall grade; overridden by the ``weights``
# section of ``config/*/dataset_quality.json`` and renormalized over the
# pillars that could actually be scored.
DEFAULT_WEIGHTS = {
    "quality": 0.35,
    "diversity": 0.30,
    "composition": 0.15,
    "hygiene": 0.20,
}

# The IQA scorers a run enables by default. Q-Align is a 7B VLM: powerful,
# but far too slow to be on unless the user asks for it.
DEFAULT_SCORERS = ("musiq", "topiq_nr", "laion_aes")

# The DINOv2 "scorer" chip: not an IQA metric, it toggles the whole
# diversity pillar (embeddings, clusters, map, outliers).
EMBEDDING_SCORER = "dinov2"

# Normalized quality under which an image is flagged as dragging the set
# down (the histogram's "below the floor" callout uses the same value).
LOW_QUALITY_FLOOR = 70.0

# Long-side floor (px) under which an image counts as low-resolution.
MIN_RESOLUTION_SIDE = 1024

# DINOv2 cosine similarity above which a pair is a near-duplicate.
NEAR_DUP_COSINE = 0.92

# Calibration of the spread score: a mean pairwise cosine distance at (or
# below) the floor scores 0 (everything is the same picture), at (or above)
# the ceiling scores 100. Same-subject LoRA datasets live in between.
SPREAD_FLOOR = 0.10
SPREAD_CEILING = 0.55

# How much of the diversity pillar the raw embedding spread carries; the
# rest is the perceptual-hash uniqueness (a re-encoded copy is invisible to
# a spread average but obvious to a phash).
_SPREAD_WEIGHT = 0.75
_UNIQUENESS_WEIGHT = 0.25

# Calibration of the composition score: the depth-signature spread (mean
# pairwise cosine distance over the Depth-Anything V2 vectors) mapped to
# 0-100. Depth signatures live closer together than DINOv2 features — the
# grid is coarse and style-invariant — so the floor/ceiling are tighter than
# the diversity spread's. FIRST-PASS values: recalibrate against real depth
# spreads once a few datasets have been indexed.
COMP_SPREAD_FLOOR = 0.05
COMP_SPREAD_CEILING = 0.45

# A pair is a "re-skin" — same composition, different style — when the depth
# signal reads them as close (:data:`RESKIN_DEPTH_MIN`) but DINOv2 does *not*
# already read them as a near-duplicate (:data:`NEAR_DUP_COSINE`). This is a
# touch looser than the Auto-build Studio's "strong re-skin" line (its layout
# uses ``dinoS < 0.85`` to pull nodes together): the report only wants to
# *flag* the same-composition redundancy DINOv2 misses, so it catches the
# whole band up to the near-dup line — a pair at depth 0.99 / DINOv2 0.88
# (same framing, moderately restyled) would otherwise fall through both the
# near-dup list (needs >= 0.92) and the re-skin count and go unreported.
# A pair DINOv2 already calls a near-dup (>= 0.92) stays a near-dup, never a
# re-skin, so the two categories never overlap.
RESKIN_DEPTH_MIN = 0.90
RESKIN_DINO_MAX = NEAR_DUP_COSINE

# Fallback recommended dataset size when the target type has none.
DEFAULT_SIZE_RANGE = (20, 150)

# Grade bands of the global score, descending.
GRADE_BANDS = ((90.0, "A"), (83.0, "B+"), (75.0, "B"), (65.0, "C"))
_LOWEST_GRADE = "D"

# Verdict bands of the global score, descending.
VERDICT_BANDS = (
    (80.0, "Good — ready to train"),
    (65.0, "Trainable — fixable gaps"),
)
_LOWEST_VERDICT = "Needs work before training"

# Histogram buckets of the per-media quality, ``(low, high, label)`` with
# ``high`` exclusive (the last one catches a perfect 100).
DISTRIBUTION_BUCKETS = (
    (0.0, 60.0, "<60"),
    (60.0, 70.0, "60–69"),
    (70.0, 80.0, "70–79"),
    (80.0, 90.0, "80–89"),
    (90.0, 100.01, "90–100"),
)

# How many recommendations the card shows at most.
_MAX_RECOMMENDATIONS = 5

# A framing bucket whose share deviates from its target by more than this
# many points is worth a rebalance recommendation.
_FRAMING_DRIFT = 12.0


@dataclass(frozen=True)
class Row:
    """One labelled line inside a pillar card.

    Attributes
    ----------
    label : str
        What was measured ("captioned", "near-dup pairs"…).
    value : str
        The formatted measurement, shown in mono.
    tone : str
        ``"ok"`` | ``"warn"`` | ``"danger"`` | ``"muted"`` — the colour the
        front-end paints the value with.
    """

    label: str
    value: str
    tone: str = "muted"


@dataclass(frozen=True)
class Pillar:
    """One scored pillar of the report."""

    key: str
    label: str
    score: float | None
    detail: str
    rows: tuple = ()


@dataclass(frozen=True)
class Bucket:
    """One bar of the quality histogram."""

    label: str
    count: int
    midpoint: float


@dataclass(frozen=True)
class MapPoint:  # pylint: disable=too-many-instance-attributes
    """One dot of the diversity map."""

    id: int
    name: str
    x: float
    y: float
    cluster: int
    outlier: bool
    near_dup: bool
    quality: float | None
    width: int | None
    height: int | None


@dataclass(frozen=True)
class CompositionPoint:  # pylint: disable=too-many-instance-attributes
    """One dot of the composition map (framing space, coloured by style).

    Positioned by the 2-D projection of the depth signature (so images that
    share a framing cluster together even when their style differs) and
    coloured by ``style`` — the mixing of styles inside one framing is what
    makes a re-skin legible.
    """

    id: int
    name: str
    x: float
    y: float
    framing: int
    style: str
    reskin: bool
    quality: float | None
    width: int | None
    height: int | None


@dataclass(frozen=True)
class Recommendation:
    """One concrete suggestion of the recommendations card."""

    head: str
    body: str


@dataclass(frozen=True)
class Snapshot:  # pylint: disable=too-many-instance-attributes
    """Everything a report needs, already read from the repository.

    Attributes
    ----------
    images : tuple of dict
        The dataset's image dicts (see
        :func:`src.sqlite_store.media_in_dataset`), each augmented with its
        ``phash``/``dhash`` hex strings when it was indexed.
    video_count : int
        How many dataset media are videos (counted, never analyzed).
    missing_count : int
        How many dataset media have no file on disk anymore.
    scorers : tuple of str
        The enabled IQA metric ids (plus :data:`EMBEDDING_SCORER` when the
        diversity pillar is enabled).
    vectors_by_id : dict
        ``{media_id: vector}`` DINOv2 embeddings.
    depth_vectors_by_id : dict
        ``{media_id: vector}`` Depth-Anything V2 composition signatures.
        Empty when the depth index step has not run — the composition pillar
        and map then stay diagnostic-only.
    styles_by_id : dict
        ``{media_id: style bucket}`` — the coarse hue bucket colouring each
        composition-map node (see :mod:`src.hue_bucket`). Only the media
        carrying a depth signature need one.
    stats_by_id : dict
        ``{media_id: src.image_stats.analyze(...)}``.
    captions_by_id : dict
        ``{media_id: caption text}`` for the report's caption type.
    tags_by_id : dict
        ``{media_id: [tag names]}`` for the framing distribution.
    buckets : dict
        The framing-bucket config (see
        :func:`src.config.load_autobuild_config`).
    target_ratios : dict
        The target type's ``{bucket: weight}`` ratios (may be empty).
    size_range : tuple of int
        The recommended ``(lo, hi)`` image count for the target type.
    settings : dict
        The merged ``dataset_quality.json`` config.
    """

    images: tuple = ()
    video_count: int = 0
    missing_count: int = 0
    scorers: tuple = DEFAULT_SCORERS
    vectors_by_id: dict = field(default_factory=dict)
    depth_vectors_by_id: dict = field(default_factory=dict)
    styles_by_id: dict = field(default_factory=dict)
    stats_by_id: dict = field(default_factory=dict)
    captions_by_id: dict = field(default_factory=dict)
    tags_by_id: dict = field(default_factory=dict)
    buckets: dict = field(default_factory=dict)
    target_ratios: dict = field(default_factory=dict)
    size_range: tuple = DEFAULT_SIZE_RANGE
    settings: dict = field(default_factory=dict)


@dataclass(frozen=True)
class QualityReport:  # pylint: disable=too-many-instance-attributes
    """The full outcome of a dataset evaluation."""

    total: int = 0
    images: int = 0
    videos: int = 0
    missing: int = 0
    favorites: int = 0
    overall: float | None = None
    grade: str = _LOWEST_GRADE
    verdict: str = _LOWEST_VERDICT
    summary: str = ""
    weights: dict = field(default_factory=dict)
    scorers: tuple = ()
    pillars: tuple = ()
    distribution: tuple = ()
    map_points: tuple = ()
    clusters: int = 0
    spread: float = 0.0
    composition_map: tuple = ()
    composition_links: tuple = ()
    framings: int = 0
    reskins: int = 0
    issues: tuple = ()
    recommendations: tuple = ()
    framing: tuple = ()


# --- config ---------------------------------------------------------------


def resolve_settings(config: dict | None) -> dict:
    """Return the report thresholds, config overrides applied.

    Every knob of the engine is overridable by ``dataset_quality.json``
    (see :func:`src.config.load_dataset_quality_config`); a missing key
    falls back to this module's constant.
    """
    config = config or {}
    weights = dict(DEFAULT_WEIGHTS)
    weights.update(config.get("weights") or {})
    return {
        "weights": weights,
        "low_quality_floor": float(
            config.get("low_quality_floor", LOW_QUALITY_FLOOR)
        ),
        "min_resolution_side": int(
            config.get("min_resolution_side", MIN_RESOLUTION_SIDE)
        ),
        "near_duplicate_cosine": float(
            config.get("near_duplicate_cosine", NEAR_DUP_COSINE)
        ),
        "near_duplicate_similarity": float(
            config.get(
                "near_duplicate_similarity", lookalike.DEFAULT_SIMILARITY
            )
        ),
        "outlier_sigma": float(
            config.get("outlier_sigma", embedding_map.OUTLIER_SIGMA)
        ),
    }


def recommended_size_range(config: dict | None, target_type: str) -> tuple:
    """Return the ``(lo, hi)`` image count recommended for a target type."""
    sizes = (config or {}).get("recommended_sizes") or {}
    found = sizes.get(target_type)
    if not found or len(found) != 2:
        return DEFAULT_SIZE_RANGE
    return (int(found[0]), int(found[1]))


# --- per-media quality ----------------------------------------------------


def normalized_scores(item: dict, scorers) -> dict:
    """Return an image's ``{metric_id: 0-100}`` for the enabled scorers."""
    stored = item.get("quality_scores") or {}
    normalized = {}
    for metric_id in scorers:
        if metric_id not in stored:
            continue
        value = quality.normalize_score(metric_id, stored[metric_id])
        if value is not None:
            normalized[metric_id] = value
    return normalized


def normalized_quality(item: dict, scorers) -> float | None:
    """Return an image's mean 0-100 quality over the enabled scorers.

    Parameters
    ----------
    item : dict
        A media dict carrying the ``quality_scores`` map.
    scorers : iterable of str
        The :data:`src.quality.QUALITY_METRICS` keys the run enabled.

    Returns
    -------
    float or None
        The mean normalized percentage, or None when the media carries no
        usable score for any enabled scorer.
    """
    values = list(normalized_scores(item, scorers).values())
    return sum(values) / len(values) if values else None


def scorer_label(metric_id: str) -> str:
    """Return a scorer's short label ("MUSIQ", not "MUSIQ (KonIQ-10k)")."""
    if metric_id == EMBEDDING_SCORER:
        return "DINOv2 (diversity)"
    return quality.get_metric_info(metric_id).label.split(" (")[0]


def _iqa_scorers(scorers) -> tuple:
    """Return the IQA metric ids of a scorer selection (DINOv2 dropped)."""
    return tuple(
        metric_id
        for metric_id in scorers
        if metric_id in quality.QUALITY_METRICS
    )


def _tone(score: float | None) -> str:
    """Return the colour tone of a 0-100 score (the badge scale)."""
    if score is None:
        return "muted"
    if score >= 90:
        return "info"
    if score >= 75:
        return "ok"
    if score >= 60:
        return "warn"
    return "danger"


# --- pillars --------------------------------------------------------------


def build_quality_pillar(snapshot: Snapshot, quality_by_id: dict) -> Pillar:
    """Score the image quality: the mean of the enabled IQA scorers."""
    scorers = _iqa_scorers(snapshot.scorers)
    scored = [value for value in quality_by_id.values() if value is not None]
    rows = []
    for metric_id in scorers:
        values = [
            normalized_scores(item, (metric_id,)).get(metric_id)
            for item in snapshot.images
        ]
        values = [value for value in values if value is not None]
        mean = sum(values) / len(values) if values else None
        rows.append(
            Row(
                label=scorer_label(metric_id),
                value="—" if mean is None else f"{mean:.0f}",
                tone=_tone(mean),
            )
        )
    flagged = sum(
        1
        for item in snapshot.images
        if image_stats.is_flagged(snapshot.stats_by_id.get(item["id"]))
    )
    rows.append(
        Row(
            label="blur / noise flags",
            value=str(flagged),
            tone="warn" if flagged else "ok",
        )
    )
    if not scored:
        return Pillar(
            key="quality",
            label="Image quality",
            score=None,
            detail="No stored quality score to average — run the scorers.",
            rows=tuple(rows),
        )
    mean = sum(scored) / len(scored)
    unscored = len(snapshot.images) - len(scored)
    detail = f"Mean of {len(scorers)} scorer(s) over {len(scored)} image(s)."
    if unscored:
        detail += f" {unscored} unscored image(s) left out."
    return Pillar(
        key="quality",
        label="Image quality",
        score=mean,
        detail=detail,
        rows=tuple(rows),
    )


def _uniqueness(snapshot: Snapshot, similarity: float) -> tuple:
    """Return ``(score, redundant, compared)`` for the perceptual hashes."""
    hashed = [
        item
        for item in snapshot.images
        if item.get("phash") and item.get("dhash")
    ]
    if len(hashed) < 2:
        return None, 0, 0
    groups = lookalike.detect(hashed, similarity=similarity).groups
    redundant = sum(len(group.media) - 1 for group in groups)
    return 100.0 * (1.0 - redundant / len(hashed)), redundant, len(hashed)


def spread_score(spread: float) -> float:
    """Map a mean pairwise cosine distance to a 0-100 spread score.

    The calibration of the diversity pillar, exposed because the dataset
    composer projects the very same score over ``dataset + selection``
    (see :mod:`src.dataset_compose`).
    """
    span = SPREAD_CEILING - SPREAD_FLOOR
    return max(0.0, min(100.0, (spread - SPREAD_FLOOR) / span * 100.0))


def composition_score(spread: float) -> float:
    """Map a depth-signature spread to a 0-100 composition score.

    The composition analogue of :func:`spread_score`, on its own calibration
    (:data:`COMP_SPREAD_FLOOR`/:data:`COMP_SPREAD_CEILING`): the framing/pose
    variety the depth signatures cover, higher being more varied.
    """
    span = COMP_SPREAD_CEILING - COMP_SPREAD_FLOOR
    return max(0.0, min(100.0, (spread - COMP_SPREAD_FLOOR) / span * 100.0))


def build_diversity_pillar(
    snapshot: Snapshot,
    result,
    settings: dict,
    near_dup_pairs: int,
) -> Pillar:
    """Score the visual spread and the redundancy of the dataset."""
    unique, redundant, compared = _uniqueness(
        snapshot, settings["near_duplicate_similarity"]
    )
    if result is None:
        detail = "Fewer than two embedded images — no spread to measure."
        unhashed: tuple = ()
        if compared:
            unhashed = (
                Row(
                    "perceptual duplicates",
                    f"{redundant} / {compared}",
                    "warn" if redundant else "ok",
                ),
            )
        return Pillar(
            "diversity", "Diversity (DINOv2)", None, detail, unhashed
        )
    spread = spread_score(result.spread)
    score = spread
    if unique is not None:
        score = _SPREAD_WEIGHT * spread + _UNIQUENESS_WEIGHT * unique
    rows = [
        Row("mean pairwise distance", f"{result.spread:.2f}", _tone(spread)),
        Row("clusters found", str(result.cluster_count), "muted"),
        Row(
            "near-dup pairs",
            str(near_dup_pairs),
            "warn" if near_dup_pairs else "ok",
        ),
        Row(
            "outliers",
            str(len(result.outliers)),
            "info" if result.outliers else "ok",
        ),
    ]
    if compared:
        rows.append(
            Row(
                "perceptual duplicates",
                f"{redundant} / {compared}",
                "warn" if redundant else "ok",
            )
        )
    return Pillar(
        key="diversity",
        label="Diversity (DINOv2)",
        score=score,
        detail=(
            f"Spread over {len(result.ids)} embedded image(s) — higher is "
            "more varied."
        ),
        rows=tuple(rows),
    )


def build_composition_pillar(result, reskin_count: int) -> Pillar:
    """Score the framing/pose variety of the dataset from its depth signatures.

    ``result`` is the composition map (:func:`build_composition_map`), or None
    when fewer than two media carry a depth signature — the pillar is then
    diagnostic-only (no score), which drops it from the global grade and
    renormalizes the other weights. ``reskin_count`` is the number of re-skin
    pairs (:func:`reskin_pairs`).
    """
    if result is None:
        return Pillar(
            key="composition",
            label="Composition",
            score=None,
            detail=(
                "No depth signature yet — run the Composition index "
                "(Libraries → Index) to score framing variety."
            ),
        )
    score = composition_score(result.spread)
    rows = (
        Row("distinct framings", str(result.cluster_count), "muted"),
        Row("depth spread", f"{result.spread:.2f}", _tone(score)),
        Row(
            "composition re-skins",
            str(reskin_count),
            "composition" if reskin_count else "ok",
        ),
    )
    return Pillar(
        key="composition",
        label="Composition",
        score=score,
        detail=(
            "Depth-Anything V2 depth signatures — framing & pose variety, "
            "style-invariant."
        ),
        rows=rows,
    )


def _degenerate_count(issues) -> int:
    """Return how many captions the integrity heuristics reject."""
    return sum(1 for issue in issues if issue.kind == "caption")


def _exact_duplicates(snapshot: Snapshot) -> int:
    """Return how many images share a content hash with an earlier one."""
    seen: set = set()
    duplicates = 0
    for item in snapshot.images:
        digest = item.get("sha256")
        if digest in seen:
            duplicates += 1
        elif digest:
            seen.add(digest)
    return duplicates


def build_hygiene_pillar(snapshot: Snapshot, issues, settings: dict) -> Pillar:
    """Score the structural readiness of the dataset."""
    images = snapshot.images
    if not images:
        return Pillar(
            key="hygiene",
            label="Hygiene",
            score=None,
            detail="No image in this dataset.",
        )
    total = len(images)
    min_side = settings["min_resolution_side"]
    captioned = sum(
        1
        for item in images
        if (snapshot.captions_by_id.get(item["id"]) or "").strip()
    )
    sized = [
        item for item in images if item.get("width") and item.get("height")
    ]
    big = sum(
        1 for item in sized if max(item["width"], item["height"]) >= min_side
    )
    degenerate = _degenerate_count(issues)
    duplicates = _exact_duplicates(snapshot)
    components = [
        100.0 * captioned / total,
        100.0 * (1.0 - degenerate / total),
        100.0 * (1.0 - duplicates / total),
    ]
    # An un-indexed dataset has no dimensions to check: the resolution
    # floor is *unknown*, not satisfied — crediting it 100 would inflate
    # the pillar for exactly the datasets that were never prepared.
    if sized:
        components.append(100.0 * big / len(sized))
    rows = (
        Row(
            "captioned",
            f"{captioned} / {total}",
            "ok" if captioned == total else "warn",
        ),
        Row(
            f"resolution ≥ {min_side}px",
            f"{big} / {len(sized)}" if sized else "not indexed",
            "ok" if sized and big == len(sized) else "warn",
        ),
        Row(
            "degenerate captions",
            str(degenerate),
            "danger" if degenerate else "ok",
        ),
        Row(
            "exact duplicates",
            str(duplicates),
            "danger" if duplicates else "ok",
        ),
    )
    return Pillar(
        key="hygiene",
        label="Hygiene",
        score=sum(components) / len(components),
        detail="Caption coverage, resolution floor, caption health, dupes.",
        rows=rows,
    )


# --- global score ---------------------------------------------------------


def overall_score(pillars, weights: dict) -> float | None:
    """Return the weighted 0-100 grade over the scorable pillars.

    Pillars whose score is None are skipped and the remaining weights
    renormalized; None when nothing was scorable.
    """
    scored = [
        (pillar.score, weights.get(pillar.key, 0.0))
        for pillar in pillars
        if pillar.score is not None
    ]
    total = sum(weight for _, weight in scored)
    if not scored or not total:
        return None
    return sum(score * weight for score, weight in scored) / total


def grade_of(score: float | None) -> str:
    """Return the letter grade of a global score."""
    if score is None:
        return "—"
    for floor, letter in GRADE_BANDS:
        if score >= floor:
            return letter
    return _LOWEST_GRADE


def verdict_of(score: float | None) -> str:
    """Return the one-line verdict of a global score."""
    if score is None:
        return "Not evaluated yet"
    for floor, text in VERDICT_BANDS:
        if score >= floor:
            return text
    return _LOWEST_VERDICT


def _summary(report_images: int, issues, overall: float | None) -> str:
    """Return the one-sentence summary under the global gauge."""
    if overall is None:
        return "Run an evaluation to score this dataset."
    open_issues = len(issues)
    if not open_issues:
        return (
            f"{report_images} image(s), nothing flagged — this set is as "
            "clean as the scorers can tell."
        )
    kinds = {issue.kind for issue in issues}
    named = ", ".join(sorted(kind.replace("_", " ") for kind in kinds))
    return (
        f"{open_issues} finding(s) across {report_images} image(s) "
        f"({named}) — resolve them to lift the grade."
    )


# --- distribution & map ---------------------------------------------------


def distribution(quality_by_id: dict) -> tuple:
    """Return the histogram of the per-media quality."""
    buckets = []
    for low, high, label in DISTRIBUTION_BUCKETS:
        count = sum(
            1
            for value in quality_by_id.values()
            if value is not None and low <= value < high
        )
        buckets.append(
            Bucket(
                label=label, count=count, midpoint=(low + min(high, 100)) / 2
            )
        )
    return tuple(buckets)


def map_points(snapshot: Snapshot, result, issues, quality_by_id) -> tuple:
    """Return one dot per embedded media, flagged and coloured."""
    if result is None:
        return ()
    paired = {
        media_id
        for issue in issues
        if issue.kind == "near_dup"
        for media_id in issue.media_ids
    }
    known = {item["id"]: item for item in snapshot.images}
    outliers = set(result.outliers)
    points = []
    for index, media_id in enumerate(result.ids):
        item = known.get(media_id)
        if item is None:
            continue
        x, y = result.coords[index]
        points.append(
            MapPoint(
                id=media_id,
                name=item["name"],
                x=x,
                y=y,
                cluster=result.labels[index],
                outlier=media_id in outliers,
                near_dup=media_id in paired,
                quality=quality_by_id.get(media_id),
                width=item.get("width"),
                height=item.get("height"),
            )
        )
    return tuple(points)


def build_composition_map(snapshot: Snapshot, settings: dict):
    """Return the clustered projection of the dataset's depth signatures.

    The composition analogue of :func:`build_map`: same deterministic
    projection/clustering (:func:`src.embedding_map.build`), run on the
    Depth-Anything V2 signatures instead of the DINOv2 vectors, so images that
    share a framing cluster together even when their appearance differs.
    Returns None when fewer than two media carry a depth signature.
    """
    pairs = [
        (item["id"], snapshot.depth_vectors_by_id[item["id"]])
        for item in snapshot.images
        if item["id"] in snapshot.depth_vectors_by_id
    ]
    if len(pairs) < 2:
        return None
    return embedding_map.build(
        [media_id for media_id, _ in pairs],
        [vector for _, vector in pairs],
        sigma=settings["outlier_sigma"],
    )


def reskin_pairs(dino_result, comp_result) -> list:
    """Return the re-skin ``(a, b)`` id pairs — same framing, different style.

    A pair is a re-skin when the depth signal reads it as close
    (:data:`RESKIN_DEPTH_MIN`) but DINOv2 reads it as far apart
    (:data:`RESKIN_DINO_MAX`) — a redundancy DINOv2 alone would miss. Scans
    only the media carrying *both* an appearance and a composition vector,
    reading each cosine from the respective similarity matrix.
    """
    if dino_result is None or comp_result is None:
        return []
    dino_index = {mid: row for row, mid in enumerate(dino_result.ids)}
    pairs = []
    ids = comp_result.ids
    for first, left in enumerate(ids):
        if left not in dino_index:
            continue
        for offset, right in enumerate(ids[first + 1 :], start=first + 1):
            if right not in dino_index:
                continue
            depth_sim = float(comp_result.similarity[first, offset])
            dino_sim = float(
                dino_result.similarity[dino_index[left], dino_index[right]]
            )
            if depth_sim >= RESKIN_DEPTH_MIN and dino_sim < RESKIN_DINO_MAX:
                pairs.append((left, right))
    return pairs


def composition_points(
    snapshot: Snapshot, result, reskin_ids: set, quality_by_id: dict
) -> tuple:
    """Return one composition-map dot per depth-embedded media."""
    if result is None:
        return ()
    known = {item["id"]: item for item in snapshot.images}
    points = []
    for index, media_id in enumerate(result.ids):
        item = known.get(media_id)
        if item is None:
            continue
        x, y = result.coords[index]
        points.append(
            CompositionPoint(
                id=media_id,
                name=item["name"],
                x=x,
                y=y,
                framing=result.labels[index],
                style=snapshot.styles_by_id.get(media_id, "neutral"),
                reskin=media_id in reskin_ids,
                quality=quality_by_id.get(media_id),
                width=item.get("width"),
                height=item.get("height"),
            )
        )
    return tuple(points)


def framing_rows(snapshot: Snapshot) -> tuple:
    """Return the ``(bucket, count, share, target_share)`` framing rows."""
    images = snapshot.images
    buckets = snapshot.buckets
    ratios = snapshot.target_ratios
    if not images or not buckets:
        return ()
    counts: dict = {}
    for item in images:
        bucket = framing.classify(
            snapshot.tags_by_id.get(item["id"], ()), buckets
        )
        counts[bucket] = counts.get(bucket, 0) + 1
    ratio_total = sum(
        value for key, value in ratios.items() if key != framing.ANY_BUCKET
    )
    rows = []
    for key in list(buckets) + [framing.UNKNOWN_BUCKET]:
        count = counts.get(key, 0)
        if not count and key not in ratios:
            continue
        target = (
            100.0 * ratios[key] / ratio_total
            if key in ratios and ratio_total
            else None
        )
        rows.append((key, count, 100.0 * count / len(images), target))
    return tuple(rows)


# --- recommendations ------------------------------------------------------


def _projected_score(quality_by_id, pillars, weights, floor):
    """Return the global score once every below-floor image is pruned.

    Only the quality pillar moves: pruning the floor raises the mean it
    averages. The diversity and hygiene pillars are left as measured —
    an honest lower bound on what the pruning buys.
    """
    scored = [v for v in quality_by_id.values() if v is not None]
    kept = [value for value in scored if value >= floor]
    if not kept or len(kept) == len(scored):
        return None
    pruned = tuple(
        Pillar(
            key=pillar.key,
            label=pillar.label,
            score=(
                sum(kept) / len(kept)
                if pillar.key == "quality"
                else pillar.score
            ),
            detail=pillar.detail,
            rows=pillar.rows,
        )
        for pillar in pillars
    )
    return overall_score(pruned, weights)


def _framing_advice(rows) -> Recommendation | None:
    """Suggest a framing rebalance when a bucket drifts from its target."""
    drifted = [
        (bucket, share, target)
        for bucket, _, share, target in rows
        if target is not None and abs(share - target) > _FRAMING_DRIFT
    ]
    if not drifted:
        return None
    bucket, share, target = max(drifted, key=lambda row: abs(row[1] - row[2]))
    direction = "over" if share > target else "under"
    return Recommendation(
        head="Rebalance framing",
        body=(
            f"{bucket} is {direction}-represented ({share:.0f}% vs a "
            f"{target:.0f}% target) — add or prune media until the shares "
            "match the target type."
        ),
    )


def _pruning_advice(issues, projected) -> Recommendation | None:
    """Suggest pruning the low-quality floor, with the projected score."""
    low = [issue for issue in issues if issue.kind == "low_quality"]
    if not low:
        return None
    gain = (
        ""
        if projected is None
        else f" — the grade would reach {projected:.0f}"
    )
    return Recommendation(
        head="Prune the floor",
        body=(
            f"{len(low)} image(s) score under the quality floor{gain}. "
            "Remove them from the dataset, or recapture them."
        ),
    )


def _dup_advice(issues) -> Recommendation | None:
    """Suggest breaking the near-duplicate pairs, naming the keeper."""
    dups = [issue for issue in issues if issue.kind == "near_dup"]
    if not dups:
        return None
    first = dups[0].detail.get("metrics") or [{}]
    keeper = first[0].get("name", "the sharper one")
    return Recommendation(
        head="Break near-dup pairs",
        body=(
            f"{len(dups)} pair(s) teach the model the same picture twice. "
            f"Keep the best of each (e.g. {keeper}) and unlink the other."
        ),
    )


def _caption_advice(snapshot: Snapshot, issues) -> Recommendation | None:
    """Suggest filling the caption gaps and fixing the degenerate ones."""
    missing = sum(
        1
        for item in snapshot.images
        if not (snapshot.captions_by_id.get(item["id"]) or "").strip()
    )
    broken = sum(1 for issue in issues if issue.kind == "caption")
    if not missing and not broken:
        return None
    parts = []
    if missing:
        parts.append(f"{missing} image(s) have no caption")
    if broken:
        parts.append(f"{broken} caption(s) are degenerate")
    return Recommendation(
        head="Fill the caption gaps",
        body=(
            f"{' and '.join(parts)}. Every training image needs one clean "
            "caption — regenerate them before deploying."
        ),
    )


def _size_advice(snapshot: Snapshot) -> Recommendation | None:
    """Suggest growing or trimming the set against the target range."""
    low, high = snapshot.size_range
    count = len(snapshot.images)
    if count < low:
        return Recommendation(
            head="Grow the set",
            body=(
                f"{count} image(s) — below the ~{low}-{high} usually "
                "recommended for this target type."
            ),
        )
    if count > high:
        return Recommendation(
            head="Curation beats volume",
            body=(
                f"{count} images — above the ~{low}-{high} sweet spot; "
                "pruning the weakest usually trains better than adding."
            ),
        )
    return None


def recommendations(snapshot, issues, rows, projected) -> tuple:
    """Return the concrete suggestions of the recommendations card."""
    found = [
        _dup_advice(issues),
        _pruning_advice(issues, projected),
        _caption_advice(snapshot, issues),
        _framing_advice(rows),
        _size_advice(snapshot),
    ]
    return tuple(item for item in found if item)[:_MAX_RECOMMENDATIONS]


# --- entry point ----------------------------------------------------------


def _issue_context(snapshot, quality_by_id, result, settings):
    """Build the detector context from a snapshot and its computed map."""
    return dataset_issues.IssueContext(
        images=tuple(snapshot.images),
        quality_by_id=quality_by_id,
        scores_by_id={
            item["id"]: normalized_scores(item, _iqa_scorers(snapshot.scorers))
            for item in snapshot.images
        },
        stats_by_id=snapshot.stats_by_id,
        captions_by_id=snapshot.captions_by_id,
        map_result=result,
        near_dup_cosine=settings["near_duplicate_cosine"],
        low_quality_floor=settings["low_quality_floor"],
    )


def build_map(snapshot: Snapshot, settings: dict):
    """Return the clustered projection of the dataset's embeddings."""
    if EMBEDDING_SCORER not in snapshot.scorers:
        return None
    pairs = [
        (item["id"], snapshot.vectors_by_id[item["id"]])
        for item in snapshot.images
        if item["id"] in snapshot.vectors_by_id
    ]
    if len(pairs) < 2:
        return None
    return embedding_map.build(
        [media_id for media_id, _ in pairs],
        [vector for _, vector in pairs],
        sigma=settings["outlier_sigma"],
    )


def evaluate(snapshot: Snapshot) -> QualityReport:
    """Run every pillar and assemble the dataset's quality report."""
    settings = resolve_settings(snapshot.settings)
    scorers = _iqa_scorers(snapshot.scorers)
    quality_by_id = {
        item["id"]: normalized_quality(item, scorers)
        for item in snapshot.images
    }
    result = build_map(snapshot, settings)
    comp_result = build_composition_map(snapshot, settings)
    reskins = reskin_pairs(result, comp_result)
    reskin_ids = {media_id for pair in reskins for media_id in pair}
    context = _issue_context(snapshot, quality_by_id, result, settings)
    scored = [v for v in quality_by_id.values() if v is not None]
    mean = sum(scored) / len(scored) if scored else None
    issues = dataset_issues.detect(
        context, settings["near_duplicate_similarity"], mean
    )
    pairs = sum(1 for issue in issues if issue.kind == "near_dup")
    pillars = (
        build_quality_pillar(snapshot, quality_by_id),
        build_diversity_pillar(snapshot, result, settings, pairs),
        build_composition_pillar(comp_result, len(reskins)),
        build_hygiene_pillar(snapshot, issues, settings),
    )
    weights = settings["weights"]
    overall = overall_score(pillars, weights)
    rows = framing_rows(snapshot)
    projected = _projected_score(
        quality_by_id, pillars, weights, settings["low_quality_floor"]
    )
    return QualityReport(
        total=len(snapshot.images) + snapshot.video_count,
        images=len(snapshot.images),
        videos=snapshot.video_count,
        missing=snapshot.missing_count,
        favorites=sum(1 for item in snapshot.images if item.get("favorite")),
        overall=overall,
        grade=grade_of(overall),
        verdict=verdict_of(overall),
        summary=_summary(len(snapshot.images), issues, overall),
        weights=weights,
        scorers=tuple(snapshot.scorers),
        pillars=pillars,
        distribution=distribution(quality_by_id),
        map_points=map_points(snapshot, result, issues, quality_by_id),
        clusters=result.cluster_count if result else 0,
        spread=result.spread if result else 0.0,
        composition_map=composition_points(
            snapshot, comp_result, reskin_ids, quality_by_id
        ),
        composition_links=tuple(reskins),
        framings=comp_result.cluster_count if comp_result else 0,
        reskins=len(reskins),
        issues=issues,
        recommendations=recommendations(snapshot, issues, rows, projected),
        framing=rows,
    )


def to_dict(report: QualityReport) -> dict:
    """Return the report as a JSON-serialisable dict (the stored blob)."""
    payload = asdict(report)
    payload["framing"] = [list(row) for row in report.framing]
    payload["composition_links"] = [
        list(pair) for pair in report.composition_links
    ]
    return payload
