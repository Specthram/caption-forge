"""Auto-build Studio engine: a live, explainable dataset proposal.

The Studio (Datasets → "Auto-build a dataset…") proposes a whole training
set from a library and explains *why* every picture was picked. It is the
dataset composer (:mod:`src.dataset_compose`) run in reverse: instead of
the user hand-picking against an existing dataset, the engine picks against
an empty one from a *recipe* (subject, quality, framing, size), and the
user then edits the proposal — drop a pick, swap it for a visual neighbour,
or force one in.

Nothing new is computed here. Every signal comes from an engine the app
already runs at index time:

* the IQA quality of :mod:`src.quality` (the ``Q`` reason and the quality
  pillar),
* the DINOv2 vectors of :mod:`src.embeddings` (the diversity gain, the
  coverage map, the uncovered zones and the clusters, via
  :mod:`src.dataset_compose` / :mod:`src.embedding_map`),
* the perceptual hashes of :mod:`src.lookalike` (the near-duplicate skip),
* the SigLIP 2 relevance of :mod:`src.siglip_grounding` (the ``⌕`` subject
  match), the WD14 tags of :mod:`src.tagger` (the ``#`` locked-tag filter
  and the framing buckets of :mod:`src.framing`),
* the grade bands and pillar weights of :mod:`src.dataset_quality`.

A candidate's **subject match** is the mean of the recipe's active ranking
signals (semantic query, example seeds); locked tags are a hard include
filter, not a ranking signal. The **selection score** is a composite of
quality, diversity gain and subject match. Pure functions over
plain dicts: the caller reads the pool, the vectors, the tags and the
recipe from the repository and passes them in (see the ``autobuild``
runner).
"""

from dataclasses import dataclass, field

import numpy as np

from src import dataset_compose, dataset_quality, embedding_map, framing
from src import image_stats, quality
from src.media import is_video_file

# --- selection score ------------------------------------------------------
# The composite that ranks an eligible candidate, per the design handoff.
# With at least one subject signal active the three terms share the weight;
# with none (pure-diversity build) quality and diversity split it alone.
SUBJECT_WEIGHTS = (0.32, 0.40, 0.28)  # quality, diversity, subject
NEUTRAL_WEIGHTS = (0.45, 0.55)  # quality, diversity

# A candidate whose subject match falls below this never enters a
# subject-driven build: it simply is not what the user asked for.
ELIGIBILITY_GATE = 0.24

# The diversity gain saturates at this cosine distance from the selection —
# the ceiling the spread score and the composer's gain both calibrate on.
GAIN_CEILING = dataset_compose.GAIN_CEILING

# A candidate this close (DINOv2 cosine) to an already-picked media is a
# near-duplicate and is skipped, unless the pool leaves no alternative.
NEAR_DUP_COSINE = dataset_quality.NEAR_DUP_COSINE

# A grid cell holding at least this many candidates but no pick is drawn as
# an uncovered zone of the corpus.
ZONE_MIN_CANDIDATES = 3

# --- proximity graph ------------------------------------------------------
# The Proximity view materialises the resemblance links between the picks.
# Only pairs at or above this cosine floor are sent (kNN-style sparsity);
# the view then lets the user raise the threshold further, up to just below
# 1.0. The app-wide near-duplicate line (:data:`NEAR_DUP_COSINE`, 0.92) is
# the red "near-identical" band inside that range.
PROXIMITY_FLOOR = 0.70

# The composition (depth) similarity is fused with the DINOv2 cosine as
# ``max(dinoS, COMP_W · depthS)`` — depth is trusted a touch less than the
# appearance signal, so its weight is just under 1. A pair reaches the
# Proximity floor (and is materialised) on either signal, which is how a
# re-skin — far in DINOv2, close in composition — becomes a visible edge.
COMP_W = 0.95

# A pick is flagged "borderline" when its subject match, once a subject is
# active, sits within this band of the gate — the triage queue's material.
FLAG_SUBJECT_MARGIN = 0.12

# How many dominant tags a cluster or an uncovered zone names.
TOP_TAGS = 3

# --- rebalance ------------------------------------------------------------
# The "⟲ Rebalance" re-pick penalises candidates sitting in an area the
# selection already crowds. A candidate's *near* count is how many picks
# fall within :data:`REBALANCE_RADIUS` (a cosine distance) of it; once more
# than :data:`REBALANCE_MIN_SHARE` of the picks are that close, its score is
# docked by :data:`REBALANCE_WEIGHT` × the excess. Mirrors the prototype's
# ``−0.85 × max(0, near/picked − 0.22)``; only engages past a few picks so
# the very first ones are not penalised against an empty selection.
REBALANCE_RADIUS = GAIN_CEILING
REBALANCE_MIN_SHARE = 0.22
REBALANCE_WEIGHT = 0.85
REBALANCE_MIN_PICKS = 4

# A cluster tag is "distinctive" when its share inside the cluster exceeds
# its share across the whole selection by at least this much — the fix for
# clusters labelled by the tags every pick shares.
CLUSTER_TAG_LIFT = 0.08


@dataclass(frozen=True)
class Recipe:  # pylint: disable=too-many-instance-attributes
    """Every parameter of one Studio proposal.

    Saved with the dataset it creates (a "living" recipe re-runs after each
    index); the same recipe over the same library always proposes the same
    set. ``dropped``/``forced``/``kept`` are the user's manual edits,
    replayed on every recompute.
    """

    media_type: str = dataset_compose.MEDIA_TYPE_IMAGE
    semantic_q: str = ""
    locked_tags: tuple = ()  # hard include filter: media must carry all
    exclude_tags: tuple = ()  # hard filter: media carrying any are cut
    seed_media_ids: tuple = ()
    library_ids: tuple = ()  # corpus scope; empty = every library
    size: int = 50
    metric: str = quality.DEFAULT_METRIC
    min_score: float = 60.0
    exclude_blur: bool = True
    framing_preset: str = "balanced"
    ratios: dict = field(default_factory=dict)
    buckets: dict = field(default_factory=dict)
    live: bool = True
    dropped: tuple = ()
    forced: tuple = ()
    kept: tuple = ()
    rebal: bool = False  # re-pick with a density penalty on crowded areas


@dataclass
class Candidate:  # pylint: disable=too-many-instance-attributes
    """One pool media, annotated for the selection and the map."""

    id: int
    name: str
    favorite: bool
    is_video: bool
    width: int | None
    height: int | None
    quality: float | None  # 0-100 for the recipe's metric, or None
    bucket: str
    tags: tuple
    subject: float  # 0-1 mean of the active subject signals
    signals: dict  # {"semantic"|"locked"|"seed": 0-1} of the active ones
    eligible: bool
    excluded: str  # "" when eligible, else why it was dropped
    has_vector: bool


# --- subject signals ------------------------------------------------------


def _rank_normalize(relevance: dict) -> dict:
    """Return ``{id: 0-1}`` from raw cosines — best match 1, worst 0.

    An absolute SigLIP cosine is not comparable across queries, but its
    rank within one query is (see :mod:`src.dataset_compose`). The best
    match scores 1, the worst 0, ties share the mean rank.
    """
    if not relevance:
        return {}
    order = sorted(relevance, key=lambda mid: relevance[mid])
    if len(order) == 1:
        return {order[0]: 1.0}
    return {mid: index / (len(order) - 1) for index, mid in enumerate(order)}


def _seed_proximity(vectors: dict, seed_ids) -> dict:
    """Return ``{id: 0-1}`` cosine proximity to the nearest example seed."""
    seeds = [vectors[sid] for sid in seed_ids if sid in vectors]
    if not seeds:
        return {}
    matrix = np.stack(seeds)
    proximity = {}
    for media_id, vector in vectors.items():
        best = float(np.max(matrix @ vector))
        proximity[media_id] = max(0.0, min(1.0, best))
    return proximity


def subject_matches(recipe: Recipe, ids, names_by_id, vectors, relevance):
    """Return ``{id: (match, signals)}`` — the subject match of each media.

    The match is the mean of the recipe's *active ranking* signals: the
    semantic rank (when a query is typed and SigLIP vectors exist) and the
    proximity to the nearest seed. Locked tags are a hard pool filter
    (applied upstream, see :func:`server.runners.autobuild`), not a ranking
    signal — they are recorded in ``signals`` only for the ``#`` reason chip
    and the trigger-word hint. A media with no active ranking signal returns
    ``(0.0, signals)`` and the build falls back to the neutral (quality +
    diversity) score. ``names_by_id`` maps a media id to its WD14 tag names.
    """
    semantic = _rank_normalize(relevance) if relevance else {}
    seed = _seed_proximity(vectors, recipe.seed_media_ids)
    locked = {framing.normalize_tag(name) for name in recipe.locked_tags}
    result = {}
    for media_id in ids:
        signals = {}
        ranking = []
        if recipe.semantic_q.strip() and media_id in semantic:
            signals["semantic"] = semantic[media_id]
            ranking.append(semantic[media_id])
        if recipe.seed_media_ids and media_id in seed:
            signals["seed"] = seed[media_id]
            ranking.append(seed[media_id])
        if locked:
            present = {
                framing.normalize_tag(name)
                for name in names_by_id.get(media_id, ())
            }
            signals["locked"] = len(locked & present) / len(locked)
        match = sum(ranking) / len(ranking) if ranking else 0.0
        result[media_id] = (match, signals)
    return result


def subject_active(recipe: Recipe) -> bool:
    """Return whether the recipe carries a ranking subject signal.

    Locked tags do not count: they are a hard pool filter, not a soft
    ranking signal, so a tags-only recipe builds on quality + diversity
    over the media that already carry every locked tag.
    """
    return bool(recipe.semantic_q.strip() or recipe.seed_media_ids)


# --- candidate preparation ------------------------------------------------


def _excluded_reason(item, recipe: Recipe, score, subject, active) -> str:
    """Return why a candidate is ineligible, or "" when it can be picked."""
    if recipe.min_score > 0 and score is not None and score < recipe.min_score:
        return "quality"
    stats = item.get("stats")
    if recipe.exclude_blur and stats:
        if stats["sharpness"] < image_stats.BLUR_FLOOR:
            return "blur"
    if active and subject < ELIGIBILITY_GATE:
        return "subject"
    return ""


def prepare(recipe: Recipe, pool, tags_by_id, vectors, relevance):
    """Return the annotated :class:`Candidate` list of one pool.

    Classifies every pool media's framing bucket, scores its subject match
    and decides whether it is eligible for the selection (a media dropped
    by the quality floor, the blur filter or the subject gate stays in the
    pool for the map, marked with why it was excluded).
    """
    ids = [item["id"] for item in pool]
    active = subject_active(recipe)
    # The store hands tags back as ``{id: [{"name": ...}]}`` rows; the
    # subject signal and the framing classifier both want bare names.
    names_by_id = {
        media_id: [row["name"] for row in tags_by_id.get(media_id, ())]
        for media_id in ids
    }
    matches = subject_matches(recipe, ids, names_by_id, vectors, relevance)
    candidates = []
    for item in pool:
        media_id = item["id"]
        names = names_by_id[media_id]
        score = dataset_compose.metric_score(item, recipe.metric)
        subject, signals = matches[media_id]
        excluded = _excluded_reason(item, recipe, score, subject, active)
        candidates.append(
            Candidate(
                id=media_id,
                name=item["name"],
                favorite=bool(item.get("favorite")),
                is_video=is_video_file(f"x.{item['file_extension']}"),
                width=item.get("width"),
                height=item.get("height"),
                quality=score,
                bucket=framing.classify(names, recipe.buckets),
                tags=tuple(names),
                subject=subject,
                signals=signals,
                eligible=not excluded,
                excluded=excluded,
                has_vector=media_id in vectors,
            )
        )
    return candidates


# --- the greedy, subject-aware selector -----------------------------------


class _Picker:  # pylint: disable=too-many-instance-attributes
    """Incremental farthest-point picker with a subject term.

    Mirrors :class:`src.dataset_builder._GreedyPicker` — one matrix-vector
    product per pick keeps a run linear — but the composite it maximizes
    folds in the precomputed subject match, and a pick that near-duplicates
    the selection is skipped unless nothing else is eligible.
    """

    def __init__(self, corpus, candidates, active: bool, rebal: bool = False):
        self._corpus = corpus
        self._candidates = candidates
        self._active = active
        self._rebal = rebal
        self._by_id = {cand.id: index for index, cand in enumerate(candidates)}
        vectors, has = self._matrix(corpus, candidates)
        self._vectors = vectors
        self._has = has
        self._min_dist = np.where(has, 1.0, 0.0)
        self._nearest_cos = np.zeros(len(candidates), dtype=np.float32)
        self._near = np.zeros(len(candidates), dtype=np.float32)
        self._taken = np.zeros(len(candidates), dtype=bool)
        self._n_taken = 0

    @staticmethod
    def _matrix(corpus, candidates):
        """Return the ``(vectors, has_vector)`` arrays of the candidates."""
        dim = next(
            (len(v) for v in corpus.vectors.values()),
            1,
        )
        vectors = np.zeros((len(candidates), dim), dtype=np.float32)
        has = np.zeros(len(candidates), dtype=bool)
        for row, cand in enumerate(candidates):
            vector = corpus.vectors.get(cand.id)
            if vector is not None:
                vectors[row] = vector
                has[row] = True
        return vectors, has

    def _fold(self, row: int) -> None:
        """Fold a newly-picked vector into the running distances."""
        self._n_taken += 1
        if not self._has[row]:
            return
        cos = self._vectors @ self._vectors[row]
        distance = np.clip(1.0 - cos, 0.0, None)
        self._min_dist = np.minimum(
            self._min_dist, np.where(self._has, distance, 0.0)
        )
        self._nearest_cos = np.maximum(
            self._nearest_cos, np.where(self._has, cos, 0.0)
        )
        self._near += self._has & (distance < REBALANCE_RADIUS)

    def take(self, media_id: int) -> None:
        """Mark a media picked (a forced/favorite pick outside the loop)."""
        row = self._by_id.get(media_id)
        if row is None or self._taken[row]:
            return
        self._taken[row] = True
        self._fold(row)

    def gain(self, media_id: int) -> float:
        """Return a candidate's current diversity gain, 0-1."""
        row = self._by_id.get(media_id)
        if row is None:
            return 0.0
        return float(min(1.0, self._min_dist[row] / GAIN_CEILING))

    def _is_near_dup(self, row: int) -> bool:
        """Return whether a row near-duplicates the current selection."""
        cand = self._candidates[row]
        twins = self._corpus.hash_twins.get(cand.id, set())
        if any(self._taken[self._by_id[t]] for t in twins if t in self._by_id):
            return True
        return (
            bool(self._has[row]) and self._nearest_cos[row] >= NEAR_DUP_COSINE
        )

    def _score(self, row: int) -> float:
        """Return the composite selection score of a candidate row."""
        cand = self._candidates[row]
        grade = (cand.quality or 0.0) / 100.0
        gain = float(min(1.0, self._min_dist[row] / GAIN_CEILING))
        if self._active:
            weight_q, weight_d, weight_s = SUBJECT_WEIGHTS
            base = weight_q * grade + weight_d * gain + weight_s * cand.subject
        else:
            weight_q, weight_d = NEUTRAL_WEIGHTS
            base = weight_q * grade + weight_d * gain
        return base + self._rebalance_penalty(row)

    def _rebalance_penalty(self, row: int) -> float:
        """Return the density penalty docking a crowded candidate (≤ 0).

        Off unless the recipe asked to rebalance and enough picks are down;
        then a candidate whose *near* share exceeds
        :data:`REBALANCE_MIN_SHARE` is docked in proportion to the excess.
        """
        if not self._rebal or self._n_taken < REBALANCE_MIN_PICKS:
            return 0.0
        near_share = float(self._near[row]) / self._n_taken
        return -REBALANCE_WEIGHT * max(0.0, near_share - REBALANCE_MIN_SHARE)

    def pick(self, rows, allow_near_dup: bool) -> int | None:
        """Pick the best untaken row, skipping near-dups unless allowed."""
        best_row = None
        best_key = None
        for row in rows:
            if self._taken[row]:
                continue
            if not allow_near_dup and self._is_near_dup(row):
                continue
            cand = self._candidates[row]
            key = (
                self._score(row),
                cand.quality is not None,
                cand.quality or 0.0,
                -cand.id,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_row = row
        if best_row is None:
            return None
        self._taken[best_row] = True
        self._fold(best_row)
        return best_row

    def row_of(self, media_id: int) -> int | None:
        """Return the picker row of a media id, or None."""
        return self._by_id.get(media_id)


def _quotas(size: int, ratios: dict) -> dict:
    """Split ``size`` into per-bucket quotas honoring the ratio weights.

    Only positive weights count; rounding leftovers go to the largest
    fractional parts (ties to declaration order). Mirrors the split of
    :mod:`src.dataset_builder` — the two selectors weight framing the same.
    """
    active = {k: float(v) for k, v in ratios.items() if float(v or 0) > 0}
    total = sum(active.values())
    if not active or total <= 0 or size <= 0:
        return {}
    shares = {k: size * v / total for k, v in active.items()}
    quotas = {k: int(share) for k, share in shares.items()}
    leftover = size - sum(quotas.values())
    by_fraction = sorted(
        active, key=lambda k: shares[k] - quotas[k], reverse=True
    )
    for key in by_fraction[:leftover]:
        quotas[key] += 1
    return quotas


def select(corpus, candidates, recipe: Recipe):
    """Return the ordered picks of one recipe over prepared candidates.

    Order: the user's forced ids, then the guaranteed favorites, then a
    greedy farthest-point fill per framing bucket (honoring the preset
    ratios) that skips near-duplicates, then a redistribution pass over
    every eligible candidate. Dropped ids never enter; kept ids are picked
    normally but are exempt from the borderline flag downstream.

    Returns
    -------
    tuple of (list, dict)
        The picked ids in order, and ``{id: {"gain", "near_dup"}}`` — the
        per-pick diversity gain at pick time and whether the pick had to
        break the near-duplicate rule (an empty pool fallback).
    """
    active = subject_active(recipe)
    dropped = set(recipe.dropped)
    forced = [mid for mid in recipe.forced if mid not in dropped]
    by_id = {cand.id: cand for cand in candidates}
    eligible = {
        cand.id
        for cand in candidates
        if cand.eligible and cand.id not in dropped
    }
    picker = _Picker(corpus, candidates, active, recipe.rebal)
    picks: list = []
    meta: dict = {}

    def commit(media_id: int, near_dup: bool) -> None:
        meta[media_id] = {"gain": picker.gain(media_id), "near_dup": near_dup}
        picker.take(media_id)
        picks.append(media_id)

    for media_id in forced:
        if media_id in by_id and media_id not in meta:
            commit(media_id, near_dup=False)
    favorites = sorted(
        (
            cand
            for cand in candidates
            if cand.favorite and cand.id in eligible and cand.id not in meta
        ),
        key=lambda c: (c.quality is not None, c.quality or 0.0, -c.id),
        reverse=True,
    )
    for cand in favorites[: max(0, recipe.size - len(picks))]:
        commit(cand.id, near_dup=False)

    quotas = _quotas(recipe.size, recipe.ratios)
    for key, quota in quotas.items():
        rows = _bucket_rows(picker, candidates, eligible, key)
        taken_here = 0
        while len(picks) < recipe.size and taken_here < quota:
            row = picker.pick(rows, allow_near_dup=False)
            if row is None:
                break
            commit(candidates[row].id, near_dup=False)
            taken_here += 1

    all_rows = _bucket_rows(picker, candidates, eligible, framing.ANY_BUCKET)
    for allow in (False, True):
        while len(picks) < recipe.size:
            row = picker.pick(all_rows, allow_near_dup=allow)
            if row is None:
                break
            commit(candidates[row].id, near_dup=allow)
    return picks, meta


def _bucket_rows(picker, candidates, eligible, key) -> list:
    """Return the picker rows eligible for one framing/any bucket."""
    rows = []
    for cand in candidates:
        if cand.id not in eligible:
            continue
        if key not in (framing.ANY_BUCKET, cand.bucket):
            continue
        row = picker.row_of(cand.id)
        if row is not None:
            rows.append(row)
    return rows


# --- per-pick reasons and flags -------------------------------------------


def reasons_for(cand: Candidate, meta: dict, recipe: Recipe) -> list:
    """Return the reason chips shown under a pick's name.

    Each chip is ``{"icon", "label", "title"}``: the dominant selection
    driver first (diversity or subject), then quality, then the manual/
    favorite markers — exactly the chips the design handoff lists.
    """
    chips = []
    gain = meta.get(cand.id, {}).get("gain", 0.0)
    if cand.id in recipe.forced:
        chips.append(_chip("⇄", "manual", "Forced in by you"))
    if cand.favorite:
        chips.append(_chip("♥", "favorite", "A guaranteed favorite"))
    if gain >= 0.5:
        chips.append(
            _chip("◆", "diversity", f"Adds visual variety (gain {gain:.2f})")
        )
    if "semantic" in cand.signals:
        value = cand.signals["semantic"]
        chips.append(
            _chip("⌕", f"{value:.2f}", "Matches the semantic query (SigLIP2)")
        )
    if cand.signals.get("locked"):
        hit = next(
            (
                name
                for name in cand.tags
                if framing.normalize_tag(name)
                in {framing.normalize_tag(t) for t in recipe.locked_tags}
            ),
            "tag",
        )
        chips.append(_chip("#", hit, "Carries a required tag"))
    if cand.signals.get("seed"):
        chips.append(_chip("◈", "seed", "Close to one of your example seeds"))
    if cand.quality is not None:
        chips.append(
            _chip("Q", f"{cand.quality:.0f}", "IQA quality for the metric")
        )
    return chips


def _chip(icon: str, label: str, title: str) -> dict:
    """Return one reason chip payload."""
    return {"icon": icon, "label": label, "title": title}


def flag_for(cand: Candidate, meta: dict, recipe: Recipe) -> dict | None:
    """Return a pick's borderline flag, or None when it is a clean pick.

    A pick is borderline when it broke the near-duplicate rule (the pool
    forced it), when its quality sits just above the floor, or when its
    subject match barely cleared the gate — the triage queue works these.
    """
    if cand.id in recipe.kept or cand.id in recipe.forced:
        return None
    info = meta.get(cand.id, {})
    if info.get("near_dup"):
        return {"kind": "near_dup", "why": "Near-duplicate of another pick"}
    if (
        subject_active(recipe)
        and cand.subject < ELIGIBILITY_GATE + FLAG_SUBJECT_MARGIN
    ):
        return {"kind": "subject", "why": "Weak subject match"}
    if (
        recipe.min_score > 0
        and cand.quality is not None
        and cand.quality < recipe.min_score + 8
    ):
        return {"kind": "quality", "why": "Quality just above the floor"}
    return None


# --- clusters and uncovered zones -----------------------------------------

# The palette the front-end paints clusters with (mirrors the report map).
CLUSTER_COLORS = (
    "#e8935a",
    "#6fa8dc",
    "#8bc48a",
    "#e0b356",
    "#c58ad0",
    "#5ac7c0",
)


def _top_tags(candidates) -> list:
    """Return the most frequent tag names of a group of candidates."""
    counts: dict = {}
    for cand in candidates:
        for name in cand.tags:
            if name in framing.ANY_BUCKET:
                continue
            counts[name] = counts.get(name, 0) + 1
    ranked = sorted(counts, key=lambda name: (-counts[name], name))
    return ranked[:TOP_TAGS]


def _tag_shares(candidates) -> dict:
    """Return ``{tag: fraction of candidates carrying it}`` (0-1)."""
    total = len(candidates)
    if not total:
        return {}
    counts: dict = {}
    for cand in candidates:
        for name in set(cand.tags):
            counts[name] = counts.get(name, 0) + 1
    return {name: count / total for name, count in counts.items()}


def _distinctive_tags(members, overall: dict, locked: set) -> list:
    """Return a cluster's over-represented tags vs the whole selection.

    A tag is distinctive when its share inside the cluster exceeds its share
    across every pick by more than :data:`CLUSTER_TAG_LIFT`; locked tags
    (shared by construction) never qualify. Sorted by that lift, top
    :data:`TOP_TAGS`. This is what separates one visual sub-family from the
    next — never the tags every pick shares.
    """
    inside = _tag_shares(members)
    ranked = sorted(
        (
            (name, share - overall.get(name, 0.0))
            for name, share in inside.items()
            if framing.normalize_tag(name) not in locked
        ),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return [name for name, lift in ranked if lift > CLUSTER_TAG_LIFT][
        :TOP_TAGS
    ]


def clusters(corpus, picks, recipe: Recipe) -> list:
    """Return the visual sub-families of the *selection* (not the pool).

    k-means over the picks' DINOv2 vectors — k auto (:func:`suggested_k`) —
    each group a colour, a distinctive-tag label, its share and members,
    sorted largest first. A cluster is a visual variation of the same
    subject (framing, setting, angle…); its purpose is reading dataset
    balance, so it is labelled by what sets it apart, not by the shared
    subject. ``picks`` is the picked :class:`Candidate` list.
    """
    embedded = [cand for cand in picks if cand.id in corpus.vectors]
    if len(embedded) < 2:
        return []
    matrix = np.stack([corpus.vectors[cand.id] for cand in embedded])
    labels, _ = embedding_map.kmeans(
        matrix, embedding_map.suggested_k(len(embedded))
    )
    groups: dict = {}
    for cand, label in zip(embedded, labels.tolist()):
        groups.setdefault(label, []).append(cand)
    overall = _tag_shares(picks)
    locked = {framing.normalize_tag(name) for name in recipe.locked_tags}
    ordered = sorted(groups.values(), key=len, reverse=True)
    total = len(picks)
    result = []
    for index, members in enumerate(ordered):
        distinct = _distinctive_tags(members, overall, locked)
        result.append(
            {
                "id": index,
                "color": CLUSTER_COLORS[index % len(CLUSTER_COLORS)],
                "label": (
                    " · ".join(distinct)
                    if distinct
                    else "mixed — close to the rest"
                ),
                "top_tags": distinct,
                "count": len(members),
                "pct": round(100.0 * len(members) / total) if total else 0,
                "media_ids": [cand.id for cand in members],
            }
        )
    return result


def cluster_of(corpus, picks, recipe: Recipe) -> dict:
    """Return ``{media_id: cluster index}`` over the picks' sub-families."""
    mapping = {}
    for entry in clusters(corpus, picks, recipe):
        for media_id in entry["media_ids"]:
            mapping[media_id] = entry["id"]
    return mapping


def proximity_edges(corpus, picks, floor: float = PROXIMITY_FLOOR) -> list:
    """Return the fused resemblance links among the picks.

    Every pair of picked media that both carry a DINOv2 vector yields an edge
    ``[a, b, dino_sim, depth_sim]`` (``a < b``, both cosines rounded to three
    decimals) whenever the *fused* similarity ``max(dino_sim, COMP_W ·
    depth_sim)`` reaches ``floor``. ``dino_sim`` is the DINOv2 appearance
    cosine; ``depth_sim`` the Depth-Anything V2 composition cosine, or ``0.0``
    when either endpoint lacks a depth signature. Materialising on the fused
    value is what lets a re-skin — far in DINOv2, close in composition — cross
    the floor and reach the front, which then classifies each edge by the
    signal that carries it (appearance first) and can raise the threshold or
    toggle the composition signal off client-side. The vectors are already
    unit-normalised (see :func:`src.dataset_compose.build_corpus`), so each
    cosine is a dot product. The list is sorted for a stable wire payload;
    picks without a DINOv2 vector contribute no edge.

    Parameters
    ----------
    corpus : Corpus
        The session geometry; ``corpus.vectors`` holds the DINOv2 unit
        vectors and ``corpus.depth_vectors`` the composition ones.
    picks : iterable of int
        The selected media ids.
    floor : float, optional
        The minimum *fused* similarity an edge must reach to be materialised.

    Returns
    -------
    list of list
        ``[[a, b, dino_sim, depth_sim], ...]`` — the sparse fused graph of the
        picks.
    """
    ids = [media_id for media_id in picks if media_id in corpus.vectors]
    depth = corpus.depth_vectors
    edges = []
    for index, first in enumerate(ids):
        vector = corpus.vectors[first]
        first_depth = depth.get(first)
        for second in ids[index + 1 :]:
            dino = float(vector @ corpus.vectors[second])
            second_depth = depth.get(second)
            comp = (
                float(first_depth @ second_depth)
                if first_depth is not None and second_depth is not None
                else 0.0
            )
            if max(dino, COMP_W * comp) >= floor:
                low, high = (
                    (first, second) if first < second else (second, first)
                )
                edges.append([low, high, round(dino, 3), round(comp, 3)])
    edges.sort()
    return edges


_ZONE_REASONS = {
    "quality": "under the quality floor",
    "blur": "excluded as blurry",
    "subject": "off-subject",
}


def uncovered_zones(corpus, candidates, picks) -> list:
    """Return the corpus cells holding candidates but no pick.

    A cell of the composer's 6×4 grid with at least
    :data:`ZONE_MIN_CANDIDATES` candidates and no picked media is a region
    of the visual space the proposal misses. Each zone names its dominant
    tags and *why* it is empty (how many of its candidates each filter
    dropped) so the user can act — loosen a threshold or grow the size.
    """
    picked = set(picks)
    width = dataset_compose.MAP_WIDTH / dataset_compose.GRID_COLUMNS
    height = dataset_compose.MAP_HEIGHT / dataset_compose.GRID_ROWS
    radius = min(width, height) * dataset_compose.ZONE_RADIUS_FACTOR
    cells: dict = {}
    covered = set()
    for cand in candidates:
        point = corpus.xy.get(cand.id)
        if point is None:
            continue
        cell = _cell_of(point)
        cells.setdefault(cell, []).append(cand)
        if cand.id in picked:
            covered.add(cell)
    zones = []
    for cell, members in sorted(cells.items()):
        if cell in covered or len(members) < ZONE_MIN_CANDIDATES:
            continue
        column, row = cell
        zones.append(
            {
                "x": round((column + 0.5) * width, 2),
                "y": round((row + 0.5) * height, 2),
                "r": round(radius, 2),
                "count": len(members),
                "top_tags": _top_tags(members),
                "why": _zone_why(members),
            }
        )
    return zones


def _cell_of(point) -> tuple:
    """Return the ``(column, row)`` grid cell a map point falls in."""
    column = min(
        dataset_compose.GRID_COLUMNS - 1,
        int(
            point[0] / dataset_compose.MAP_WIDTH * dataset_compose.GRID_COLUMNS
        ),
    )
    row = min(
        dataset_compose.GRID_ROWS - 1,
        int(point[1] / dataset_compose.MAP_HEIGHT * dataset_compose.GRID_ROWS),
    )
    return column, row


def _zone_why(members) -> str:
    """Return the one-line "why uncovered" of a zone's candidates."""
    counts: dict = {}
    for cand in members:
        if cand.excluded:
            counts[cand.excluded] = counts.get(cand.excluded, 0) + 1
    if not counts:
        return "eligible but not picked — raise the target size"
    parts = [
        f"{count} {_ZONE_REASONS.get(reason, reason)}"
        for reason, count in sorted(counts.items())
    ]
    return " · ".join(parts)
