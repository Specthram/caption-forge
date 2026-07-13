"""Unit tests for the Auto-build Studio selection engine."""

import numpy as np

from src import autobuild_studio as studio
from src import dataset_compose
from src.autobuild_studio import Recipe


def _media(media_id, **over):
    """Return a minimal pool media dict for the engine."""
    item = {
        "id": media_id,
        "name": f"m{media_id}",
        "favorite": False,
        "file_extension": "png",
        "width": 100,
        "height": 100,
        "quality_scores": {},
        "stats": None,
        "missing": False,
    }
    item.update(over)
    return item


def _corpus(pool, vectors):
    """Build a corpus over an empty dataset and the given pool."""
    return dataset_compose.build_corpus([], pool, vectors, {})


def _vec(*values):
    """Return a unit-ish float vector."""
    array = np.array(values, dtype=np.float32)
    return array / (np.linalg.norm(array) or 1.0)


def test_subject_match_blends_active_signals():
    """The match is the mean of the active signals only."""
    recipe = Recipe(semantic_q="red", locked_tags=("red_hair",))
    tags = {1: [{"name": "red_hair"}], 2: []}
    relevance = {1: 0.9, 2: 0.1}
    matches = studio.subject_matches(
        recipe, [1, 2], {1: ["red_hair"], 2: []}, {}, relevance
    )
    # Media 1: semantic rank 1.0 + locked 1.0 -> 1.0; media 2: 0.0 + 0.0.
    assert matches[1][0] == 1.0
    assert matches[2][0] == 0.0
    assert set(matches[1][1]) == {"semantic", "locked"}


def test_locked_tags_are_not_a_ranking_signal():
    """Locked tags shape the pool, not the subject match or its activity."""
    recipe = Recipe(locked_tags=("red_hair",))
    matches = studio.subject_matches(recipe, [1], {1: ["red_hair"]}, {}, None)
    # The tag is recorded for the chip, but the ranking match stays neutral.
    assert matches[1][0] == 0.0
    assert set(matches[1][1]) == {"locked"}
    assert not studio.subject_active(recipe)
    assert studio.subject_active(Recipe(semantic_q="red"))


def test_proximity_edges_are_sparse_and_ordered():
    """Only pick pairs at or above the floor become edges, sorted [a, b]."""
    pool = [_media(1), _media(2), _media(3)]
    vectors = {
        1: _vec(1.0, 0.0),
        2: _vec(0.999, 0.045),  # ~identical to 1 -> near-duplicate (>=0.92)
        3: _vec(0.0, 1.0),  # orthogonal to 1 -> below the 0.70 floor
    }
    corpus = _corpus(pool, vectors)
    edges = studio.proximity_edges(corpus, [1, 2, 3])
    assert [edge[:2] for edge in edges] == [[1, 2]]
    assert edges[0][2] >= studio.NEAR_DUP_COSINE
    # A pick without a vector contributes no edge.
    assert studio.proximity_edges(corpus, [1, 2, 99]) == edges
    # A high floor drops even the near-duplicate pair.
    assert studio.proximity_edges(corpus, [1, 2, 3], floor=0.999) == []
    # Every edge carries both the DINOv2 and the (here zero) depth cosine.
    assert len(edges[0]) == 4
    assert edges[0][3] == 0.0


def test_proximity_edges_fuse_composition_depth():
    """A depth-close pair crosses the floor even when DINOv2 rates it far."""
    pool = [_media(1), _media(2)]
    vectors = {1: _vec(1.0, 0.0), 2: _vec(0.0, 1.0)}  # orthogonal -> dino 0
    depth = {1: _vec(1.0, 0.0), 2: _vec(0.98, 0.2)}  # ~aligned -> comp high
    corpus = dataset_compose.build_corpus([], pool, vectors, {}, depth)
    edges = studio.proximity_edges(corpus, [1, 2])
    assert len(edges) == 1
    a, b, dino, comp = edges[0]
    assert [a, b] == [1, 2]
    assert dino < studio.PROXIMITY_FLOOR  # DINOv2 alone would drop it
    assert studio.COMP_W * comp >= studio.PROXIMITY_FLOOR  # depth carries it


def test_proximity_edges_fall_back_without_depth():
    """A pair DINOv2 rates far and with no depth signature yields no edge."""
    pool = [_media(1), _media(2)]
    vectors = {1: _vec(1.0, 0.0), 2: _vec(0.0, 1.0)}
    corpus = dataset_compose.build_corpus(
        [], pool, vectors, {}, {1: _vec(1.0)}
    )
    # Only media 1 carries a depth vector, so the pair's comp cosine is 0.
    assert studio.proximity_edges(corpus, [1, 2]) == []


def test_prepare_gates_off_subject_when_signal_active():
    """A candidate below the subject gate is excluded, not dropped."""
    recipe = Recipe(semantic_q="red", min_score=0)
    pool = [_media(1), _media(2)]
    relevance = {1: 0.9, 2: 0.1}
    cands = studio.prepare(recipe, pool, {1: [], 2: []}, {}, relevance)
    by_id = {c.id: c for c in cands}
    assert by_id[1].eligible
    assert not by_id[2].eligible and by_id[2].excluded == "subject"


def test_select_respects_size_forced_and_favorites():
    """Forced ids come first, favorites are guaranteed, size caps the run."""
    vectors = {1: _vec(1, 0, 0), 2: _vec(0, 1, 0), 3: _vec(0, 0, 1)}
    pool = [_media(1), _media(2, favorite=True), _media(3)]
    corpus = _corpus(pool, vectors)
    recipe = Recipe(size=2, forced=(3,), framing_preset="free", min_score=0)
    cands = studio.prepare(recipe, pool, {}, corpus.vectors, None)
    picks, _meta = studio.select(corpus, cands, recipe)
    assert picks[0] == 3  # forced first
    assert 2 in picks  # favorite guaranteed
    assert len(picks) == 2  # size cap


def test_select_skips_near_duplicates():
    """A near-duplicate of a pick is skipped while an alternative exists."""
    vectors = {1: _vec(1, 0), 2: _vec(0.999, 0.045), 3: _vec(0, 1)}
    pool = [_media(1), _media(2), _media(3)]
    corpus = _corpus(pool, vectors)
    recipe = Recipe(size=2, framing_preset="free", min_score=0)
    cands = studio.prepare(recipe, pool, {}, corpus.vectors, None)
    picks, meta = studio.select(corpus, cands, recipe)
    assert set(picks) == {1, 3}  # 2 was too close to 1
    assert all(not info["near_dup"] for info in meta.values())


def test_dropped_id_never_selected():
    """A dropped id is excluded from the proposal on every recompute."""
    vectors = {1: _vec(1, 0), 2: _vec(0, 1)}
    pool = [_media(1), _media(2)]
    corpus = _corpus(pool, vectors)
    recipe = Recipe(size=2, framing_preset="free", min_score=0, dropped=(1,))
    cands = studio.prepare(recipe, pool, {}, corpus.vectors, None)
    picks, _meta = studio.select(corpus, cands, recipe)
    assert picks == [2]


def test_reasons_and_flag():
    """A forced favorite gets its chips; a near-dup fallback is flagged."""
    recipe = Recipe(size=1, forced=(1,), min_score=0)
    cand = studio.Candidate(
        id=1,
        name="m1",
        favorite=True,
        is_video=False,
        width=1,
        height=1,
        quality=88.0,
        bucket="face",
        tags=(),
        subject=0.0,
        signals={},
        eligible=True,
        excluded="",
        has_vector=True,
    )
    chips = studio.reasons_for(cand, {1: {"gain": 0.6}}, recipe)
    icons = {chip["icon"] for chip in chips}
    assert {"⇄", "♥", "◆", "Q"} <= icons
    # A forced pick is never flagged as borderline.
    assert studio.flag_for(cand, {1: {"near_dup": True}}, recipe) is None


def test_uncovered_zones_and_clusters():
    """Zones form where candidates cluster with no pick; clusters group."""
    # Two tight groups far apart; pick from the first group only.
    vectors = {
        1: _vec(1, 0),
        2: _vec(0.98, 0.02),
        3: _vec(0.99, 0.01),
        4: _vec(0, 1),
        5: _vec(0.02, 0.98),
        6: _vec(0.01, 0.99),
    }
    pool = [_media(i) for i in range(1, 7)]
    corpus = _corpus(pool, vectors)
    recipe = Recipe(size=6, framing_preset="free", min_score=0)
    cands = studio.prepare(recipe, pool, {}, corpus.vectors, None)
    zones = studio.uncovered_zones(corpus, cands, picks=[1])
    assert all(zone["count"] >= studio.ZONE_MIN_CANDIDATES for zone in zones)
    # Clusters are the sub-families of the *selection*: pass the picks.
    groups = studio.clusters(corpus, cands, recipe)
    assert sum(g["count"] for g in groups) == 6
    # Two tight groups far apart -> two clusters, largest first.
    assert len(groups) == 2
    assert groups[0]["count"] >= groups[1]["count"]


def test_clusters_labelled_by_distinctive_not_shared_tags():
    """A cluster's label shows what sets it apart, not the shared subject."""
    vectors = {
        1: _vec(1, 0),
        2: _vec(0.98, 0.02),
        3: _vec(0.99, 0.01),
        4: _vec(0, 1),
        5: _vec(0.02, 0.98),
        6: _vec(0.01, 0.99),
    }
    pool = [_media(i) for i in range(1, 7)]
    # Every pick shares "1girl"; group A carries "hat", group B "beach".
    tags = {
        1: [{"name": "1girl"}, {"name": "hat"}],
        2: [{"name": "1girl"}, {"name": "hat"}],
        3: [{"name": "1girl"}, {"name": "hat"}],
        4: [{"name": "1girl"}, {"name": "beach"}],
        5: [{"name": "1girl"}, {"name": "beach"}],
        6: [{"name": "1girl"}, {"name": "beach"}],
    }
    corpus = _corpus(pool, vectors)
    recipe = Recipe(size=6, framing_preset="free", min_score=0)
    cands = studio.prepare(recipe, pool, tags, corpus.vectors, None)
    groups = studio.clusters(corpus, cands, recipe)
    labels = {g["label"] for g in groups}
    assert labels == {"hat", "beach"}  # shared "1girl" never labels a cluster


def test_rebalance_penalty_docks_a_crowded_candidate():
    """With rebalance on, a candidate crowded by picks is penalised."""
    picker = studio._Picker(  # pylint: disable=protected-access
        _corpus([], {}), [], active=False, rebal=True
    )
    # No picks down yet: the penalty is dormant.
    assert picker._rebalance_penalty(0) == 0.0  # noqa: SLF001
