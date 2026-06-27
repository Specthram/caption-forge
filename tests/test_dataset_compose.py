"""Tests for the dataset composer engine (:mod:`src.dataset_compose`).

Pure functions over hand-built media dicts and unit vectors: no database,
no model. The vectors live on the unit circle of a 2-D plane embedded in a
3-D space, so "close" and "far" are readable by eye and every cosine below
is exact.
"""

import math

import numpy as np
import pytest

from src import dataset_compose, dataset_quality, image_stats


def _vector(degrees: float) -> np.ndarray:
    """Return a unit vector at ``degrees`` on the (x, y) unit circle."""
    radians = math.radians(degrees)
    return np.array(
        [math.cos(radians), math.sin(radians), 0.0], dtype=np.float32
    )


def _media(media_id: int, **overrides) -> dict:
    """Return a media dict shaped like the repository's ``_media_dict``."""
    item = {
        "id": media_id,
        "sha256": f"{media_id:064x}",
        "name": f"img_{media_id:03d}.png",
        "file_extension": "png",
        "favorite": False,
        "width": 1600,
        "height": 2400,
        "quality_scores": {"musiq": 60.0},
        "quality_score": None,
        "quality_metric": None,
        "stats": {"sharpness": 80.0, "clipping": 1.0, "cleanliness": 90.0},
        "eff_path": None,
        "missing": False,
    }
    item.update(overrides)
    return item


def _corpus(dataset, pool, angles, hashes=None) -> dataset_compose.Corpus:
    """Build a corpus whose vectors are unit-circle angles, in degrees."""
    return dataset_compose.build_corpus(
        dataset,
        pool,
        {media_id: _vector(angle) for media_id, angle in angles.items()},
        hashes or {},
    )


class TestMetricScore:
    """Tests for :func:`dataset_compose.metric_score`."""

    def test_normalizes_a_single_metric(self):
        """A raw MUSIQ score comes back on the 0-100 scale."""
        item = _media(1, quality_scores={"musiq": 50.0})
        assert dataset_compose.metric_score(item, "musiq") == pytest.approx(
            50.0, abs=1.0
        )

    def test_none_when_the_metric_was_never_scored(self):
        """A metric with no stored row scores None, never zero."""
        item = _media(1, quality_scores={"musiq": 50.0})
        assert dataset_compose.metric_score(item, "topiq_nr") is None

    def test_average_means_every_stored_metric(self):
        """The average pseudo-metric folds the stored metrics together."""
        item = _media(1, quality_scores={"musiq": 100.0, "laion_aes": 1.0})
        average = dataset_compose.metric_score(item, "average")
        alone = dataset_compose.metric_score(item, "musiq")
        assert average is not None and alone is not None
        assert average < alone


class TestNearDuplicate:
    """Tests for :func:`dataset_compose.near_duplicate`."""

    def test_a_close_cosine_raises_the_alert(self):
        """A candidate within NEAR_DUP_COSINE of the dataset is flagged."""
        dataset, pool = [_media(1)], [_media(2)]
        corpus = _corpus(dataset, pool, {1: 0.0, 2: 5.0})
        alert = dataset_compose.near_duplicate(corpus, 2, [1])
        assert alert is not None
        assert alert["kind"] == "cosine"
        assert alert["media_id"] == 1
        assert alert["cosine"] >= dataset_quality.NEAR_DUP_COSINE

    def test_a_distant_candidate_is_not_flagged(self):
        """Beyond the cosine threshold there is no alert at all."""
        corpus = _corpus([_media(1)], [_media(2)], {1: 0.0, 2: 80.0})
        assert dataset_compose.near_duplicate(corpus, 2, [1]) is None

    def test_a_shared_hash_group_wins_over_the_cosine(self):
        """A re-encoded copy reads as a hash duplicate, not a neighbour."""
        twins = {1: ("f" * 16, "e" * 16), 2: ("f" * 16, "e" * 16)}
        corpus = _corpus([_media(1)], [_media(2)], {1: 0.0, 2: 80.0}, twins)
        alert = dataset_compose.near_duplicate(corpus, 2, [1])
        assert alert is not None and alert["kind"] == "hash"

    def test_only_the_references_are_compared(self):
        """A twin outside the dataset and the selection raises nothing."""
        corpus = _corpus([], [_media(1), _media(2)], {1: 0.0, 2: 1.0})
        assert dataset_compose.near_duplicate(corpus, 2, []) is None


class TestDiversityGain:
    """Tests for :func:`dataset_compose.diversity_gain`."""

    def test_a_twin_of_the_dataset_adds_nothing(self):
        """A candidate on top of a dataset media gains ~0."""
        corpus = _corpus([_media(1)], [_media(2)], {1: 0.0, 2: 0.0})
        assert dataset_compose.diversity_gain(corpus, 2, [1]) == 0.0

    def test_an_orthogonal_candidate_saturates_the_gain(self):
        """A cosine distance past the ceiling clamps at 1."""
        corpus = _corpus([_media(1)], [_media(2)], {1: 0.0, 2: 90.0})
        assert dataset_compose.diversity_gain(corpus, 2, [1]) == 1.0

    def test_gain_falls_once_a_neighbour_is_selected(self):
        """Selecting a lookalike shrinks the next candidate's gain."""
        corpus = _corpus(
            [_media(1)], [_media(2), _media(3)], {1: 0.0, 2: 60.0, 3: 62.0}
        )
        alone = dataset_compose.diversity_gain(corpus, 3, [1])
        crowded = dataset_compose.diversity_gain(corpus, 3, [1, 2])
        assert crowded < alone


class TestZones:
    """Tests for the empty zones of the corpus."""

    def test_a_cell_with_candidates_but_no_dataset_media_is_empty(self):
        """The zone sits where the corpus has no dataset coverage."""
        zones = dataset_compose.empty_zones([(5.0, 5.0)], [(95.0, 70.0)])
        assert len(zones) == 1
        assert zones[0].x > dataset_compose.MAP_WIDTH / 2

    def test_a_covered_cell_is_never_a_zone(self):
        """A cell holding a dataset media is covered."""
        assert dataset_compose.empty_zones([(5.0, 5.0)], [(6.0, 6.0)]) == ()


class TestCandidates:
    """Tests for :func:`dataset_compose.candidates`."""

    def test_the_score_floor_drops_a_weak_candidate(self):
        """A scored candidate under the floor leaves the grid."""
        pool = [
            _media(2, quality_scores={"musiq": 20.0}),
            _media(3, quality_scores={"musiq": 90.0}),
        ]
        corpus = _corpus([], pool, {2: 0.0, 3: 90.0})
        filters = dataset_compose.Filters(metric="musiq", min_score=60.0)
        rows = dataset_compose.candidates(corpus, filters, set())
        assert [row["id"] for row in rows] == [3]

    def test_an_unscored_candidate_survives_the_floor(self):
        """A library indexed without the quality step still shows up."""
        pool = [_media(2, quality_scores={})]
        corpus = _corpus([], pool, {2: 0.0})
        filters = dataset_compose.Filters(metric="musiq", min_score=90.0)
        rows = dataset_compose.candidates(corpus, filters, set())
        assert [row["id"] for row in rows] == [2]
        assert rows[0]["score"] is None

    def test_a_selected_candidate_survives_every_filter(self):
        """Moving a slider never silently drops a picked image."""
        pool = [_media(2, quality_scores={"musiq": 20.0})]
        corpus = _corpus([], pool, {2: 0.0})
        filters = dataset_compose.Filters(metric="musiq", min_score=90.0)
        rows = dataset_compose.candidates(corpus, filters, {2})
        assert [row["id"] for row in rows] == [2]

    def test_blur_exclusion_ignores_an_unanalyzed_media(self):
        """A media with no statistics is never read as blurry."""
        blurry = _media(
            2,
            stats={
                "sharpness": image_stats.BLUR_FLOOR - 1.0,
                "clipping": 0.0,
                "cleanliness": 90.0,
            },
        )
        unknown = _media(3, stats=None)
        corpus = _corpus([], [blurry, unknown], {2: 0.0, 3: 90.0})
        filters = dataset_compose.Filters(exclude_blur=True)
        rows = dataset_compose.candidates(corpus, filters, set())
        assert [row["id"] for row in rows] == [3]

    def test_the_resolution_floor_uses_the_long_side(self):
        """A portrait 800x2000 passes a 1024 floor on its long side."""
        pool = [
            _media(2, width=800, height=2000),
            _media(3, width=900, height=900),
        ]
        corpus = _corpus([], pool, {2: 0.0, 3: 90.0})
        filters = dataset_compose.Filters(min_side=1024)
        rows = dataset_compose.candidates(corpus, filters, set())
        assert [row["id"] for row in rows] == [2]

    def test_hide_near_dups_drops_the_flagged_candidates(self):
        """The checkbox removes what the red banner would have announced."""
        corpus = _corpus(
            [_media(1)], [_media(2), _media(3)], {1: 0.0, 2: 2.0, 3: 90.0}
        )
        filters = dataset_compose.Filters(hide_near_dups=True)
        rows = dataset_compose.candidates(corpus, filters, set())
        assert [row["id"] for row in rows] == [3]

    def test_similar_to_selection_needs_a_selection(self):
        """With nothing picked the mode keeps nothing."""
        corpus = _corpus([], [_media(2), _media(3)], {2: 0.0, 3: 2.0})
        filters = dataset_compose.Filters(similar_to_selection=True)
        assert dataset_compose.candidates(corpus, filters, set()) == []

    def test_similar_to_selection_keeps_the_neighbours(self):
        """A picked media pulls its visual neighbours in."""
        corpus = _corpus(
            [], [_media(2), _media(3), _media(4)], {2: 0.0, 3: 2.0, 4: 90.0}
        )
        filters = dataset_compose.Filters(similar_to_selection=True)
        rows = dataset_compose.candidates(corpus, filters, {2})
        assert sorted(row["id"] for row in rows) == [2, 3]

    def test_sorting_by_gain_puts_the_freshest_first(self):
        """The diversity sort ranks by distance to what is already in."""
        corpus = _corpus(
            [_media(1)], [_media(2), _media(3)], {1: 0.0, 2: 10.0, 3: 89.0}
        )
        filters = dataset_compose.Filters(sort=dataset_compose.SORT_GAIN)
        rows = dataset_compose.candidates(corpus, filters, set())
        assert [row["id"] for row in rows] == [3, 2]

    def test_sorting_by_name_is_alphabetical(self):
        """The name sort ignores every score."""
        pool = [_media(2, name="b.png"), _media(3, name="a.png")]
        corpus = _corpus([], pool, {2: 0.0, 3: 90.0})
        filters = dataset_compose.Filters(sort=dataset_compose.SORT_NAME)
        rows = dataset_compose.candidates(corpus, filters, set())
        assert [row["name"] for row in rows] == ["a.png", "b.png"]

    def test_a_semantic_query_keeps_the_best_matches(self):
        """The relevance map narrows the pool to its top fraction."""
        pool = [_media(i) for i in range(2, 12)]
        corpus = _corpus([], pool, {i: i * 5.0 for i in range(2, 12)})
        relevance = {item["id"]: 0.1 for item in pool}
        relevance[6], relevance[7] = 0.8, 0.9
        rows = dataset_compose.candidates(
            corpus, dataset_compose.Filters(), set(), relevance
        )
        assert sorted(row["id"] for row in rows) == [6, 7]

    def test_a_semantic_query_always_keeps_one_match(self):
        """A tiny pool never comes back empty."""
        corpus = _corpus([], [_media(2)], {2: 0.0})
        rows = dataset_compose.candidates(
            corpus, dataset_compose.Filters(), set(), {2: 0.01}
        )
        assert [row["id"] for row in rows] == [2]


class TestPreview:
    """Tests for :func:`dataset_compose.preview`."""

    def test_an_empty_selection_projects_the_dataset_itself(self):
        """With nothing picked the delta is exactly zero."""
        dataset = [_media(1), _media(2)]
        corpus = _corpus(dataset, [], {1: 0.0, 2: 60.0})
        result = dataset_compose.preview(corpus, set(), "musiq", {}, {})
        assert result["delta"] == pytest.approx(0.0)
        assert result["size"]["total"] == 2

    def test_a_strong_pick_lifts_the_projected_score(self):
        """Adding a high-quality, distant image raises the grade."""
        dataset = [_media(1, quality_scores={"musiq": 40.0})]
        pool = [_media(2, quality_scores={"musiq": 75.0})]
        corpus = _corpus(dataset, pool, {1: 0.0, 2: 90.0})
        result = dataset_compose.preview(corpus, {2}, "musiq", {}, {})
        assert result["delta"] > 0

    def test_a_duplicate_pick_costs_the_hygiene_pillar(self):
        """Each selected near-duplicate is penalised and announced."""
        corpus = _corpus([_media(1)], [_media(2)], {1: 0.0, 2: 1.0})
        result = dataset_compose.preview(corpus, {2}, "musiq", {}, {})
        assert result["pillars"]["duplicates"] == 1
        assert result["pillars"]["hygiene"] == pytest.approx(
            100.0 - dataset_compose.HYGIENE_PENALTY
        )
        assert result["advice"][0]["tone"] == "danger"

    def test_a_balanced_selection_says_so(self):
        """Nothing to flag yields the single green advice card."""
        dataset = [_media(i, quality_scores={"musiq": 90.0}) for i in (1, 2)]
        corpus = _corpus(dataset, [], {1: 0.0, 2: 90.0})
        result = dataset_compose.preview(corpus, set(), "musiq", {}, {})
        assert [row["tone"] for row in result["advice"]] == ["ok"]

    def test_framing_splits_the_dataset_from_the_selection(self):
        """The stacked bar knows what is already in and what is added."""
        buckets = {"face": ["portrait"], "full_body": ["full_body"]}
        tags = {1: ["portrait"], 2: ["full_body"]}
        corpus = _corpus([_media(1)], [_media(2)], {1: 0.0, 2: 90.0})
        result = dataset_compose.preview(corpus, {2}, "musiq", tags, buckets)
        rows = {row["bucket"]: row for row in result["framing"]}
        assert rows["face"]["base"] == 1 and rows["face"]["added"] == 0
        assert rows["full_body"]["base"] == 0
        assert rows["full_body"]["added"] == 1

    def test_the_size_bar_flags_an_oversized_set(self):
        """Past the recommended maximum the bar goes over."""
        dataset = [_media(i) for i in range(1, 4)]
        corpus = _corpus(dataset, [], {i: i * 30.0 for i in range(1, 4)})
        result = dataset_compose.preview(
            corpus, set(), "musiq", {}, {}, size_range=(1, 2)
        )
        assert result["size"]["over"] is True
