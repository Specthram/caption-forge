"""Tests for :mod:`src.dataset_quality` (dataset evaluation engine).

Pure-function tests over synthetic media dicts, hand-picked perceptual
hashes and low-dimensional vectors — no database, no model. Quality scores
use MUSIQ, whose native 0-100 range makes raw and normalized values
identical.
"""

import json

import numpy as np
import pytest

from src import dataset_quality

_BUCKETS = {
    "face": ["portrait"],
    "full_body": ["full_body"],
}


def _hex(value: int) -> str:
    """Return a 16-hex-char (64-bit) hash string for an integer."""
    return format(value, "016x")


def _media(  # pylint: disable=too-many-arguments
    media_id,
    score=None,
    phash=None,
    dhash=None,
    width=2000,
    height=2000,
    favorite=False,
    missing=False,
    sha256=None,
):
    """Return a media dict shaped like ``media_in_dataset`` output."""
    return {
        "id": media_id,
        "sha256": sha256 or f"sha{media_id}",
        "name": f"{media_id}.png",
        "eff_path": None if missing else f"/x/{media_id}.png",
        "missing": missing,
        "favorite": favorite,
        "width": width,
        "height": height,
        "quality_scores": {} if score is None else {"musiq": score},
        "quality_score": score,
        "quality_metric": "musiq" if score is not None else None,
        "phash": _hex(phash) if phash is not None else None,
        "dhash": _hex(dhash) if dhash is not None else None,
    }


def _snapshot(images, **kwargs):
    """Return a snapshot over ``images``, MUSIQ + DINOv2 enabled."""
    kwargs.setdefault("scorers", ("musiq", dataset_quality.EMBEDDING_SCORER))
    return dataset_quality.Snapshot(images=tuple(images), **kwargs)


class TestNormalizedQuality:
    """Tests for the per-media normalized quality helpers."""

    def test_single_metric(self):
        """A stored MUSIQ score reads back as its 0-100 value."""
        item = _media(1, score=72.0)
        assert dataset_quality.normalized_quality(item, ("musiq",)) == 72.0

    def test_unscored_media_is_none(self):
        """A media with no stored score has no quality."""
        item = _media(1)
        assert dataset_quality.normalized_quality(item, ("musiq",)) is None

    def test_disabled_scorer_is_ignored(self):
        """A stored score whose scorer is off never enters the mean."""
        item = _media(1, score=72.0)
        assert dataset_quality.normalized_quality(item, ("topiq_nr",)) is None

    def test_mean_over_several_scorers(self):
        """Enabled scorers are averaged after normalization."""
        item = _media(1, score=60.0)
        item["quality_scores"]["topiq_nr"] = 0.8  # native 0-1 -> 80
        value = dataset_quality.normalized_quality(item, ("musiq", "topiq_nr"))
        assert value == 70.0


class TestQualityPillar:
    """Tests for the image-quality pillar."""

    def test_mean_of_scored_images(self):
        """The pillar score is the mean over the scored images."""
        snapshot = _snapshot([_media(1, 60.0), _media(2, 80.0), _media(3)])
        pillar = dataset_quality.evaluate(snapshot).pillars[0]
        assert pillar.key == "quality"
        assert pillar.score == 70.0

    def test_no_score_leaves_the_pillar_unscored(self):
        """Without a single stored score the pillar carries no score."""
        pillar = dataset_quality.evaluate(_snapshot([_media(1)])).pillars[0]
        assert pillar.score is None

    def test_per_scorer_row(self):
        """One row per enabled IQA scorer, plus the blur/noise flags."""
        snapshot = _snapshot([_media(1, 60.0)], scorers=("musiq",))
        pillar = dataset_quality.evaluate(snapshot).pillars[0]
        labels = [row.label for row in pillar.rows]
        assert labels == ["MUSIQ", "blur / noise flags"]


class TestDiversityPillar:
    """Tests for the DINOv2 diversity pillar."""

    def test_orthogonal_vectors_score_high(self):
        """Fully unrelated images fill the spread scale."""
        vectors = {
            1: np.array([1.0, 0.0, 0.0]),
            2: np.array([0.0, 1.0, 0.0]),
            3: np.array([0.0, 0.0, 1.0]),
        }
        snapshot = _snapshot(
            [_media(1, 70.0), _media(2, 70.0), _media(3, 70.0)],
            vectors_by_id=vectors,
        )
        pillar = dataset_quality.evaluate(snapshot).pillars[1]
        assert pillar.score == 100.0

    def test_identical_vectors_score_zero(self):
        """Three copies of the same picture have no spread at all."""
        vector = np.array([1.0, 0.0, 0.0])
        vectors = {1: vector, 2: vector, 3: vector}
        snapshot = _snapshot(
            [_media(1, 70.0), _media(2, 70.0), _media(3, 70.0)],
            vectors_by_id=vectors,
        )
        pillar = dataset_quality.evaluate(snapshot).pillars[1]
        assert pillar.score == 0.0

    def test_no_embedding_leaves_the_pillar_unscored(self):
        """Fewer than two embeddings: nothing to measure."""
        snapshot = _snapshot([_media(1, 70.0), _media(2, 70.0)])
        assert dataset_quality.evaluate(snapshot).pillars[1].score is None

    def test_disabled_embedding_scorer_skips_the_map(self):
        """Turning DINOv2 off drops the map even when vectors exist."""
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
        snapshot = _snapshot(
            [_media(1, 70.0), _media(2, 70.0)],
            scorers=("musiq",),
            vectors_by_id=vectors,
        )
        report = dataset_quality.evaluate(snapshot)
        assert report.map_points == ()
        assert report.pillars[1].score is None

    def test_perceptual_duplicates_temper_the_score(self):
        """Two hash-identical images cost a quarter of the pillar."""
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
        images = [
            _media(1, 70.0, phash=0, dhash=0),
            _media(2, 70.0, phash=0, dhash=0),
        ]
        snapshot = _snapshot(images, vectors_by_id=vectors)
        pillar = dataset_quality.evaluate(snapshot).pillars[1]
        # spread 100 * 0.75 + uniqueness 50 * 0.25
        assert pillar.score == 87.5


class TestCompositionPillar:
    """Tests for the Depth-Anything V2 composition pillar and map."""

    def test_no_depth_leaves_the_pillar_unscored(self):
        """Without depth signatures the pillar is diagnostic-only."""
        snapshot = _snapshot([_media(1, 70.0), _media(2, 70.0)])
        pillar = dataset_quality.evaluate(snapshot).pillars[2]
        assert pillar.key == "composition"
        assert pillar.score is None

    def test_varied_framings_score_high(self):
        """Orthogonal depth signatures fill the composition scale."""
        depth = {
            1: np.array([1.0, 0.0, 0.0]),
            2: np.array([0.0, 1.0, 0.0]),
            3: np.array([0.0, 0.0, 1.0]),
        }
        snapshot = _snapshot(
            [_media(1, 70.0), _media(2, 70.0), _media(3, 70.0)],
            depth_vectors_by_id=depth,
        )
        report = dataset_quality.evaluate(snapshot)
        assert report.pillars[2].score == 100.0
        assert report.framings >= 1
        assert len(report.composition_map) == 3

    def test_reskin_is_close_in_depth_far_in_appearance(self):
        """Same framing, different style: depth close, DINOv2 far."""
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
        depth = {1: np.array([1.0, 0.0]), 2: np.array([0.999, 0.045])}
        snapshot = _snapshot(
            [_media(1, 70.0), _media(2, 70.0)],
            vectors_by_id=vectors,
            depth_vectors_by_id=depth,
        )
        report = dataset_quality.evaluate(snapshot)
        assert report.reskins == 1
        assert report.composition_links == ((1, 2),)
        assert all(point.reskin for point in report.composition_map)
        rows = {row.label: row.value for row in report.pillars[2].rows}
        assert rows["composition re-skins"] == "1"

    def test_reskin_catches_the_resemblance_band(self):
        """Same framing, DINOv2 in 0.85–0.92: a re-skin, not a near-dup.

        The pair DINOv2 reads at 0.88 (moderately restyled, same framing)
        used to fall through both lists; the report now flags it as a
        composition re-skin.
        """
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.88, 0.475])}
        depth = {1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])}
        report = dataset_quality.evaluate(
            _snapshot(
                [_media(1, 70.0), _media(2, 70.0)],
                vectors_by_id=vectors,
                depth_vectors_by_id=depth,
            )
        )
        assert report.reskins == 1

    def test_near_duplicate_is_not_a_reskin(self):
        """A pair DINOv2 calls a near-dup (>=0.92) is never a re-skin."""
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.95, 0.312])}
        depth = {1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])}
        report = dataset_quality.evaluate(
            _snapshot(
                [_media(1, 70.0), _media(2, 70.0)],
                vectors_by_id=vectors,
                depth_vectors_by_id=depth,
            )
        )
        assert report.reskins == 0

    def test_style_bucket_rides_the_points(self):
        """Each map dot carries its media's style bucket, neutral default."""
        depth = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
        snapshot = _snapshot(
            [_media(1, 70.0), _media(2, 70.0)],
            depth_vectors_by_id=depth,
            styles_by_id={1: "warm"},
        )
        points = {
            point.id: point.style
            for point in dataset_quality.evaluate(snapshot).composition_map
        }
        assert points == {1: "warm", 2: "neutral"}


class TestHygienePillar:
    """Tests for the structural-readiness pillar."""

    def test_full_coverage_scores_100(self):
        """Captioned, large, clean and unique: a perfect hygiene score."""
        snapshot = _snapshot(
            [_media(1, 70.0), _media(2, 70.0)],
            captions_by_id={1: "A cat.", 2: "A dog."},
        )
        assert dataset_quality.evaluate(snapshot).pillars[3].score == 100.0

    def test_missing_caption_lowers_coverage(self):
        """An uncaptioned image costs half of the coverage component."""
        snapshot = _snapshot(
            [_media(1, 70.0), _media(2, 70.0)],
            captions_by_id={1: "A cat.", 2: ""},
        )
        pillar = dataset_quality.evaluate(snapshot).pillars[3]
        assert pillar.score == 87.5  # (50 + 100 + 100 + 100) / 4

    def test_small_images_lower_the_resolution_row(self):
        """An image under the floor moves the resolution row."""
        snapshot = _snapshot(
            [_media(1, 70.0, width=512, height=512)],
            captions_by_id={1: "A cat."},
        )
        pillar = dataset_quality.evaluate(snapshot).pillars[3]
        rows = {row.label: row.value for row in pillar.rows}
        assert rows["resolution ≥ 1024px"] == "0 / 1"
        assert pillar.score == 75.0  # (100 + 100 + 100 + 0) / 4

    def test_unindexed_dataset_gets_no_free_resolution_credit(self):
        """Unknown dimensions are dropped, never counted as satisfied.

        The regression this guards: crediting the resolution component 100
        when nothing is indexed inflates the pillar for exactly the
        datasets that were never prepared.
        """
        snapshot = _snapshot(
            [_media(1, 70.0, width=None, height=None)],
            captions_by_id={1: ""},
        )
        pillar = dataset_quality.evaluate(snapshot).pillars[3]
        rows = {row.label: row.value for row in pillar.rows}
        assert rows["resolution ≥ 1024px"] == "not indexed"
        assert pillar.score == pytest.approx(200 / 3)  # (0 + 100 + 100) / 3

    def test_exact_duplicates_are_counted(self):
        """Two media sharing a content hash are an exact duplicate."""
        images = [
            _media(1, 70.0, sha256="same"),
            _media(2, 70.0, sha256="same"),
        ]
        snapshot = _snapshot(images, captions_by_id={1: "A.", 2: "B."})
        pillar = dataset_quality.evaluate(snapshot).pillars[3]
        rows = {row.label: row.value for row in pillar.rows}
        assert rows["exact duplicates"] == "1"


class TestOverallScore:
    """Tests for the weighted global grade."""

    def test_weighted_mean_of_the_pillars(self):
        """The grade weights the four pillars 0.35/0.30/0.15/0.20."""
        pillars = (
            dataset_quality.Pillar("quality", "", 100.0, ""),
            dataset_quality.Pillar("diversity", "", 0.0, ""),
            dataset_quality.Pillar("composition", "", 100.0, ""),
            dataset_quality.Pillar("hygiene", "", 100.0, ""),
        )
        score = dataset_quality.overall_score(
            pillars, dataset_quality.DEFAULT_WEIGHTS
        )
        # 100*0.35 + 0*0.30 + 100*0.15 + 100*0.20 = 70.
        assert score == 70.0

    def test_unscored_pillars_renormalize_the_weights(self):
        """A pillar with no signal is dropped, not counted as zero.

        A dataset with no depth signature (composition None) therefore scores
        almost exactly as it did before the pillar existed — the remaining
        weights renormalize back to roughly the old 0.40/0.35/0.25 split.
        """
        pillars = (
            dataset_quality.Pillar("quality", "", 80.0, ""),
            dataset_quality.Pillar("diversity", "", None, ""),
            dataset_quality.Pillar("composition", "", None, ""),
            dataset_quality.Pillar("hygiene", "", 60.0, ""),
        )
        score = dataset_quality.overall_score(
            pillars, dataset_quality.DEFAULT_WEIGHTS
        )
        assert round(score, 4) == round((80 * 0.35 + 60 * 0.20) / 0.55, 4)

    def test_nothing_scorable_is_none(self):
        """No scorable pillar means no grade."""
        pillars = (dataset_quality.Pillar("quality", "", None, ""),)
        assert (
            dataset_quality.overall_score(
                pillars, dataset_quality.DEFAULT_WEIGHTS
            )
            is None
        )

    def test_grade_and_verdict_bands(self):
        """Grades and verdicts follow the documented bands."""
        assert dataset_quality.grade_of(92.0) == "A"
        assert dataset_quality.grade_of(84.0) == "B+"
        assert dataset_quality.grade_of(76.0) == "B"
        assert dataset_quality.grade_of(66.0) == "C"
        assert dataset_quality.grade_of(40.0) == "D"
        assert dataset_quality.verdict_of(81.0).startswith("Good")
        assert dataset_quality.verdict_of(70.0).startswith("Trainable")
        assert dataset_quality.verdict_of(10.0).startswith("Needs work")


class TestDistribution:
    """Tests for the quality histogram."""

    def test_buckets_count_each_image_once(self):
        """Every scored image lands in exactly one bucket."""
        images = [_media(1, 55.0), _media(2, 72.0), _media(3, 100.0)]
        report = dataset_quality.evaluate(_snapshot(images))
        counts = {bucket.label: bucket.count for bucket in report.distribution}
        assert counts == {
            "<60": 1,
            "60–69": 0,
            "70–79": 1,
            "80–89": 0,
            "90–100": 1,
        }

    def test_unscored_images_are_left_out(self):
        """An image with no score is in no bucket."""
        report = dataset_quality.evaluate(_snapshot([_media(1)]))
        assert sum(bucket.count for bucket in report.distribution) == 0


class TestIssues:
    """Tests for the flagged-media list assembled by the report."""

    def test_low_quality_under_the_floor(self):
        """An image under the floor is flagged, one above it is not."""
        images = [_media(1, 40.0), _media(2, 90.0)]
        report = dataset_quality.evaluate(
            _snapshot(images, captions_by_id={1: "A.", 2: "B."})
        )
        flagged = [i for i in report.issues if i.kind == "low_quality"]
        assert [issue.media_ids for issue in flagged] == [(1,)]

    def test_near_dup_pair_picks_the_best(self):
        """The higher-quality image of a near-duplicate pair is kept."""
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.999, 0.045])}
        images = [_media(1, 60.0), _media(2, 90.0)]
        report = dataset_quality.evaluate(
            _snapshot(images, vectors_by_id=vectors)
        )
        dup = next(i for i in report.issues if i.kind == "near_dup")
        assert dup.key == "dup:1:2"
        assert dup.detail["best_id"] == 2
        assert dup.detail["loser_id"] == 1

    def test_degenerate_caption_is_flagged(self):
        """A looping caption is a CAPTION issue carrying its phrase."""
        report = dataset_quality.evaluate(
            _snapshot([_media(1, 90.0)], captions_by_id={1: "a red " * 12})
        )
        issue = next(i for i in report.issues if i.kind == "caption")
        assert issue.detail["phrase"] is not None

    def test_empty_caption_is_not_a_caption_issue(self):
        """A missing caption is a coverage gap, not a degenerate caption."""
        report = dataset_quality.evaluate(
            _snapshot([_media(1, 90.0)], captions_by_id={1: ""})
        )
        assert not [i for i in report.issues if i.kind == "caption"]

    def test_issues_are_sorted_by_impact(self):
        """The list is ordered by impact, strongest first."""
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.999, 0.045])}
        images = [_media(1, 30.0), _media(2, 90.0)]
        report = dataset_quality.evaluate(
            _snapshot(images, vectors_by_id=vectors)
        )
        impacts = [issue.impact for issue in report.issues]
        assert impacts == sorted(impacts, reverse=True)

    def test_fingerprint_tracks_the_measurement(self):
        """A caption issue's fingerprint changes with the caption text."""
        first = dataset_quality.evaluate(
            _snapshot([_media(1, 90.0)], captions_by_id={1: "a red " * 12})
        ).issues[0]
        second = dataset_quality.evaluate(
            _snapshot([_media(1, 90.0)], captions_by_id={1: "a blue " * 12})
        ).issues[0]
        assert first.key == second.key
        assert first.fingerprint != second.fingerprint


class TestFramingAndRecommendations:
    """Tests for the framing rows and the recommendations card."""

    def test_framing_rows_compare_to_the_target(self):
        """Each bucket reports its share and the target type's quota."""
        images = [_media(1, 70.0), _media(2, 70.0)]
        snapshot = _snapshot(
            images,
            tags_by_id={1: ["portrait"], 2: ["full_body"]},
            buckets=_BUCKETS,
            target_ratios={"face": 50, "full_body": 50},
        )
        rows = {
            row[0]: (row[2], row[3])
            for row in dataset_quality.evaluate(snapshot).framing
        }
        assert rows["face"] == (50.0, 50.0)
        assert rows["full_body"] == (50.0, 50.0)

    def test_recommendations_name_the_findings(self):
        """A dataset with a floor and a gap gets both recommendations."""
        images = [_media(1, 40.0), _media(2, 90.0)]
        report = dataset_quality.evaluate(
            _snapshot(images, captions_by_id={1: "A.", 2: ""})
        )
        heads = [item.head for item in report.recommendations]
        assert "Prune the floor" in heads
        assert "Fill the caption gaps" in heads

    def test_a_clean_dataset_still_advises_on_size(self):
        """Nothing flagged: the only advice left is the size range."""
        images = [_media(index, 95.0) for index in range(1, 4)]
        report = dataset_quality.evaluate(
            _snapshot(
                images,
                scorers=("musiq",),
                captions_by_id={index: "A cat." for index in range(1, 4)},
            )
        )
        heads = [item.head for item in report.recommendations]
        assert heads == ["Grow the set"]


class TestSerialisation:
    """The report must survive a round-trip through the stored blob."""

    def test_to_dict_is_json_ready(self):
        """Every nested dataclass and tuple becomes JSON-serialisable."""
        vectors = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
        report = dataset_quality.evaluate(
            _snapshot(
                [_media(1, 40.0), _media(2, 90.0)],
                vectors_by_id=vectors,
                captions_by_id={1: "a red " * 12, 2: "A dog."},
            )
        )
        payload = json.loads(json.dumps(dataset_quality.to_dict(report)))
        assert payload["grade"] == report.grade
        assert len(payload["issues"]) == len(report.issues)
        assert payload["pillars"][0]["key"] == "quality"
