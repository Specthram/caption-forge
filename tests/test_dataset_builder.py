"""Tests for the auto-build selection engine (:mod:`src.dataset_builder`).

Pure-function tests over synthetic media dicts and low-dimensional
vectors — no database, no model. The quality metric used throughout is
the normalized average pseudo-metric, whose 0-100 range makes the raw
scores and the normalized ones identical.
"""

import numpy as np

from src import dataset_builder
from src import framing
from src import quality
from src.embeddings import vector_to_blob

_BUCKETS = {
    "face": ["portrait"],
    "upper_body": ["upper_body"],
    "full_body": ["full_body"],
}


def _media(
    media_id,
    phash,
    favorite=False,
    score=50.0,
    width=100,
    height=100,
):
    """Return a synthetic pool media dict (hashes + display quality)."""
    return {
        "id": media_id,
        "sha256": f"sha{media_id}",
        "favorite": favorite,
        "phash": phash,
        "dhash": phash,
        "width": width,
        "height": height,
        "quality_score": score,
        "quality_metric": quality.AVERAGE_METRIC_ID,
    }


def _unit(x, y):
    """Return a 2D unit vector as an embedding BLOB."""
    vector = np.array([x, y], dtype=np.float32)
    return vector_to_blob(vector / np.linalg.norm(vector))


def _distinct_hash(index: int) -> str:
    """Return one of 4 maximally-distant 16-hex-char hashes."""
    return ["0" * 16, "f" * 16, ("0f" * 8), ("f0" * 8)][index % 4]


def _tags(mapping):
    """Return a ``tags_for_media_bulk``-shaped map from ``{id: [names]}``."""
    return {
        media_id: [{"name": name} for name in names]
        for media_id, names in mapping.items()
    }


class TestQuotas:
    """Tests for the ratio-to-quota split."""

    def test_quotas_sum_to_the_size(self):
        """Rounding leftovers land on the largest fractional parts."""
        quotas = dataset_builder._quotas(  # pylint: disable=protected-access
            10, {"face": 50, "upper_body": 30, "full_body": 20}
        )
        assert quotas == {"face": 5, "upper_body": 3, "full_body": 2}

    def test_zero_and_negative_weights_are_ignored(self):
        """Only positive weights earn a quota."""
        quotas = dataset_builder._quotas(  # pylint: disable=protected-access
            4, {"face": 100, "upper_body": 0, "full_body": -3}
        )
        assert quotas == {"face": 4}


class TestPrepareCandidates:
    """Tests for the pool-preparation pass."""

    def test_quality_bar_drops_low_and_unscored(self):
        """An active bar drops scored-below and unscored media alike."""
        media = [
            _media(1, _distinct_hash(0), score=80.0),
            _media(2, _distinct_hash(1), score=20.0),
            _media(3, _distinct_hash(2), score=None),
        ]
        candidates, stats = dataset_builder.prepare_candidates(
            media, {}, {}, _BUCKETS, min_quality=50.0
        )
        assert [c["id"] for c in candidates] == [1]
        assert stats.quality_dropped == 1
        assert stats.unscored_dropped == 1

    def test_dedup_keeps_the_favorite_of_a_group(self):
        """A favorite survives its lookalike group over a better score."""
        media = [
            _media(1, "0" * 16, score=90.0),
            _media(2, "0" * 16, favorite=True, score=40.0),
            _media(3, _distinct_hash(1), score=50.0),
        ]
        candidates, stats = dataset_builder.prepare_candidates(
            media, {}, {}, _BUCKETS
        )
        assert {c["id"] for c in candidates} == {2, 3}
        assert stats.duplicates_dropped == 1

    def test_classifies_framing_and_counts_unknown(self):
        """Tags map to buckets; untagged media count as unknown."""
        media = [
            _media(1, _distinct_hash(0)),
            _media(2, _distinct_hash(1)),
        ]
        candidates, stats = dataset_builder.prepare_candidates(
            media, _tags({1: ["portrait"]}), {}, _BUCKETS
        )
        by_id = {c["id"]: c["bucket"] for c in candidates}
        assert by_id == {1: "face", 2: framing.UNKNOWN_BUCKET}
        assert stats.unknown_count == 1


class TestSelect:
    """Tests for the quota + farthest-point selection."""

    def _candidate(self, cid, bucket, vector=None, favorite=False, score=50.0):
        """Return a prepared candidate dict."""
        return {
            "id": cid,
            "favorite": favorite,
            "quality": score,
            "bucket": bucket,
            "vector": vector,
        }

    def test_ratios_are_respected(self):
        """Each bucket fills its quota when candidates abound."""
        candidates = [self._candidate(i, "face") for i in range(1, 11)] + [
            self._candidate(i, "full_body") for i in range(11, 21)
        ]
        result = dataset_builder.select(
            candidates, 10, {"face": 70, "full_body": 30}
        )
        assert result.buckets["face"].selected == 7
        assert result.buckets["full_body"].selected == 3
        assert len(result.media_ids) == 10

    def test_favorites_are_guaranteed_but_quota_capped(self):
        """Favorites fill first, best quality first, never past the quota."""
        candidates = [
            self._candidate(1, "face", favorite=True, score=30.0),
            self._candidate(2, "face", favorite=True, score=90.0),
            self._candidate(3, "face", favorite=True, score=60.0),
            self._candidate(4, "face", score=99.0),
        ]
        result = dataset_builder.select(candidates, 2, {"face": 100})
        assert set(result.media_ids) == {2, 3}
        assert result.buckets["face"].favorites == 2

    def test_farthest_point_avoids_near_duplicates(self):
        """The second pick is the most distant vector, not the closest."""
        base = np.array([1.0, 0.0], dtype=np.float32)
        near = np.array([0.999, 0.045], dtype=np.float32)
        near /= np.linalg.norm(near)
        far = np.array([0.0, 1.0], dtype=np.float32)
        candidates = [
            self._candidate(1, "face", vector=base),
            self._candidate(2, "face", vector=near),
            self._candidate(3, "face", vector=far),
        ]
        result = dataset_builder.select(candidates, 2, {"face": 100})
        assert set(result.media_ids) == {1, 3}

    def test_redistribution_fills_a_starved_bucket(self):
        """A short bucket's slots go to the other ratio-eligible media."""
        candidates = [self._candidate(1, "face")] + [
            self._candidate(i, "full_body") for i in range(2, 12)
        ]
        result = dataset_builder.select(
            candidates, 6, {"face": 50, "full_body": 50}
        )
        assert len(result.media_ids) == 6
        assert result.redistributed == 2
        assert result.shortfall == 0

    def test_any_ratio_takes_unknown_media(self):
        """An ``any`` ratio draws from the whole pool, unknown included."""
        candidates = [
            self._candidate(1, framing.UNKNOWN_BUCKET, score=90.0),
            self._candidate(2, "face", score=10.0),
        ]
        result = dataset_builder.select(
            candidates, 2, {framing.ANY_BUCKET: 100}
        )
        assert set(result.media_ids) == {1, 2}

    def test_shortfall_is_reported(self):
        """Too few candidates leave a shortfall, never an error."""
        candidates = [self._candidate(1, "face")]
        result = dataset_builder.select(candidates, 5, {"face": 100})
        assert result.shortfall == 4

    def test_vectorless_candidates_compete_on_quality(self):
        """With no embeddings at all the pick order is by quality."""
        candidates = [
            self._candidate(1, "face", score=10.0),
            self._candidate(2, "face", score=90.0),
            self._candidate(3, "face", score=50.0),
        ]
        result = dataset_builder.select(candidates, 2, {"face": 100})
        assert result.media_ids == [2, 3]

    def test_selection_is_deterministic(self):
        """The same pool and parameters always select the same media."""
        rng = np.random.default_rng(7)
        candidates = []
        for cid in range(1, 30):
            vector = rng.normal(size=3).astype(np.float32)
            vector /= np.linalg.norm(vector)
            candidates.append(
                self._candidate(
                    cid,
                    "face" if cid % 2 else "full_body",
                    vector=vector,
                    score=float(cid),
                )
            )
        first = dataset_builder.select(
            candidates, 10, {"face": 60, "full_body": 40}
        )
        second = dataset_builder.select(
            candidates, 10, {"face": 60, "full_body": 40}
        )
        assert first.media_ids == second.media_ids


class TestBuild:
    """End-to-end engine test (prepare + select)."""

    def test_full_pipeline(self):
        """Dedup, quality bar, ratios and diversity compose correctly."""
        media = [
            _media(1, "0" * 16, score=90.0),  # dup group with 2
            _media(2, "0" * 16, favorite=True, score=80.0),
            _media(3, "f" * 16, score=85.0),
            _media(4, "0f" * 8, score=10.0),  # below the bar
            _media(5, "f0" * 8, score=70.0),
        ]
        tags = _tags(
            {
                1: ["portrait"],
                2: ["portrait"],
                3: ["full_body"],
                4: ["full_body"],
                5: ["full_body"],
            }
        )
        vectors = {
            2: _unit(1.0, 0.0),
            3: _unit(0.0, 1.0),
            5: _unit(1.0, 1.0),
        }
        result = dataset_builder.build(
            media,
            tags,
            vectors,
            3,
            {"face": 40, "full_body": 60},
            _BUCKETS,
            min_quality=50.0,
        )
        assert result.duplicates_dropped == 1
        assert result.quality_dropped == 1
        assert set(result.media_ids) == {2, 3, 5}
        assert result.buckets["face"].favorites == 1
