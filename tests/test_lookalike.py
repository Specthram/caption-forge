"""Tests for :mod:`src.lookalike` (perceptual near-duplicate engine).

Pure in-memory: media are plain dicts carrying hand-picked hex hashes, so
the tests pin the Hamming maths, the pHash/dHash cross-validation, the
score/distance mapping, the union-find grouping and the sort orders without
any database or image decoding.
"""

import numpy as np

from src import lookalike


def _hex(value: int) -> str:
    """Return a 16-hex-char (64-bit) hash string for an integer."""
    return format(value, "016x")


def _media(media_id, phash, dhash, quality=None, width=10, height=10):
    """Return a media dict shaped like ``media_with_hashes`` output."""
    return {
        "id": media_id,
        "sha256": f"sha{media_id}",
        "phash": _hex(phash),
        "dhash": _hex(dhash),
        "quality_score": quality,
        "quality_metric": "musiq" if quality is not None else None,
        "width": width,
        "height": height,
        "eff_path": f"/x/{media_id}.png",
        "name": f"{media_id}.png",
    }


class TestHammingMatrix:
    """Tests for the block popcount helper."""

    def test_known_distances(self):
        """XOR + popcount matches hand-computed bit differences."""
        left = np.array([0b0000, 0b1111], dtype=np.uint64)
        right = np.array([0b0000, 0b1010, 0b1111], dtype=np.uint64)
        matrix = lookalike._hamming_matrix(left, right)
        assert matrix.tolist() == [[0, 2, 4], [4, 2, 0]]

    def test_full_64_bit_distance(self):
        """A hash and its complement differ in all 64 bits."""
        left = np.array([0], dtype=np.uint64)
        right = np.array([np.uint64(0xFFFFFFFFFFFFFFFF)], dtype=np.uint64)
        assert lookalike._hamming_matrix(left, right)[0, 0] == 64


class TestScoreDistanceMapping:
    """Tests for the centralized similarity <-> distance mapping."""

    def test_distance_zero_is_100_percent(self):
        """Bit-identical hashes read as a perfect 100% similarity."""
        assert lookalike.distance_to_similarity(0) == 100

    def test_similarity_to_distance_default(self):
        """The 88% default maps to a max Hamming distance of 4."""
        assert lookalike.similarity_to_distance(88) == 4

    def test_mapping_round_trips(self):
        """distance -> similarity -> distance is stable on exact points."""
        for distance in range(0, 8):
            similarity = lookalike.distance_to_similarity(distance)
            assert lookalike.similarity_to_distance(similarity) == distance

    def test_similarity_100_requires_identical(self):
        """A 100% threshold allows only a zero distance."""
        assert lookalike.similarity_to_distance(100) == 0

    def test_similarity_clamped_to_zero(self):
        """A large distance never yields a negative percentage."""
        assert lookalike.distance_to_similarity(40) == 0


class TestCrossValidation:
    """A pair must match on both hashes to be kept."""

    def test_matching_phash_but_far_dhash_is_dropped(self):
        """A close pHash with a far dHash is not a lookalike."""
        media = [
            _media(1, 0x0, 0x0),
            _media(2, 0x1, 0xFF),  # pHash d=1 (ok), dHash d=8 (too far)
        ]
        result = lookalike.detect(media, similarity=88)  # max distance 4
        assert result.groups == ()

    def test_both_close_is_kept(self):
        """Both hashes within the threshold groups the pair."""
        media = [
            _media(1, 0x0, 0x0),
            _media(2, 0x1, 0x1),
        ]
        result = lookalike.detect(media, similarity=88)
        assert len(result.groups) == 1
        assert len(result.groups[0].media) == 2


class TestUnionFindGrouping:
    """Transitive pairs collapse into a single group."""

    def test_chain_a_b_c_is_one_group(self):
        """A-B and B-C matching (but not A-C) still yields one group."""
        # At similarity 97 the max distance is 1: A-B and B-C are distance
        # 1, A-C is distance 2, so only the chain links connect them.
        media = [
            _media(1, 0x0, 0x0),
            _media(2, 0x1, 0x1),
            _media(3, 0x3, 0x3),
        ]
        result = lookalike.detect(media, similarity=97)
        assert len(result.groups) == 1
        ids = {m.media_id for m in result.groups[0].media}
        assert ids == {1, 2, 3}


class TestSorting:
    """Intra-group and inter-group ordering."""

    def test_intra_group_orders_by_quality_then_resolution(self):
        """Best quality first, a never-scored media last, area breaks ties."""
        media = [
            _media(1, 0x0, 0x0, quality=50, width=100, height=100),
            _media(2, 0x0, 0x0, quality=90, width=10, height=10),
            _media(3, 0x0, 0x0, quality=None, width=999, height=999),
            _media(4, 0x0, 0x0, quality=90, width=200, height=200),
        ]
        result = lookalike.detect(media, similarity=100)
        order = [m.media_id for m in result.groups[0].media]
        # 90-quality media first (larger area wins the tie), then 50, then
        # the unscored one despite its huge resolution.
        assert order == [4, 2, 1, 3]

    def test_representative_similarity_is_100(self):
        """The group representative is 100% similar to itself."""
        media = [
            _media(1, 0x0, 0x0, quality=90),
            _media(2, 0x1, 0x1, quality=50),
        ]
        result = lookalike.detect(media, similarity=88)
        rep = result.groups[0].media[0]
        assert rep.media_id == 1 and rep.similarity == 100
        # The other member is one bit off -> 97%.
        assert result.groups[0].media[1].similarity == 97

    def test_groups_sorted_by_best_quality_desc(self):
        """The group holding the highest-quality media comes first."""
        media = [
            _media(1, 0x0, 0x0, quality=40),
            _media(2, 0x1, 0x1, quality=30),
            _media(3, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, quality=95),
            _media(4, 0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFE, quality=20),
        ]
        result = lookalike.detect(media, similarity=88)
        assert len(result.groups) == 2
        assert result.groups[0].best_quality == 95
        assert result.groups[1].best_quality == 40


class TestEmptyCases:
    """Degenerate inputs return clean empty results."""

    def test_no_media(self):
        """No media at all -> no groups, zero hashed."""
        result = lookalike.detect([])
        assert result.groups == ()
        assert result.hashed_count == 0

    def test_single_media(self):
        """A lone media cannot pair with anything."""
        result = lookalike.detect([_media(1, 0x0, 0x0)])
        assert result.groups == ()
        assert result.hashed_count == 1

    def test_no_matches(self):
        """Two far-apart media form no group."""
        media = [
            _media(1, 0x0, 0x0),
            _media(2, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
        ]
        result = lookalike.detect(media, similarity=88)
        assert result.groups == ()
        assert result.hashed_count == 2

    def test_media_missing_a_hash_is_ignored(self):
        """A media whose hashes are None is dropped before comparison."""
        media = [
            _media(1, 0x0, 0x0),
            {"id": 2, "sha256": "s2", "phash": None, "dhash": None},
        ]
        result = lookalike.detect(media, similarity=88)
        assert result.hashed_count == 1
        assert result.groups == ()
