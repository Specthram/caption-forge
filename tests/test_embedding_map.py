"""Tests for :mod:`src.embedding_map` (projection, clusters, outliers).

Pure NumPy over hand-built vectors: the map must be deterministic (the
same dataset always draws the same picture) and must not invent outliers
in a uniform dataset.
"""

import numpy as np

from src import embedding_map


def _blob(center: np.ndarray, count: int, spread: float = 0.02):
    """Return ``count`` vectors jittered around ``center`` (seeded)."""
    rng = np.random.default_rng(7)
    noise = rng.normal(scale=spread, size=(count, center.size))
    return list(center + noise)


class TestCosine:
    """Tests for the similarity helpers."""

    def test_diagonal_is_zeroed(self):
        """A media is never its own near-duplicate."""
        vectors = np.eye(3)
        matrix = embedding_map.cosine_matrix(vectors)
        assert np.allclose(np.diag(matrix), 0.0)

    def test_orthogonal_rows_have_distance_one(self):
        """Orthogonal vectors sit at cosine distance 1."""
        matrix = embedding_map.cosine_matrix(np.eye(3))
        assert embedding_map.mean_pairwise_distance(matrix) == 1.0

    def test_identical_rows_have_distance_zero(self):
        """Copies of one vector have no spread."""
        vectors = np.tile(np.array([1.0, 0.0, 0.0]), (4, 1))
        matrix = embedding_map.cosine_matrix(vectors)
        assert embedding_map.mean_pairwise_distance(matrix) == 0.0

    def test_zero_row_survives_normalization(self):
        """A zero vector does not divide by zero."""
        vectors = np.array([[0.0, 0.0], [1.0, 0.0]])
        assert np.isfinite(embedding_map.cosine_matrix(vectors)).all()


class TestSuggestedK:
    """Tests for the cluster-count heuristic."""

    def test_tiny_dataset_gets_one_cluster(self):
        """Under four media there is nothing to cluster."""
        assert embedding_map.suggested_k(3) == 1

    def test_k_grows_with_the_dataset(self):
        """k follows sqrt(n / 2), inside the documented bounds."""
        assert embedding_map.suggested_k(8) == 2
        assert embedding_map.suggested_k(50) == 5

    def test_k_is_capped(self):
        """A huge dataset never draws more than MAX_CLUSTERS colours."""
        assert embedding_map.suggested_k(5000) == embedding_map.MAX_CLUSTERS


class TestBuild:
    """Tests for the assembled map."""

    def test_fewer_than_two_media_has_no_map(self):
        """One embedded image cannot be projected."""
        assert embedding_map.build([1], [np.array([1.0, 0.0])]) is None

    def test_two_blobs_split_into_two_clusters(self):
        """Two separated groups get two cluster labels."""
        left = _blob(np.array([1.0, 0.0, 0.0]), 5)
        right = _blob(np.array([0.0, 1.0, 0.0]), 5)
        result = embedding_map.build(range(10), left + right)
        assert result.cluster_count == 2
        assert len(set(result.labels[:5])) == 1
        assert result.labels[0] != result.labels[5]

    def test_coordinates_are_normalized(self):
        """Every dot lands inside the unit box the front-end draws in."""
        vectors = _blob(np.array([1.0, 0.0, 0.0]), 6) + _blob(
            np.array([0.0, 0.0, 1.0]), 6
        )
        result = embedding_map.build(range(12), vectors)
        coords = np.array(result.coords)
        assert coords.min() >= 0.0
        assert coords.max() <= 1.0

    def test_uniform_dataset_has_no_outlier(self):
        """Identical neighbor distances must not flag anyone."""
        vectors = np.tile(np.array([1.0, 0.0, 0.0]), (5, 1))
        result = embedding_map.build(range(5), list(vectors))
        assert result.outliers == ()

    def test_a_lonely_vector_is_an_outlier(self):
        """An image with no close neighbor is flagged."""
        vectors = _blob(np.array([1.0, 0.0, 0.0]), 9, spread=0.001)
        vectors.append(np.array([0.0, 0.0, 1.0]))
        result = embedding_map.build(range(10), vectors, sigma=2.0)
        assert result.outliers == (9,)

    def test_a_singleton_cluster_cannot_hide_an_outlier(self):
        """k-means may give the outlier its own centroid; kNN still sees it.

        The regression this guards: scoring the outlier by its distance to
        its *own* centroid reads zero the moment k-means hands it a cluster
        of one, and the report silently stops flagging anything.
        """
        vectors = _blob(np.array([1.0, 0.0, 0.0]), 9, spread=0.001)
        vectors.append(np.array([0.0, 0.0, 1.0]))
        result = embedding_map.build(range(10), vectors, sigma=2.0)
        # The lone vector did capture a cluster of its own...
        assert result.labels.count(result.labels[9]) == 1
        # ... and is flagged all the same.
        assert 9 in result.outliers

    def test_the_map_is_deterministic(self):
        """The same vectors always draw the same map."""
        vectors = _blob(np.array([1.0, 0.0, 0.0]), 4) + _blob(
            np.array([0.0, 1.0, 0.0]), 4
        )
        first = embedding_map.build(range(8), vectors)
        second = embedding_map.build(range(8), vectors)
        assert first.labels == second.labels
        assert first.coords == second.coords
