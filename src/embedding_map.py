"""2-D projection, clustering and outlier detection over DINOv2 vectors.

The dataset quality report shows the embedded media as a scatter plot: one
dot per image, coloured by cluster, near-duplicate pairs and outliers
ringed. Everything here is pure NumPy — a PCA projection (truncated SVD),
a seeded k-means and a sigma threshold on the distance to the nearest
centroid — so the report needs no UMAP/HDBSCAN dependency and stays
deterministic: the same dataset always draws the same map.

Distances are cosine distances (``1 - cosine similarity``) over the
L2-normalized vectors :func:`src.embeddings.embed_image` stores.
"""

from dataclasses import dataclass

import numpy as np

# Cluster count bounds. k grows with the dataset (``sqrt(n / 2)``), which
# keeps a 20-image character set at 2-3 clusters and a 300-image style set
# at the ceiling — enough colour separation to read the map, never so many
# that every dot is its own cluster.
MIN_CLUSTERS = 2
MAX_CLUSTERS = 6

# A media whose mean cosine distance to its NEIGHBOR_COUNT nearest
# neighbors exceeds ``mean + OUTLIER_SIGMA * std`` is an outlier: nothing in
# the dataset looks like it, which in a training set means "off-concept".
# The neighbors, not the cluster centroid: on the 16-to-50-image datasets a
# LoRA is trained on, k-means happily gives a lone outlier a centroid of its
# own, and its distance to that centroid is then zero. The distance to what
# is actually *near* it cannot be gamed that way.
OUTLIER_SIGMA = 2.5
NEIGHBOR_COUNT = 3

# k-means: fixed seed and iteration cap (the vectors are L2-normalized and
# the k is small, so it converges well before the cap).
_SEED = 20240517
_MAX_ITERATIONS = 40


@dataclass(frozen=True)
class MapResult:  # pylint: disable=too-many-instance-attributes
    """The scatter plot of an embedded dataset.

    ``ids`` row order for every array here; ``coords`` one ``(x, y)`` per
    media, each axis min-max to ``[0, 1]``; ``labels`` cluster index each;
    ``cluster_count`` clusters fitted; ``outlier_score`` mean cosine distance
    to the :data:`NEIGHBOR_COUNT` nearest neighbors (how alone);
    ``outlier_threshold`` the cutoff; ``outliers`` outlying ids; ``spread``
    mean pairwise cosine distance; ``similarity`` all-pairs cosine matrix,
    diagonal zeroed (the near-duplicate detector reuses it).
    """

    ids: tuple
    coords: tuple
    labels: tuple
    cluster_count: int
    outlier_score: tuple
    outlier_threshold: float
    outliers: tuple
    spread: float
    similarity: np.ndarray


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Return the vectors L2-normalized row-wise (zero rows kept zero)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return the all-pairs cosine similarity matrix, diagonal zeroed.

    The stored vectors are already L2-normalized, but they are normalized
    again defensively so the similarities stay inside ``[-1, 1]`` whatever
    the source.
    """
    unit = unit_rows(vectors)
    matrix = unit @ unit.T
    np.fill_diagonal(matrix, 0.0)
    return matrix


def mean_pairwise_distance(matrix: np.ndarray) -> float:
    """Return the mean cosine distance over every unordered pair."""
    count = matrix.shape[0]
    if count < 2:
        return 0.0
    upper = matrix[np.triu_indices(count, k=1)]
    return float(1.0 - upper.mean())


def suggested_k(count: int) -> int:
    """Return the cluster count to fit for ``count`` media."""
    if count < 2 * MIN_CLUSTERS:
        return 1
    k = round((count / 2.0) ** 0.5)
    return int(max(MIN_CLUSTERS, min(MAX_CLUSTERS, k, count - 1)))


def _init_centroids(unit: np.ndarray, k: int, rng) -> np.ndarray:
    """Return k-means++ seeds: spread-out starting centroids."""
    first = int(rng.integers(unit.shape[0]))
    centroids = [unit[first]]
    for _ in range(1, k):
        distances = 1.0 - np.max(
            unit @ np.stack(centroids).T, axis=1  # cosine distance
        )
        distances = np.clip(distances, 0.0, None)
        total = float(distances.sum())
        if total <= 0:
            centroids.append(unit[int(rng.integers(unit.shape[0]))])
            continue
        pick = int(rng.choice(unit.shape[0], p=distances / total))
        centroids.append(unit[pick])
    return np.stack(centroids)


def kmeans(vectors: np.ndarray, k: int) -> tuple:
    """Fit a seeded spherical k-means; return ``(labels, centroids)``.

    Spherical because the vectors live on the unit sphere: assignment
    maximizes cosine similarity and the recomputed centroids are
    re-normalized, which is the natural metric for DINOv2 features.
    """
    unit = unit_rows(vectors)
    if k <= 1:
        centroid = unit_rows(unit.mean(axis=0, keepdims=True))
        return np.zeros(unit.shape[0], dtype=int), centroid
    rng = np.random.default_rng(_SEED)
    centroids = _init_centroids(unit, k, rng)
    labels = np.zeros(unit.shape[0], dtype=int)
    for _ in range(_MAX_ITERATIONS):
        new_labels = np.argmax(unit @ centroids.T, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for index in range(k):
            members = unit[labels == index]
            if members.size:
                centroids[index] = members.mean(axis=0)
        centroids = unit_rows(centroids)
    return labels, centroids


def project(vectors: np.ndarray) -> np.ndarray:
    """Return a ``(n, 2)`` PCA projection, each axis scaled to ``[0, 1]``.

    A truncated SVD of the centered vectors: the two leading principal
    components carry the dominant visual variation, which is all the map
    needs to separate clusters at a glance.
    """
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return np.zeros((centered.shape[0], 2))
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    axes = components[:2]
    if axes.shape[0] == 1:  # a single non-degenerate direction
        axes = np.vstack([axes, np.zeros_like(axes)])
    coords = centered @ axes.T
    span = coords.max(axis=0) - coords.min(axis=0)
    span[span == 0] = 1.0
    return (coords - coords.min(axis=0)) / span


def neighbor_distances(similarity: np.ndarray) -> np.ndarray:
    """Return each row's mean cosine distance to its nearest neighbors.

    The diagonal of ``similarity`` is zeroed, so a media would otherwise
    count itself as a distance-1 neighbor; it is masked out with ``-inf``
    before the top-k selection.
    """
    count = similarity.shape[0]
    neighbors = min(NEIGHBOR_COUNT, count - 1)
    masked = similarity.copy()
    np.fill_diagonal(masked, -np.inf)
    nearest = np.sort(masked, axis=1)[:, -neighbors:]
    return 1.0 - nearest.mean(axis=1)


def build(ids, vectors, sigma: float = OUTLIER_SIGMA) -> MapResult | None:
    """Project, cluster and flag the outliers of an embedded dataset.

    ``ids`` aligned with ``vectors`` (the DINOv2 vectors); ``sigma`` how many
    std devs above the mean an outlier sits (:data:`OUTLIER_SIGMA`). Returns a
    :class:`MapResult`, or None when fewer than two media are embedded.
    """
    ids = tuple(int(media_id) for media_id in ids)
    if len(ids) < 2:
        return None
    stacked = np.stack([np.asarray(v, dtype=np.float32) for v in vectors])
    unit = unit_rows(stacked.astype(np.float64))
    similarity = cosine_matrix(stacked.astype(np.float64))
    k = suggested_k(len(ids))
    labels, _ = kmeans(unit, k)
    scores = neighbor_distances(similarity)
    deviation = float(scores.std())
    threshold = float(scores.mean() + sigma * deviation)
    # A uniform dataset (every media equally close to its neighbors, e.g.
    # two identical images) has no outlier — without this guard the
    # threshold would collapse onto the mean and flag half the dataset.
    outliers = (
        ()
        if deviation <= 1e-9
        else tuple(
            media_id
            for media_id, score in zip(ids, scores)
            if score > threshold
        )
    )
    coords = project(unit)
    return MapResult(
        ids=ids,
        coords=tuple((float(x), float(y)) for x, y in coords),
        labels=tuple(int(label) for label in labels),
        cluster_count=int(len(set(labels.tolist()))),
        outlier_score=tuple(float(score) for score in scores),
        outlier_threshold=threshold,
        outliers=outliers,
        spread=mean_pairwise_distance(similarity),
        similarity=similarity,
    )
