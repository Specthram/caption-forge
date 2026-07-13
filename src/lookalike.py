"""Perceptual near-duplicate ("lookalike") detection engine.

The Libraries tab's "Index" action stores two perceptual hashes per media
(``phash``/``dhash``, see :mod:`src.perceptual_hash`). Unlike a content
``sha256`` — where a single changed byte gives a completely different hash —
these stay close for the *same* picture re-encoded, resized or recompressed.
This module turns those hashes into groups of lookalike media.

The pipeline is, in order:

1. Convert every media's hex hash into a 64-bit integer.
2. Compute all-pairs Hamming distances (bit differences) with XOR +
   popcount, in blocks so the :math:`O(N^2)` scan never materializes the
   whole distance matrix — bounded memory at ~100k media.
3. Keep a pair only when *both* distances are within the threshold
   (cross-validating pHash against dHash cuts false positives).
4. Union-find the kept pairs into connected groups.

It is read-only: it detects and describes groups, and never hides or
deletes anything.
"""

import math
import os
from dataclasses import dataclass, field

import numpy as np

# Similarity mapping. Hamming distance 0 (bit-identical) = 100%; each differing
# bit costs _SIMILARITY_SLOPE points. Steep on purpose so the default threshold
# maps to a tiny distance. Centralized so engine + UI slider agree.
_SIMILARITY_SLOPE = 3

# The default "minimum similarity" the UI slider opens on: 88% maps to a max
# Hamming distance of 4 (``(100 - 88) / 3``).
DEFAULT_SIMILARITY = 88

# Rows compared per block in the pairwise scan: a block pair holds a
# ``_BLOCK x _BLOCK`` distance sub-matrix in memory (~a few MB at 1024),
# keeping the quadratic scan's footprint flat whatever N is.
_BLOCK = 1024

# Population-count lookup for a single byte, used to popcount a uint64 array
# by viewing it as 8 bytes and summing their set-bit counts.
_POPCOUNT_TABLE = np.array(
    [bin(value).count("1") for value in range(256)], dtype=np.uint8
)


@dataclass(frozen=True)
class LookalikeMedia:  # pylint: disable=too-many-instance-attributes
    """One media within a lookalike group.

    ``media_id`` db id; ``sha256`` content hash (also thumb key);
    ``similarity`` 0-100 vs the group representative (itself 100);
    ``quality_score``/``quality_metric`` stored score + its metric;
    ``width``/``height`` indexed dims; ``eff_path`` effective file (None when
    missing); ``name`` display label.
    """

    media_id: int
    sha256: str
    similarity: int
    quality_score: float | None
    quality_metric: str | None
    width: int | None
    height: int | None
    eff_path: str | None
    name: str


@dataclass(frozen=True)
class LookalikeGroup:
    """A connected group of mutually-lookalike media.

    ``media`` best first (quality desc, unscored last, then area desc); the
    first is the representative. ``best_quality`` the group's top score (None
    when no member was scored) — the key groups rank by.
    """

    media: tuple[LookalikeMedia, ...]
    best_quality: float | None = None


@dataclass(frozen=True)
class LookalikeResult:
    """The outcome of a detection run.

    ``groups`` best first (by :attr:`LookalikeGroup.best_quality` desc), empty
    when nothing matched; ``hashed_count`` the scan's N; ``similarity`` the
    threshold used, percent.
    """

    groups: tuple[LookalikeGroup, ...] = field(default_factory=tuple)
    hashed_count: int = 0
    similarity: int = DEFAULT_SIMILARITY


def similarity_to_distance(similarity: float) -> int:
    """Return the max Hamming distance for a minimum similarity percentage.

    Inverse of the similarity mapping (see module docstring): a pair matches
    when its distance is at most the returned value. 100% maps to 0
    (bit-identical only); lower thresholds allow more differing bits.
    ``similarity`` is 0-100; returns the largest matching distance (>= 0).
    """
    span = (100.0 - similarity) / _SIMILARITY_SLOPE
    # A tiny epsilon absorbs float error so e.g. 88 -> 12/3 = 4 exactly.
    return max(0, int(math.floor(span + 1e-9)))


def distance_to_similarity(distance: int) -> int:
    """Return the similarity percentage for a Hamming distance.

    Distance 0 -> 100%; each differing bit subtracts
    :data:`_SIMILARITY_SLOPE` points, clamped to ``[0, 100]``.
    """
    return int(max(0, min(100, 100 - distance * _SIMILARITY_SLOPE)))


class _UnionFind:
    """Minimal union-find (disjoint set) over ``n`` integer elements."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, item: int) -> int:
        """Return ``item``'s set root, compressing the path on the way."""
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, first: int, second: int) -> None:
        """Merge the sets containing ``first`` and ``second``."""
        root_a, root_b = self.find(first), self.find(second)
        if root_a != root_b:
            self._parent[root_b] = root_a

    def groups(self) -> list[list[int]]:
        """Return the members of each set of size two or more."""
        buckets: dict[int, list[int]] = {}
        for item in range(len(self._parent)):
            buckets.setdefault(self.find(item), []).append(item)
        return [members for members in buckets.values() if len(members) > 1]


def _to_uint64(hex_values) -> np.ndarray:
    """Return a uint64 array from 16-hex-char perceptual hash strings."""
    return np.array([int(value, 16) for value in hex_values], dtype=np.uint64)


def _hamming_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the pairwise Hamming distances between two uint64 vectors.

    ``left`` (length ``a``) against ``right`` (length ``b``) gives an
    ``a x b`` matrix of bit-difference counts, via XOR then a byte-wise
    popcount (each uint64 viewed as 8 bytes summed through
    :data:`_POPCOUNT_TABLE`).
    """
    xor = left[:, None] ^ right[None, :]
    as_bytes = xor.reshape(-1).view(np.uint8).reshape(xor.shape + (8,))
    return _POPCOUNT_TABLE[as_bytes].sum(axis=-1)


def _matched_pairs(
    phash: np.ndarray, dhash: np.ndarray, max_distance: int
) -> list[tuple[int, int]]:
    """Return the index pairs ``(i, j)``, ``i < j``, matching on both hashes.

    Scans the upper triangle in ``_BLOCK``-sized tiles so no distance
    sub-matrix bigger than one tile ever exists at once. A pair is kept
    only when the pHash *and* the dHash distances are within
    ``max_distance``.
    """
    n = len(phash)
    pairs: list[tuple[int, int]] = []
    for i0 in range(0, n, _BLOCK):
        i1 = min(i0 + _BLOCK, n)
        for j0 in range(i0, n, _BLOCK):
            j1 = min(j0 + _BLOCK, n)
            dp = _hamming_matrix(phash[i0:i1], phash[j0:j1])
            dd = _hamming_matrix(dhash[i0:i1], dhash[j0:j1])
            mask = (dp <= max_distance) & (dd <= max_distance)
            local_i, local_j = np.nonzero(mask)
            global_i = local_i + i0
            global_j = local_j + j0
            keep = global_i < global_j
            pairs.extend(zip(global_i[keep].tolist(), global_j[keep].tolist()))
    return pairs


def _area(item: dict) -> int:
    """Return a media dict's pixel area, or 0 when a dimension is missing."""
    width = item.get("width") or 0
    height = item.get("height") or 0
    return int(width) * int(height)


def _quality_sort_key(item: dict):
    """Return a "best quality, then largest" descending sort key.

    A media never scored (``quality_score`` is None) sorts after every
    scored one, matching how the grids rank an un-indexed media last.
    """
    score = item.get("quality_score")
    scored = score is not None
    return (scored, score if scored else 0.0, _area(item))


def _build_group(
    members: list[dict], phash_int: dict, dhash_int: dict
) -> LookalikeGroup:
    """Return a sorted :class:`LookalikeGroup` for one connected component."""
    ordered = sorted(members, key=_quality_sort_key, reverse=True)
    representative = ordered[0]
    rep_p = phash_int[representative["id"]]
    rep_d = dhash_int[representative["id"]]
    media = []
    for item in ordered:
        distance = max(
            (phash_int[item["id"]] ^ rep_p).bit_count(),
            (dhash_int[item["id"]] ^ rep_d).bit_count(),
        )
        media.append(
            LookalikeMedia(
                media_id=item["id"],
                sha256=item["sha256"],
                similarity=distance_to_similarity(distance),
                quality_score=item.get("quality_score"),
                quality_metric=item.get("quality_metric"),
                width=item.get("width"),
                height=item.get("height"),
                eff_path=item.get("eff_path"),
                name=item.get("name")
                or os.path.basename(item.get("eff_path") or ""),
            )
        )
    best = representative.get("quality_score")
    return LookalikeGroup(media=tuple(media), best_quality=best)


def detect(media, similarity: float = DEFAULT_SIMILARITY) -> LookalikeResult:
    """Group perceptually near-duplicate media at a similarity threshold.

    ``media`` are hashed media dicts (see
    :func:`src.sqlite_store.media_with_hashes`) carrying ``id``/``sha256``/
    ``phash``/``dhash`` plus index columns + ``eff_path`` for ranking/display;
    media missing a hash are ignored. ``similarity`` is the min percent
    (default :data:`DEFAULT_SIMILARITY`). Returns a :class:`LookalikeResult`
    (groups best first, N compared, threshold); empty when < 2 hashed or no
    match.
    """
    usable = [
        item for item in media if item.get("phash") and item.get("dhash")
    ]
    hashed_count = len(usable)
    max_distance = similarity_to_distance(similarity)
    if hashed_count < 2:
        return LookalikeResult(
            groups=(),
            hashed_count=hashed_count,
            similarity=int(similarity),
        )
    phash = _to_uint64(item["phash"] for item in usable)
    dhash = _to_uint64(item["dhash"] for item in usable)
    pairs = _matched_pairs(phash, dhash, max_distance)
    union = _UnionFind(hashed_count)
    for i, j in pairs:
        union.union(i, j)
    phash_int = {item["id"]: int(item["phash"], 16) for item in usable}
    dhash_int = {item["id"]: int(item["dhash"], 16) for item in usable}
    groups = [
        _build_group(
            [usable[index] for index in members], phash_int, dhash_int
        )
        for members in union.groups()
    ]
    # Rank groups by their best member's quality (a group with no scored
    # member sorts last), then by size, then by representative id for a
    # stable order.
    groups.sort(
        key=lambda group: (
            group.best_quality is not None,
            group.best_quality if group.best_quality is not None else 0.0,
            len(group.media),
            -group.media[0].media_id,
        ),
        reverse=True,
    )
    return LookalikeResult(
        groups=tuple(groups),
        hashed_count=hashed_count,
        similarity=int(similarity),
    )
