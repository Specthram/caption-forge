"""Tests for :mod:`src.perceptual_hash` (pHash/dHash of media).

These exercise the real ``imagehash`` code path on small generated images
and a tiny video, so no model weights or network access are involved. The
foundation of the future "lookalike" feature only computes the hashes, so
the tests cover stability, resolution invariance and graceful failure —
not comparison or clustering, which live in later steps.
"""

import cv2
import imagehash
import numpy as np
import pytest
from PIL import Image

from src import perceptual_hash


def _write_pattern_image(path, size=256):
    """Write a deterministic, non-flat RGB image and return its path.

    A smooth diagonal gradient with a darker corner block gives the
    perceptual hashes real structure to lock onto (a flat color would make
    every hash trivially identical and prove nothing).
    """
    coords = np.arange(size)
    gradient = (coords[:, None] + coords[None, :]) % 256
    array = np.stack([gradient] * 3, axis=-1).astype(np.uint8)
    array[: size // 4, : size // 4] = 20
    Image.fromarray(array).save(path)
    return path


def _write_video(path, size=(32, 24)):
    """Write a tiny two-frame mp4 to ``path`` and return it."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, size
    )
    for _ in range(2):
        writer.write(np.full((size[1], size[0], 3), 180, dtype=np.uint8))
    writer.release()
    return path


class TestComputeHashes:
    """Tests for :func:`perceptual_hash.compute_hashes`."""

    def test_hashes_are_16_char_hex(self, tmp_path):
        """Both hashes are the 16-hex-char strings imagehash serializes."""
        source = _write_pattern_image(tmp_path / "a.png")
        phash, dhash = perceptual_hash.compute_hashes(source)
        assert len(phash) == 16 and len(dhash) == 16
        assert all(c in "0123456789abcdef" for c in phash + dhash)

    def test_hash_is_stable_for_the_same_file(self, tmp_path):
        """Hashing the same file twice yields the exact same hex."""
        source = _write_pattern_image(tmp_path / "a.png")
        first = perceptual_hash.compute_hashes(source)
        second = perceptual_hash.compute_hashes(source)
        assert first == second

    def test_phash_is_resolution_invariant(self, tmp_path):
        """The same image at half the resolution keeps a near-equal pHash."""
        big = _write_pattern_image(tmp_path / "big.png", size=256)
        small = tmp_path / "small.png"
        with Image.open(big) as image:
            image.resize((128, 128)).save(small)
        big_phash = imagehash.hex_to_hash(
            perceptual_hash.compute_hashes(big)[0]
        )
        small_phash = imagehash.hex_to_hash(
            perceptual_hash.compute_hashes(small)[0]
        )
        assert big_phash - small_phash <= 2

    def test_scores_a_video_from_its_first_frame(self, tmp_path):
        """A video is hashed from its first frame, not skipped."""
        source = _write_video(tmp_path / "clip.mp4")
        phash, dhash = perceptual_hash.compute_hashes(source)
        assert phash is not None and dhash is not None

    def test_corrupt_image_returns_none_pair(self, tmp_path):
        """A corrupt/unsupported file fails gracefully instead of raising."""
        bogus = tmp_path / "bogus.png"
        bogus.write_bytes(b"not an image")
        assert perceptual_hash.compute_hashes(bogus) == (None, None)

    def test_corrupt_video_returns_none_pair(self, tmp_path):
        """A file with a video extension but no readable frames is a pair."""
        bogus = tmp_path / "bogus.mp4"
        bogus.write_bytes(b"not a video")
        assert perceptual_hash.compute_hashes(bogus) == (None, None)
