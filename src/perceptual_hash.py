"""Perceptual hashing of media, the foundation of "lookalike" detection.

The future "lookalike" feature will surface near-duplicate media — the same
picture at a different resolution, crop-free re-encode or compression level.
Unlike a content sha256 (which changes with a single altered byte), a
perceptual hash stays stable across such transforms, so two visually
identical files land on the same — or a very close — hash.

This module only *computes* the hashes; it neither compares nor clusters
them. It produces two complementary ``imagehash`` fingerprints per media:
``phash`` (a DCT-based hash, robust to scaling and light compression) and
``dhash`` (a gradient hash, cheap and sensitive to structure), each
serialized as a 16-character hex string. Images are hashed from their
*original* file (never a downscaled thumbnail, which would throw away the
signal); a video is hashed from its first frame — the same frame
:mod:`src.thumbnails` uses for its preview. An unreadable or corrupt file
yields ``(None, None)`` without raising, mirroring how
:func:`src.quality.score_media` degrades.
"""

import logging

import imagehash
from PIL import Image

from src.media import is_video_file
from src.thumbnails import first_frame

logger = logging.getLogger(__name__)


def _hash_image(image: Image.Image) -> tuple[str, str]:
    """Return the ``(phash, dhash)`` hex pair for an open PIL image."""
    return str(imagehash.phash(image)), str(imagehash.dhash(image))


def compute_hashes(source_path) -> tuple[str | None, str | None]:
    """Return a media's perceptual hashes as ``(phash, dhash)`` hex strings.

    Hashes the original file (a video's first frame), never a thumbnail.
    Returns the 16-char hex ``phash`` and ``dhash``, or ``(None, None)`` when
    the file couldn't be read/decoded.
    """
    try:
        return _hash_source(source_path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # A single unreadable/corrupt file must not abort a batch index.
        logger.warning(
            "Perceptual hashing failed for %s: %s", source_path, exc
        )
        return None, None


def _hash_source(source_path) -> tuple[str | None, str | None]:
    """Open a media's original file and hash it (first frame for a video)."""
    if is_video_file(str(source_path)):
        image = first_frame(source_path)
        if image is None:
            return None, None
        return _hash_image(image)
    with Image.open(source_path) as image:
        image.load()
        return _hash_image(image)
