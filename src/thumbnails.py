"""Thumbnail cache for the media grids.

The Libraries/Medias/Datasets/Caption grids (see :mod:`src.media_grid` and
:mod:`src.gallery`) send an image straight from disk to the browser; a
source photo can be many megapixels, which is by far the heaviest part of
painting a page of cards. This module generates a small, resized copy of
each unique media — keyed by its content sha256, so duplicates reached
through several files/libraries share one thumbnail — cached under
:func:`get_thumbnails_dir`, sharded into 256 ``<2-hex-prefix>/`` sub-folders
(see :func:`thumbnail_path`) so a large library never dumps tens of
thousands of files into one folder.

Generation happens in exactly one place: the Libraries tab's "Index" action
(see :mod:`src.libraries_gallery`), never during a scan or upload and never
on the fly while painting a grid — a card just checks whether the cached
file already exists (:func:`thumbnail_path`) and falls back to the original
otherwise. Zooming a card (the lightbox, ``src/js/lightbox.js``) swaps back
to the original file, since a thumbnail is a downscaled preview only.
"""

import logging

import cv2
from PIL import Image, ImageOps

from src.constants import THUMBNAILS_DIR
from src.media import is_video_file

logger = logging.getLogger(__name__)

# Long side of a generated thumbnail, in pixels: comfortably above a grid
# card's rendered size (even on a high-DPI screen) while cutting a
# multi-megapixel source down to a browser-friendly size.
THUMBNAIL_SIZE = 512
_JPEG_QUALITY = 85


def get_thumbnails_dir():
    """Return the thumbnail cache directory (a function so it is patchable).

    A harness redirects it like :func:`src.deploy.deploy_root`.
    """
    return THUMBNAILS_DIR


def cached_sha256() -> set:
    """Return the content hashes that already have a cached thumbnail.

    The cache holds no DB row (the file *is* the record), so the Index panel
    counts the "thumbs" step by intersecting this with a library's live hashes.
    One directory walk, cheap next to a stat per media.
    """
    root = get_thumbnails_dir()
    if not root.is_dir():
        return set()
    return {path.stem for path in root.glob("*/*.jpg")}


def thumbnail_path(sha256: str):
    """Return a media's cache path, sharded by sha256 prefix (may not exist).

    Sharded into ``<2 hex>/<sha256>.jpg`` (256 buckets) so one folder never
    holds tens of thousands of files. Keyed by content hash, so a media shared
    across libraries resolves to one thumbnail, never duplicated.
    """
    return get_thumbnails_dir() / sha256[:2] / f"{sha256}.jpg"


def _save(image: Image.Image, dest) -> None:
    """Resize ``image`` to fit :data:`THUMBNAIL_SIZE` and save it as a JPEG."""
    image = ImageOps.exif_transpose(image)
    image.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, "JPEG", quality=_JPEG_QUALITY)


def first_frame(source_path):
    """Return a video's first frame as a PIL image, or None if unreadable.

    Factored out so the perceptual hasher (:mod:`src.perceptual_hash`) reuses
    the extraction instead of duplicating the OpenCV/BGR-to-RGB dance. Returns
    an RGB image, or None when the file can't be opened or has no frame.
    """
    capture = cv2.VideoCapture(str(source_path))
    try:
        if not capture.isOpened():
            return None
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _generate_video(source_path, dest) -> bool:
    """Save a video's first frame as its thumbnail; return whether it did."""
    image = first_frame(source_path)
    if image is None:
        return False
    _save(image, dest)
    return True


def _generate_image(source_path, dest) -> bool:
    """Save a resized copy of an image as its thumbnail; always succeeds."""
    with Image.open(source_path) as image:
        _save(image, dest)
    return True


def ensure_thumbnail(source_path, sha256: str, force: bool = False):
    """Return the cached thumbnail for a media, generating it if missing.

    ``sha256`` is the cache key; ``force`` regenerates even when cached (the
    "Force re-index" option — a changed source, or a new
    :data:`THUMBNAIL_SIZE`). Returns the cached path, or None when
    ``source_path`` couldn't be read (caller then falls back to the original).
    """
    dest = thumbnail_path(sha256)
    if dest.exists() and not force:
        return dest
    try:
        generated = (
            _generate_video(source_path, dest)
            if is_video_file(str(source_path))
            else _generate_image(source_path, dest)
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # A single unreadable/corrupt file must not abort a batch scan.
        logger.warning(
            "Thumbnail generation failed for %s: %s", source_path, exc
        )
        return None
    return dest if generated else None


def probe_dimensions(source_path):
    """Return a media's original ``(width, height)`` in pixels, or None.

    Fills the Index ``dimension`` sort column — reads only the header (images)
    or container metadata (videos), never full pixels, so it stays cheap.
    """
    if is_video_file(str(source_path)):
        capture = cv2.VideoCapture(str(source_path))
        try:
            if not capture.isOpened():
                return None
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        return (width, height) if width and height else None
    try:
        with Image.open(source_path) as image:
            return image.size
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("Dimension probe failed for %s: %s", source_path, exc)
        return None
