"""Virtual crops: a rectangle of a media, materialized on demand.

A crop is a **non-destructive alias** of an existing media. It creates no
file in any library: the database holds a ``media`` row whose
``parent_media_id`` points at the source and whose ``crop_rect`` records the
rectangle, in percentages of the source image. The pixels are produced on
the fly, the first time something asks for them (a grid thumbnail, a quality
score, an auto-tag pass, a deploy), and cached under
:func:`get_crops_dir` as a lossless PNG.

The cache is keyed by the crop's *synthetic* content hash
(:func:`crop_sha256`), a digest of the parent's hash and the rectangle. Two
identical crops of the same parent therefore collapse onto one row and one
cached file, and re-framing a crop naturally invalidates both: the hash
changes, so the old PNG (and the old deployed copy, named after the hash) is
pruned rather than silently reused.

Because a crop *is* a media row whose effective file resolves to its cached
PNG (see :func:`src.sqlite_store.media.effective_file`), every engine that
takes a path — thumbnails, quality, the WD14 tagger, SigLIP grounding, the
deploy copier — works on the cropped pixels with no change at all. What the
rest of the codebase must do instead is keep crops *out* of the listings
that describe files on disk (the Media grid, the library scan, the index,
the lookalike detection), which is what the ``parent_media_id IS NULL``
predicate does there.
"""

import hashlib
import json
import logging
import os
from pathlib import Path

from src.constants import CROPS_DIR

logger = logging.getLogger(__name__)

# The rectangle's keys, in the order they are hashed (see crop_sha256).
RECT_KEYS = ("x", "y", "w", "h")

# Smallest crop side, as a percentage of the source. Mirrors the overlay's
# ``MIN`` so a rectangle dragged to nothing stays renderable.
MIN_SIDE_PCT = 1.0

# Aspect ratios the overlay offers; "free" means the rectangle is unlocked.
RATIOS = ("free", "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16")


def get_crops_dir() -> Path:
    """Return the rendered-crop cache dir (a function so it is patchable).

    A harness redirects it like :func:`src.thumbnails.get_thumbnails_dir`.
    """
    return CROPS_DIR


def normalize_rect(rect) -> dict:
    """Return a crop rectangle clamped inside the source image.

    ``rect`` is a ``{"x", "y", "w", "h"}`` mapping in percentages; missing or
    unparsable values fall back to the full frame. Returns floats with each
    side at least :data:`MIN_SIDE_PCT`, wholly inside ``[0, 100]``.
    """
    defaults = {"x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0}
    values = {}
    for key in RECT_KEYS:
        try:
            values[key] = float((rect or {}).get(key, defaults[key]))
        except (TypeError, ValueError):
            values[key] = defaults[key]
    width = min(100.0, max(MIN_SIDE_PCT, values["w"]))
    height = min(100.0, max(MIN_SIDE_PCT, values["h"]))
    left = min(max(0.0, values["x"]), 100.0 - width)
    top = min(max(0.0, values["y"]), 100.0 - height)
    return {"x": left, "y": top, "w": width, "h": height}


def normalize_ratio(ratio) -> str:
    """Return a known aspect-ratio key, defaulting to ``"free"``."""
    return ratio if ratio in RATIOS else "free"


def is_full_frame(rect) -> bool:
    """Return whether a rectangle covers the whole source (a no-op crop)."""
    box = normalize_rect(rect)
    return (
        box["x"] < 0.01
        and box["y"] < 0.01
        and box["w"] > 99.99
        and box["h"] > 99.99
    )


def rect_to_json(rect) -> str:
    """Serialize a normalized rectangle for the ``media.crop_rect`` column."""
    box = normalize_rect(rect)
    return json.dumps({key: round(box[key], 4) for key in RECT_KEYS})


def rect_from_json(raw):
    """Return the rectangle stored in ``media.crop_rect``, or None.

    A row with no ``crop_rect`` (an ordinary media) or an unreadable one
    yields ``None``, which every caller reads as "not a crop".
    """
    if not raw:
        return None
    try:
        return normalize_rect(json.loads(raw))
    except (TypeError, ValueError):
        return None


def crop_sha256(parent_sha256: str, rect) -> str:
    """Return a crop's synthetic content hash.

    A crop's identity is its parent plus its rectangle, so their digest *is*
    its content hash everywhere the real one serves: deploy stem, thumbnail
    key, rendered-PNG key, and ``media.sha256`` uniqueness (an identical crop
    of the same parent is the same media). Returns a 64-char hex digest.
    """
    box = normalize_rect(rect)
    payload = "|".join(
        [f"crop:{parent_sha256}"]
        + [f"{key}={box[key]:.4f}" for key in RECT_KEYS]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def crop_path(sha256: str) -> Path:
    """Return the cached PNG path of a crop hash (sharded, may not exist).

    Sharded into 256 ``<2-hex-prefix>/`` sub-folders like the thumbnail
    cache, so a large dataset of crops never dumps thousands of files into a
    single directory.
    """
    return get_crops_dir() / sha256[:2] / f"{sha256}.png"


def pixel_rect(width: int, height: int, rect) -> tuple:
    """Return the ``(left, top, right, bottom)`` pixel box of a rectangle.

    Percentages are resolved against the *oriented* source size (the one the
    user framed on screen). Each side is at least one pixel, and the box
    never leaves the image.
    """
    box = normalize_rect(rect)
    left = int(round(box["x"] / 100.0 * width))
    top = int(round(box["y"] / 100.0 * height))
    right = int(round((box["x"] + box["w"]) / 100.0 * width))
    bottom = int(round((box["y"] + box["h"]) / 100.0 * height))
    left = min(max(0, left), max(0, width - 1))
    top = min(max(0, top), max(0, height - 1))
    right = min(max(left + 1, right), width)
    bottom = min(max(top + 1, bottom), height)
    return left, top, right, bottom


def source_size(source_path) -> tuple:
    """Return a source image's ``(width, height)`` after EXIF orientation.

    Only the header and the orientation tag are read, so this stays cheap on
    a multi-megapixel photo.
    """
    # pylint: disable=import-outside-toplevel  # Pillow is only needed here.
    from PIL import Image

    with Image.open(source_path) as img:
        orientation = img.getexif().get(0x0112, 1)
        if orientation in {5, 6, 7, 8}:
            return img.height, img.width
        return img.width, img.height


def crop_size(source_path, rect) -> tuple:
    """Return the ``(width, height)`` in pixels a crop renders to."""
    width, height = source_size(source_path)
    left, top, right, bottom = pixel_rect(width, height, rect)
    return right - left, bottom - top


def render(source_path, rect, dest: Path) -> Path:
    """Write the cropped pixels of ``source_path`` to ``dest`` as PNG.

    The EXIF orientation is baked in before cropping (the rectangle was
    framed on the oriented image, and PNG carries no orientation tag), and
    the result is saved losslessly — a crop is an intermediate the deploy
    resize reads back, so it must not accumulate JPEG artifacts.
    """
    # pylint: disable=import-outside-toplevel  # Pillow is only needed here.
    from PIL import Image, ImageOps

    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode == "CMYK":
            img = img.convert("RGB")
        img.crop(pixel_rect(img.width, img.height, rect)).save(
            dest, format="PNG"
        )
    return dest


def ensure_render(source_path, sha256: str, rect):
    """Return a crop's cached PNG, rendering it when it is absent.

    ``source_path`` is the *parent* file, ``sha256`` the crop's synthetic cache
    key, ``rect`` the rectangle. Returns the cached PNG, or ``None`` when the
    source is unreadable (missing parent, unsupported format) — the crop is
    then *missing*, like a media whose every file vanished.
    """
    dest = crop_path(sha256)
    if dest.is_file():
        return dest
    if not source_path or not os.path.exists(source_path):
        return None
    try:
        return render(source_path, rect, dest)
    except (OSError, ValueError) as exc:
        logger.warning("crop render failed for %s: %s", source_path, exc)
        return None


def delete_render(sha256: str) -> bool:
    """Delete a crop's cached PNG; return whether a file was removed."""
    dest = crop_path(sha256)
    try:
        dest.unlink()
        return True
    except OSError:
        return False
