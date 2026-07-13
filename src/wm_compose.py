"""Virtual watermark removal: compose inpaint patches over an original.

The key invariant of the whole feature: **the source file is never modified**.
A watermark zone is erased by an edit *patch* — a small PNG covering the
dilated bounding box — stored under :data:`~src.constants.PATCHES_DIR` and
pasted over the original at every point the media is shown or deployed. This
module owns that geometry and the two caches it needs:

* the **patch** cache, one ``<zone_id>.png`` per zone (a regeneration
  overwrites it), written by the edit engine and read here;
* the **composite** cache, one PNG per (source, patch-set), keyed by a
  synthetic hash so a patch regeneration naturally invalidates it — exactly
  the way a re-framed crop re-hashes its cache (see :mod:`src.crops`).

Both patch generation (:mod:`src.watermark_flux`) and composition apply the
same EXIF orientation and the same :func:`dilated_pixel_rect`, so a patch
always lands back on the pixels it was cut from.
"""

import hashlib
import logging
import os
from pathlib import Path

from src.constants import COMPOSED_DIR, PATCHES_DIR

logger = logging.getLogger(__name__)

# Extra pixels the patch covers *beyond* the detection box, pasted back too, so
# anti-aliased letter edges just outside a tight OWLv2/YOLO box are still
# hidden. Kept small; the FLUX context margin (``dilate_px``) extends past this
# again, so the edit still has clean background to reason from. Both the edit
# engine (:func:`src.watermark_flux.edit`) and composition use this value, so
# the generated patch and the paste rectangle always line up.
PATCH_COVER_PX = 5


def get_patches_dir() -> Path:
    """Return the patch cache dir (a function so a harness can redirect it)."""
    return PATCHES_DIR


def get_composed_dir() -> Path:
    """Return the composite cache dir (patchable like the patch dir)."""
    return COMPOSED_DIR


def patch_path(zone_id: int) -> Path:
    """Return a zone's patch PNG path (may not exist yet)."""
    return get_patches_dir() / f"{int(zone_id)}.png"


def has_patch(zone) -> bool:
    """Return whether a zone dict carries a usable patch on disk."""
    return bool(zone.get("patch_sha")) and patch_path(zone["id"]).is_file()


def oriented_open(source_path):
    """Return the source as an EXIF-transposed RGB PIL image.

    Every geometry computation and the composition itself work on the
    *oriented* pixels — the ones the user framed on screen — so a rotated
    JPEG never pastes a patch in the wrong corner.
    """
    # pylint: disable=import-outside-toplevel  # Pillow only needed here.
    from PIL import Image, ImageOps

    with Image.open(source_path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def oriented_size(source_path) -> tuple:
    """Return the source's oriented ``(width, height)`` (header read only)."""
    # pylint: disable=import-outside-toplevel
    from PIL import Image

    with Image.open(source_path) as img:
        orientation = img.getexif().get(0x0112, 1)
        if orientation in {5, 6, 7, 8}:
            return img.height, img.width
        return img.width, img.height


def dilated_pixel_rect(box, width: int, height: int, dilate_px: int) -> tuple:
    """Return the ``(left, top, right, bottom)`` px box a patch covers.

    The watermark box (fractions of the source) is resolved to pixels and
    grown by ``dilate_px`` on every side (the mask dilation that lets the
    inpainter blend past the watermark's edge), then clamped to the image.
    """
    left = int(round(box["x"] * width)) - dilate_px
    top = int(round(box["y"] * height)) - dilate_px
    right = int(round((box["x"] + box["w"]) * width)) + dilate_px
    bottom = int(round((box["y"] + box["h"]) * height)) + dilate_px
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    return left, top, right, bottom


def write_patch(zone_id: int, patch_image) -> str:
    """Write a zone's patch PNG and return the sha256 of its bytes.

    The digest is the composition cache key: any regeneration produces
    different bytes and so a different sha, invalidating every composite that
    used the old patch.
    """
    dest = patch_path(zone_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    patch_image.save(dest, format="PNG")
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def delete_patch(zone_id: int) -> None:
    """Remove a zone's patch PNG (a no-op when absent)."""
    try:
        patch_path(zone_id).unlink()
    except OSError:
        pass


def _patched_zones(zones) -> list:
    """Return the zones that carry a usable patch, ordered by id."""
    return sorted(
        (zone for zone in zones if has_patch(zone)),
        key=lambda zone: zone["id"],
    )


def composed_sha(media_sha: str, zones) -> str | None:
    """Return the synthetic hash of a source and its patch set, or None.

    None means "no patched zone" — the caller then serves the original
    untouched. Otherwise the digest folds in every patched zone's box,
    dilation and patch sha, so moving a box or regenerating a patch yields a
    new composite identity (and a new cache file).
    """
    patched = _patched_zones(zones)
    if not patched:
        return None
    payload = [f"compose:{media_sha}"]
    for zone in patched:
        box = zone["box"]
        payload.append(
            f"{zone['id']}:{box['x']:.5f}:{box['y']:.5f}:"
            f"{box['w']:.5f}:{box['h']:.5f}:{zone['dilate_px']}:"
            f"{zone['patch_sha']}"
        )
    return hashlib.sha256("|".join(payload).encode("utf-8")).hexdigest()


def composed_path(sha: str) -> Path:
    """Return a composite's cache path, sharded by sha prefix like crops."""
    return get_composed_dir() / sha[:2] / f"{sha}.png"


def compose_image(source_path, zones):
    """Return the EXIF-upright original with every patched zone pasted over it.

    The shared paste geometry behind both the cached composite
    (:func:`render_composed`) and the "flatten to disk" bake
    (:func:`src.watermark.flatten_media`). Returns a fresh RGB PIL image; the
    source file is not touched.
    """
    image = oriented_open(source_path)
    width, height = image.size
    # pylint: disable=import-outside-toplevel
    from PIL import Image

    for zone in _patched_zones(zones):
        # The patch covers the detection box grown by PATCH_COVER_PX (so tight
        # boxes still hide letter edges); the wider FLUX context margin around
        # it was never pasted (see :func:`src.watermark_flux.edit`).
        left, top, right, bottom = dilated_pixel_rect(
            zone["box"], width, height, PATCH_COVER_PX
        )
        with Image.open(patch_path(zone["id"])) as patch:
            patch = patch.convert("RGB").resize(
                (right - left, bottom - top), Image.Resampling.LANCZOS
            )
            image.paste(patch, (left, top))
    return image


def render_composed(source_path, zones, dest: Path) -> Path:
    """Compose every patched zone over the original and save it as PNG."""
    image = compose_image(source_path, zones)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, format="PNG")
    return dest


def ensure_composed(source_path, media_sha: str, zones):
    """Return the composited image path, or None when nothing is patched.

    Cached by :func:`composed_sha`; a hit returns immediately, a miss renders
    once. ``None`` (no patch, or an unreadable source) tells the caller to
    serve the original file as-is.
    """
    sha = composed_sha(media_sha, zones)
    if sha is None:
        return None
    dest = composed_path(sha)
    if dest.is_file():
        return dest
    if not source_path or not os.path.exists(source_path):
        return None
    try:
        return render_composed(source_path, zones, dest)
    except (OSError, ValueError) as exc:
        logger.warning("watermark compose failed for %s: %s", source_path, exc)
        return None
