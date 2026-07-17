"""Coarse dominant-hue bucket of an image (model-free, for the report map).

The Quality report's composition map colours each node by *visual style* so
that re-skins — same framing, different style — read as different colours
inside a single framing cluster. The app has no learned style signal, and a
coarse hue bucket is enough: this module reduces an image to one of five
buckets straight from its pixels with OpenCV — no model, no VRAM, a few
milliseconds — from the saturation-weighted circular mean of its hue.

Buckets, on OpenCV's ``0-179`` hue wheel (half the usual ``0-359``):

* **neutral** — too little saturation to carry a hue at all (greys, B&W).
* **warm** — reds, oranges and yellows.
* **green** — yellow-greens through greens.
* **cool** — cyans and blues.
* **pink** — magentas and violets.

The buckets match the composition map's five-colour palette exactly (see
``web/src/design/tokens.ts``); the projection groups the nodes by framing,
this colours them by style, and the two together make a re-skin legible.
"""

import math

import numpy as np

# The bucket names, aligned with the front-end style palette.
BUCKETS = ("warm", "green", "cool", "pink", "neutral")

# Mean saturation (0-1) under which an image has no meaningful hue: it is a
# greyscale / desaturated picture and goes to the neutral bucket.
_SATURATION_FLOOR = 0.12

# The analysis downscales to this long side first — the mean hue is a coarse,
# scale-free statistic, so a small working resolution keeps the cost flat.
_ANALYSIS_SIDE = 256

# Hue-wheel cutoffs (OpenCV 0-179). Reds wrap around both ends, so warm owns
# the top of the wheel as well as the bottom.
_WARM_MAX = 35.0
_GREEN_MAX = 85.0
_COOL_MAX = 140.0
_PINK_MAX = 160.0


def _read_hsv(source_path):
    """Return the image as a downscaled HSV ``uint8`` array, or None.

    Uses ``numpy.fromfile`` + ``cv2.imdecode`` (not ``cv2.imread``, which
    cannot open a non-ASCII path on Windows — the norm in a media library).
    Returns an ``H x W x 3`` HSV array, long side at most
    :data:`_ANALYSIS_SIDE`; None when the file could not be decoded.
    """
    import cv2  # pylint: disable=import-outside-toplevel

    try:
        raw = np.fromfile(str(source_path), dtype=np.uint8)
    except OSError:
        return None
    if raw.size == 0:
        return None
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    long_side = max(image.shape[:2])
    if long_side > _ANALYSIS_SIDE:
        factor = _ANALYSIS_SIDE / long_side
        image = cv2.resize(
            image,
            (
                max(1, round(image.shape[1] * factor)),
                max(1, round(image.shape[0] * factor)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def _bucket(hue: float) -> str:
    """Return the style bucket a ``0-179`` hue falls in."""
    if hue < _WARM_MAX or hue >= _PINK_MAX:
        return "warm"
    if hue < _GREEN_MAX:
        return "green"
    if hue < _COOL_MAX:
        return "cool"
    return "pink"


def classify(source_path) -> str | None:
    """Return the dominant-style bucket of an image file, or None.

    The bucket is the saturation-weighted circular mean of the image's hue,
    mapped through :func:`_bucket`; a picture with too little saturation
    overall is :data:`neutral`. ``source_path`` is any decodable image path
    (videos are not supported — the caller keeps them out). Returns None when
    the file cannot be decoded.
    """
    hsv = _read_hsv(source_path)
    if hsv is None:
        return None
    hue = hsv[..., 0].reshape(-1).astype(np.float64)
    sat = hsv[..., 1].reshape(-1).astype(np.float64) / 255.0
    weight = float(sat.sum())
    if float(sat.mean()) < _SATURATION_FLOOR or weight <= 0.0:
        return "neutral"
    angle = hue / 180.0 * 2.0 * math.pi
    mean_x = float((np.cos(angle) * sat).sum())
    mean_y = float((np.sin(angle) * sat).sum())
    mean_hue = (math.degrees(math.atan2(mean_y, mean_x)) % 360.0) / 2.0
    return _bucket(mean_hue)
