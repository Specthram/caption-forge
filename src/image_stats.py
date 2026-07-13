"""Cheap, model-free per-image statistics (sharpness, clipping, noise).

The dataset quality report explains *why* an image is weak, which the
learned IQA metrics of :mod:`src.quality` cannot do on their own: a MUSIQ
score of 41 says "bad", not "out of focus". These three signals are
computed straight from the pixels with OpenCV and NumPy — no model, no
VRAM, a few milliseconds per image — and feed the report's low-quality
and near-duplicate inspectors.

* **sharpness** — the variance of the image Laplacian, log-mapped to
  0-100 (blur flattens the second derivative).
* **clipping** — the percentage of pixels crushed to pure black or blown
  to pure white (lost highlight/shadow detail).
* **cleanliness** — 100 minus Immerkaer's noise-sigma estimate, mapped to
  0-100 (grain and compression mush both raise sigma).

Every value is a percentage: higher is always better, except ``clipping``
where lower is better. OpenCV is imported lazily so the module stays free
to import in the API process before the heavy backends are needed.
"""

import math

import numpy as np

# The analysis downscales to this long side first: the statistics are
# scale-sensitive, so a fixed working resolution keeps a 4K photo and a
# 1024px crop comparable, and keeps the cost flat across a dataset.
ANALYSIS_SIDE = 512

# Laplacian variance of a tack-sharp 512px image sits well above this;
# the log mapping saturates here so a sharper image cannot outscore a
# sharp one by an order of magnitude.
_SHARPNESS_CEILING = 2000.0

# Luminance levels counted as crushed / blown.
_CLIP_LOW = 2
_CLIP_HIGH = 253

# Noise sigma (0-255 scale) mapped to 0 cleanliness. A clean render sits
# near 1-2, a heavily compressed JPEG near 6-8.
_NOISE_CEILING = 12.0

# Immerkaer's noise-estimation kernel: a discrete Laplacian whose response
# to a smooth image is zero, so what it measures is the noise floor.
_NOISE_KERNEL = np.array(
    [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
    dtype=np.float64,
)

# Below these percentages the report flags an image as blurry / noisy.
BLUR_FLOOR = 35.0
NOISE_FLOOR = 45.0


def _read_gray(source_path):
    """Return the image as a downscaled float grayscale array, or None.

    Uses ``numpy.fromfile`` + ``cv2.imdecode``, not ``cv2.imread``, which
    can't open a non-ASCII path on Windows (the norm in a media library).
    Returns a 2-D ``float64`` 0-255 array, long side at most
    :data:`ANALYSIS_SIDE`; None when undecodable.
    """
    import cv2  # pylint: disable=import-outside-toplevel

    try:
        raw = np.fromfile(str(source_path), dtype=np.uint8)
    except OSError:
        return None
    if raw.size == 0:
        return None
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None
    long_side = max(image.shape[:2])
    if long_side > ANALYSIS_SIDE:
        factor = ANALYSIS_SIDE / long_side
        image = cv2.resize(
            image,
            (
                max(1, round(image.shape[1] * factor)),
                max(1, round(image.shape[0] * factor)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    return image.astype(np.float64)


def sharpness(gray: np.ndarray) -> float:
    """Return the 0-100 sharpness of a grayscale array.

    The variance of the Laplacian is the classic focus measure; it spans
    several orders of magnitude, so it is mapped through a logarithm
    saturating at :data:`_SHARPNESS_CEILING`.
    """
    import cv2  # pylint: disable=import-outside-toplevel

    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    scaled = math.log1p(variance) / math.log1p(_SHARPNESS_CEILING)
    return float(max(0.0, min(100.0, scaled * 100.0)))


def clipping(gray: np.ndarray) -> float:
    """Return the percentage of crushed-black or blown-white pixels."""
    crushed = np.count_nonzero(gray <= _CLIP_LOW)
    blown = np.count_nonzero(gray >= _CLIP_HIGH)
    return float(100.0 * (crushed + blown) / gray.size)


def cleanliness(gray: np.ndarray) -> float:
    """Return the 0-100 freedom from noise / compression mush.

    Uses Immerkaer's estimator: the mean absolute response of a smoothness
    -cancelling Laplacian kernel, scaled so that a noise-free image yields
    a sigma of zero. The sigma is mapped to 0-100, saturating at
    :data:`_NOISE_CEILING`.
    """
    import cv2  # pylint: disable=import-outside-toplevel

    height, width = gray.shape[:2]
    if height < 3 or width < 3:
        return 100.0
    response = cv2.filter2D(gray, cv2.CV_64F, _NOISE_KERNEL)
    inner = response[1:-1, 1:-1]
    sigma = math.sqrt(math.pi / 2.0) * float(np.abs(inner).sum())
    sigma /= 6.0 * (width - 2) * (height - 2)
    return float(max(0.0, min(100.0, 100.0 * (1.0 - sigma / _NOISE_CEILING))))


def analyze(source_path) -> dict | None:
    """Return the three statistics of one image file.

    Videos aren't supported (the caller skips them). Returns
    ``{"sharpness", "clipping", "cleanliness"}`` percentages, or None when the
    file couldn't be decoded.
    """
    gray = _read_gray(source_path)
    if gray is None:
        return None
    return {
        "sharpness": sharpness(gray),
        "clipping": clipping(gray),
        "cleanliness": cleanliness(gray),
    }


def is_flagged(stats: dict | None) -> bool:
    """Return whether an image's statistics read as blurry or noisy."""
    if not stats:
        return False
    return (
        stats["sharpness"] < BLUR_FLOOR or stats["cleanliness"] < NOISE_FLOOR
    )
