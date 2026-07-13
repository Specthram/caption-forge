"""Image quality scoring via pyiqa: pluggable metrics, chosen per run.

Drives the Libraries "Quality evaluation" action: the user picks one or more
models to score each media's *original* file (never a thumbnail — a resized
copy would bias the perceptual score); a video is scored from its first frame.

Metrics live in :data:`QUALITY_METRICS`: MUSIQ (fast default), TOPIQ-NR
(accurate, same cost), Q-Align (7B VLM, full or 8/4-bit) and LAION Aesthetic
(appeal, not distortions). Each carries its native range, an approximate VRAM
need and labeled colored bands for the grid badge (:func:`badge_style`). The
badge always shows the score normalized to 0-100, so grids stay comparable.
Each metric's score is its own ``media_quality`` row, so a media can carry
several; the badge reads the bands of whichever metric the grids display.
"""

import gc
import logging
import threading
from dataclasses import dataclass

from src.media import is_video_file

logger = logging.getLogger(__name__)


def _install_clip_pkg_resources_shim() -> None:
    """Expose ``packaging`` under ``pkg_resources`` for the old clip package.

    pyiqa's CLIP metrics (CLIPIQA, LAION Aesthetic) pull in OpenAI's old
    ``clip``, which does ``from pkg_resources import packaging`` — gone in
    setuptools 70+, raising "cannot import name 'packaging'" on first load.
    Aliasing the standalone ``packaging`` (already installed) keeps it working,
    no setuptools pin or third-party patch needed.

    Called from :func:`_get_metric` just before pyiqa is imported (not at
    module import), so this module stays cheap for the registry-only callers.
    """
    import pkg_resources  # pylint: disable=import-outside-toplevel

    if not hasattr(pkg_resources, "packaging"):
        import packaging  # pylint: disable=import-outside-toplevel

        # ``import packaging`` doesn't load submodules, and ``clip`` reads
        # ``packaging.version.parse`` at import; without this explicit import
        # the shim dies with "module 'packaging' has no attribute 'version'"
        # unless something else imported it first.
        import packaging.version  # noqa: F401  pylint: disable=import-outside-toplevel

        pkg_resources.packaging = packaging


# Longest side (px) an image is downscaled to before scoring: MUSIQ/TOPIQ are
# multi-scale metrics, so a huge photo costs quadratic memory/time for no
# extra signal. Candidate for a Settings-tab value later.
_MAX_SCORE_SIDE = 1920

# Shared tier palette (red -> orange -> green -> dark green -> blue), reused
# across every metric's bands so the badges stay visually consistent: blue is
# reserved for an exceptional score, matching the ramp's original design.
_RED = "#d93025"
_ORANGE = "#f9a825"
_GREEN = "#34a853"
_DARK_GREEN = "#0b8043"
_BLUE = "#4285f4"


@dataclass(frozen=True)
class QualityBand:
    """One labeled, colored tier of a quality metric's score range.

    ``min_score`` inclusive lower bound (native scale), ``label`` tier name,
    ``color`` CSS badge background.
    """

    min_score: float
    label: str
    color: str


@dataclass(frozen=True)
class QualityMetric:  # pylint: disable=too-many-instance-attributes
    """Static facts about one pyiqa metric offered in the Settings tab.

    ``pyiqa_name`` passed to ``pyiqa.create_metric``; ``label`` dropdown label;
    ``score_min``/``score_max`` native range (clamp + normalize to 0-100);
    ``min_vram_gb`` advisory floor (not enforced); ``vram_note`` +
    ``description`` shown in the UI; ``bands`` ascending tiers —
    :func:`badge_style` picks the
    last whose ``min_score`` is at or below the clamped score.
    """

    pyiqa_name: str
    label: str
    score_min: float
    score_max: float
    min_vram_gb: float
    vram_note: str
    description: str
    bands: tuple[QualityBand, ...]


# Every quality metric offered in the Settings tab, in dropdown order (cheap
# and light first, Q-Align's variants together, the aesthetic predictor
# last). MUSIQ stays the factory default: it is the cheapest metric here and
# the one existing indexed libraries were scored with.
QUALITY_METRICS = {
    "musiq": QualityMetric(
        pyiqa_name="musiq",
        label="MUSIQ (KonIQ-10k)",
        score_min=0.0,
        score_max=100.0,
        min_vram_gb=0.5,
        vram_note="~0.5 GB VRAM — runs even on CPU.",
        description=(
            "Multi-scale transformer trained on KonIQ-10k. Detects blur, "
            "noise and compression artifacts. Fast and light: the best "
            "choice for indexing large libraries continuously."
        ),
        bands=(
            QualityBand(0.0, "low", _RED),
            QualityBand(60.0, "fair", _ORANGE),
            QualityBand(75.0, "good", _GREEN),
            QualityBand(90.0, "excellent", _BLUE),
        ),
    ),
    "topiq_nr": QualityMetric(
        pyiqa_name="topiq_nr",
        label="TOPIQ-NR (KonIQ)",
        score_min=0.0,
        score_max=1.0,
        min_vram_gb=1.0,
        vram_note="~1 GB VRAM.",
        description=(
            "CFANet (ResNet-50 backbone) trained on KonIQ-10k. Usually "
            "correlates better with human judgment than MUSIQ on academic "
            "benchmarks, at a near-identical VRAM cost — a good default "
            "alternative when VRAM is not a concern."
        ),
        bands=(
            QualityBand(0.0, "low", _RED),
            QualityBand(0.60, "fair", _ORANGE),
            QualityBand(0.80, "good", _GREEN),
            QualityBand(0.90, "very good", _DARK_GREEN),
            QualityBand(0.95, "excellent", _BLUE),
        ),
    ),
    "qalign": QualityMetric(
        pyiqa_name="qalign",
        label="Q-Align (full precision)",
        score_min=1.0,
        score_max=5.0,
        min_vram_gb=16.0,
        vram_note="~15-17 GB VRAM (7B VLM in fp16).",
        description=(
            "Large vision-language model (mPLUG-Owl2, 7B) aligned to human "
            "judgment on 5 textual levels (bad/poor/fair/good/excellent — "
            "hence the 1/2/3/4/5 bounds below). The most reliable of the "
            "list, but slow and heavy: reserved for a 16 GB+ GPU or small, "
            "finely sorted batches."
        ),
        bands=(
            QualityBand(1.0, "low", _RED),
            QualityBand(2.0, "fair", _ORANGE),
            QualityBand(3.0, "good", _GREEN),
            QualityBand(4.0, "very good", _DARK_GREEN),
            QualityBand(4.5, "excellent", _BLUE),
        ),
    ),
    "qalign_8bit": QualityMetric(
        pyiqa_name="qalign_8bit",
        label="Q-Align (8-bit)",
        score_min=1.0,
        score_max=5.0,
        min_vram_gb=9.0,
        vram_note="~8-9 GB VRAM (bitsandbytes 8-bit).",
        description=(
            "The same Q-Align, loaded in 8-bit (bitsandbytes). A little "
            "slower per image and slightly reduced accuracy, for almost "
            "half the VRAM of full precision."
        ),
        bands=(
            QualityBand(1.0, "low", _RED),
            QualityBand(2.0, "fair", _ORANGE),
            QualityBand(3.0, "good", _GREEN),
            QualityBand(4.0, "very good", _DARK_GREEN),
            QualityBand(4.5, "excellent", _BLUE),
        ),
    ),
    "qalign_4bit": QualityMetric(
        pyiqa_name="qalign_4bit",
        label="Q-Align (4-bit)",
        score_min=1.0,
        score_max=5.0,
        min_vram_gb=6.0,
        vram_note="~5-6 GB VRAM (bitsandbytes 4-bit).",
        description=(
            "Q-Align quantized to 4-bit: brings the model within reach of "
            "consumer GPUs, at the cost of a slightly noisier judgment than "
            "the 8-bit/full-precision variants."
        ),
        bands=(
            QualityBand(1.0, "low", _RED),
            QualityBand(2.0, "fair", _ORANGE),
            QualityBand(3.0, "good", _GREEN),
            QualityBand(4.0, "very good", _DARK_GREEN),
            QualityBand(4.5, "excellent", _BLUE),
        ),
    ),
    "laion_aes": QualityMetric(
        pyiqa_name="laion_aes",
        label="LAION Aesthetic (recommended)",
        score_min=1.0,
        score_max=10.0,
        min_vram_gb=1.5,
        vram_note="~1.5 GB VRAM.",
        description=(
            "CLIP ViT-L/14 aesthetic predictor, the one LAION used to "
            "filter LAION-5B. Does not measure blur/noise like MUSIQ or "
            "TOPIQ-NR, but the perceived appeal of the image — often a more "
            "relevant signal than technical quality when picking LoRA "
            "training images. Light, and complements one of the two above "
            "rather than replacing it. Community thresholds, not an "
            "official standard."
        ),
        bands=(
            QualityBand(1.0, "low", _RED),
            QualityBand(4.5, "fair", _ORANGE),
            QualityBand(6.0, "good", _GREEN),
            QualityBand(7.0, "very good", _DARK_GREEN),
            QualityBand(8.0, "excellent", _BLUE),
        ),
    ),
}

# Factory default metric: the cheapest one, and the one every pre-existing
# indexed library was scored with before this registry existed.
DEFAULT_METRIC = "musiq"

# Pseudo-metric id for the normalized average across a media's metrics. NOT a
# real pyiqa metric (never scored, never in Settings): computed on read from
# the per-metric rows. Only surfaces as a grid "displayed quality" choice, so
# it needs its own bands — generic tiers on the shared 0-100 scale.
AVERAGE_METRIC_ID = "average"

AVERAGE_METRIC = QualityMetric(
    pyiqa_name="",
    label="Average (all metrics)",
    score_min=0.0,
    score_max=100.0,
    min_vram_gb=0.0,
    vram_note="",
    description=(
        "Mean of every metric scored for the media, each normalized to "
        "0-100 before being averaged."
    ),
    bands=(
        QualityBand(0.0, "low", _RED),
        QualityBand(60.0, "fair", _ORANGE),
        QualityBand(75.0, "good", _GREEN),
        QualityBand(85.0, "very good", _DARK_GREEN),
        QualityBand(92.0, "excellent", _BLUE),
    ),
)

# Lazy singleton: the active metric (torch weights, downloaded once by pyiqa)
# loads on first score and stays for the process, reloading only if the metric
# or device changes. The lock stops a double load. Mutable globals, lowercase.
_lock = threading.Lock()
_metric = None  # pylint: disable=invalid-name
_metric_key = None  # pylint: disable=invalid-name


def get_metric_info(metric_id: str | None) -> QualityMetric:
    """Return the :class:`QualityMetric` for ``metric_id``, or the default.

    The pseudo-metric :data:`AVERAGE_METRIC_ID` returns :data:`AVERAGE_METRIC`
    (its own 0-100 bands). Any other unknown id (a stale value from before a
    metric was removed, or a typo) silently falls back to
    :data:`DEFAULT_METRIC` rather than raising, since this is read on hot
    UI-rendering paths.
    """
    if metric_id == AVERAGE_METRIC_ID:
        return AVERAGE_METRIC
    return QUALITY_METRICS.get(metric_id, QUALITY_METRICS[DEFAULT_METRIC])


def normalize_score(metric_id: str | None, score) -> float | None:
    """Return a score normalized from a metric's native range to 0-100.

    The mapping :func:`badge_style` displays, exposed so the repo can average
    metrics onto one scale (the :data:`AVERAGE_METRIC_ID` pseudo-metric)
    without the heavy scoring machinery. Raw Q-Align ``3.0`` (1-5) and raw
    MUSIQ ``50`` (0-100) both give ``50.0``. Unknown ``metric_id`` falls back
    to :data:`DEFAULT_METRIC`; ``None`` score returns ``None``.
    """
    if score is None:
        return None
    metric = get_metric_info(metric_id)
    raw = float(score)
    clamped = max(metric.score_min, min(metric.score_max, raw))
    span = metric.score_max - metric.score_min
    return (clamped - metric.score_min) / span * 100.0


def normalization_bounds() -> dict[str, tuple[float, float]]:
    """Return each real metric's ``(score_min, score_max)`` native range.

    Lets the repository build the SQL that normalizes and averages scores
    for the :data:`AVERAGE_METRIC_ID` sort (see
    :func:`src.sqlite_store.base._sort_order_by`) without depending on the
    full :class:`QualityMetric` records. The average pseudo-metric is
    excluded (it has no rows of its own).
    """
    return {
        key: (metric.score_min, metric.score_max)
        for key, metric in QUALITY_METRICS.items()
    }


def metric_choices() -> list[tuple[str, str]]:
    """Return ``(label, id)`` pairs for the Settings tab's metric dropdown."""
    return [(metric.label, key) for key, metric in QUALITY_METRICS.items()]


def badge_style(
    quality_score, metric_id: str | None = None
) -> tuple[str, str]:
    """Return ``(display_text, color)`` for a stored quality score.

    Text is the score normalized to 0-100 so badges read the same whatever
    metric produced them; color comes from the metric's native-scale bands.
    ``metric_id`` should be the metric the score was computed with (stored
    alongside it), so the badge reflects its bands even after the Settings
    selection changes. Falls back to :data:`DEFAULT_METRIC` when None/unknown.
    """
    metric = get_metric_info(metric_id)
    raw = float(quality_score)
    clamped = max(metric.score_min, min(metric.score_max, raw))
    color = metric.bands[0].color
    for band in metric.bands:
        if clamped >= band.min_score:
            color = band.color
        else:
            break
    span = metric.score_max - metric.score_min
    percent = (clamped - metric.score_min) / span * 100.0
    return f"{percent:.0f}%", color


def detect_vram_gb() -> float | None:
    """Return the primary CUDA GPU's total VRAM in GB, or None off-CUDA."""
    import torch  # pylint: disable=import-outside-toplevel

    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024**3)


# Vendor prefixes stripped from the reported device name so the topbar chip
# reads "RTX 4090", not "NVIDIA GeForce RTX 4090".
_GPU_NAME_NOISE = ("NVIDIA ", "GeForce ", "AMD ", "Radeon ")


def gpu_status() -> dict | None:
    """Return the primary CUDA GPU's name and live memory, or None off-CUDA.

    Memory from ``torch.cuda.mem_get_info`` — the driver's whole-device view,
    so ``used`` reflects every process (what a gauge should show). All in GB.
    ``{"name", "total_gb", "used_gb", "free_gb"}`` on CUDA, else None.
    """
    import torch  # pylint: disable=import-outside-toplevel

    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    for noise in _GPU_NAME_NOISE:
        name = name.replace(noise, "")
    free, total = torch.cuda.mem_get_info(0)
    gib = 1024**3
    return {
        "name": name.strip(),
        "total_gb": total / gib,
        "used_gb": (total - free) / gib,
        "free_gb": free / gib,
    }


def vram_advice_markdown() -> str:
    """Return Settings-tab Markdown recommending metrics for this GPU.

    Ranks every :data:`QUALITY_METRICS` entry against the detected VRAM (or
    notes that none was detected, e.g. a CPU-only machine), so the user can
    pick a metric without knowing the models beforehand. Purely informative:
    nothing here is enforced, a metric loads regardless of the estimate.
    """
    vram = detect_vram_gb()
    if vram is None:
        lines = [
            "**No CUDA GPU detected** — only MUSIQ, TOPIQ-NR and LAION "
            "Aesthetic stay realistic (Q-Align is a 7B vision-language "
            "model, impractical on CPU)."
        ]
    else:
        lines = [f"**GPU detected: ~{vram:.1f} GB VRAM.**"]
    for metric in QUALITY_METRICS.values():
        fits = vram is None or vram >= metric.min_vram_gb
        mark = "✅" if fits else "⚠️"
        lines.append(f"- {mark} **{metric.label}** — {metric.vram_note}")
        lines.append(f"  {metric.description}")
    return "\n".join(lines)


def _get_metric(metric_id: str):
    """Return the cached metric, (re)loaded for the id/device in use."""
    global _metric, _metric_key  # pylint: disable=global-statement

    with _lock:
        import torch  # pylint: disable=import-outside-toplevel

        # pyiqa is heavy (pulls scikit-image); imported on first score only.
        # The shim must run before pyiqa (and thus clip) is imported.
        _install_clip_pkg_resources_shim()
        import pyiqa  # pylint: disable=import-outside-toplevel

        # Best available accelerator: NVIDIA CUDA, Apple MPS, else CPU.
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        pyiqa_name = get_metric_info(metric_id).pyiqa_name
        key = (pyiqa_name, device)
        if _metric is None or _metric_key != key:
            logger.info("Loading %s quality metric on %s", pyiqa_name, device)
            _metric = pyiqa.create_metric(
                pyiqa_name, device=torch.device(device)
            )
            _metric_key = key
        return _metric


def unload_metric() -> None:
    """Release the cached metric and return its memory to the system.

    Batch callers call this when their run ends: the singleton otherwise keeps
    the weights resident (~15 GB for Q-Align), starving a VLM loaded next.
    ``gc.collect`` breaks pyiqa's reference cycles before ``empty_cache`` hands
    blocks back to the driver. No-op when nothing is loaded; the next score
    reloads.
    """
    global _metric, _metric_key  # pylint: disable=global-statement

    with _lock:
        if _metric is None:
            return
        import torch  # pylint: disable=import-outside-toplevel

        logger.info("Unloading %s quality metric", _metric_key)
        _metric = None
        _metric_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def warm_metric_stream(
    metric_id: str = DEFAULT_METRIC, poll_seconds: float = 0.3
):  # pylint: disable=redefined-variable-type
    """Load (and cache) the metric, yielding download/load progress.

    First use downloads weights (~100 MB MUSIQ to ~15 GB Q-Align) and loads
    them — minutes on a cold cache with no score produced. Batch callers drive
    this generator first to paint a live progress bar instead of one stuck at
    zero.

    The load runs in a worker thread while ``tqdm`` is instrumented process-
    wide: every downloader (torch hub, ``clip``, huggingface_hub) reports
    through a ``tqdm`` bar, so sampling the live bar covers them all. The
    indicator is cosmetic; scores are unaffected. ``poll_seconds`` sets the
    snapshot cadence. Yields ``{"desc", "n", "total"}`` (bytes for download
    bars, zeros until a bar exists). Re-raises whatever the load raised.
    """
    import tqdm as tqdm_module  # pylint: disable=import-outside-toplevel

    state = {"desc": "", "n": 0.0, "total": 0.0}
    state_lock = threading.Lock()
    original_init = tqdm_module.tqdm.__init__
    original_update = tqdm_module.tqdm.update

    def record(progress_bar):
        """Publish one bar's counters as the latest snapshot."""
        with state_lock:
            state["desc"] = getattr(progress_bar, "desc", "") or ""
            state["n"] = float(getattr(progress_bar, "n", 0) or 0)
            state["total"] = float(getattr(progress_bar, "total", 0) or 0)

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        record(self)

    def patched_update(self, n=1):
        result = original_update(self, n)
        record(self)
        return result

    errors = []

    def worker():
        try:
            _get_metric(metric_id)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            errors.append(exc)

    tqdm_module.tqdm.__init__ = patched_init
    tqdm_module.tqdm.update = patched_update
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            # Snapshot under the lock, yield outside it: the generator
            # suspends at the yield, and holding the lock across that
            # suspension would block the worker's next record().
            with state_lock:
                snapshot = dict(state)
            yield snapshot
            thread.join(poll_seconds)
    finally:
        # Monkeypatch teardown: re-binding the saved originals over the
        # patch functions trips redefined-variable-type (disabled on the
        # def line — pylint ignores block pragmas inside a finally).
        tqdm_module.tqdm.__init__ = original_init
        tqdm_module.tqdm.update = original_update
    if errors:
        raise errors[0]


def _first_frame(source_path):
    """Return a video's first frame as a PIL image, or None if unreadable."""
    import cv2  # pylint: disable=import-outside-toplevel
    from PIL import Image  # pylint: disable=import-outside-toplevel

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


def score_media(source_path, metric_id: str = DEFAULT_METRIC):
    """Return the quality score for a media's original file.

    ``source_path`` is the original file (never a thumbnail — a resized copy
    would bias the score). ``metric_id`` is a :data:`QUALITY_METRICS` key
    (default :data:`DEFAULT_METRIC`). Returns the raw native-range score, or
    None if the file couldn't be read/scored.
    """
    try:
        return _score_source(source_path, metric_id)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("Quality scoring failed for %s: %s", source_path, exc)
        return None


def _score_source(source_path, metric_id: str):
    """Open, downscale and score a media's original file (or first frame)."""
    import torch  # pylint: disable=import-outside-toplevel
    from PIL import Image  # pylint: disable=import-outside-toplevel

    cap = (_MAX_SCORE_SIDE, _MAX_SCORE_SIDE)
    with torch.inference_mode():
        if is_video_file(str(source_path)):
            image = _first_frame(source_path)
            if image is None:
                return None
            image.thumbnail(cap, Image.Resampling.LANCZOS)
            return float(_get_metric(metric_id)(image).item())

        with Image.open(source_path) as image:
            image.load()
            image.thumbnail(cap, Image.Resampling.LANCZOS)
            return float(_get_metric(metric_id)(image).item())
