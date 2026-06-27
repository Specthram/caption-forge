"""WD (Waifu Diffusion) tagger backend for media auto-tagging.

Runs one of SmilingWolf's booru taggers (ONNX) on an image and returns
danbooru-style tags above the caller's thresholds, split by kind (*general* vs
*character*; *rating* rows ignored). Gradio-free like the other backends.

Generations are picked from :data:`KNOWN_TAGGERS`, or a user folder via
:data:`LOCAL_SOURCE`. Known models aren't bundled: :func:`download` fetches the
two files (weights + tag list) into ``models/taggers/<repo>/`` once, and
:func:`is_available` gates the UI until both are present. Inference runs on GPU
when onnxruntime exposes the CUDA provider (``onnxruntime-gpu`` build), else
the CPU provider (see :func:`_providers`).
"""

import csv
import logging
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from src.constants import MODELS_DIR

logger = logging.getLogger(__name__)

# Every known SmilingWolf tagger generation offered in the UI, in listing
# order; EVA02-Large is the best-scoring v3 variant and stays the factory
# default.
KNOWN_TAGGERS = {
    "SmilingWolf/wd-swinv2-tagger-v3": "WD SwinV2 Tagger v3",
    "SmilingWolf/wd-eva02-large-tagger-v3": "WD EVA02-Large Tagger v3",
    "SmilingWolf/wd-vit-large-tagger-v3": "WD ViT-Large Tagger v3",
    "SmilingWolf/wd-v1-4-swinv2-tagger-v2": "WD v1.4 SwinV2 Tagger v2",
    "SmilingWolf/wd-vit-tagger-v3": "WD ViT Tagger v3",
}
DEFAULT_REPO_ID = "SmilingWolf/wd-eva02-large-tagger-v3"

# Sentinel "source" value selecting a user-chosen local folder instead of one
# of the known repos above.
LOCAL_SOURCE = "__local__"

MODEL_FILENAME = "model.onnx"
TAGS_FILENAME = "selected_tags.csv"

# Bundled models folder (app-owned), one sub-folder per repo so generations
# coexist. The caption-model scanner only looks at its own top level, so this
# never shows in the Caption dropdown.
TAGGERS_DIR = MODELS_DIR / "taggers"

# Community-standard starting thresholds for the v3 taggers: general tags are
# kept from a fairly low confidence, character tags only when near-certain.
DEFAULT_GENERAL_THRESHOLD = 0.35
DEFAULT_CHARACTER_THRESHOLD = 0.85

# ``selected_tags.csv`` category codes.
_CATEGORY_GENERAL = 0
_CATEGORY_CHARACTER = 4

# Lazy singleton: ONNX session (~2 GB) + labels loaded on first use, kept for
# the process, keyed by (source, local_dir) so switching model in the UI
# reloads instead of reusing a stale session. The lock stops a double load.
# Mutable module globals, hence lowercase.
_lock = threading.Lock()
_session = None  # pylint: disable=invalid-name
_labels = None  # pylint: disable=invalid-name
_loaded_key = None  # pylint: disable=invalid-name


def repo_dir(repo_id: str) -> Path:
    """Return the local download folder for a known tagger repo."""
    return TAGGERS_DIR / repo_id.rsplit("/", maxsplit=1)[-1]


def resolve_dir(source: str, local_dir: str = "") -> Path | None:
    """Return the folder the given tagger source's files live in.

    ``source`` is a :data:`KNOWN_TAGGERS` key or :data:`LOCAL_SOURCE`;
    ``local_dir`` is used only for the latter. None when :data:`LOCAL_SOURCE`
    and ``local_dir`` is blank.
    """
    if source == LOCAL_SOURCE:
        local_dir = (local_dir or "").strip()
        return Path(local_dir) if local_dir else None
    return repo_dir(source)


def model_path(
    source: str = DEFAULT_REPO_ID, local_dir: str = ""
) -> Path | None:
    """Return the local path of the ONNX model file for a source."""
    folder = resolve_dir(source, local_dir)
    return (folder / MODEL_FILENAME) if folder else None


def tags_path(
    source: str = DEFAULT_REPO_ID, local_dir: str = ""
) -> Path | None:
    """Return the local path of the tag list CSV for a source."""
    folder = resolve_dir(source, local_dir)
    return (folder / TAGS_FILENAME) if folder else None


def is_available(source: str = DEFAULT_REPO_ID, local_dir: str = "") -> bool:
    """Return whether the tagger files for ``source`` are present on disk."""
    model_file = model_path(source, local_dir)
    tags_file = tags_path(source, local_dir)
    return (
        model_file is not None
        and model_file.is_file()
        and tags_file is not None
        and tags_file.is_file()
    )


def download(source: str) -> None:
    """Download a known tagger's model + tag list from Hugging Face.

    Files land in :func:`repo_dir`. Idempotent. ``source`` is a
    :data:`KNOWN_TAGGERS` key; raises ``ValueError`` for :data:`LOCAL_SOURCE`.
    """
    if source == LOCAL_SOURCE:
        raise ValueError("A local tagger has no model to download.")
    # Imported here so the module stays cheap to import for the UI checks.
    from huggingface_hub import hf_hub_download  # pylint: disable=C0415

    target = repo_dir(source)
    target.mkdir(parents=True, exist_ok=True)
    for filename in (MODEL_FILENAME, TAGS_FILENAME):
        hf_hub_download(
            repo_id=source, filename=filename, local_dir=str(target)
        )


def release() -> None:
    """Drop the cached ONNX session (frees its memory)."""
    global _session, _labels, _loaded_key  # pylint: disable=global-statement
    with _lock:
        _session = None
        _labels = None
        _loaded_key = None


def _load_labels(tags_file: Path) -> list:
    """Return the tag list as ``(name, category)`` tuples, in model order.

    The CSV row order matches the model's output vector, so the row index is
    the score index.
    """
    with tags_file.open(encoding="utf-8", newline="") as handle:
        return [
            (row["name"], int(row["category"]))
            for row in csv.DictReader(handle)
        ]


def _providers(onnxruntime) -> list:
    """Return the ONNX execution providers to try, GPU first when present.

    The CUDA provider is only listed by onnxruntime when the
    ``onnxruntime-gpu`` build is installed; on the plain CPU build this is
    just ``["CPUExecutionProvider"]``. CPU is always kept as the last
    fallback so a missing CUDA/cuDNN runtime degrades gracefully instead of
    failing the run.
    """
    available = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _new_session(onnxruntime, model_file):
    """Create the ONNX session, retrying on CPU if the GPU provider fails.

    Listing the CUDA provider does not guarantee its runtime DLLs load
    (a mismatched or absent cuDNN, for instance); on any such failure we
    fall back to a CPU-only session so tagging still runs.
    """
    try:
        return onnxruntime.InferenceSession(
            str(model_file), providers=_providers(onnxruntime)
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(
            "WD tagger GPU session failed (%s); falling back to CPU.", exc
        )
        return onnxruntime.InferenceSession(
            str(model_file), providers=["CPUExecutionProvider"]
        )


def _get_session(source: str, local_dir: str):
    """Return the cached ONNX session (and labels) for ``source``.

    Reloads whenever the requested source differs from the one currently
    cached, so switching the selected tagger in the UI takes effect on the
    next call.
    """
    global _session, _labels, _loaded_key  # pylint: disable=global-statement
    key = (source, local_dir)
    with _lock:
        if _session is None or _loaded_key != key:
            # Imported lazily: onnxruntime is only needed once the user
            # actually tags something.
            import onnxruntime  # pylint: disable=import-outside-toplevel

            model_file = model_path(source, local_dir)
            logger.info("Loading WD tagger from %s", model_file)
            _session = _new_session(onnxruntime, model_file)
            logger.info("WD tagger running on %s", _session.get_providers())
            _labels = _load_labels(tags_path(source, local_dir))
            _loaded_key = key
        return _session, _labels


def _prepare_image(path, size: int) -> np.ndarray:
    """Return an image as the tagger's input tensor.

    WD taggers expect a white-composited, square-padded image resized to the
    model size, as float32 BGR in 0-255 (no normalization), NHWC.
    """
    with Image.open(path) as img:
        image = img.convert("RGBA")
    canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
    canvas.alpha_composite(image)
    image = canvas.convert("RGB")

    side = max(image.size)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(
        image, ((side - image.width) // 2, (side - image.height) // 2)
    )
    square = square.resize((size, size), Image.Resampling.BICUBIC)

    array = np.asarray(square, dtype=np.float32)
    return array[:, :, ::-1][np.newaxis, :]  # RGB -> BGR, add batch dim.


def _split_scores(labels, scores, general_threshold, character_threshold):
    """Return the above-threshold tags split by kind, best first.

    ``{"general": [(name, score), ...], "character": [...]}``, each sorted by
    descending score. Rating rows ignored.
    """
    result = {"general": [], "character": []}
    for (name, category), score in zip(labels, scores):
        if category == _CATEGORY_GENERAL and score >= general_threshold:
            result["general"].append((name, float(score)))
        elif category == _CATEGORY_CHARACTER and score >= character_threshold:
            result["character"].append((name, float(score)))
    for values in result.values():
        values.sort(key=lambda pair: pair[1], reverse=True)
    return result


def tag_image(
    path,
    general_threshold: float = DEFAULT_GENERAL_THRESHOLD,
    character_threshold: float = DEFAULT_CHARACTER_THRESHOLD,
    source: str = DEFAULT_REPO_ID,
    local_dir: str = "",
) -> dict:
    """Tag one image; return the confident tags split by kind.

    ``general_threshold``/``character_threshold`` are the per-kind score
    floors; ``source`` is a :data:`KNOWN_TAGGERS` key or :data:`LOCAL_SOURCE`
    (then ``local_dir``). Returns the :func:`_split_scores` dict. Raises
    ``RuntimeError`` when the tagger files aren't available yet.
    """
    if not is_available(source, local_dir):
        raise RuntimeError(
            "The tagger model is not available — download it first, or "
            "check the local folder path."
        )
    session, labels = _get_session(source, local_dir)
    model_input = session.get_inputs()[0]
    size = model_input.shape[1]  # NHWC: (batch, height, width, 3).
    tensor = _prepare_image(path, size)
    scores = session.run(None, {model_input.name: tensor})[0][0]
    return _split_scores(
        labels, scores, general_threshold, character_threshold
    )
