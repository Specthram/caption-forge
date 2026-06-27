"""Watermark detection: a SigLIP pre-filter and a YOLO box detector.

Two model-backed signals feed the Watermark Lab's batch scan (the third,
scanning existing captions/tags for the word "watermark", is text-only and
lives in :mod:`src.watermark`):

* **SigLIP pre-filter** (:func:`watermark_score`) — a zero-shot "does this
  image carry a watermark/logo/text overlay?" score reusing the SigLIP 2
  checkpoint already in the app (:mod:`src.siglip_grounding`). It has no box
  head, so it only *screens* — a clean image scoring low is skipped before the
  expensive detector runs, discarding the bulk of a healthy library.
* **YOLO detector** (:func:`detect_yolo`) — a fine-tuned watermark YOLO in
  Ultralytics ``.pt`` format, the only signal that yields bounding boxes. The
  weights are user-provided: any ``.pt`` file dropped in the configured models
  folder (defaulting to the app ``models/`` directory) is offered in the rail,
  and the selected one is run through Ultralytics. With none selected the
  detector reports unavailable and the scan falls back to the pre-filter +
  caption/tag scan + manual zones.

The model lifecycle (load / unload around a job) is owned by the runner, as
for grounding — these functions assume the relevant model is already loaded.
"""

import logging

from src import settings, siglip_grounding
from src.constants import DEFAULT_OWLV2_MODEL, OWLV2_MODEL_IDS

logger = logging.getLogger(__name__)

# The positive prompt the SigLIP pre-filter scores an image against. Its
# calibrated 0-100 correspondence (see src.siglip_grounding.ground_image) is
# compared to the pre-filter threshold.
WATERMARK_PROMPT = "a watermark, logo, signature or text overlay on the image"

_yolo_model = None  # pylint: disable=invalid-name
_yolo_key = None  # pylint: disable=invalid-name


def yolo_models_dir():
    """Return the folder scanned for watermark YOLO ``.pt`` weights.

    Configured in Settings (:func:`src.settings.get_watermark_models_dir`),
    the app ``models/`` folder by default.
    """
    return settings.get_watermark_models_dir()


def list_yolo_models() -> list:
    """Return the ``.pt`` filenames available in the models folder, sorted."""
    folder = yolo_models_dir()
    if not folder.is_dir():
        return []
    return sorted(path.name for path in folder.glob("*.pt"))


def selected_yolo_path():
    """Return the selected YOLO ``.pt`` path, or None when unset/missing."""
    name = settings.get_watermark_prefs()["yolo_model"]
    if not name:
        return None
    path = yolo_models_dir() / name
    return path if path.is_file() else None


def is_yolo_available() -> bool:
    """Return whether a selected watermark YOLO ``.pt`` exists on disk."""
    return selected_yolo_path() is not None


def watermark_score(source_path) -> float:
    """Return the SigLIP 0-100 likelihood the image carries a watermark.

    The SigLIP checkpoint must already be loaded (see
    :func:`src.siglip_grounding.load_model`); the runner owns the load/unload.
    """
    results = siglip_grounding.ground_image(
        source_path, [WATERMARK_PROMPT], with_heat=False
    )
    return float(results[0]["score"]) if results else 0.0


def _load_yolo():
    """Load (once) and return the selected Ultralytics model, or None.

    Reloads when the selected weights change, so switching the rail's model
    picker takes effect on the next scan.
    """
    global _yolo_model, _yolo_key  # pylint: disable=global-statement
    path = selected_yolo_path()
    if path is None:
        return None
    key = str(path)
    if _yolo_model is not None and _yolo_key == key:
        return _yolo_model
    try:
        from ultralytics import YOLO  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.warning("ultralytics not installed — YOLO detection disabled.")
        return None
    _yolo_model = YOLO(key)
    _yolo_key = key
    return _yolo_model


def unload_yolo() -> None:
    """Drop the YOLO model and free its VRAM (freed when detect phase ends)."""
    global _yolo_model, _yolo_key  # pylint: disable=global-statement
    if _yolo_model is None:
        return
    _yolo_model = None
    _yolo_key = None
    try:
        import torch  # pylint: disable=import-outside-toplevel

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def detect_yolo(source_path, confidence_min: float = 25.0) -> list:
    """Return watermark boxes the selected YOLO finds, or ``[]`` when none.

    ``confidence_min`` is a 0-100 floor; a box under it is dropped. Returns
    ``{"box": {"x","y","w","h"}, "score", "detector": "yolo"}`` dicts, box in
    source fractions. Empty when no model is available or nothing is found.
    """
    model = _load_yolo()
    if model is None:
        return []
    results = model.predict(
        source=str(source_path),
        conf=confidence_min / 100.0,
        verbose=False,
    )
    detections = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for xyxyn, conf in zip(boxes.xyxyn.tolist(), boxes.conf.tolist()):
            left, top, right, bottom = xyxyn
            detections.append(
                {
                    "box": {
                        "x": max(0.0, min(1.0, left)),
                        "y": max(0.0, min(1.0, top)),
                        "w": max(0.0, min(1.0, right - left)),
                        "h": max(0.0, min(1.0, bottom - top)),
                    },
                    "score": round(float(conf) * 100.0, 1),
                    "detector": "yolo",
                }
            )
    return detections


# --- OWLv2 zero-shot open-vocabulary detector (the v2 default) ---
# OWLv2 takes free-text queries and returns a box + score per query, so a
# watermark is found by *describing* it ("watermark", "copyright text", …)
# rather than by a fine-tuned class head. The ensemble large checkpoint is
# auto-downloaded and cached by transformers on first scan (~1.7 GB). Same
# lifecycle discipline as the SigLIP grounder: lazy heavy imports, the runner
# owns the load/unload pair so the weights never starve the FLUX edit queued
# behind them.

OWLV2_MODEL_ID = DEFAULT_OWLV2_MODEL


def clamp_owlv2_model(model_id) -> str:
    """Return a known OWLv2 model id, falling back to the default."""
    return model_id if model_id in OWLV2_MODEL_IDS else OWLV2_MODEL_ID


_owlv2_model = None  # pylint: disable=invalid-name
_owlv2_processor = None  # pylint: disable=invalid-name
_owlv2_id = None  # pylint: disable=invalid-name


def load_owlv2(model_id=None):
    """Load (once) an OWLv2 detector and processor; return ``(model, proc)``.

    Auto-downloaded and cached by transformers on first use (the configured HF
    token, if any, is applied first). Reloads when ``model_id`` changes so
    switching the rail's checkpoint takes effect on the next scan; a repeat
    call for the same id reuses the resident model. Heavy imports stay lazy;
    the runner owns :func:`unload_owlv2`.
    """
    # pylint: disable=global-statement
    global _owlv2_model, _owlv2_processor, _owlv2_id
    settings.apply_hf_token()
    wanted = clamp_owlv2_model(model_id or OWLV2_MODEL_ID)
    if _owlv2_model is not None and _owlv2_id == wanted:
        return _owlv2_model, _owlv2_processor
    if _owlv2_model is not None:
        unload_owlv2()
    # pylint: disable=import-outside-toplevel
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    cuda = torch.cuda.is_available()
    _owlv2_processor = Owlv2Processor.from_pretrained(wanted)
    model = Owlv2ForObjectDetection.from_pretrained(
        wanted, dtype=torch.float16 if cuda else torch.float32
    )
    _owlv2_model = model.to("cuda" if cuda else "cpu").eval()
    _owlv2_id = wanted
    return _owlv2_model, _owlv2_processor


def unload_owlv2() -> None:
    """Drop the OWLv2 model and free its VRAM (at detect-phase end)."""
    # pylint: disable=global-statement
    global _owlv2_model, _owlv2_processor, _owlv2_id
    if _owlv2_model is None:
        return
    _owlv2_model = None
    _owlv2_processor = None
    _owlv2_id = None
    try:
        import torch  # pylint: disable=import-outside-toplevel

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _owlv2_queries_out(results, queries: list) -> list:
    """Return the matched query string of each detection, in result order.

    Transformers renamed the grounded post-process output over releases
    (``text_labels`` strings now, a ``labels`` index tensor before); this
    reads whichever is present so a version bump never drops the term that
    matched.
    """
    text_labels = results.get("text_labels")
    if text_labels is not None:
        return [str(label) for label in text_labels]
    labels = results.get("labels")
    if labels is not None:
        out = []
        for label in labels.tolist():
            index = int(label)
            out.append(queries[index] if 0 <= index < len(queries) else None)
        return out
    return [None] * len(results.get("scores", []))


def _box_overlap(a, b) -> tuple:
    """Return ``(iou, contained)`` of two fraction boxes.

    ``contained`` is the share of ``a`` that lies inside ``b`` — used so a
    small sub-box nested in a larger detection of the same watermark is merged
    away even when their IoU is low.
    """
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    ih = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = iw * ih
    area_a = a["w"] * a["h"]
    union = area_a + b["w"] * b["h"] - inter
    return (
        inter / union if union > 0 else 0.0,
        inter / area_a if area_a > 0 else 0.0,
    )


def _nms(
    detections: list, iou_max: float = 0.5, contain_max: float = 0.6
) -> list:
    """Greedily drop overlapping detections, keeping the highest score.

    OWLv2 fires several boxes on one watermark (different queries, nested
    crops); merging them leaves one clean zone per mark instead of a stack.
    """
    kept: list = []
    for det in sorted(detections, key=lambda d: -d["score"]):
        keep = True
        for other in kept:
            iou, contained = _box_overlap(det["box"], other["box"])
            if iou > iou_max or contained > contain_max:
                keep = False
                break
        if keep:
            kept.append(det)
    return kept


def detect_owlv2(source_path, queries, confidence_min: float = 20.0) -> list:
    """Return watermark boxes OWLv2 finds for the given text queries.

    Zero-shot: each query is scored independently; a box under
    ``confidence_min`` (0-100) is dropped at detection time (the v2 rule — no
    weak zone is ever created). Returns ``{"box", "score", "detector":
    "owlv2", "query"}`` dicts, box in source fractions. The model must already
    be loaded (:func:`load_owlv2`).
    """
    queries = [q.strip() for q in (queries or []) if q and str(q).strip()]
    if not queries or _owlv2_model is None:
        return []
    # pylint: disable=import-outside-toplevel
    import torch
    from PIL import Image, ImageOps

    with Image.open(source_path) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
    width, height = rgb.size
    inputs = _owlv2_processor(text=[queries], images=rgb, return_tensors="pt")
    inputs = {key: val.to(_owlv2_model.device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = _owlv2_model(**inputs)
    # OWLv2 pads every image to a square before resizing, and its predicted
    # boxes are normalised over that padded square (not the original). Passing
    # the original (h, w) here mislocates boxes on non-square images — a corner
    # logo drifts off the edge. So post-process against the square (the longest
    # side) and divide by the original dimensions: the image sits at the
    # square's top-left, so a box in square pixels maps straight to a source
    # fraction once scaled by the real width/height.
    side = max(width, height)
    target = torch.tensor([[side, side]], device=_owlv2_model.device)
    post = getattr(
        _owlv2_processor, "post_process_grounded_object_detection", None
    ) or getattr(_owlv2_processor, "post_process_object_detection")
    results = post(
        outputs=outputs,
        target_sizes=target,
        threshold=confidence_min / 100.0,
    )[0]
    terms = _owlv2_queries_out(results, queries)
    detections = []
    for index, (box, score) in enumerate(
        zip(results["boxes"].tolist(), results["scores"].tolist())
    ):
        left, top, right, bottom = box
        detections.append(
            {
                "box": {
                    "x": max(0.0, min(1.0, left / width)),
                    "y": max(0.0, min(1.0, top / height)),
                    "w": max(0.0, min(1.0, (right - left) / width)),
                    "h": max(0.0, min(1.0, (bottom - top) / height)),
                },
                "score": round(float(score) * 100.0, 1),
                "detector": "owlv2",
                "query": terms[index] if index < len(terms) else None,
            }
        )
    return _nms(detections)
