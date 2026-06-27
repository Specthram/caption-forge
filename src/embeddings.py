"""Visual embedding backend (DINOv2-small) for the dataset auto-builder.

Turns an image into a compact, L2-normalized feature vector whose cosine
distance reflects visual/pose similarity — the signal the auto-builder's
diversity selection maximizes (see :mod:`src.dataset_builder`). DINOv2 is
a self-supervised vision transformer: unlike CLIP, its features are
purely visual (composition, pose, framing), which is what "pick the most
varied pictures" needs.

Gradio-free, same lifecycle discipline as :mod:`src.quality`: the model
imports and loads lazily behind a lock, and :func:`unload_model` must be
called when a run ends so the weights never starve the VLM loaded next.
Images only — a video's single frame is a poor diversity signal, so the
callers keep videos out (like the lookalike detection does).
"""

import threading

import numpy as np

# The identifier stored in ``media_embedding.model_id`` — bump alongside
# ``_HF_REPO`` if the model ever changes, so stale vectors are never
# compared against new ones.
MODEL_ID = "dinov2-small"

# Hugging Face repository of the model weights (~90 MB).
_HF_REPO = "facebook/dinov2-small"

# The vector length DINOv2-small produces (its hidden size).
VECTOR_DIM = 384

_lock = threading.Lock()
_model = None  # pylint: disable=invalid-name
_processor = None  # pylint: disable=invalid-name


def vector_to_blob(vector) -> bytes:
    """Return a vector's float32 bytes for the ``media_embedding`` BLOB."""
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob) -> np.ndarray:
    """Return the float32 vector stored in a ``media_embedding`` BLOB."""
    return np.frombuffer(blob, dtype=np.float32)


def load_model() -> None:
    """Load (or reuse) the embedding model, CUDA first with CPU fallback.

    Heavy imports happen here, not at module import (house rule: a tab
    render never pays for torch/transformers). Idempotent behind the
    module lock.
    """
    global _model, _processor  # pylint: disable=global-statement

    with _lock:
        if _model is not None:
            return
        # pylint: disable=import-outside-toplevel
        import torch
        from transformers import AutoImageProcessor, AutoModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _processor = AutoImageProcessor.from_pretrained(_HF_REPO)
        _model = AutoModel.from_pretrained(_HF_REPO).to(device).eval()


def unload_model() -> None:
    """Drop the loaded model and free its VRAM.

    Call when an embedding run ends (even interrupted) — same rationale
    as :func:`src.quality.unload_metric`: kept loaded, the weights would
    hold their VRAM for the process lifetime.
    """
    global _model, _processor  # pylint: disable=global-statement

    with _lock:
        if _model is None:
            return
        _model = None
        _processor = None
        import torch  # pylint: disable=import-outside-toplevel

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def embed_image(source_path) -> np.ndarray:
    """Return the image's L2-normalized DINOv2 feature vector.

    Loads the model on first use (:func:`load_model`). The vector is the pooled
    CLS output, L2-normalized so two embeddings' cosine is their dot product.
    Returns a unit-norm float32 vector of :data:`VECTOR_DIM` components.
    """
    load_model()
    # pylint: disable=import-outside-toplevel
    import torch
    from PIL import Image

    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
    inputs = _processor(images=rgb, return_tensors="pt")
    inputs = {key: val.to(_model.device) for key, val in inputs.items()}
    with torch.no_grad():
        pooled = _model(**inputs).pooler_output[0]
    vector = pooled.float().cpu().numpy()
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = vector / norm
    return vector.astype(np.float32)
