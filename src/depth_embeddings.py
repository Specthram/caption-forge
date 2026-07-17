"""Composition embedding backend (Depth-Anything V2) for the Studio.

Turns an image into a compact, L2-normalized *depth signature*: the model's
monocular depth map reduced to a small grid and flattened. Because depth
encodes geometry — the pose, the framing, the camera, where the subject sits
in space — and not colour or texture, two pictures of the *same composition in
a different style* (a photo and its anime re-skin, a changed outfit) score a
high cosine here even when the DINOv2 appearance signal rates them far apart.
The Proximity view of the auto-builder fuses this with DINOv2 to surface those
re-skins, still redundant for a LoRA (the model memorises the shared layout).

Gradio-free, same lifecycle discipline as :mod:`src.embeddings`: the model
imports and loads lazily behind a lock, and :func:`unload_model` must be
called when a run ends so the weights never starve the model loaded next.
Images only — a video's single frame is a poor composition signal, so the
callers keep videos out (like the DINOv2 embedding does). The vectors share
the ``media_embedding`` table, keyed by :data:`MODEL_ID`, so they coexist
with the DINOv2 and SigLIP vectors without a schema of their own.
"""

import threading

import numpy as np

# The identifier stored in ``media_embedding.model_id`` — bump alongside
# ``_HF_REPO`` or ``_GRID`` if either ever changes, so stale signatures are
# never compared against new ones.
MODEL_ID = "depth-anything-v2-small"

# Hugging Face repository of the model weights (~100 MB).
_HF_REPO = "depth-anything/Depth-Anything-V2-Small-hf"

# The depth map is reduced to a ``_GRID`` × ``_GRID`` block of average depth,
# flattened into the signature. Small enough to be style-invariant (fine
# texture is averaged away, only the coarse spatial layout survives), large
# enough to separate distinct compositions.
_GRID = 16

# The vector length the reduction produces (its flattened grid).
VECTOR_DIM = _GRID * _GRID

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
    """Load (or reuse) the depth model, CUDA first with CPU fallback.

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
        from transformers import AutoImageProcessor
        from transformers import AutoModelForDepthEstimation

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _processor = AutoImageProcessor.from_pretrained(_HF_REPO)
        _model = (
            AutoModelForDepthEstimation.from_pretrained(_HF_REPO)
            .to(device)
            .eval()
        )


def unload_model() -> None:
    """Drop the loaded model and free its VRAM.

    Call when a depth run ends (even interrupted) — same rationale as
    :func:`src.embeddings.unload_model`: kept loaded, the weights would
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


def _reduce(depth) -> np.ndarray:
    """Return the L2-normalized composition signature of a depth map.

    The raw depth map is normalized to ``[0, 1]`` (so the signature does not
    depend on the absolute depth scale), average-pooled to a
    :data:`_GRID` × :data:`_GRID` block (fine texture averaged away — only the
    coarse spatial layout survives, which is what makes it style-invariant),
    mean-centred and L2-normalized so two signatures' cosine is their dot
    product. ``depth`` is a 2-D float array.
    """
    array = np.asarray(depth, dtype=np.float32)
    low = float(array.min())
    high = float(array.max())
    if high > low:
        array = (array - low) / (high - low)
    else:
        array = np.zeros_like(array)
    rows = np.array_split(array, _GRID, axis=0)
    grid = np.stack(
        [
            np.stack([block.mean() for block in np.array_split(row, _GRID, 1)])
            for row in rows
        ]
    )
    vector = grid.reshape(-1).astype(np.float32)
    vector = vector - float(vector.mean())
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = vector / norm
    return vector.astype(np.float32)


def embed_image(source_path) -> np.ndarray:
    """Return the image's L2-normalized Depth-Anything V2 composition vector.

    Loads the model on first use (:func:`load_model`). Runs monocular depth
    estimation, then reduces the depth map to the fixed-length signature of
    :func:`_reduce`. Returns a unit-norm float32 vector of :data:`VECTOR_DIM`
    components.
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
        depth = _model(**inputs).predicted_depth[0]
    return _reduce(depth.float().cpu().numpy())
