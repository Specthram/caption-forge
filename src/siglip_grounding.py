"""SigLIP 2 visual grounding — how well a short text matches an image.

The engine behind the Caption and Media grounding modals. It answers one
question per text: *does the image support this statement?* — a caption
claim ("a red ball on the grass") or a tag wrapped in :data:`TAG_PROMPT`.

Two signals come out of a single image forward pass:

* a **score**, the model's own sigmoid probability for the (image, text)
  pair. SigLIP is trained with a pairwise sigmoid loss, so every text is
  judged independently and many can be true at once — never a softmax
  distribution over the texts.
* a **heat grid**, one cosine value per image patch. SigLIP has no box
  head and no CLS token: its image embedding is produced by a multi-head
  attention pooling head (MAP). Feeding a *single* patch to that head
  collapses its attention to identity, so the head reduces to
  ``out_proj(v_proj(x)) + mlp(layernorm(...))`` — a per-patch embedding in
  the joint image/text space, obtained without a second forward pass (see
  :func:`_dense_patch_embeddings`). Its cosine with the text embedding is
  the localization map the UI paints.

Every fixed-resolution SigLIP 2 checkpoint declares ``model_type:
"siglip"`` and loads through the SigLIP 1 classes; only the NaFlex ones
use the ``siglip2`` architecture, and their variable patch grid is why
they are not offered here.

Gradio-free, same lifecycle discipline as :mod:`src.quality` and
:mod:`src.embeddings`: heavy imports are lazy, loading is guarded by a
module lock, and :func:`unload_model` must run when a job ends so the
weights never starve the VLM loaded next. Images only.
"""

import base64
import threading

import numpy as np

# The tag pre-prompt of the Media grounding mode: a bare booru tag is a
# poor caption, so each one is injected into a fixed sentence before it is
# scored. No LLM is involved on that path.
TAG_PROMPT = "a photo that contains {tag}"

# SigLIP's text tower is trained with a fixed 64-token context, padded to
# its full length — a shorter padding shifts the embeddings.
TEXT_CONTEXT = 64

# --- Score calibration ---
# SigLIP's own sigmoid is tuned for retrieval vs a huge negative set: its bias
# (~-16) crushes almost every "is this true of the image?" to 0, even correct
# claims. The signal is in the *cosine*, not the probability — so a claim is
# scored relative to a bank of generic prompts on the same image: how far its
# cosine beats what random text scores. Keeps SigLIP's independence (no
# softmax) while spreading scores over a usable 0-100 with a stable threshold.
_NEG_PROMPTS = (
    "a photo",
    "an image",
    "a picture",
    "something",
    "a random object",
    "a texture",
    "an abstract shape",
    "a blank background",
    "noise",
    "a screenshot",
    "text",
    "a color",
)
# Logistic slope over the cosine gap (claim − reference). Tuned so a clear
# match (~0.06 above the reference) saturates near 100 and a non-match sits
# low, on real photographs and hand-drawn images alike.
CALIB_SLOPE = 90.0
# The reference is a high quantile of the negatives' cosines — a claim must
# beat what nearly every contentless prompt scores on this image to pass.
CALIB_REF_QUANTILE = 0.9

# Every offered checkpoint is a ``patch16`` one, so the patch grid a heat
# map is drawn on is always ``resolution // 16`` squared: 256 -> 16x16,
# 384 -> 24x24, 512 -> 32x32. Finer grid, sharper heat map, more VRAM.
PATCH_SIZE = 16

# The four size tiers, smallest first. ``vram`` is the rough fp16 cost of
# the loaded weights, shown in Settings next to the choice.
MODEL_SIZES = {
    "base": {
        "label": "Base",
        "params": "0.4B",
        "vram": "~1.5-2 GB",
        "resolutions": (256, 384, 512),
    },
    "large": {
        "label": "Large",
        "params": "0.9B",
        "vram": "~2.5-3 GB",
        "resolutions": (256, 384, 512),
    },
    "so400m": {
        "label": "SO400M",
        "params": "1B",
        "vram": "~3-4 GB",
        "resolutions": (256, 384, 512),
    },
    "giant-opt": {
        "label": "Giant",
        "params": "2B",
        "vram": "~5-6 GB",
        "resolutions": (256, 384),
    },
}

DEFAULT_SIZE = "so400m"
DEFAULT_RESOLUTION = 512

_lock = threading.Lock()
_model = None  # pylint: disable=invalid-name
_processor = None  # pylint: disable=invalid-name
_model_id = None  # pylint: disable=invalid-name
# Text embeddings are pure functions of (checkpoint, text): a tag batch
# re-scores the same few prompts across hundreds of media, so they are
# memoised for the lifetime of the loaded model and dropped with it.
_text_cache: dict = {}


def clamp_size(size) -> str:
    """Return a known model size, falling back to :data:`DEFAULT_SIZE`."""
    return size if size in MODEL_SIZES else DEFAULT_SIZE


def clamp_resolution(size, resolution) -> int:
    """Return a resolution the given size ships, else its largest one.

    ``size`` is a :data:`MODEL_SIZES` key (unknown falls back first);
    ``resolution`` the requested square input. Returns it when the checkpoint
    exists (Giant tops at 384), else the size's largest.
    """
    offered = MODEL_SIZES[clamp_size(size)]["resolutions"]
    try:
        value = int(resolution)
    except (TypeError, ValueError):
        return max(offered)
    return value if value in offered else max(offered)


def repo_id(size, resolution) -> str:
    """Return the Hugging Face repository for a size / resolution pair."""
    size = clamp_size(size)
    return f"google/siglip2-{size}-patch{PATCH_SIZE}-{resolution}"


def grid_side(resolution) -> int:
    """Return the number of patches per side at a given resolution."""
    return int(resolution) // PATCH_SIZE


def tag_prompt(tag: str) -> str:
    """Return the sentence a bare tag is scored through."""
    return TAG_PROMPT.format(tag=tag)


def loaded_model_id():
    """Return the repository id currently loaded, or None."""
    return _model_id


def load_model(size=DEFAULT_SIZE, resolution=DEFAULT_RESOLUTION) -> str:
    """Load (or reuse) a SigLIP 2 checkpoint; return its repository id.

    Heavy imports happen here, never at module import (house rule: a tab render
    never pays for torch). Idempotent behind the module lock; a different
    checkpoint unloads the current one first. ``size`` a :data:`MODEL_SIZES`
    key, ``resolution`` a square input the size ships (clamped otherwise).
    """
    global _model, _processor, _model_id  # pylint: disable=global-statement

    size = clamp_size(size)
    resolution = clamp_resolution(size, resolution)
    wanted = repo_id(size, resolution)
    with _lock:
        if _model_id == wanted:
            return wanted
    if _model_id is not None:
        unload_model()
    with _lock:
        # pylint: disable=import-outside-toplevel
        import torch
        from transformers import AutoModel, AutoProcessor

        cuda = torch.cuda.is_available()
        _processor = AutoProcessor.from_pretrained(wanted)
        _model = AutoModel.from_pretrained(
            wanted, dtype=torch.float16 if cuda else torch.float32
        )
        _model = _model.to("cuda" if cuda else "cpu").eval()
        _model_id = wanted
    return wanted


def unload_model() -> None:
    """Drop the loaded checkpoint and free its VRAM.

    Call when a grounding run ends, even interrupted — kept loaded, the
    weights would hold their VRAM for the process lifetime and the VLM
    queued behind would fail to allocate.
    """
    global _model, _processor, _model_id  # pylint: disable=global-statement

    with _lock:
        if _model is None:
            return
        _model = None
        _processor = None
        _model_id = None
        _text_cache.clear()
        import torch  # pylint: disable=import-outside-toplevel

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _dense_patch_embeddings(hidden, head):
    """Return one joint-space embedding per patch token.

    SigLIP pools its patch tokens with a learned probe through
    ``nn.MultiheadAttention`` and reads the single output token. Attention
    over a *single* key is a softmax over one element, i.e. the identity,
    so the pooled embedding of a one-patch image is exactly
    ``out_proj(v_proj(x))`` pushed through the head's residual MLP. Doing that
    for every patch at once yields the dense map — no extra forward pass.
    ``hidden`` is ``(batch, patches, dim)`` (the tower's post-layernormed
    ``last_hidden_state``), ``head`` its attention pooling head. Returns
    ``(batch, patches, dim)`` in the ``image_embeds`` space.
    """
    attention = head.attention
    dim = hidden.shape[-1]
    # The value slice of the packed q/k/v projection.
    weight = attention.in_proj_weight[2 * dim :]
    bias = attention.in_proj_bias[2 * dim :]
    value = hidden @ weight.t() + bias
    attended = attention.out_proj(value)
    return attended + head.mlp(head.layernorm(attended))


def _pooled(output):
    """Return the pooled embedding from a text/vision feature call.

    Transformers 5.x wraps ``get_text_features`` / ``get_image_features`` in
    a ``BaseModelOutputWithPooling`` (the projected embedding sits at
    ``pooler_output``); older releases handed the tensor back directly. This
    accepts either, so a transformers bump does not silently break scoring.
    """
    return getattr(output, "pooler_output", output)


def _encode_texts(texts):
    """Return the L2-normalized text embeddings, memoised per checkpoint."""
    import torch  # pylint: disable=import-outside-toplevel

    missing = [text for text in texts if text not in _text_cache]
    if missing:
        inputs = _processor(
            text=missing,
            padding="max_length",
            max_length=TEXT_CONTEXT,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: val.to(_model.device) for key, val in inputs.items()}
        with torch.no_grad():
            features = _pooled(_model.get_text_features(**inputs))
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        for text, vector in zip(missing, features.float().cpu()):
            _text_cache[text] = vector
    return torch.stack([_text_cache[text] for text in texts])


def _encode_image(source_path):
    """Return an image's ``(pooled, dense)`` normalized embeddings."""
    # pylint: disable=import-outside-toplevel
    import torch
    from PIL import Image

    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
    inputs = _processor(images=rgb, return_tensors="pt")
    pixels = inputs["pixel_values"].to(_model.device, dtype=_model.dtype)
    vision = _model.vision_model
    with torch.no_grad():
        outputs = vision(pixel_values=pixels)
        pooled = outputs.pooler_output
        dense = _dense_patch_embeddings(outputs.last_hidden_state, vision.head)
    pooled = pooled.float().cpu()
    dense = dense.float().cpu()
    pooled = pooled / pooled.norm(p=2, dim=-1, keepdim=True)
    dense = dense / dense.norm(p=2, dim=-1, keepdim=True)
    return pooled, dense


def _heat_grid(similarity, side):
    """Return a patch cosine row min-max normalized to a ``uint8`` square.

    The absolute cosine range of the dense map is checkpoint-dependent and
    narrow, so the grid carries *relative* evidence within one image: 255
    is "the patch that supports this text most", 0 the least. How loudly
    the map is painted is the caller's business — the UI scales its
    opacity by the text's own :func:`ground_image` score, so a hallucinated
    claim stays dim even though its brightest patch reads 255.
    """
    low = float(similarity.min())
    high = float(similarity.max())
    span = high - low
    if span <= 0.0:
        normalized = np.zeros_like(similarity)
    else:
        normalized = (similarity - low) / span
    grid = np.round(normalized * 255.0).astype(np.uint8)
    return grid.reshape(side, side)


def _encode_grid(grid) -> str:
    """Return a heat grid as base64 (row-major, one byte per patch)."""
    return base64.b64encode(grid.tobytes()).decode("ascii")


def ground_image(source_path, texts, with_heat: bool = True) -> list[dict]:
    """Score every text against an image, with an optional heat map.

    One image forward pass serves every text: the pooled embedding gives the
    scores, the dense patch embeddings the maps. Model must already be loaded
    (:func:`load_model`) — the caller owns load/unload so a batch pays once.
    ``texts`` are short statements (a claim, or :func:`tag_prompt` output);
    ``with_heat`` computes the per-patch grids (skipped by the score-only batch
    scorer). Returns one ``{"text", "score", "heat", "side"}`` per text in
    order — ``score`` calibrated 0-100 (:func:`_calibrate_scores`), ``heat`` a
    base64 ``uint8`` ``side*side`` grid (None when ``with_heat`` false).
    """
    texts = list(texts)
    if not texts:
        return []
    pooled, dense = _encode_image(source_path)
    text_embeds = _encode_texts(texts)
    claim_cos = (text_embeds @ pooled.t()).squeeze(-1)
    scores = _calibrate_scores(claim_cos, pooled)
    side = int(round(dense.shape[1] ** 0.5))
    patch_similarity = None
    if with_heat:
        patch_similarity = (dense[0] @ text_embeds.t()).numpy()
    results = []
    for index, text in enumerate(texts):
        heat = None
        if patch_similarity is not None:
            heat = _encode_grid(_heat_grid(patch_similarity[:, index], side))
        results.append(
            {
                "text": text,
                "score": float(scores[index]),
                "heat": heat,
                "side": side,
            }
        )
    return results


def embed_image(source_path):
    """Return an image's L2-normalized pooled embedding as a NumPy vector.

    The joint-space vector SigLIP scores a text against — stored once per media
    by the ``siglip`` index step so the composer can rank a whole library
    against a typed query without re-reading a file. Model must already be
    loaded (:func:`load_model`). Returns a 1-D ``float32`` unit vector.
    """
    pooled, _dense = _encode_image(source_path)
    return pooled[0].numpy().astype("float32")


def embed_text(text: str):
    """Return a query's L2-normalized text embedding as a NumPy vector.

    The counterpart of :func:`embed_image`: the cosine of the two is the
    text-to-image relevance the composer's semantic search sorts by. The
    model must already be loaded (see :func:`load_model`).
    """
    return _encode_texts([text])[0].numpy().astype("float32")


def caption_cosines(source_path, texts) -> list[float]:
    """Return the raw image-text cosine of each text, uncalibrated.

    The reference-free caption score (:mod:`src.caption_score`) needs SigLIP's
    *raw* cosine, not the per-claim grounding score: the negative-bank
    calibration :func:`ground_image` applies saturates on a whole caption. One
    image forward pass, same loaded checkpoint. The model must already be
    loaded (see :func:`load_model`); the caller owns the load/unload pair.
    """
    texts = list(texts)
    if not texts:
        return []
    pooled, _dense = _encode_image(source_path)
    text_embeds = _encode_texts(texts)
    claim_cos = (text_embeds @ pooled.t()).squeeze(-1)
    return [float(value) for value in claim_cos]


def _calibrate_scores(claim_cos, pooled):
    """Return 0-100 scores from claim cosines, referenced to a negative bank.

    ``claim_cos`` is each claim's cosine with the image; ``pooled`` the
    image embedding. The generic :data:`_NEG_PROMPTS` are scored on the same
    image and their :data:`CALIB_REF_QUANTILE` quantile is the reference a
    claim's cosine is measured against: ``sigmoid(slope * (cos - ref))``.
    Per-image, so a picture that happens to match a "negative" (text on it,
    say) simply raises its own bar rather than skewing the scale.
    """
    import torch  # pylint: disable=import-outside-toplevel

    neg_embeds = _encode_texts(_NEG_PROMPTS)
    neg_cos = (neg_embeds @ pooled.t()).squeeze(-1)
    ref = torch.quantile(neg_cos, CALIB_REF_QUANTILE)
    return torch.sigmoid(CALIB_SLOPE * (claim_cos - ref)) * 100.0
