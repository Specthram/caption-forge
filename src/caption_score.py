"""Reference-free caption scoring — how well a whole caption fits an image.

The "zero-reference" companion to :mod:`src.siglip_grounding`. Grounding
decomposes a caption into atomic claims and scores each; this module skips
the LLM entirely and asks a blunter question of several vision-language
encoders at once: *taken whole, does this caption match the pixels?* — with
no ground-truth reference caption to compare against (hence reference-free,
the CLIPScore family of metrics).

Three encoders answer, one line each in the Caption tab's score card:

* **SigLIP2** — reuses the grounding checkpoint and its calibrated score
  (see :func:`src.siglip_grounding.ground_image`), so the two features share
  one model.
* **CLIP (OpenAI)** — the original ``openai/clip-vit-*`` contrastive model.
* **BLIP** — Salesforce's ``blip-itm-*`` retrieval model, read through its
  image-text contrastive (ITC) cosine, not its ITM classification head.

Every encoder is scored the same way, so their numbers are broadly
comparable: the caption's cosine with the image is measured *relative to a
bank of generic, contentless prompts* run on the same image, then squashed
to 0-100. A raw cosine is not comparable across encoders (each has its own
range and bias); the gap over a per-image reference is far steadier. This is
the same calibration :mod:`src.siglip_grounding` applies to a claim.

Gradio-free, same lifecycle discipline as the grounding engine: heavy
imports are lazy and every model a scorer loads is unloaded in a ``finally``
so it never starves the model queued behind it. Images only — a caption
score needs a still frame, so a video is skipped upstream.
"""

from contextlib import contextmanager

from src import siglip_grounding

# The model families offered, smallest checkpoint first inside each. SigLIP2
# is not listed here: its checkpoint is the grounding one (Settings → Caption
# grounding), reused so the app never loads two SigLIPs. ``vram`` is the rough
# fp16 cost of the loaded weights, shown in Settings next to the choice.
CLIP_SIZES = {
    "base": {
        "label": "Base",
        "repo": "openai/clip-vit-base-patch32",
        "params": "0.15B",
        "vram": "~0.6 GB",
    },
    "large": {
        "label": "Large",
        "repo": "openai/clip-vit-large-patch14",
        "params": "0.4B",
        "vram": "~1.7 GB",
    },
}
BLIP_SIZES = {
    "base": {
        "label": "Base",
        "repo": "Salesforce/blip-itm-base-coco",
        "params": "0.2B",
        "vram": "~0.9 GB",
    },
    "large": {
        "label": "Large",
        "repo": "Salesforce/blip-itm-large-coco",
        "params": "0.5B",
        "vram": "~1.8 GB",
    },
}

DEFAULT_CLIP_SIZE = "large"
DEFAULT_BLIP_SIZE = "large"

# The three lines the score card renders, in display order. ``siglip2`` reuses
# the grounding checkpoint; the other two carry their own size catalogue so
# Settings can offer a base / large choice per family.
KINDS = ("siglip2", "clip", "blip")
LABELS = {
    "siglip2": "SigLIP2",
    "clip": "CLIP (OpenAI)",
    "blip": "BLIP",
}

# The Settings catalogue: only the two families that have a size selector of
# their own (SigLIP2 reuses the grounding card's selector).
CATALOGUE = {
    "clip": {"label": LABELS["clip"], "sizes": CLIP_SIZES},
    "blip": {"label": LABELS["blip"], "sizes": BLIP_SIZES},
}

# --- Score calibration ---
# The raw image-caption cosine, linearly rescaled to 0-100 per encoder. A
# negative-prompt bank was tried and abandoned: a real caption always beats
# generic prompts, pinning every caption to ~99-100 (useless for ranking). The
# raw cosine *does* discriminate, in a narrow per-encoder band, so each family
# maps its realistic ``(low, high)`` cosine window onto 0-100. Tune these for
# your data: ``low`` maps to 0 (noise), ``high`` to 100 (tight + complete).
COSINE_BOUNDS = {
    # SigLIP2 cosines are small (its sigmoid bias lives elsewhere).
    "siglip2": (0.02, 0.16),
    # CLIP ViT-L/14: a matched caption sits ~0.25-0.32, a weak one ~0.15.
    "clip": (0.15, 0.32),
    # BLIP ITC cosines run higher and wider than CLIP's.
    "blip": (0.25, 0.55),
}


def _rescale(cosine: float, kind: str) -> float:
    """Map a raw cosine to 0-100 through the encoder's realistic window."""
    low, high = COSINE_BOUNDS[kind]
    score = (float(cosine) - low) / (high - low) * 100.0
    return max(0.0, min(score, 100.0))


def _pooled(output):
    """Return the pooled embedding of a features call, transformers-agnostic.

    Transformers 5.x wraps ``get_image_features`` / ``get_text_features`` in a
    ``BaseModelOutputWithPooling`` (the projected embedding is at
    ``pooler_output``); older releases returned the tensor directly. Accepting
    either is what a bare CLIP forward missed before — the wrapped output has
    no ``.norm`` and the scorer silently failed, leaving the CLIP line blank.
    """
    return getattr(output, "pooler_output", output)


def _cosine(image_embed, text_embed) -> float:
    """Return the cosine of one image embedding against one text embedding."""
    image_embed = image_embed / image_embed.norm(p=2, dim=-1, keepdim=True)
    text_embed = text_embed / text_embed.norm(p=2, dim=-1, keepdim=True)
    return float((image_embed * text_embed).sum())


def clamp_clip_size(size) -> str:
    """Return a known CLIP size, falling back to :data:`DEFAULT_CLIP_SIZE`."""
    return size if size in CLIP_SIZES else DEFAULT_CLIP_SIZE


def clamp_blip_size(size) -> str:
    """Return a known BLIP size, falling back to :data:`DEFAULT_BLIP_SIZE`."""
    return size if size in BLIP_SIZES else DEFAULT_BLIP_SIZE


def clip_repo(size) -> str:
    """Return the Hugging Face repository for a CLIP size tier."""
    return CLIP_SIZES[clamp_clip_size(size)]["repo"]


def blip_repo(size) -> str:
    """Return the Hugging Face repository for a BLIP size tier."""
    return BLIP_SIZES[clamp_blip_size(size)]["repo"]


def _open_rgb(path):
    """Return the image at ``path`` as an RGB PIL image."""
    from PIL import Image  # pylint: disable=import-outside-toplevel

    with Image.open(path) as image:
        return image.convert("RGB")


def _device_dtype():
    """Return ``(device, dtype)`` — fp16 on CUDA, fp32 on CPU."""
    import torch  # pylint: disable=import-outside-toplevel

    cuda = torch.cuda.is_available()
    return (
        "cuda" if cuda else "cpu",
        torch.float16 if cuda else torch.float32,
    )


def _free_cuda() -> None:
    """Empty the CUDA allocator cache when a GPU is present."""
    import torch  # pylint: disable=import-outside-toplevel

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def _siglip_session(spec):
    """Yield a ``score_one(path, caption)`` over a loaded SigLIP2 checkpoint.

    Reads SigLIP's *raw* image-text cosine, not the per-claim grounding score
    (the grounding calibration saturates on a whole caption). Owns the
    load/unload pair so a batch pays for the weights once.
    """
    siglip_grounding.load_model(spec["size"], spec["resolution"])
    try:

        def score_one(path, caption):
            cosines = siglip_grounding.caption_cosines(path, [caption])
            return _rescale(cosines[0] if cosines else 0.0, "siglip2")

        yield score_one
    finally:
        siglip_grounding.unload_model()


@contextmanager
def _clip_session(spec):
    """Yield a ``score_one`` over a loaded OpenAI CLIP checkpoint."""
    # pylint: disable=import-outside-toplevel
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device, dtype = _device_dtype()
    repo = spec["model_id"]
    model = CLIPModel.from_pretrained(repo, dtype=dtype).to(device).eval()
    processor = CLIPProcessor.from_pretrained(repo)
    try:

        def score_one(path, caption):
            inputs = processor(
                text=[caption],
                images=_open_rgb(path),
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            with torch.no_grad():
                image_embed = _pooled(
                    model.get_image_features(
                        pixel_values=inputs["pixel_values"].to(dtype)
                    )
                )
                text_embed = _pooled(
                    model.get_text_features(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                )
            return _rescale(_cosine(image_embed[0], text_embed[0]), "clip")

        yield score_one
    finally:
        del model, processor
        _free_cuda()


@contextmanager
def _blip_session(spec):
    """Yield a ``score_one`` over a loaded BLIP retrieval checkpoint.

    Reads BLIP's image-text *contrastive* cosine (``use_itm_head=False``),
    not its ITM classification head.
    """
    # pylint: disable=import-outside-toplevel
    import torch
    from transformers import AutoProcessor, BlipForImageTextRetrieval

    device, dtype = _device_dtype()
    repo = spec["model_id"]
    model = (
        BlipForImageTextRetrieval.from_pretrained(repo, dtype=dtype)
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(repo)
    try:

        def score_one(path, caption):
            inputs = processor(
                images=_open_rgb(path),
                text=[caption],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            with torch.no_grad():
                output = model(
                    pixel_values=inputs["pixel_values"].to(dtype),
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_itm_head=False,
                )
            # itm_score is the (1, 1) image-text cosine here.
            cosine = float(output.itm_score.float().cpu().reshape(-1)[0])
            return _rescale(cosine, "blip")

        yield score_one
    finally:
        del model, processor
        _free_cuda()


_SESSIONS = {
    "siglip2": _siglip_session,
    "clip": _clip_session,
    "blip": _blip_session,
}


def score_caption(path, caption, specs, progress=None) -> list[dict]:
    """Score one caption with every requested encoder, one at a time.

    Encoders load and free in sequence (never co-resident), so a tight GPU
    pays for one checkpoint at a time. A family that fails to load/score
    doesn't sink the run: its line returns ``score`` None + an ``error``.
    ``specs`` are ``{"kind", "label", "model_id", ...}`` dicts (from
    :func:`src.storage.caption_score_specs`); ``progress`` names the encoder
    in flight. Returns one ``{"kind", "model_id", "score", "error"}`` per spec,
    ``score`` a 0-100 float or None on error.
    """
    results = []
    for spec in specs:
        if progress is not None:
            progress(sub=f"{spec['label']}…")
        try:
            with _SESSIONS[spec["kind"]](spec) as score_one:
                score = score_one(path, caption)
            error = None
        except Exception as exc:  # pylint: disable=broad-except
            score, error = None, str(exc)
        results.append(
            {
                "kind": spec["kind"],
                "model_id": spec["model_id"],
                "score": score,
                "error": error,
            }
        )
    return results


def score_dataset(items, specs, progress=None) -> dict:
    """Score many captions with every encoder, loading each model once.

    The batch inversion of :func:`score_caption`: the outer loop is the
    *encoder* (loaded once, run over every item, freed), so N captions pay for
    each checkpoint once. A family that fails to load is skipped whole.
    ``items`` are ``{"revision_id", "path", "caption"}`` dicts; ``progress``
    total is items × encoders. Returns ``{revision_id: {model_kind: score}}``
    for the families that ran.
    """
    items = list(items)
    scores: dict = {item["revision_id"]: {} for item in items}
    total = len(items) * len(specs)
    done = 0
    for spec in specs:
        kind, label = spec["kind"], spec["label"]
        try:
            with _SESSIONS[kind](spec) as score_one:
                for index, item in enumerate(items, start=1):
                    scores[item["revision_id"]][kind] = score_one(
                        item["path"], item["caption"]
                    )
                    done += 1
                    if progress is not None:
                        progress(
                            done=done,
                            total=total,
                            sub=f"{label} {index}/{len(items)}",
                        )
        except Exception:  # pylint: disable=broad-except
            done += len(items)
            if progress is not None:
                progress(done=done, total=total, sub=f"{label} failed")
    return scores
