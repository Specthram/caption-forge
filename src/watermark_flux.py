"""FLUX.2 klein edit engine: generate a watermark-erasing patch.

Replaces the old LaMa/OpenCV inpainters. Given a source image and one
watermark box, :func:`edit` crops the dilated bounding box, downscales it so
the chosen side fits the resolution cap (hard limit 1536x1536, ratio kept),
runs the crop through FLUX.2 klein as an image-to-image edit with a text
instruction ("remove any watermark, logo or brand"), re-upscales the returned
image to the crop size and hands it back as the *patch* — which
:func:`src.wm_compose.write_patch` stores and composition pastes over the
original. The source file is never modified.

The model is heavy, so it follows the loader lifecycle of the other engines
(:mod:`src.siglip_grounding`): the pipeline is built lazily behind a lock and
:func:`unload_model` must run when a job ends. ``enable_model_cpu_offload``
keeps only the active sub-model (Qwen3 text encoder, then the transformer,
then the VAE) resident, so a 9B set fits a 32 GB card and a 4B set leaves ample
headroom.

Two sources feed a load:

* **Hugging Face** — the official ``black-forest-labs/FLUX.2-klein-*`` repos
  (see :data:`_HF_REPOS`), transformer + Qwen3 text encoder + VAE downloaded
  and cached automatically. Precision (std / fp8 / nvfp4) and the KV
  quick-edit variant each pick a different repo.
* **Local** — a user ``.safetensors`` / ``.gguf`` transformer plus a matching
  Qwen3 text encoder file; the VAE is read from an ``ae.safetensors`` sitting
  next to the transformer when present, else from the matching official repo.
"""

import logging
import threading

from src import wm_compose

logger = logging.getLogger(__name__)

# --- Model catalogue -------------------------------------------------------

MODEL_9B = "9b"
MODEL_4B = "4b"

PRECISION_STD = "std"
PRECISION_FP8 = "fp8"
PRECISION_NVFP4 = "nvfp4"

SOURCE_HF = "hf"
SOURCE_LOCAL = "local"

# The FLUX.2 hard cap: a crop is never sent larger than this on any side.
MAX_EDIT_SIDE = 1536

# klein is step-distilled to four inference steps; the KV variant is the same
# body tuned for quick edits. These are the knobs the pipe call needs and the
# single place to retune once the weights are exercised end to end.
_STEPS = 4
_GUIDANCE = 1.0

# (model, precision, kv) -> official repo id. KV exists only for 9B std/fp8;
# nvfp4 and 4B never carry it (handoff rule). Standard repos capitalise the
# size (``9B``/``4B``), the quantised ones lower-case it (``9b``/``4b``).
_HF_REPOS = {
    (MODEL_9B, PRECISION_STD, True): "black-forest-labs/FLUX.2-klein-9b-kv",
    (MODEL_9B, PRECISION_FP8, True): (
        "black-forest-labs/FLUX.2-klein-9b-kv-fp8"
    ),
    (MODEL_9B, PRECISION_STD, False): "black-forest-labs/FLUX.2-klein-9B",
    (MODEL_9B, PRECISION_FP8, False): "black-forest-labs/FLUX.2-klein-9b-fp8",
    (MODEL_9B, PRECISION_NVFP4, False): (
        "black-forest-labs/FLUX.2-klein-9b-nvfp4"
    ),
    (MODEL_4B, PRECISION_STD, False): "black-forest-labs/FLUX.2-klein-4B",
    (MODEL_4B, PRECISION_FP8, False): "black-forest-labs/FLUX.2-klein-4b-fp8",
    (MODEL_4B, PRECISION_NVFP4, False): (
        "black-forest-labs/FLUX.2-klein-4b-nvfp4"
    ),
}

# The Qwen3 text encoder each model size requires (bundled in the official
# repos; only overridden when the user picks a specific version/file).
_ENCODER_REPOS = {
    MODEL_9B: "Qwen/Qwen3-8B",
    MODEL_4B: "Qwen/Qwen3-4B",
}

_lock = threading.Lock()
_pipe = None  # pylint: disable=invalid-name
_pipe_key = None  # pylint: disable=invalid-name


# --- Preference resolution (pure, no torch) --------------------------------


def normalize_kv(model: str, precision: str, kv: bool) -> bool:
    """Return whether the KV variant is actually usable for this combo.

    KV exists only for 9B in std or fp8; anywhere else the flag is dropped so
    a stale preference never resolves to a repo that does not exist.
    """
    usable = (PRECISION_STD, PRECISION_FP8)
    return bool(kv) and model == MODEL_9B and precision in usable


def resolve_repo(model: str, precision: str, kv: bool):
    """Return the official HF repo id for a combo, or None when invalid."""
    key = (model, precision, normalize_kv(model, precision, kv))
    return _HF_REPOS.get(key)


def encoder_repo(model: str) -> str:
    """Return the Qwen3 text-encoder repo matching a model size."""
    return _ENCODER_REPOS.get(model, _ENCODER_REPOS[MODEL_9B])


def model_label(prefs) -> str:
    """Return a short human label of the engine, e.g. ``klein 9B KV fp8``."""
    model = prefs.get("model", MODEL_4B)
    precision = prefs.get("precision", PRECISION_STD)
    kv = normalize_kv(model, precision, prefs.get("kv", False))
    size = "9B" if model == MODEL_9B else "4B"
    parts = ["klein", size]
    if kv:
        parts.append("KV")
    if precision != PRECISION_STD:
        parts.append(precision)
    return " ".join(parts)


def local_transformer_path(prefs):
    """Return the chosen local transformer file, or None (HF or unset)."""
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    if prefs.get("source") != SOURCE_LOCAL:
        return None
    raw = str(prefs.get("local_model_path") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


def is_available(prefs) -> bool:
    """Return whether a load can proceed with the current preferences.

    HF loads download on demand, so they are always "available"; a local load
    needs its transformer file to exist and its combo to resolve.
    """
    if prefs.get("source") == SOURCE_LOCAL:
        return local_transformer_path(prefs) is not None
    return (
        resolve_repo(
            prefs.get("model", MODEL_4B),
            prefs.get("precision", PRECISION_STD),
            prefs.get("kv", False),
        )
        is not None
    )


def _pipe_identity(prefs) -> tuple:
    """Return a hashable key of everything that changes the loaded pipeline."""
    encoder = prefs.get("text_encoder") or {}
    return (
        prefs.get("source", SOURCE_HF),
        prefs.get("model", MODEL_4B),
        prefs.get("precision", PRECISION_STD),
        normalize_kv(
            prefs.get("model", MODEL_4B),
            prefs.get("precision", PRECISION_STD),
            prefs.get("kv", False),
        ),
        str(prefs.get("local_model_path") or ""),
        encoder.get("source", SOURCE_HF),
        str(encoder.get("version") or ""),
        str(encoder.get("path") or ""),
    )


# --- Model lifecycle -------------------------------------------------------


def _dtype():
    """Return the torch compute dtype (bf16; quantised repos self-describe)."""
    import torch  # pylint: disable=import-outside-toplevel

    # fp8/nvfp4 repos ship pre-quantised weights; compute still runs in bf16.
    return torch.bfloat16


def _device() -> str:
    """Return the inference device, GPU when a CUDA build sees one."""
    import torch  # pylint: disable=import-outside-toplevel

    return "cuda" if torch.cuda.is_available() else "cpu"


def _pipeline_class(kv: bool):
    """Return the diffusers pipeline class (KV quick-edit or plain klein)."""
    # pylint: disable=import-outside-toplevel
    from diffusers import Flux2KleinKVPipeline, Flux2KleinPipeline

    return Flux2KleinKVPipeline if kv else Flux2KleinPipeline


def _base_pipeline_repo(model: str) -> str:
    """Return the full-pipeline base repo for a size (std, no KV).

    Only the ``-9B`` / ``-4B`` repos ship a complete diffusers pipeline
    (``model_index.json`` + VAE + Qwen3 text encoder + scheduler); every
    quantised or KV repo is a single transformer ``.safetensors`` file. So a
    variant load takes its scaffold from the base repo and swaps the
    transformer in.
    """
    return _HF_REPOS[(model, PRECISION_STD, False)]


def _hf_variant_transformer(repo: str, base_repo: str, dtype):
    """Load the transformer from a single-file variant repo (fp8/nvfp4/kv).

    The quantised and KV repos hold one ``.safetensors`` transformer and no
    pipeline metadata, so ``from_pretrained`` cannot build a pipeline from
    them; the weight is fetched and loaded through ``from_single_file``, its
    architecture config read from the base repo's ``transformer/`` folder.
    """
    # pylint: disable=import-outside-toplevel
    from diffusers import Flux2Transformer2DModel
    from huggingface_hub import HfApi, hf_hub_download

    weights = [
        name
        for name in HfApi().list_repo_files(repo)
        if name.endswith(".safetensors")
    ]
    if not weights:
        raise ValueError(f"{repo}: no .safetensors transformer weight found")
    local = hf_hub_download(repo, weights[0])
    return Flux2Transformer2DModel.from_single_file(
        local, config=base_repo, subfolder="transformer", torch_dtype=dtype
    )


def _load_hf(prefs):
    """Build a klein pipeline from official HF repos (auto-downloaded).

    A standard, non-KV combo reads a complete pipeline straight from its base
    repo. Every other combo (fp8/nvfp4, or the KV variant) is published as a
    single-file transformer repo, so its weights are swapped into the base
    pipeline's scaffold.
    """
    model = prefs.get("model", MODEL_4B)
    precision = prefs.get("precision", PRECISION_STD)
    kv = normalize_kv(model, precision, prefs.get("kv", False))
    repo = resolve_repo(model, precision, kv)
    if repo is None:
        raise ValueError(f"no FLUX.2 klein repo for {model}/{precision}")
    pipe_cls = _pipeline_class(kv)
    dtype = _dtype()
    base_repo = _base_pipeline_repo(model)
    if precision == PRECISION_STD and not kv:
        pipe = pipe_cls.from_pretrained(repo, torch_dtype=dtype)
    else:
        transformer = _hf_variant_transformer(repo, base_repo, dtype)
        pipe = pipe_cls.from_pretrained(
            base_repo, transformer=transformer, torch_dtype=dtype
        )
    _apply_encoder_override(pipe, prefs, model)
    return pipe


def _encoder_load_kwargs(version: str, default_dtype) -> dict:
    """Return the ``from_pretrained`` kwargs for an HF encoder version.

    ``fp16`` loads the base Qwen3 repo in float16; ``fp8`` loads it 8-bit
    through bitsandbytes (the readily available quantisation — a true fp8-mixed
    checkpoint would ship as its own repo); anything else keeps the pipeline's
    dtype. GGUF encoders are chosen as a local file, not here. This is the one
    spot to retune once the encoder variants are exercised end to end.
    """
    import torch  # pylint: disable=import-outside-toplevel

    if version == "fp16":
        return {"torch_dtype": torch.float16}
    if version == "fp8":
        # pylint: disable=import-outside-toplevel
        from transformers import BitsAndBytesConfig

        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    return {"torch_dtype": default_dtype}


def _apply_encoder_override(pipe, prefs, model) -> None:
    """Swap the bundled Qwen3 encoder for a user-chosen version/file.

    A no-op unless the text-encoder preference points somewhere other than the
    repo default; the pipeline keeps its bundled encoder otherwise. An HF
    override reloads the matching Qwen3 repo at the chosen precision; a local
    override loads the user's file at the pipeline dtype.
    """
    encoder = prefs.get("text_encoder") or {}
    source = encoder.get("source", SOURCE_HF)
    path = str(encoder.get("path") or "").strip()
    version = str(encoder.get("version") or "").strip()
    if source == SOURCE_HF and not version:
        return
    if source == SOURCE_LOCAL and not path:
        return
    # pylint: disable=import-outside-toplevel
    from transformers import AutoModelForCausalLM

    if source == SOURCE_LOCAL:
        ref = path
        kwargs = {"torch_dtype": pipe.text_encoder.dtype}
    else:
        ref = encoder_repo(model)
        kwargs = _encoder_load_kwargs(version, pipe.text_encoder.dtype)
    pipe.text_encoder = AutoModelForCausalLM.from_pretrained(ref, **kwargs)


def _load_local(prefs):
    """Build a klein pipeline from a local transformer (+ encoder, VAE)."""
    path = local_transformer_path(prefs)
    if path is None:
        raise FileNotFoundError("no local FLUX.2 transformer selected")
    model = prefs.get("model", MODEL_4B)
    precision = prefs.get("precision", PRECISION_STD)
    kv = normalize_kv(model, precision, prefs.get("kv", False))
    # pylint: disable=import-outside-toplevel
    from diffusers import Flux2Transformer2DModel

    dtype = _dtype()
    transformer = Flux2Transformer2DModel.from_single_file(
        str(path), **_local_quant(path, dtype)
    )
    # The pipeline scaffold (VAE, text encoder, scheduler) always comes from
    # the full base repo — the quantised/KV repos are single-file only.
    base_repo = _base_pipeline_repo(model)
    kwargs = {"transformer": transformer, "torch_dtype": dtype}
    vae = _local_vae(path, dtype)
    if vae is not None:
        kwargs["vae"] = vae
    pipe = _pipeline_class(kv).from_pretrained(base_repo, **kwargs)
    _apply_encoder_override(pipe, prefs, model)
    return pipe


def _local_quant(path, dtype) -> dict:
    """Return the from_single_file kwargs for a ``.gguf`` file (else empty)."""
    if path.suffix.lower() != ".gguf":
        return {"torch_dtype": dtype}
    # pylint: disable=import-outside-toplevel
    from diffusers import GGUFQuantizationConfig

    return {
        "quantization_config": GGUFQuantizationConfig(compute_dtype=dtype),
        "torch_dtype": dtype,
    }


def _local_vae(transformer_path, dtype):
    """Return a VAE read from ``ae.safetensors`` beside the model, or None."""
    ae = transformer_path.with_name("ae.safetensors")
    if not ae.is_file():
        return None
    # pylint: disable=import-outside-toplevel
    from diffusers import AutoencoderKLFlux2

    return AutoencoderKLFlux2.from_single_file(str(ae), torch_dtype=dtype)


def load_model(prefs) -> None:
    """Build the klein pipeline for ``prefs`` (idempotent, lazy, locked).

    Rebuilds when the preferences that shape the pipeline change (a different
    model, precision, KV flag or source) so switching the rail takes effect on
    the next scan. Raises when the combo is invalid or a local file is missing.
    """
    global _pipe, _pipe_key  # pylint: disable=global-statement
    key = _pipe_identity(prefs)
    with _lock:
        if _pipe is not None and _pipe_key == key:
            return
        _pipe = None
        _pipe_key = None
        if prefs.get("source") == SOURCE_LOCAL:
            pipe = _load_local(prefs)
        else:
            pipe = _load_hf(prefs)
        pipe.enable_model_cpu_offload()
        _pipe = pipe
        _pipe_key = key


def unload_model() -> None:
    """Drop the pipeline so it never starves the next model of VRAM."""
    global _pipe, _pipe_key  # pylint: disable=global-statement
    with _lock:
        _pipe = None
        _pipe_key = None
    try:
        import torch  # pylint: disable=import-outside-toplevel

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --- Edit ------------------------------------------------------------------


# FLUX.2's image processor rejects any side below this; a thin watermark strip
# (e.g. a wide 336x48 logo) would otherwise fail the edit.
_MIN_EDIT_SIDE = 64


def _target_size(crop_w: int, crop_h: int, max_res: int, res_side: str):
    """Return the size the crop is sent at: chosen side <= cap, ratio kept.

    Downscales so the chosen side fits the cap, then guarantees *both* sides
    reach :data:`_MIN_EDIT_SIDE` — a thin strip is scaled up (ratio kept) until
    its short side clears the floor, and only a truly extreme aspect ratio is
    stretched (the long side clamped to the cap). Each side is snapped to a
    multiple of 16 (the FLUX.2 VAE stride) and clamped to ``[64, cap]``.
    """
    cap = max(_MIN_EDIT_SIDE, min(int(max_res), MAX_EDIT_SIDE))
    chosen = max(crop_w, crop_h) if res_side == "long" else min(crop_w, crop_h)
    scale = min(1.0, cap / float(max(1, chosen)))
    width, height = crop_w * scale, crop_h * scale
    shortest = min(width, height)
    if shortest < _MIN_EDIT_SIDE:
        boost = _MIN_EDIT_SIDE / shortest
        width, height = width * boost, height * boost

    def _snap(value: float) -> int:
        return max(_MIN_EDIT_SIDE, min(cap, int(round(value)) // 16 * 16))

    return _snap(width), _snap(height)


def _run_pipe(pipe, image, prompt: str, seed):
    """Run one FLUX.2 klein image edit and return the PIL result.

    The single spot the diffusers call lives, so retuning steps/guidance or the
    editing argument name once weights are exercised touches only here.
    """
    import torch  # pylint: disable=import-outside-toplevel

    generator = None
    if seed is not None:
        generator = torch.Generator(_device()).manual_seed(int(seed))
    result = pipe(
        prompt=prompt or "remove any watermark, logo or brand",
        image=image,
        num_inference_steps=_STEPS,
        guidance_scale=_GUIDANCE,
        generator=generator,
    )
    return result.images[0]


def edit(
    source_path,
    box,
    dilate_px: int = 8,
    prompt: str = "remove any watermark, logo or brand",
    seed=None,
    max_res: int = 1024,
    res_side: str = "long",
):
    """Return the FLUX-edited patch (PIL image) for one watermark box.

    The pipeline must already be loaded (:func:`load_model`); the runner owns
    the load/unload. The crop *sent* to FLUX is the box grown by ``dilate_px``
    + :data:`~src.wm_compose.PATCH_COVER_PX`, giving the edit clean background
    to reason from. The returned patch is the box grown by only
    ``PATCH_COVER_PX`` (composition pastes exactly that): a small margin that
    hides anti-aliased letter edges just outside a tight box, while the wider
    ``dilate_px`` of pure context beyond it keeps the fill coherent (no
    half-edited borders bleeding in).
    """
    with _lock:
        pipe = _pipe
    if pipe is None:
        raise RuntimeError("FLUX.2 klein pipeline is not loaded")
    image = wm_compose.oriented_open(source_path)
    width, height = image.size
    dilate_px = max(0, int(dilate_px))
    cover_px = wm_compose.PATCH_COVER_PX
    left, top, right, bottom = wm_compose.dilated_pixel_rect(
        box, width, height, dilate_px + cover_px
    )
    crop = image.crop((left, top, right, bottom))
    crop_w, crop_h = crop.size
    target = _target_size(crop_w, crop_h, max_res, res_side)
    # pylint: disable=import-outside-toplevel
    from PIL import Image

    sent = (
        crop
        if target == (crop_w, crop_h)
        else crop.resize(target, Image.Resampling.LANCZOS)
    )
    edited = _run_pipe(pipe, sent, prompt, seed)
    if edited.size != (crop_w, crop_h):
        edited = edited.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
    # Keep the box grown by PATCH_COVER_PX (the pasted region); the extra
    # dilate_px around it was context only.
    inner = wm_compose.dilated_pixel_rect(box, width, height, cover_px)
    patch = edited.crop(
        (inner[0] - left, inner[1] - top, inner[2] - left, inner[3] - top)
    )
    return patch.convert("RGB")
