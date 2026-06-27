"""Application settings: layered load, diff-based save and typed getters.

Effective = factory defaults < ``config/default/settings.json`` <
``config/user`` (see :mod:`src.config`). Only values differing from the
default layer are persisted, so a new default is picked up automatically.

Per-model-type generation prefs (prompt, think mode, temperature) live in the
per-type ``models/`` config files, not here; the getters below wrap that layer.
"""

import copy
import os
import re
from pathlib import Path

from src import config
from src.constants import (
    DEFAULT_CAPTION_EXTENSIONS,
    DEFAULT_GGUF_N_CTX,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINK_MODE,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_MAX_SECONDS,
    DEFAULT_VIDEO_PROMPT,
    DEFAULT_VIDEO_RESOLUTION,
    DEFAULT_OWLV2_MODEL,
    DEFAULT_OWLV2_QUERIES,
    DEFAULT_WATERMARK_TAGS,
    MAX_IMAGE_SIZE,
    OWLV2_MODEL_IDS,
    MODELS_DIR,
    STORAGE_DIR,
)
from src.index_steps import QUALITY_METRIC_IDS, STEP_KEYS
from src.media import MEDIA_EXTENSIONS
from src.quality import AVERAGE_METRIC_ID, QUALITY_METRICS
from src.caption_score import (
    DEFAULT_BLIP_SIZE,
    DEFAULT_CLIP_SIZE,
    blip_repo,
    clamp_blip_size,
    clamp_clip_size,
    clip_repo,
)
from src.siglip_grounding import (
    DEFAULT_RESOLUTION,
    DEFAULT_SIZE,
    clamp_resolution,
    clamp_size,
    repo_id,
)
from src.tagger import (
    DEFAULT_CHARACTER_THRESHOLD,
    DEFAULT_GENERAL_THRESHOLD,
    DEFAULT_REPO_ID,
    KNOWN_TAGGERS,
    LOCAL_SOURCE,
)

# Sane bounds for the caption image-size cap (longest side, px).
MIN_IMAGE_SIZE = 256
MAX_ALLOWED_IMAGE_SIZE = 4096

# Sane bounds for video frame sampling.
MIN_VIDEO_FPS = 0.1
MAX_VIDEO_FPS = 30.0
MIN_VIDEO_SECONDS = 0.5
MAX_VIDEO_SECONDS = 3600.0

# Sane bounds for the GGUF context window (tokens).
MIN_GGUF_N_CTX = 2048
MAX_GGUF_N_CTX = 131072

# SigLIP score (0-100) a claim/tag must reach to count as grounded. Applied
# on read, never baked into a stored score, so moving the slider re-reads.
DEFAULT_GROUNDING_THRESHOLD = 55.0

# Factory defaults: fallback when a key is absent from both files.
# ``model_dir`` points at shipped ``models/`` so the app runs with no setup;
# computed at runtime (machine-independent) and overridable. Empty path =
# "unset" (never scanned).
_FACTORY_DEFAULTS = {
    "model_dir": str(MODELS_DIR),
    # Where drag-drop uploads (internal library) are written. Defaults to
    # shipped ``storage/``; internal library row kept pointed at it (see
    # libraries_gallery.sync_internal_library_dir).
    "internal_media_dir": str(STORAGE_DIR),
    "device": "cuda",  # "cuda" | "cpu" — enforced, never a silent fallback
    "gguf_n_ctx": DEFAULT_GGUF_N_CTX,
    "caption_image_size": MAX_IMAGE_SIZE,
    "video_resolution": DEFAULT_VIDEO_RESOLUTION,
    "video_fps": DEFAULT_VIDEO_FPS,
    "video_max_seconds": DEFAULT_VIDEO_MAX_SECONDS,
    "video_prompt": DEFAULT_VIDEO_PROMPT,
    "caption_extensions": list(DEFAULT_CAPTION_EXTENSIONS),
    # Deploy target. Empty = unset; each dataset mirrors into
    # ``<deploy_dir>/<dataset_name>/``.
    "deploy_dir": "",
    # Folder of watermark YOLO ``.pt`` weights (see src.watermark_detect).
    # Defaults to app models/; the selected filename is a Lab pref, not here.
    "watermark_models_dir": str(MODELS_DIR),
    # Selected WD tagger: a src.tagger.KNOWN_TAGGERS repo id, or
    # src.tagger.LOCAL_SOURCE to use "autotag_local_dir".
    "autotag_source": DEFAULT_REPO_ID,
    "autotag_local_dir": "",
    # WD14 confidence floors. New tag names land in "Uncategorized"; a known
    # name reuses its tag. No configurable target category.
    "autotag_general": DEFAULT_GENERAL_THRESHOLD,
    "autotag_character": DEFAULT_CHARACTER_THRESHOLD,
    # Which index scans this machine may run. Per-install — turning a step off
    # greys out everything depending on it app-wide.
    "index_steps": {key: True for key in STEP_KEYS},
    # Which IQA metrics the quality step computes here. Subset of
    # src.quality.QUALITY_METRICS; heavy VLM scorers (Q-Align) opt-in.
    # Empty/invalid falls back to the factory trio (QUALITY_METRIC_IDS).
    "index_quality_metrics": list(QUALITY_METRIC_IDS),
    # SigLIP grounding (see src.siglip_grounding). size/resolution pick the
    # checkpoint (bigger = finer, both cost VRAM). Thresholds split: a claim
    # and a pre-prompted tag don't score on the same scale.
    "grounding_model_size": DEFAULT_SIZE,
    "grounding_resolution": DEFAULT_RESOLUTION,
    "grounding_threshold_caption": DEFAULT_GROUNDING_THRESHOLD,
    "grounding_threshold_tags": DEFAULT_GROUNDING_THRESHOLD,
    # VLM that splits a caption into claims (SigLIP only scores). Empty = use
    # loaded model; a scanned model name is auto-loaded when the slot is empty.
    "grounding_claim_model": "",
    # Beta. Off hides every SigLIP grounding surface app-wide; the
    # reference-free caption/tag scores are separate and stay. Default on.
    "grounding_enabled": True,
    # Caption editor auto-saves a dirty draft after a short typing pause.
    # Off restores the manual Save button. Default on.
    "autosave_enabled": True,
    # Reference-free caption score (see src.caption_score). SigLIP2 reuses the
    # grounding checkpoint; CLIP/BLIP pick base/large here (bigger = costlier).
    "caption_score_clip_size": DEFAULT_CLIP_SIZE,
    "caption_score_blip_size": DEFAULT_BLIP_SIZE,
    # Hugging Face access token used for every model download (OWLv2, FLUX.2,
    # SigLIP, WD14…). Empty = anonymous. Applied to the process environment
    # (see apply_hf_token) so transformers / diffusers / huggingface_hub all
    # pick it up. Needed for gated or rate-limited repos.
    "hf_token": "",
}

# The settings keys a save from the Settings tab may write. Any other user key
# (e.g. ``last_caption_type``, written on its own through
# set_last_caption_type) is preserved untouched across a settings save.
_SETTINGS_KEYS = tuple(_FACTORY_DEFAULTS)

# Caption extensions must be safe filename suffixes and must never collide with
# a media extension (writing "image.jpg" as a caption would clobber the image).
_MEDIA_SUFFIXES = {ext.lstrip(".").lower() for ext in MEDIA_EXTENSIONS}

# Cached effective settings. Getters run on hot paths (once per image in batch
# captioning), so re-reading JSON every call would be wasteful. Invalidated on
# every write through this module; an external JSON edit needs a restart.
# Mutable module global, not a constant, hence lowercase.
_settings_cache = None  # pylint: disable=invalid-name


def _invalidate_cache():
    """Drop the cached settings so the next load re-reads from disk."""
    global _settings_cache  # pylint: disable=global-statement
    _settings_cache = None


def load_settings() -> dict:
    """Return the effective settings (factory < default file < user file).

    The merged result is cached and a deep copy is returned, so callers may
    mutate the returned dict without corrupting the cache.
    """
    global _settings_cache  # pylint: disable=global-statement
    if _settings_cache is None:
        _settings_cache = config.deep_merge(
            _FACTORY_DEFAULTS, config.load_layered_settings()
        )
    return copy.deepcopy(_settings_cache)


def save_settings(settings: dict) -> None:
    """Persist settings, writing only values that differ from the defaults.

    Keys equal to the default layer are dropped from the user file; any other
    user key is preserved. ``settings`` is the full effective dict from the UI.
    """
    base = config.deep_merge(_FACTORY_DEFAULTS, config.read_default_settings())
    user = config.read_user_settings()
    for key in _SETTINGS_KEYS:
        if key in settings and base.get(key) != settings[key]:
            user[key] = settings[key]
        else:
            user.pop(key, None)
    config.save_user_settings(user)
    _invalidate_cache()


# --- Per-model-type generation settings ---
# These wrap the ``models/`` config layer (not the global settings cache), so
# they read/write the per-type config file directly.


def _model_settings(model_type: str) -> dict:
    """Return the merged ``settings`` block for a model type."""
    return config.load_model_config(model_type).get("settings", {})


def get_selected_prompt(model_type: str) -> str | None:
    """Return the last-selected prompt title for ``model_type``, or None."""
    return _model_settings(model_type).get("selected_prompt")


def set_selected_prompt(model_type: str, title: str) -> None:
    """Persist the user's selected prompt title for ``model_type``."""
    config.set_user_model_setting(model_type, "selected_prompt", title)


def get_model_think_mode(model_type: str) -> str:
    """Return the saved thinking mode for ``model_type`` (factory fallback)."""
    return _model_settings(model_type).get("think_mode", DEFAULT_THINK_MODE)


def set_model_think_mode(model_type: str, think_mode: str) -> None:
    """Persist the thinking mode for ``model_type``."""
    config.set_user_model_setting(model_type, "think_mode", think_mode)


def get_model_temperature(model_type: str) -> float:
    """Return the saved temperature for ``model_type`` (factory fallback)."""
    value = _model_settings(model_type).get("temperature", DEFAULT_TEMPERATURE)
    try:
        return float(value)
    except (TypeError, ValueError):
        return DEFAULT_TEMPERATURE


def set_model_temperature(model_type: str, temperature: float) -> None:
    """Persist the temperature for ``model_type``."""
    config.set_user_model_setting(
        model_type, "temperature", float(temperature)
    )


# --- Validation / clamping ---


def clamp_image_size(value, default: int = MAX_IMAGE_SIZE) -> int:
    """Parse a user-supplied image size and clamp it to a sane range."""
    try:
        size = int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default
    return max(MIN_IMAGE_SIZE, min(size, MAX_ALLOWED_IMAGE_SIZE))


def clamp_video_resolution(
    value, default: int = DEFAULT_VIDEO_RESOLUTION
) -> int:
    """Parse a user-supplied video frame size and clamp it to a sane range."""
    try:
        size = int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default
    return max(64, min(size, MAX_ALLOWED_IMAGE_SIZE))


def clamp_video_fps(value, default: float = DEFAULT_VIDEO_FPS) -> float:
    """Parse a user-supplied sampling FPS and clamp it to a sane range."""
    try:
        fps = float(str(value).strip())
    except (ValueError, TypeError):
        return default
    return max(MIN_VIDEO_FPS, min(fps, MAX_VIDEO_FPS))


def clamp_video_max_seconds(
    value, default: float = DEFAULT_VIDEO_MAX_SECONDS
) -> float:
    """Parse a user-supplied max duration (seconds) and clamp it to a range."""
    try:
        secs = float(str(value).strip())
    except (ValueError, TypeError):
        return default
    return max(MIN_VIDEO_SECONDS, min(secs, MAX_VIDEO_SECONDS))


def clamp_gguf_n_ctx(value, default: int = DEFAULT_GGUF_N_CTX) -> int:
    """Parse a user-supplied GGUF context size and clamp it to a sane range."""
    try:
        n_ctx = int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default
    return max(MIN_GGUF_N_CTX, min(n_ctx, MAX_GGUF_N_CTX))


def estimate_video_frames(fps, seconds) -> int:
    """Return how many frames sampling at ``fps`` over ``seconds`` yields.

    Count is ``fps * seconds + 1`` (t=0 included). Inputs pass the fps/seconds
    clamps first, so an out-of-range UI value still gives a sane number. Always
    at least 1.
    """
    rate = clamp_video_fps(fps)
    duration = clamp_video_max_seconds(seconds)
    return max(1, int(round(rate * duration)) + 1)


def clamp_caption_extensions(
    value, default: list[str] | None = None
) -> list[str]:
    """Normalize a caption-extension list.

    Lowercased, dot stripped, alphanumeric only, de-duplicated, media
    extensions dropped. Empty result falls back to ``default`` (factory
    extensions). Accepts a list/tuple or a comma/newline string.
    """
    default = (
        list(default)
        if default is not None
        else list(DEFAULT_CAPTION_EXTENSIONS)
    )

    if isinstance(value, str):
        items = re.split(r"[,\n]", value)
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return default

    seen, out = set(), []
    for item in items:
        ext = str(item).strip().lstrip(".").lower()
        if not ext or not re.fullmatch(r"[a-z0-9_]+", ext):
            continue
        if ext in _MEDIA_SUFFIXES or ext in seen:
            continue
        seen.add(ext)
        out.append(ext)

    return out or default


# --- Typed getters ---


def get_caption_extensions() -> list[str]:
    """Return the validated caption extensions the user can switch between."""
    return clamp_caption_extensions(
        load_settings().get("caption_extensions", DEFAULT_CAPTION_EXTENSIONS)
    )


def get_model_dir() -> Path | None:
    """Return the configured model directory, or None when unset.

    Empty = "not chosen yet"; callers treat None as an empty model list, not a
    scan of the cwd.
    """
    raw = str(load_settings().get("model_dir", "")).strip()
    return Path(raw) if raw else None


def get_internal_media_dir() -> Path | None:
    """Return the folder new internal (drag-drop) uploads are written to.

    Defaults to shipped ``storage/``. The internal library is kept pointed here
    (see :func:`src.libraries_gallery.sync_internal_library_dir`); already
    uploaded media keep their paths. None when cleared.
    """
    raw = str(load_settings().get("internal_media_dir", "")).strip()
    return Path(raw) if raw else None


def get_deploy_dir() -> Path | None:
    """Return the configured deploy directory, or None when unset.

    Datasets mirror into ``<deploy_dir>/<dataset_name>/``. Empty = unset.
    """
    raw = str(load_settings().get("deploy_dir", "")).strip()
    return Path(raw) if raw else None


def get_hf_token() -> str:
    """Return the configured Hugging Face access token, or "" when unset."""
    return str(load_settings().get("hf_token", "")).strip()


def apply_hf_token() -> None:
    """Export the configured HF token to the process environment (idempotent).

    transformers, diffusers and ``huggingface_hub`` all read ``HF_TOKEN`` /
    ``HUGGING_FACE_HUB_TOKEN`` at download time, so setting the env var once
    covers every model load without threading a token through each call. Called
    at startup and after a settings save; a cleared token removes the vars.
    """
    token = get_hf_token()
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if token:
            os.environ[name] = token
        else:
            os.environ.pop(name, None)


def get_watermark_models_dir() -> Path:
    """Return the folder scanned for watermark YOLO ``.pt`` weights.

    The configured directory, or the app ``models/`` folder when unset.
    """
    raw = str(load_settings().get("watermark_models_dir", "")).strip()
    return Path(raw) if raw else MODELS_DIR


def get_autotag_source() -> str:
    """Return the selected WD tagger source (a known repo id, or local)."""
    value = str(load_settings().get("autotag_source", DEFAULT_REPO_ID))
    if value != LOCAL_SOURCE and value not in KNOWN_TAGGERS:
        return DEFAULT_REPO_ID
    return value


def set_autotag_source(source: str) -> None:
    """Persist the selected WD tagger source."""
    user = config.read_user_settings()
    user["autotag_source"] = source
    config.save_user_settings(user)
    _invalidate_cache()


def get_autotag_local_dir() -> str:
    """Return the configured local tagger folder, or "" when unset."""
    return str(load_settings().get("autotag_local_dir", "")).strip()


def clamp_threshold(value, default: float) -> float:
    """Parse a user-supplied confidence threshold, clamped to ``[0, 1]``."""
    try:
        threshold = float(str(value).strip())
    except (ValueError, TypeError):
        return default
    return max(0.0, min(threshold, 1.0))


def get_autotag_general() -> float:
    """Return the general-tag confidence floor of the auto-tag step."""
    return clamp_threshold(
        load_settings().get("autotag_general"), DEFAULT_GENERAL_THRESHOLD
    )


def get_autotag_character() -> float:
    """Return the character-tag confidence floor of the auto-tag step."""
    return clamp_threshold(
        load_settings().get("autotag_character"), DEFAULT_CHARACTER_THRESHOLD
    )


def clamp_index_steps(value) -> dict:
    """Normalize the per-machine index-scan toggles.

    Unknown keys dropped, a missing one defaults to enabled (a step added
    later is on out of the box). Returns ``{step_key: bool}`` for every
    :data:`src.index_steps.STEP_KEYS`.
    """
    raw = value if isinstance(value, dict) else {}
    return {key: bool(raw.get(key, True)) for key in STEP_KEYS}


def get_index_steps() -> dict:
    """Return which index scans this machine may run."""
    return clamp_index_steps(load_settings().get("index_steps"))


def clamp_index_quality_metrics(value) -> list:
    """Return the IQA metrics the quality index step scores.

    Keeps only known :data:`~src.quality.QUALITY_METRICS` keys, in order,
    deduped. Empty/invalid falls back to the factory trio so the step is never
    silently a no-op (turn the whole step off in Settings instead).
    """
    picked = []
    if isinstance(value, (list, tuple)):
        for metric in value:
            if metric in QUALITY_METRICS and metric not in picked:
                picked.append(metric)
    return picked or list(QUALITY_METRIC_IDS)


def get_index_quality_metrics() -> list:
    """Return the IQA metrics scored by the quality index step."""
    return clamp_index_quality_metrics(
        load_settings().get("index_quality_metrics")
    )


def is_index_step_enabled(key: str) -> bool:
    """Return whether one index scan is enabled on this machine."""
    return get_index_steps().get(key, False)


def set_autotag_local_dir(path: str) -> None:
    """Persist the local tagger folder path ("" clears it)."""
    user = config.read_user_settings()
    path = (path or "").strip()
    if path:
        user["autotag_local_dir"] = path
    else:
        user.pop("autotag_local_dir", None)
    config.save_user_settings(user)
    _invalidate_cache()


def get_last_caption_type() -> str:
    """Return the remembered last-selected caption type, or "" if none.

    Written on its own (like ``current_dataset``), so a Settings-tab save
    leaves it untouched.
    """
    return str(load_settings().get("last_caption_type", "")).strip()


def set_last_caption_type(name: str) -> None:
    """Persist (or clear) the remembered gallery-wide caption type."""
    user = config.read_user_settings()
    name = (name or "").strip()
    if name:
        user["last_caption_type"] = name
    else:
        user.pop("last_caption_type", None)
    config.save_user_settings(user)
    _invalidate_cache()


def get_quality_display_metric() -> str:
    """Return the metric the grids display and sort quality by.

    View-only pref written by the grids' dropdown on its own (like
    ``last_caption_type``), so it survives a Settings-tab save. A
    :data:`~src.quality.QUALITY_METRICS` key or the
    :data:`~src.quality.AVERAGE_METRIC_ID` pseudo-metric; anything else falls
    back to the average. Defaults to the average.
    """
    value = str(
        load_settings().get("quality_display_metric", AVERAGE_METRIC_ID)
    )
    if value == AVERAGE_METRIC_ID or value in QUALITY_METRICS:
        return value
    return AVERAGE_METRIC_ID


def set_quality_display_metric(metric_id: str) -> None:
    """Persist the grids' displayed-quality metric selection.

    Written on its own (like :func:`set_last_caption_type`), not part of the
    Settings-tab save. Unknown value stored as-is, normalized on read.
    """
    user = config.read_user_settings()
    user["quality_display_metric"] = str(metric_id or AVERAGE_METRIC_ID)
    config.save_user_settings(user)
    _invalidate_cache()


def get_review_after_generation() -> bool:
    """Return whether "Generate all" runs an integrity pass afterwards."""
    return bool(load_settings().get("review_after_generation", False))


def set_review_after_generation(enabled: bool) -> None:
    """Persist the "Review after generation" batch toggle."""
    user = config.read_user_settings()
    user["review_after_generation"] = bool(enabled)
    config.save_user_settings(user)
    _invalidate_cache()


# --- SigLIP grounding ---


def get_grounding_model_size() -> str:
    """Return the configured SigLIP size tier (clamped to a known one)."""
    return clamp_size(load_settings().get("grounding_model_size"))


def get_grounding_resolution() -> int:
    """Return the configured SigLIP input resolution for that size.

    Clamped to a resolution the size actually ships — Giant has no 512
    checkpoint, so asking for one reads back 384.
    """
    settings = load_settings()
    return clamp_resolution(
        clamp_size(settings.get("grounding_model_size")),
        settings.get("grounding_resolution"),
    )


def get_grounding_model_id() -> str:
    """Return the Hugging Face repository of the configured checkpoint."""
    return repo_id(get_grounding_model_size(), get_grounding_resolution())


def clamp_grounding_threshold(value) -> float:
    """Parse a validation threshold, clamped to ``[0, 100]``."""
    try:
        threshold = float(str(value).strip())
    except (ValueError, TypeError):
        return DEFAULT_GROUNDING_THRESHOLD
    return max(0.0, min(threshold, 100.0))


def get_grounding_threshold_caption() -> float:
    """Return the score a caption claim must reach to count as grounded."""
    return clamp_grounding_threshold(
        load_settings().get("grounding_threshold_caption")
    )


def get_grounding_threshold_tags() -> float:
    """Return the score a tag must reach to count as present in the image."""
    return clamp_grounding_threshold(
        load_settings().get("grounding_threshold_tags")
    )


def get_grounding_claim_model() -> str:
    """Return the model that decomposes a caption into claims, or "".

    Empty = use the loaded VLM; otherwise a scanned model name (a
    :func:`src.scanner.scan_local_models` key) the grounding job auto-loads.
    """
    return str(load_settings().get("grounding_claim_model", "")).strip()


# --- Watermark Lab ---

# Persisted rail prefs, restored when the Lab reopens. One ``watermark``
# object so the whole rail round-trips in a single settings key.
DEFAULT_WATERMARK_PREFS = {
    # v2 detector choice, exclusive: OWLv2 zero-shot open-vocabulary (default)
    # or the fine-tuned YOLO ``.pt``. No fusion, no SigLIP pre-filter, no
    # caption scan — the detector alone locates the boxes.
    "detector": "owlv2",
    # OWLv2 checkpoint (see src.constants.OWLV2_MODELS): base is lighter, large
    # resolves smaller marks. Persisted so a switch sticks across sessions.
    "owlv2_model": DEFAULT_OWLV2_MODEL,
    # OWLv2 text queries (chips): each is sent verbatim as an open-vocabulary
    # query at scan time. Editable and persisted.
    "owlv2_queries": list(DEFAULT_OWLV2_QUERIES),
    # OWLv2 confidence floor (0-100): a box under it is never created. OWLv2's
    # open-vocabulary scores for watermarks are low in absolute terms (a real
    # corner logo lands ~25-45 on the large model), so the floor is low.
    "owlv2_confidence": 20,
    # Detection floor for the YOLO boxes. Fine-tuned watermark models score
    # real hits low (~0.2-0.5 except classic tiled stock marks), so a low
    # default surfaces most of them.
    "confidence_min": 25,
    # FLUX.2 klein edit engine. 4B is the pragmatic default: the goal is
    # chaining hundreds of erases fast, and the last picked model/precision
    # persists across sessions like every other rail preference.
    "model": "4b",
    "precision": "std",
    "kv": False,
    "source": "hf",
    "local_model_path": "",
    # Qwen3 text encoder (required). Source and version/file are chosen
    # independently of the model; empty version/path = the repo's bundled one.
    "text_encoder": {"source": "hf", "version": "", "path": ""},
    "prompt": "remove any watermark, logo or brand",
    "max_res": 1024,
    "res_side": "long",
    "dilate_px": 8,
    "tag_cleanup": True,
    "tags_to_remove": list(DEFAULT_WATERMARK_TAGS),
    "compare_mode": "slider",
    # Selected watermark YOLO ``.pt`` filename within the models dir (see
    # get_watermark_models_dir).
    "yolo_model": "",
}

_WM_INT_BOUNDS = {
    "owlv2_confidence": (5, 90),
    "confidence_min": (10, 95),
    "dilate_px": (0, 32),
    "max_res": (512, 1536),
}
_WM_MODELS = ("9b", "4b")
_WM_PRECISIONS = ("std", "fp8", "nvfp4")
_WM_SOURCES = ("hf", "local")
_WM_SIDES = ("long", "short")
_WM_COMPARE_MODES = ("slider", "hover", "zone")
_WM_DETECTORS = ("owlv2", "yolo")


def _clean_text_encoder(value) -> dict:
    """Return the text-encoder pref as ``{source, version, path}`` cleaned."""
    raw = value if isinstance(value, dict) else {}
    source = raw.get("source")
    return {
        "source": source if source in _WM_SOURCES else "hf",
        "version": str(raw.get("version") or "").strip(),
        "path": str(raw.get("path") or "").strip(),
    }


def _clean_query_list(value) -> list:
    """Return OWLv2 queries as a de-duplicated list of trimmed strings.

    A non-list (or one emptied to nothing) falls back to the factory queries
    so a scan always has something to look for.
    """
    if not isinstance(value, list):
        return list(DEFAULT_OWLV2_QUERIES)
    seen: list = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in seen:
            seen.append(text)
    return seen or list(DEFAULT_OWLV2_QUERIES)


def _clean_choice(prefs: dict, key: str, allowed) -> None:
    """Reset a preference to its default when it is not an allowed value."""
    if prefs.get(key) not in allowed:
        prefs[key] = DEFAULT_WATERMARK_PREFS[key]


def get_watermark_prefs() -> dict:
    """Return the Watermark Lab rail prefs, defaults merged and clamped.

    Missing keys fall back to :data:`DEFAULT_WATERMARK_PREFS`, so a new field
    reads sensibly against an older stored object.
    """
    stored = load_settings().get("watermark") or {}
    prefs = dict(DEFAULT_WATERMARK_PREFS)
    if isinstance(stored, dict):
        prefs.update(stored)
    for key, (low, high) in _WM_INT_BOUNDS.items():
        try:
            prefs[key] = max(low, min(high, int(prefs[key])))
        except (TypeError, ValueError):
            prefs[key] = DEFAULT_WATERMARK_PREFS[key]
    _clean_choice(prefs, "detector", _WM_DETECTORS)
    _clean_choice(prefs, "owlv2_model", OWLV2_MODEL_IDS)
    _clean_choice(prefs, "model", _WM_MODELS)
    _clean_choice(prefs, "precision", _WM_PRECISIONS)
    _clean_choice(prefs, "source", _WM_SOURCES)
    _clean_choice(prefs, "res_side", _WM_SIDES)
    _clean_choice(prefs, "compare_mode", _WM_COMPARE_MODES)
    if not isinstance(prefs.get("tags_to_remove"), list):
        prefs["tags_to_remove"] = list(DEFAULT_WATERMARK_TAGS)
    prefs["owlv2_queries"] = _clean_query_list(prefs.get("owlv2_queries"))
    prefs["text_encoder"] = _clean_text_encoder(prefs.get("text_encoder"))
    prefs["prompt"] = str(prefs.get("prompt") or "").strip() or (
        DEFAULT_WATERMARK_PREFS["prompt"]
    )
    prefs["local_model_path"] = str(
        prefs.get("local_model_path") or ""
    ).strip()
    prefs["tag_cleanup"] = bool(prefs.get("tag_cleanup"))
    prefs["kv"] = bool(prefs.get("kv"))
    prefs["yolo_model"] = str(prefs.get("yolo_model") or "").strip()
    return prefs


def set_watermark_prefs(partial: dict) -> dict:
    """Merge a partial update into the stored Lab prefs; return the result."""
    prefs = get_watermark_prefs()
    if isinstance(partial, dict):
        prefs.update(
            {key: value for key, value in partial.items() if key in prefs}
        )
    user = config.read_user_settings()
    user["watermark"] = prefs
    config.save_user_settings(user)
    _invalidate_cache()
    return get_watermark_prefs()


# --- Beta features ---


def get_grounding_enabled() -> bool:
    """Return whether the SigLIP grounding feature is enabled (default on)."""
    return bool(load_settings().get("grounding_enabled", True))


# --- Reference-free caption score ---


def get_caption_score_clip_size() -> str:
    """Return the configured CLIP size tier (clamped to a known one)."""
    return clamp_clip_size(load_settings().get("caption_score_clip_size"))


def get_caption_score_blip_size() -> str:
    """Return the configured BLIP size tier (clamped to a known one)."""
    return clamp_blip_size(load_settings().get("caption_score_blip_size"))


def get_caption_score_clip_id() -> str:
    """Return the Hugging Face repository of the configured CLIP model."""
    return clip_repo(get_caption_score_clip_size())


def get_caption_score_blip_id() -> str:
    """Return the Hugging Face repository of the configured BLIP model."""
    return blip_repo(get_caption_score_blip_size())


# Whether captioning/auto-tagging chain a grounding pass is *not* a setting:
# like "review after generation" it's a checkbox by the run button, carried in
# the job body (``ground_after``). Not wired into Libraries index steps —
# grounding scores a captioning run's output, not a disk scan.


def get_device() -> str:
    """Return the configured compute device ("cuda" or "cpu")."""
    return load_settings().get("device", "cuda")


def get_gguf_n_ctx() -> int:
    """Return the validated GGUF context window (tokens) for llama-cpp."""
    return clamp_gguf_n_ctx(
        load_settings().get("gguf_n_ctx", DEFAULT_GGUF_N_CTX)
    )


def get_caption_image_size() -> int:
    """Return the validated longest-side cap (px) applied before captioning."""
    return clamp_image_size(
        load_settings().get("caption_image_size", MAX_IMAGE_SIZE)
    )


def get_video_resolution() -> int:
    """Return the validated longest-side size (px) for video frames."""
    return clamp_video_resolution(
        load_settings().get("video_resolution", DEFAULT_VIDEO_RESOLUTION)
    )


def get_video_fps() -> float:
    """Return the validated frames-per-second sampled from video."""
    return clamp_video_fps(load_settings().get("video_fps", DEFAULT_VIDEO_FPS))


def get_video_max_seconds() -> float:
    """Return the validated max video duration (seconds) to sample."""
    return clamp_video_max_seconds(
        load_settings().get("video_max_seconds", DEFAULT_VIDEO_MAX_SECONDS)
    )


def get_video_prompt() -> str:
    """Return the editable video-caption prompt body (factory fallback)."""
    prompt = load_settings().get("video_prompt", DEFAULT_VIDEO_PROMPT)
    text = str(prompt).strip()
    return text or DEFAULT_VIDEO_PROMPT
