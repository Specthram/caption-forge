"""Project-wide constants: filesystem paths and tuning defaults."""

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# Layered config. ``default/`` ships read-only (committed); ``user/`` holds
# runtime overrides (git-ignored), merged on top key by key.
CONFIG_DIR = ROOT_DIR / "config"
DEFAULT_CONFIG_DIR = CONFIG_DIR / "default"
USER_CONFIG_DIR = CONFIG_DIR / "user"
# Per-model-type config (prompts + generation settings). Renamed from
# ``prompts/`` once it outgrew prompts.
DEFAULT_MODELS_DIR = DEFAULT_CONFIG_DIR / "models"
USER_MODELS_DIR = USER_CONFIG_DIR / "models"
SETTINGS_FILENAME = "settings.json"

# Shipped model dir (with .gitkeep) so the app runs out of the box: drop
# weights in ``models/``. Overridable in Settings. Computed at runtime so the
# shipped settings file stays machine-independent.
MODELS_DIR = ROOT_DIR / "models"

# Exit code the "Restart server" button uses to signal an intentional
# restart. run.bat relaunches only on this exact code; any other exit (crash,
# normal close) is left alone. Picked to avoid Python's codes (0/1) and Ctrl+C.
RESTART_EXIT_CODE = 3

# SQLite datasets database. Folder ships with a .gitkeep; the file is created
# empty on first launch and git-ignored.
DATABASE_DIR = ROOT_DIR / "database"
DB_FILENAME = "cforge.db"
DB_PATH = DATABASE_DIR / DB_FILENAME
# Timestamped backups (git-ignored with the rest of ``database/``).
DB_BACKUP_DIR = DATABASE_DIR / "backups"

# Internal upload library: the one folder the app writes to. Drag-drop uploads
# land here under their original name, then referenced in place like any other
# media. Seeded as the ``internal`` library row. Ships with .gitkeep,
# git-ignored.
STORAGE_DIR = ROOT_DIR / "storage"

# HF config cache (no weights, ~10-50MB per model).
CONFIG_CACHE_DIR = ROOT_DIR / ".cache" / "model_configs"

# Thumbnail cache (gitignored, lazily rebuilt — safe to delete). A resized
# preview of each unique media (keyed by content sha256) is cached during a
# scan/upload (see :mod:`src.thumbnails`) so grids show a small image, not the
# multi-megapixel original.
CACHE_DIR = ROOT_DIR / "cache"
THUMBNAILS_DIR = CACHE_DIR / "thumbnails"
# Rendered crop pixels (gitignored, lazily rebuilt). A crop is a rectangle of
# another media, not its own file (see :mod:`src.crops`); its PNG is
# materialized here the first time something needs the pixels.
CROPS_DIR = CACHE_DIR / "crops"
# Watermark inpaint patches (gitignored, lazily rebuilt). One small PNG per
# watermark zone (see src.watermark), named ``<zone_id>.png`` — a zone owns one
# patch at a time (regeneration overwrites).
PATCHES_DIR = CACHE_DIR / "patches"
# Composited images: an original with all its patched watermark zones pasted
# over it, keyed by a synthetic hash of source + patch set (see
# src.wm_compose). Served in place of the original everywhere, so the source
# file is never modified.
COMPOSED_DIR = CACHE_DIR / "composed"
# User HF config/processor files, one sub-folder per repo id. Drop a repo's
# config + tokenizer/processor here (no weights) to skip the runtime download
# transformers otherwise triggers for a safetensors model.
HF_CONFIG_DIR = ROOT_DIR / "hf_config"

# Caption file extensions the user can switch between (no leading dot). The
# live list lives in settings.json; this is the factory default.
DEFAULT_CAPTION_EXTENSIONS = ["txt", "caption", "booru"]

# Tag categories seeded into a fresh DB (only when none exist, so a deleted
# default is never resurrected). Each is ``(name, hex color)`` for the chip.
DEFAULT_TAG_CATEGORIES = [
    ("character", "#e57373"),
    ("style", "#64b5f6"),
    ("artist", "#81c784"),
    ("content", "#ffb74d"),
]

# Per-model-type generation fallbacks when a type's config has no saved value.
# ``think_mode`` is one of "auto"/"off"/"show".
DEFAULT_THINK_MODE = "auto"
DEFAULT_TEMPERATURE = 0.7

# Watermark Lab (see src.watermark): tags stripped from a media once its every
# watermark zone is patched — the editable, persisted factory list.
DEFAULT_WATERMARK_TAGS = [
    "watermark",
    "logo",
    "artist logo",
    "artist name",
    "patreon logo",
]

# Watermark Lab v2 — the OWLv2 zero-shot detector's default text queries. Each
# is sent verbatim as an open-vocabulary query at scan time; the list is
# editable (chips) and persisted in the rail prefs.
DEFAULT_OWLV2_QUERIES = [
    "watermark",
    "stock photo watermark",
    "watermark in the corner",
    "website watermark",
    "semi-transparent watermark",
    "photographer signature",
]

# Selectable OWLv2 checkpoints (id, rail label). Base is lighter and faster;
# large resolves finer detail (better on small marks). Both are the "ensemble"
# self-trained variants. Kept here so settings and the detector share the list
# without a circular import.
OWLV2_MODELS = [
    (
        "google/owlv2-large-patch14-ensemble",
        "OWLv2 large patch14 · ensemble (~1.7 GB · best detection)",
    ),
    (
        "google/owlv2-base-patch16-ensemble",
        "OWLv2 base patch16 · ensemble (~0.6 GB · lighter, weaker)",
    ),
]
OWLV2_MODEL_IDS = tuple(model_id for model_id, _label in OWLV2_MODELS)
# Large is the default: on real corner watermarks it scores ~40 where base
# scores ~28 (below a usable threshold), so base is only the lighter option.
DEFAULT_OWLV2_MODEL = OWLV2_MODELS[0][0]

# Baked-in ("flattened") originals, one backup per media kept so a flatten is
# reversible: <media_id>/<original filename>. Under the app cache so a user
# library folder is never littered (see src.watermark.flatten_media).
WATERMARK_BACKUPS_DIR = CACHE_DIR / "wm_backups"

ITEMS_PER_PAGE = 40
ITEMS_PER_ROW = 4
MAX_IMAGE_SIZE = 1024
GEMMA_MAX_IMAGE_SIZE = 896

# Multi-frame video captioning. Frames sampled at DEFAULT_VIDEO_FPS over the
# first DEFAULT_VIDEO_MAX_SECONDS, so count = fps * seconds + 1 (t=0 included).
# Longer clips trimmed to the first DEFAULT_VIDEO_MAX_SECONDS. Sampled frames
# are fed one of three ways, by family/format:
#   * Qwen3-VL / Qwen2.5-VL / Gemma 4 (transformers): native video — the
#     processor inserts trained timestamp tokens (``<x seconds>``).
#   * Gemma 3 / 3n (transformers) and every GGUF vision model: chronological
#     sequence of stills (no native video token).
#   * Single-image models (Llava): caption each frame independently.
VIDEO_RESOLUTIONS = [256, 320, 480, 512]
DEFAULT_VIDEO_RESOLUTION = 256
DEFAULT_VIDEO_FPS = 2
DEFAULT_VIDEO_MAX_SECONDS = 5

# GGUF (llama-cpp) context window. 4096 is plenty for images, but video sends
# ~256 tokens/frame, so ~15 frames fill 4096; 8192 leaves room for a couple
# dozen. Raising it costs VRAM (KV cache grows with n_ctx); the model's trained
# context is far larger (often 128k+). Ignored by the transformers backend,
# which is bounded by the model itself.
DEFAULT_GGUF_N_CTX = 8192

# Default editable video prompt: the model-agnostic body of what a video
# caption should cover. The path-specific preamble (native video vs. stills)
# and the auto-computed timing hint are prepended at generation time, so this
# text stays free of format placeholders and is edited in Settings.
DEFAULT_VIDEO_PROMPT = (
    "In one flowing description, cover:\n"
    "- Shot & camera: framing / shot type (close-up, medium, "
    "wide...), camera angle, and any camera movement (pan, tilt, "
    "zoom, tracking, handheld, or static).\n"
    "- Style: visual style and medium (photo, 3D render, anime, "
    "illustration...), lighting, color palette and overall mood.\n"
    "- Setting: the location, background and notable environment "
    "details.\n"
    "- Subjects: the main characters or objects, their appearance, "
    "clothing and placement.\n"
    "- Motion & speed: what happens over time — the actions and "
    "gestures — AND how fast it happens. Use the elapsed time to "
    "judge the pace: say whether the motion is slow, steady, fast "
    "or sudden, and whether it is smooth or abrupt.\n\n"
    "Describe only what is visible. Do not mention frames, images, "
    "stills, panels, timestamps or that the video was sampled; "
    "write it as a single continuous video description."
)
