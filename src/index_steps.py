"""The "Index" pipeline: the scans a library index run chains.

"Index" is not a single scan: it is a fixed chain of five steps, each
backed by an engine already in ``src`` and each recorded in its own table.
This module is the single source of truth for *what* the steps are (key,
label, the models behind them, their cost); *whether* a step runs on this
machine is a setting (see :func:`src.settings.get_index_steps`), and *how*
a step runs is the job body (see :mod:`server.runners.library`).

Steps, in chain order
---------------------
``thumbs``
    Cached previews (:mod:`src.thumbnails`). The chain always probes
    dimensions, perceptual hashes and the model-free statistics of
    :mod:`src.image_stats` alongside — near-duplicates, the resolution
    floor, the blur/noise filters and the quality report depend on them —
    but the thumbnail JPEGs themselves are only written when the step is
    enabled.
``quality``
    The IQA scores of :data:`QUALITY_METRIC_IDS` (:mod:`src.quality`).
``embed``
    DINOv2 vectors (:mod:`src.embeddings`).
``depth``
    Depth-Anything V2 composition signatures (:mod:`src.depth_embeddings`),
    fused into the auto-builder's Proximity graph to catch re-skins — same
    composition, different style — that DINOv2 rates as far apart.
``siglip``
    SigLIP 2 joint-space image vectors (:mod:`src.siglip_grounding`), so a
    typed query ranks the library by meaning — the semantic search of the
    dataset composer.
``wd14``
    Booru auto-tags (:mod:`src.tagger`), written into the tag category
    configured in Settings.

Nothing here imports the settings or the database, so both may import it.
"""

# The IQA metrics the quality step scores, in the order the chain runs
# them. Mirrors :data:`src.dataset_quality.DEFAULT_SCORERS` (the quality
# report's default chips) — a media counts as scored once it carries a
# ``media_quality`` row for every one of them.
QUALITY_METRIC_IDS = ("musiq", "topiq_nr", "laion_aes")

THUMBS = "thumbs"
QUALITY = "quality"
EMBED = "embed"
DEPTH = "depth"
SIGLIP = "siglip"
WD14 = "wd14"

# Ordered catalogue: the chain runs the steps in this order (cheap CPU
# work first, then the GPU models one at a time), and the UI lists them in
# this order too.
STEPS = (
    {
        "key": THUMBS,
        "label": "Thumbnails",
        "short": "thumb",
        "models": "PIL / ffmpeg · cached previews",
        "cost": "CPU · ~180 KB / media",
        "images_only": False,
        "description": (
            "Small previews so grids never load originals. Turn off to "
            "save disk space — grids then read the full files (slower)."
        ),
    },
    {
        "key": QUALITY,
        "label": "Quality scores",
        "short": "score",
        "models": "MUSIQ · TOPIQ-NR · LAION-Aes",
        "cost": "GPU · ~0.1 s / img",
        "images_only": False,
        "description": (
            "IQA models behind the quality badges, sorting and dataset "
            "reports."
        ),
    },
    {
        "key": EMBED,
        "label": "Embeddings — similarity",
        "short": "embed",
        "models": "DINOv2 ViT-S",
        "cost": "GPU · ~0.4 GB VRAM",
        "images_only": True,
        "description": (
            "Visual signatures for near-duplicates, the diversity map and "
            "Auto-build."
        ),
    },
    {
        "key": DEPTH,
        "label": "Embeddings — composition",
        "short": "depth",
        "models": "Depth-Anything V2 Small",
        "cost": "GPU · ~0.5 GB VRAM",
        "images_only": True,
        "description": (
            "Style-invariant depth signatures — fused into Proximity to "
            "catch re-skins (same composition, different style) that DINOv2 "
            "rates as far apart. Off = fusion falls back to DINOv2 alone."
        ),
    },
    {
        "key": SIGLIP,
        "label": "Embeddings — semantic search",
        "short": "siglip",
        "models": "SigLIP 2 (the grounding checkpoint)",
        "cost": "GPU · ~1.6 GB VRAM",
        "images_only": True,
        "description": (
            "Text-to-image vectors so the dataset composer can find "
            "candidates by typing what they show."
        ),
    },
    {
        "key": WD14,
        "label": "Auto-tags",
        "short": "tags",
        "models": "WD14 EVA02-Large v3",
        "cost": "GPU · ~1.2 GB VRAM",
        "images_only": True,
        "description": "Booru tags for search and tag-guided captions.",
    },
)

STEP_KEYS = tuple(step["key"] for step in STEPS)

# Which index step each Quality-report scorer chip needs to have run.
SCORER_STEP = {
    "dinov2": EMBED,
}


def scorer_step(scorer_id: str) -> str:
    """Return the index step a report scorer depends on.

    Every IQA metric comes from the quality step; the ``dinov2`` chip needs the
    embeddings step. Returns the :data:`STEP_KEYS` key it depends on.
    """
    return SCORER_STEP.get(scorer_id, QUALITY)


def normalize_steps(requested, enabled: dict) -> list:
    """Return the steps to run: the requested ones, minus the disabled.

    ``requested`` is the caller's steps (None = the whole chain); ``enabled``
    the per-machine toggles (``{step_key: bool}``). Returns the keys in chain
    order, never a step disabled on this machine.
    """
    asked = STEP_KEYS if requested is None else tuple(requested)
    return [key for key in STEP_KEYS if key in asked and enabled.get(key)]
