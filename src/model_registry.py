"""Map a weight filename to model metadata (type + HF config repo).

Handles both ``.safetensors`` (transformers) and ``.gguf`` (llama-cpp) vision
models. The first matching rule wins.
"""

import re

# Filenames that are text-encoders or text-only LLMs — never vision models.
# Note: qwen3 *without* "vl" is the text encoder; qwen3-vl is the vision model.
# Plain "mistral" is NOT skipped here: text-only Mistrals match no vision rule
# below (so they resolve to None anyway), while the vision Mistral Small 3.2 /
# Pixtral must be allowed through to its rule.
_SKIP_PATTERNS = re.compile(
    r"(?i)(^clip|t5xxl|flant5|umt5|gner.?t5|ltx|nsfw_wan|modelst5|"
    r"zimage|josiefied|qwen_3_\d+b|qwen3.*base|^mmproj|chat_template)",
    re.IGNORECASE,
)

_VISION_RULES = [
    # Gemma 4 — its own transformers architecture (Gemma4ForConditional
    # Generation) with native video support. Must come before the gemma3
    # rule. unsloth mirrors are ungated (google/* requires HF auth → 401).
    {
        "pattern": re.compile(r"(?i)(gemma.?4)"),
        "type": "gemma4",
        "hf_config": "unsloth/gemma-4-E4B-it",
    },
    # Gemma 3n — must come before the gemma3 rule.
    {
        "pattern": re.compile(r"(?i)(gemma.?3n)"),
        "type": "gemma3n",
        "hf_config": "unsloth/gemma-3n-E4B-it",
    },
    # Gemma 3 vision (12B).
    {
        "pattern": re.compile(r"(?i)(gemma.?3|gemma312b)"),
        "type": "gemma3",
        "hf_config": "unsloth/gemma-3-12b-it",
    },
    # Qwen3-VL — size detected from filename.
    {
        "pattern": re.compile(r"(?i)(qwen3.?vl|qwen3-vl)"),
        "type": "qwen3",
        "hf_config_by_size": {
            "4b": "Qwen/Qwen3-VL-4B-Instruct",
            "8b": "Qwen/Qwen3-VL-8B-Instruct",
        },
        "hf_config_default": "Qwen/Qwen3-VL-4B-Instruct",
    },
    # Qwen3.6 — a unified multimodal model (image / text / video). Not a
    # separate "-VL" repo, so it needs its own rule ahead of any generic qwen
    # match; size read from the filename. Loads via AutoModelForImageTextToText
    # (safetensors) or the generic mtmd handler (GGUF).
    {
        "pattern": re.compile(r"(?i)(qwen.?3\.?6)"),
        "type": "qwen3.6",
        "hf_config_by_size": {
            "27b": "Qwen/Qwen3.6-27B",
            "35b": "Qwen/Qwen3.6-35B-A3B",
        },
        "hf_config_default": "Qwen/Qwen3.6-27B",
    },
    # Mistral Small 3.2 (and Pixtral) — a Mistral3ForConditionalGeneration VLM.
    # GGUF loads via llama-cpp's GenericMTMDChatHandler, which reads Mistral's
    # chat template embedded in the GGUF; safetensors uses the transformers
    # class. The ungated unsloth mirror supplies config/processor (mistralai/*
    # is gated → 401); GGUF never fetches it.
    {
        "pattern": re.compile(r"(?i)(mistral.?(small.?)?3\.?2|pixtral)"),
        "type": "mistral3",
        "hf_config": "unsloth/Mistral-Small-3.2-24B-Instruct-2506",
    },
    # JoyCaption — a LLaVA-architecture VLM (SigLIP2 vision + Llama 3.1). The
    # loader's "llava" type handles it (LlavaForConditionalGeneration for
    # safetensors, Llava16ChatHandler for GGUF + mmproj). The hf_config feeds
    # config/processor for the safetensors path; GGUF loads from local weights
    # and never fetches it.
    {
        "pattern": re.compile(r"(?i)(joy.?caption)"),
        "type": "llava",
        "hf_config": "fancyfeast/llama-joycaption-beta-one-hf-llava",
    },
]

# Quant/format tokens dropped before matching a model against its mmproj.
_MMPROJ_NOISE = {
    "gguf",
    "q8",
    "q6",
    "q4",
    "q5",
    "k",
    "0",
    "m",
    "s",
    "fp16",
    "f16",
    "bf16",
    "mmproj",
}

# Model-size tokens (parameter count): "4b", "12b", "27b", "31b" and the
# Gemma-3n/4 "effective size" forms "e2b"/"e4b". A projector built for one
# size never loads into a model of another (different vision tower, so
# llama-cpp's mtmd loader crashes), so a size clash rejects the pairing even
# when the brand/family tokens all match.
_SIZE_RE = re.compile(r"^e?\d+b$")


def _size_tokens(filename: str) -> set:
    """Return the model-size tokens in a filename (e.g. {"e4b"}, {"31b"})."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", filename.lower())
        if _SIZE_RE.match(token)
    }


def detect_model(filename: str) -> dict | None:
    """Return model metadata for a weight filename, or None if not a VLM.

    Returns ``{"type", "format", "hf_config"}`` — family, ``safetensors``/
    ``gguf``, and the HF repo id for config + processor. None when the file is
    not a recognized vision model.
    """
    stem = filename.lower()

    if _SKIP_PATTERNS.search(stem):
        return None

    if filename.endswith(".safetensors"):
        fmt = "safetensors"
    elif filename.endswith(".gguf"):
        fmt = "gguf"
    else:
        return None

    for rule in _VISION_RULES:
        if rule["pattern"].search(stem):
            entry = {"type": rule["type"], "format": fmt}
            if "hf_config_by_size" in rule:
                size = next(
                    (s for s in rule["hf_config_by_size"] if s in stem), None
                )
                entry["hf_config"] = rule["hf_config_by_size"].get(
                    size, rule["hf_config_default"]
                )
            else:
                entry["hf_config"] = rule["hf_config"]
            return entry

    return None


def _family_of(filename: str) -> str | None:
    """Return the model family a filename declares, or None if undeclared.

    Runs the vision rules without the skip patterns, so it also classifies
    ``mmproj-*`` projector names (e.g. ``mmproj-...gemma-4...`` -> gemma3n).
    """
    stem = filename.lower()
    for rule in _VISION_RULES:
        if rule["pattern"].search(stem):
            return rule["type"]
    return None


def find_mmproj(
    model_filename: str,
    all_filenames: list[str],
    model_family: str | None = None,
) -> str | None:
    """Return the mmproj GGUF best matching a vision GGUF model, or None.

    Scores the distinctive tokens (quant/format noise removed) shared between
    the model name and each candidate projector. When ``model_family`` is
    given, projectors are filtered by family so a cross-architecture one (which
    crashes llama-cpp's mtmd loader) is never paired: family matches win; if
    none match but others declare a *different* family, no pairing is returned.
    Returns the best ``mmproj-*.gguf``, or None when nothing shares a token or
    matches the family.
    """
    mmprojs = [
        f
        for f in all_filenames
        if f.lower().startswith("mmproj") and f.endswith(".gguf")
    ]
    if not mmprojs:
        return None

    if model_family is not None:
        same_family = [m for m in mmprojs if _family_of(m) == model_family]
        if same_family:
            mmprojs = same_family
        elif any(_family_of(m) is not None for m in mmprojs):
            # Every projector declares a different family — refuse to pair.
            return None

    model_tokens = set(re.findall(r"[a-z0-9]+", model_filename.lower()))
    model_tokens -= _MMPROJ_NOISE
    model_sizes = _size_tokens(model_filename)

    best, best_score = None, 0
    for mm in mmprojs:
        # A projector for a different model size (e.g. a 31B mmproj against an
        # E4B model) shares the brand tokens but crashes the mtmd loader —
        # reject it when both names carry a size and they do not overlap.
        mm_sizes = _size_tokens(mm)
        if model_sizes and mm_sizes and model_sizes.isdisjoint(mm_sizes):
            continue
        mm_tokens = set(re.findall(r"[a-z0-9]+", mm.lower())) - _MMPROJ_NOISE
        score = len(model_tokens & mm_tokens)
        if score > best_score:
            best, best_score = mm, score

    return best if best_score > 0 else None
