"""Scan the configured model directory for local vision models.

Detects ``.safetensors`` and ``.gguf`` weights. There is no HuggingFace model
list or weight download — only local files are considered.
"""

from safetensors import safe_open

from src.settings import get_model_dir
from src.model_registry import detect_model, find_mmproj


def _is_hf_vlm_safetensors(path) -> bool:
    """Return whether ``path`` is a genuine HF vision-language checkpoint.

    Reads just the safetensors header (fast) and keeps a file only if it has
    the nested layout (a ``language_model`` key) AND a vision encoder
    (``vision``/``visual``). This rejects ComfyUI text-encoder repacks (flat
    ``model.layers``, embedded tokenizer, fp8 scale tensors) and text-only
    files — none of which can caption.
    """
    try:
        with safe_open(str(path), framework="pt") as handle:
            keys = handle.keys()
    except Exception:  # pylint: disable=broad-exception-caught
        # Any unreadable/corrupt header → treat as not a usable model.
        return False

    has_language_model = any("language_model" in k for k in keys)
    has_vision = any(("vision" in k or "visual" in k) for k in keys)
    return has_language_model and has_vision


def scan_local_models() -> dict:
    """Return the vision-capable models found in the model directory.

    Maps display name (the file name) to a config dict::

            {
                "local_path": Path,
                "hf_config": "org/repo",
                "type": "gemma3n" | "gemma3" | "qwen3",
                "format": "safetensors" | "gguf",
                "mmproj_path": Path | None,  # gguf vision projector
                "source": "local",
            }
    """
    model_dir = get_model_dir()
    if model_dir is None or not model_dir.exists():
        return {}

    all_names = [p.name for p in model_dir.iterdir() if p.is_file()]

    models = {}
    for path in sorted(model_dir.iterdir()):
        if not path.is_file():
            continue
        meta = detect_model(path.name)
        if meta is None:
            continue

        # Hide ComfyUI / text-only safetensors transformers can't caption with.
        if meta["format"] == "safetensors" and not _is_hf_vlm_safetensors(
            path
        ):
            continue

        mmproj_path = None
        if meta["format"] == "gguf":
            mm = find_mmproj(path.name, all_names, meta["type"])
            mmproj_path = (model_dir / mm) if mm else None

        models[path.name] = {
            "local_path": path,
            "hf_config": meta["hf_config"],
            "type": meta["type"],
            "format": meta["format"],
            "mmproj_path": mmproj_path,
            "source": "local",
        }

    return models


def list_mmproj_files() -> list[str]:
    """Return the mmproj GGUF file names for manual pairing."""
    model_dir = get_model_dir()
    if model_dir is None or not model_dir.exists():
        return []
    return sorted(
        p.name
        for p in model_dir.iterdir()
        if p.is_file()
        and p.name.lower().startswith("mmproj")
        and p.suffix.lower() == ".gguf"
    )
