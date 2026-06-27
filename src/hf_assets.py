"""Locate and check the user-provided HF config/processor files.

A safetensors model needs its Hugging Face config + processor (config.json,
tokenizer, preprocessor...) to be instantiated by transformers. Rather than
download them silently at load time, the user may drop them under
:data:`src.constants.HF_CONFIG_DIR`, one sub-folder per repo id. This module
holds the pure path/presence logic shared by the loader (which assembles the
load directory) and the UI (which reports what is missing). It imports no heavy
backend, so it stays cheap to use from the event callbacks.
"""

from pathlib import Path

from src.constants import HF_CONFIG_DIR

# Core files checked for presence. transformers needs more (tokenizer data,
# chat template...), but these three reliably signal whether the user dropped a
# real config/processor snapshot for the repo.
REQUIRED_HF_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)

# Glob patterns excluded from the metadata download: the weights never come
# from Hugging Face (they are the user's local files).
WEIGHT_IGNORE_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.gguf", "*.h5")


def _safe_id(hf_config_id: str) -> str:
    """Turn a repo id into a filesystem-safe folder name."""
    return hf_config_id.replace("/", "--")


def hf_config_dir(hf_config_id: str) -> Path:
    """Return the local folder where ``hf_config_id``'s files are expected."""
    return HF_CONFIG_DIR / _safe_id(hf_config_id)


def missing_hf_files(hf_config_id: str) -> list[str]:
    """Return the required files absent from the repo's local config folder.

    ``hf_config_id`` is the HF repo id (e.g. ``"Qwen/Qwen3-VL-4B-Instruct"``).
    Returns the :data:`REQUIRED_HF_FILES` not present locally; empty when the
    user has provided the config (no download needed).
    """
    folder = hf_config_dir(hf_config_id)
    return [name for name in REQUIRED_HF_FILES if not (folder / name).exists()]
