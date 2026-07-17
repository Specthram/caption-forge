"""Layered configuration store.

Two layers under ``config/``: ``default/`` (shipped, read-only, committed) and
``user/`` (overrides, runtime, git-ignored). Each user file merges onto its
default counterpart key by key, so a later default is picked up automatically.

Per-model-type files are named after the type, e.g. ``models/qwen3.json``, and
hold two sections::

    {
        "prompts":  {title: {"prompt": str, "default": bool}},
        "settings": {"think_mode": str, "temperature": float, ...}
    }

The ``default`` flag (default layer only) marks the factory prompt;
``settings`` stores the type's generation prefs.
"""

import json
from pathlib import Path

from src.constants import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_MODELS_DIR,
    SETTINGS_FILENAME,
    USER_CONFIG_DIR,
    USER_MODELS_DIR,
)


def deep_merge(base: dict, override: dict) -> dict:
    """Merge ``override`` onto ``base`` recursively.

    ``override`` values win; two nested dicts merge recursively rather than the
    override replacing the base. Returns a new dict; inputs untouched.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _read_json(path: Path) -> dict:
    """Return the JSON at ``path``, or ``{}`` if missing or invalid."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    """Write ``data`` as indented UTF-8 JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8"
    )


# --- Settings ---


def read_default_settings() -> dict:
    """Return the raw default settings layer (no user overrides applied)."""
    return _read_json(DEFAULT_CONFIG_DIR / SETTINGS_FILENAME)


def read_user_settings() -> dict:
    """Return the raw user settings layer."""
    return _read_json(USER_CONFIG_DIR / SETTINGS_FILENAME)


def load_layered_settings() -> dict:
    """Return the default settings layer merged with the user layer."""
    return deep_merge(read_default_settings(), read_user_settings())


def save_user_settings(settings: dict) -> None:
    """Persist the user settings layer to ``config/user/settings.json``."""
    _write_json(USER_CONFIG_DIR / SETTINGS_FILENAME, settings)


# --- Caption-review judge prompts ---


def load_review_prompts() -> dict:
    """Return the caption-review judge prompts (default overlaid by user).

    Live in ``review.json`` of each layer: ``extract_claims`` (split a caption
    into atomic claims), ``verify_claim`` (VQA yes/no per claim), ``rewrite``
    (fix only contradicted passages). Empty when no layer ships the file.
    """
    default = _read_json(DEFAULT_CONFIG_DIR / "review.json")
    user = _read_json(USER_CONFIG_DIR / "review.json")
    return deep_merge(default, user)


# --- Model profiles ---

_PROFILES_FILENAME = "model_profiles.json"


def load_model_profiles() -> dict:
    """Return the raw model-profiles store (user layer only).

    Profiles are runtime user data (weights paths are machine-specific), so
    there is no default layer: the file lives in ``config/user/`` alone.
    Empty when never saved.
    """
    return _read_json(USER_CONFIG_DIR / _PROFILES_FILENAME)


def save_model_profiles(data: dict) -> None:
    """Persist the model-profiles store to ``config/user/``."""
    _write_json(USER_CONFIG_DIR / _PROFILES_FILENAME, data)


# --- Dataset auto-build configuration ---


def load_autobuild_config() -> dict:
    """Return the dataset auto-build config (default overlaid by user).

    Lives in ``autobuild.json`` of each layer, three sections:
    ``framing_buckets`` (bucket id to booru tags; declaration order = classify
    precedence, see :func:`src.framing.classify`), ``target_types`` (framing
    ratio presets) and ``selection`` (picker diversity/quality weights, see
    :mod:`src.dataset_builder`). Empty when no layer ships the file.
    """
    default = _read_json(DEFAULT_CONFIG_DIR / "autobuild.json")
    user = _read_json(USER_CONFIG_DIR / "autobuild.json")
    return deep_merge(default, user)


def load_dataset_quality_config() -> dict:
    """Return the dataset-quality config (default overlaid by user).

    Lives in ``dataset_quality.json`` of each layer: category ``weights`` of
    the grade, per-target-type ``recommended_sizes`` ranges,
    ``min_resolution_side`` floor and ``near_duplicate_similarity`` threshold
    (see :mod:`src.dataset_quality`). Empty when no layer ships the file.
    """
    default = _read_json(DEFAULT_CONFIG_DIR / "dataset_quality.json")
    user = _read_json(USER_CONFIG_DIR / "dataset_quality.json")
    return deep_merge(default, user)


# --- Per-model-type config (prompts + generation settings) ---


def _default_model_path(model_type: str) -> Path:
    """Return the default-layer config file path for a model type."""
    return DEFAULT_MODELS_DIR / f"{model_type}.json"


def _user_model_path(model_type: str) -> Path:
    """Return the user-layer config file path for a model type."""
    return USER_MODELS_DIR / f"{model_type}.json"


def load_model_config(model_type: str) -> dict:
    """Return the merged per-type config (default overlaid by user).

    ``model_type`` (e.g. ``"qwen3"``, ``"gemma3n"``) is the file stem. Returns
    ``{"prompts": ..., "settings": ...}``; either section absent when no layer
    provides it.
    """
    default = _read_json(_default_model_path(model_type))
    user = _read_json(_user_model_path(model_type))
    return deep_merge(default, user)


def load_prompts(model_type: str) -> dict:
    """Return the merged prompts for one model type.

    Title to entry, default overlaid by user. Empty if no config file exists.
    """
    return load_model_config(model_type).get("prompts", {})


def save_user_prompt(model_type: str, title: str, prompt: str) -> None:
    """Add or update one prompt in the user layer, preserving its settings."""
    path = _user_model_path(model_type)
    data = _read_json(path)
    data.setdefault("prompts", {})[title] = {"prompt": prompt}
    _write_json(path, data)


def set_user_model_setting(model_type: str, key: str, value) -> None:
    """Persist one per-type generation setting, preserving its prompts."""
    path = _user_model_path(model_type)
    data = _read_json(path)
    data.setdefault("settings", {})[key] = value
    _write_json(path, data)


def delete_user_prompt(model_type: str, title: str) -> bool:
    """Remove one user-layer prompt preset, preserving the rest.

    Built-in prompts live in the read-only default layer, never touched; only
    a user override with this title is removed. Returns whether one existed.
    """
    path = _user_model_path(model_type)
    data = _read_json(path)
    prompts = data.get("prompts", {})
    if title not in prompts:
        return False
    del prompts[title]
    _write_json(path, data)
    return True


# --- Legacy layout migration ---


def migrate_legacy_layout() -> None:
    """Upgrade the pre-``models/`` user layer to the current layout.

    Idempotent, safe on every startup. Two legacy shapes upgraded in place:
    (1) ``config/user/prompts/*.json`` folded into
    ``config/user/models/<type>.json`` under ``prompts``; (2) the global
    ``selected_prompts`` map moved into each type's
    ``settings.selected_prompt`` and dropped from settings.
    """
    _migrate_user_prompt_files()
    _migrate_selected_prompts()


def _migrate_user_prompt_files() -> None:
    """Fold legacy ``config/user/prompts/*.json`` into the models layer."""
    legacy_dir = USER_CONFIG_DIR / "prompts"
    if not legacy_dir.is_dir():
        return
    for path in legacy_dir.glob("*.json"):
        legacy_prompts = _read_json(path)
        if legacy_prompts:
            target = _user_model_path(path.stem)
            data = _read_json(target)
            data.setdefault("prompts", {}).update(legacy_prompts)
            _write_json(target, data)
        path.unlink()
    try:
        legacy_dir.rmdir()
    except OSError:
        pass


def _migrate_selected_prompts() -> None:
    """Move ``selected_prompts`` from settings into the models layer."""
    user_settings = read_user_settings()
    selected = user_settings.get("selected_prompts")
    if not selected:
        return
    for model_type, title in selected.items():
        set_user_model_setting(model_type, "selected_prompt", title)
    user_settings.pop("selected_prompts", None)
    save_user_settings(user_settings)
