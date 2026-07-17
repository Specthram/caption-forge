"""Model profiles: one bundle of everything needed to load and run a VLM.

A profile carries the weights file (any folder, not just the configured
model dir), its format (``gguf``/``safetensors``), the model family (auto-
detected from the filename via :mod:`src.model_registry`, manually
overridable), the GGUF vision projector (auto-detected next to the weights,
manual override), and the generation defaults the Caption panel used to
hold: temperature, context size, thinking mode, max new tokens, image
resolution and a default prompt preset (preset lists stay keyed by family).

Profiles are shared by the captioner (``active_id``) and the review judge
(``judge_id``). The store is a single user-layer JSON file (see
:func:`src.config.load_model_profiles`); the first read seeds one profile
named "Default". ``loaded_id`` (which profile's weights sit in VRAM) is
process state, not persisted.
"""

import copy
from pathlib import Path

from src import config, model_registry
from src.constants import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINK_MODE,
)

# Model families a profile may hold. ``manual`` = offered in the editor's
# manual type list; ``think`` = the family supports a thinking mode.
# ``text`` is an auto-detected fallback only (a GGUF no vision rule matched:
# loadable for text-only judging, images disabled).
FAMILIES = (
    {"key": "qwen3", "label": "Qwen3-VL", "think": True, "manual": True},
    {"key": "qwen3.6", "label": "Qwen3.6", "think": True, "manual": True},
    {"key": "gemma3", "label": "Gemma 3", "think": True, "manual": True},
    {"key": "gemma3n", "label": "Gemma 3n", "think": False, "manual": True},
    {"key": "gemma4", "label": "Gemma 4", "think": True, "manual": True},
    {
        "key": "mistral3",
        "label": "Mistral 3.2 / Pixtral",
        "think": False,
        "manual": True,
    },
    {
        "key": "llava",
        "label": "JoyCaption (LLaVA)",
        "think": False,
        "manual": True,
    },
    {
        "key": "text",
        "label": "Text-only GGUF",
        "think": False,
        "manual": False,
    },
)

TEXT_TYPE = "text"

_FAMILY_KEYS = {family["key"] for family in FAMILIES}
_THINK_KEYS = {family["key"] for family in FAMILIES if family["think"]}

_FORMATS = ("gguf", "safetensors")
_MODES = ("auto", "manual")
_THINK_MODES = ("off", "auto", "show")

# Image-resolution slider bounds (longest side, px).
MIN_IMG_RES, MAX_IMG_RES, IMG_RES_STEP = 512, 2048, 128
_MAX_TOK_BOUNDS = (16, 32768)
_TEMP_BOUNDS = (0.0, 2.0)

# Which profile's weights are resident. Process state: reset on restart,
# cross-checked against the loader before being reported (a judge swap or a
# crashed load must never leave a stale "loaded" dot).
_loaded_id = None  # pylint: disable=invalid-name


def is_think_capable(model_type: str) -> bool:
    """Return whether a family supports the thinking mode toggle."""
    return model_type in _THINK_KEYS


def detect_format(filename: str) -> str | None:
    """Return the weights format a filename implies, or None."""
    name = filename.lower()
    if name.endswith(".gguf"):
        return "gguf"
    if name.endswith(".safetensors"):
        return "safetensors"
    return None


def detect_type(filename: str) -> str:
    """Return the family a weights filename auto-detects to.

    A vision rule match wins; an unmatched GGUF is loadable as text-only
    (``text``); an unmatched safetensors is unrecognized (``""``).
    """
    meta = model_registry.detect_model(filename)
    if meta is not None:
        return meta["type"]
    if detect_format(filename) == "gguf":
        return TEXT_TYPE
    return ""


def auto_mmproj(directory: str, filename: str, family: str) -> str | None:
    """Return the mmproj filename auto-detected next to a GGUF model.

    Scans the profile's own folder (not the configured model dir) and
    delegates the family/size matching to
    :func:`src.model_registry.find_mmproj`. None when the family has no
    vision (``text``/unknown) or nothing matches.
    """
    if family in ("", TEXT_TYPE) or not directory:
        return None
    path = Path(directory)
    if not path.is_dir():
        return None
    names = [p.name for p in path.iterdir() if p.is_file()]
    return model_registry.find_mmproj(filename, names, family)


def _clamp(value, bounds, default, cast):
    """Return ``value`` cast and clamped to ``bounds``, or ``default``."""
    try:
        number = cast(value)
    except (TypeError, ValueError):
        return default
    low, high = bounds
    return max(low, min(number, high))


def _clamp_img_res(value) -> int:
    """Clamp an image resolution to the slider's range and step."""
    res = _clamp(value, (MIN_IMG_RES, MAX_IMG_RES), 1024, int)
    return MIN_IMG_RES + round((res - MIN_IMG_RES) / IMG_RES_STEP) * (
        IMG_RES_STEP
    )


def _default_profile() -> dict:
    """Return a fresh profile with factory generation defaults."""
    # pylint: disable=import-outside-toplevel  # circular: settings->config
    from src.settings import get_gguf_n_ctx, get_model_dir

    model_dir = get_model_dir()
    return {
        "id": 0,
        "name": "Default",
        "file": "",
        "dir": str(model_dir) if model_dir else "",
        "format": "gguf",
        "type": "",
        "type_mode": "auto",
        "temp": DEFAULT_TEMPERATURE,
        "n_ctx": get_gguf_n_ctx(),
        "mmproj_mode": "auto",
        "mmproj": None,
        "think": DEFAULT_THINK_MODE,
        "max_tok": DEFAULT_MAX_NEW_TOKENS,
        "img_res": 1024,
        "prompt": "",
    }


def _sanitize(raw: dict, base: dict) -> dict:
    """Return ``base`` updated with the valid fields of ``raw``.

    Unknown keys are dropped, enums fall back to ``base``, numbers are
    clamped. The effective type and mmproj are then re-derived so a stored
    profile is always internally consistent (auto modes re-run detection).
    """
    # pylint: disable=import-outside-toplevel  # circular: settings->config
    from src.settings import clamp_gguf_n_ctx

    profile = dict(base)
    for key in ("name", "file", "dir", "prompt"):
        if key in raw:
            profile[key] = str(raw[key] or "").strip()
    for key, allowed in (
        ("format", _FORMATS),
        ("type_mode", _MODES),
        ("mmproj_mode", _MODES),
        ("think", _THINK_MODES),
    ):
        if raw.get(key) in allowed:
            profile[key] = raw[key]
    if raw.get("type") in _FAMILY_KEYS or raw.get("type") == "":
        profile["type"] = raw["type"]
    if "mmproj" in raw:
        profile["mmproj"] = (
            str(raw["mmproj"]).strip() or None if (raw["mmproj"]) else None
        )
    profile["temp"] = _clamp(
        raw.get("temp", profile["temp"]), _TEMP_BOUNDS, base["temp"], float
    )
    profile["max_tok"] = _clamp(
        raw.get("max_tok", profile["max_tok"]),
        _MAX_TOK_BOUNDS,
        base["max_tok"],
        int,
    )
    profile["img_res"] = _clamp_img_res(raw.get("img_res", profile["img_res"]))
    profile["n_ctx"] = clamp_gguf_n_ctx(raw.get("n_ctx", profile["n_ctx"]))

    fmt = detect_format(profile["file"])
    if fmt is not None:
        profile["format"] = fmt
    if profile["type_mode"] == "auto":
        profile["type"] = (
            detect_type(profile["file"]) if profile["file"] else ""
        )
    has_vision_gguf = profile["format"] == "gguf" and profile["type"] not in (
        "",
        TEXT_TYPE,
    )
    if not has_vision_gguf:
        profile["mmproj"] = None
        profile["mmproj_mode"] = "auto"
    elif profile["mmproj_mode"] == "auto":
        profile["mmproj"] = auto_mmproj(
            profile["dir"], profile["file"], profile["type"]
        )
    if not profile["name"]:
        profile["name"] = Path(profile["file"]).stem or base["name"]
    return profile


def _read_store() -> dict:
    """Return the store, seeded with one "Default" profile on first read."""
    store = config.load_model_profiles()
    profiles = store.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        first = _default_profile()
        first["id"] = 1
        store = {
            "profiles": [first],
            "active_id": 1,
            "judge_id": 1,
            "next_id": 2,
        }
        config.save_model_profiles(store)
        return store
    ids = {p.get("id") for p in profiles}
    for key in ("active_id", "judge_id"):
        if store.get(key) not in ids:
            store[key] = profiles[0]["id"]
    return store


def list_profiles() -> dict:
    """Return ``{"profiles", "active_id", "judge_id", "loaded_id"}``."""
    store = _read_store()
    return {
        "profiles": copy.deepcopy(store["profiles"]),
        "active_id": store["active_id"],
        "judge_id": store["judge_id"],
        "loaded_id": get_loaded_id(),
    }


def get_profile(profile_id: int) -> dict | None:
    """Return one profile by id, or None."""
    for profile in _read_store()["profiles"]:
        if profile["id"] == profile_id:
            return copy.deepcopy(profile)
    return None


def create_profile(raw: dict, role: str | None = None) -> dict:
    """Create a profile from ``raw`` fields; return the stored profile.

    ``role`` (``"caption"``/``"judge"``) selects the new profile for that
    slot, matching the editor's save rule (created from the Caption panel →
    becomes active; from the judge picker → becomes the judge).
    """
    store = _read_store()
    raw = dict(raw)
    if not raw.get("name") and raw.get("file"):
        # Auto-name from the weights filename until the user types one.
        raw["name"] = Path(str(raw["file"])).stem
    profile = _sanitize(raw, _default_profile())
    profile["id"] = int(store.get("next_id", 1))
    store["next_id"] = profile["id"] + 1
    store["profiles"].append(profile)
    if role == "caption":
        store["active_id"] = profile["id"]
    elif role == "judge":
        store["judge_id"] = profile["id"]
    config.save_model_profiles(store)
    return copy.deepcopy(profile)


def update_profile(profile_id: int, raw: dict) -> dict | None:
    """Merge ``raw`` into an existing profile; return it, or None."""
    store = _read_store()
    for index, profile in enumerate(store["profiles"]):
        if profile["id"] == profile_id:
            merged = _sanitize(raw, {**_default_profile(), **profile})
            merged["id"] = profile_id
            store["profiles"][index] = merged
            config.save_model_profiles(store)
            return copy.deepcopy(merged)
    return None


def delete_profile(profile_id: int) -> bool:
    """Delete a profile; the last remaining one is never deleted.

    The active/judge selections fall back to the first remaining profile
    when they pointed at the deleted one.
    """
    store = _read_store()
    remaining = [p for p in store["profiles"] if p["id"] != profile_id]
    if len(remaining) == len(store["profiles"]) or not remaining:
        return False
    store["profiles"] = remaining
    for key in ("active_id", "judge_id"):
        if store[key] == profile_id:
            store[key] = remaining[0]["id"]
    config.save_model_profiles(store)
    return True


def select_profile(role: str, profile_id: int) -> bool:
    """Point the captioner (``"caption"``) or judge slot at a profile."""
    store = _read_store()
    if profile_id not in {p["id"] for p in store["profiles"]}:
        return False
    if role == "judge":
        store["judge_id"] = profile_id
    else:
        store["active_id"] = profile_id
    config.save_model_profiles(store)
    return True


def load_cfg(profile: dict) -> dict | None:
    """Return the :func:`src.loader.load_model` config for a profile.

    None when the profile has no usable weights file. The HF config repo
    comes from filename detection, or from the family when the type was
    forced manually. ``n_ctx`` rides along for the GGUF context override.
    """
    if not profile.get("file") or not profile.get("dir"):
        return None
    local_path = Path(profile["dir"]) / profile["file"]
    model_type = profile.get("type") or TEXT_TYPE
    meta = model_registry.detect_model(profile["file"])
    hf_config = (
        meta["hf_config"]
        if meta is not None and meta["type"] == model_type
        else model_registry.hf_config_for(model_type, profile["file"])
    )
    mmproj = profile.get("mmproj")
    return {
        "local_path": local_path,
        "hf_config": hf_config,
        "type": model_type,
        "format": profile.get("format") or detect_format(profile["file"]),
        "mmproj_path": (Path(profile["dir"]) / mmproj) if mmproj else None,
        "n_ctx": profile.get("n_ctx"),
        "source": "local",
    }


def set_loaded_id(profile_id: int | None) -> None:
    """Record which profile's weights are resident (None = unloaded)."""
    global _loaded_id  # pylint: disable=global-statement
    _loaded_id = profile_id


def get_loaded_id() -> int | None:
    """Return the loaded profile id, cross-checked against the loader.

    None when nothing is loaded or the resident weights no longer match the
    recorded profile (deleted, retargeted, or loaded outside profiles).
    """
    # pylint: disable=import-outside-toplevel  # heavy loader stays lazy
    from src import loader

    if _loaded_id is None or not loader.is_model_loaded():
        return None
    profile = get_profile(_loaded_id)
    if profile is None or profile["file"] != loader.loaded_name:
        return None
    return _loaded_id
