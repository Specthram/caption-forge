"""Prompt-preset routes for a model type (builtin + user presets).

Wraps the gradio-free :mod:`src.config` store (not ``src.prompts``, which
returns Gradio components). Generation preferences (selected preset,
temperature, thinking mode) live in :mod:`src.settings`, keyed by model
type.
"""

from fastapi import APIRouter

from server.schemas import GenModeBody, SavePromptBody
from src import config, settings

router = APIRouter(prefix="/api/prompts", tags=["prompts"])


@router.get("")
def list_prompts(model_type: str) -> dict:
    """Return the presets for a model type plus its generation settings."""
    prompts = config.load_prompts(model_type)
    return {
        "prompts": [
            {
                "title": title,
                "prompt": entry.get("prompt", ""),
                "builtin": bool(entry.get("default")),
            }
            for title, entry in prompts.items()
        ],
        "selected": settings.get_selected_prompt(model_type),
        "temperature": settings.get_model_temperature(model_type),
        "think_mode": settings.get_model_think_mode(model_type),
    }


@router.post("")
def save_prompt(body: SavePromptBody) -> dict:
    """Add or update a user prompt preset."""
    config.save_user_prompt(body.model_type, body.title, body.prompt)
    return {"ok": True}


@router.delete("")
def delete_prompt(model_type: str, title: str) -> dict:
    """Delete a user prompt preset (built-ins are never removed)."""
    removed = config.delete_user_prompt(model_type, title)
    return {"ok": removed}


@router.post("/settings")
def save_gen_settings(body: GenModeBody) -> dict:
    """Persist per-type generation settings (temp / thinking / preset)."""
    if body.temperature is not None:
        settings.set_model_temperature(body.model_type, body.temperature)
    if body.think_mode is not None:
        settings.set_model_think_mode(body.model_type, body.think_mode)
    if body.selected_prompt is not None:
        settings.set_selected_prompt(body.model_type, body.selected_prompt)
    return {"ok": True}
