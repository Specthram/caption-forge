"""Model-profile routes: CRUD, selection, file browsing, detection, load.

Thin adapter over :mod:`src.model_profiles`. The file browser reuses
:func:`src.fs_browse.browse_files` (weights + mmproj GGUFs; the front dims
entries that don't match the picked format). Loading is a job — one worker,
one GPU — that swaps VRAM to the profile's weights.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from server.jobs import manager
from server.runners import model as model_runner
from server.schemas import ProfileBody, ProfileDetectBody, ProfileSelectBody
from src import fs_browse, model_profiles
from src.settings import get_model_dir

router = APIRouter(prefix="/api/profiles", tags=["profiles"])

_MODEL_EXTS = ("gguf", "safetensors")


@router.get("")
def list_profiles() -> dict:
    """Return every profile, the slot selections and the family table."""
    data = model_profiles.list_profiles()
    data["families"] = list(model_profiles.FAMILIES)
    return data


@router.post("")
def create_profile(body: ProfileBody) -> dict:
    """Create a profile; ``role`` selects it for the caption/judge slot."""
    fields = body.model_dump(exclude_unset=True)
    role = fields.pop("role", None)
    return model_profiles.create_profile(fields, role)


@router.get("/browse")
def browse_models(path: str = "") -> dict:
    """Return one folder listing of the weights / mmproj file picker.

    Starts at the default models folder (Settings ▸ Directories) when no
    ``path`` is given.
    """
    start = path or str(get_model_dir() or "")
    return fs_browse.browse_files(start, _MODEL_EXTS)


@router.post("/select")
def select_profile(body: ProfileSelectBody) -> dict:
    """Point the captioner or judge slot at a profile."""
    if not model_profiles.select_profile(body.role, body.id):
        raise HTTPException(status_code=404, detail="profile not found")
    return {"ok": True}


@router.post("/detect")
def detect(body: ProfileDetectBody) -> dict:
    """Re-run type / mmproj auto-detection for a picked weights file."""
    family = model_profiles.detect_type(body.file)
    return {
        "type": family,
        "format": model_profiles.detect_format(body.file),
        "mmproj": model_profiles.auto_mmproj(body.dir, body.file, family),
        "name": Path(body.file).stem,
    }


@router.put("/{profile_id}")
def update_profile(profile_id: int, body: ProfileBody) -> dict:
    """Apply the provided fields to an existing profile."""
    fields = body.model_dump(exclude_unset=True)
    fields.pop("role", None)
    profile = model_profiles.update_profile(profile_id, fields)
    if profile is None:
        raise HTTPException(status_code=404, detail="profile not found")
    return profile


@router.delete("/{profile_id}")
def delete_profile(profile_id: int) -> dict:
    """Delete a profile (the last remaining one is refused)."""
    if not model_profiles.delete_profile(profile_id):
        raise HTTPException(
            status_code=409,
            detail="unknown profile, or the last one — cannot delete",
        )
    return {"ok": True}


@router.post("/{profile_id}/load")
def load_profile(profile_id: int) -> dict:
    """Enqueue a job swapping VRAM to the profile's weights."""
    profile = model_profiles.get_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="profile not found")
    cfg = model_profiles.load_cfg(profile)
    if cfg is None:
        raise HTTPException(
            status_code=409, detail="profile has no weights file"
        )
    job = manager.submit(
        "load-model",
        f"Load {profile['name']}",
        model_runner.load_profile_body(cfg, profile),
        sub="loading",
    )
    return {"job_id": job.id}
