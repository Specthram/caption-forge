"""Model routes: scan local models, report status, load / unload (jobs)."""

from fastapi import APIRouter, HTTPException

from server.jobs import manager
from server.runners import model as model_runner
from server.schemas import LoadModelBody
from src import loader, quality, scanner, settings

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
def list_models() -> dict:
    """Return the local models found in the configured model directory."""
    models = scanner.scan_local_models()
    return {
        "models": [
            {
                "name": name,
                "type": cfg["type"],
                "format": cfg["format"],
                "hf_config": cfg.get("hf_config"),
                "has_mmproj": bool(cfg.get("mmproj_path")),
            }
            for name, cfg in models.items()
        ]
    }


@router.get("/status")
def model_status() -> dict:
    """Return the loaded model, last status line and VRAM/device info."""
    return {
        "loaded": loader.is_model_loaded(),
        "name": loader.loaded_name,
        "type": loader.current_model_type,
        "format": loader.current_format,
        "status": loader.last_status,
        "vram_total_gb": quality.detect_vram_gb(),
        "gpu": quality.gpu_status(),
        "device": settings.get_device(),
    }


@router.post("/load")
def load_model(body: LoadModelBody) -> dict:
    """Enqueue a job loading the named model; return its job id."""
    models = scanner.scan_local_models()
    cfg = models.get(body.name)
    if cfg is None:
        raise HTTPException(status_code=404, detail="model not found")
    job = manager.submit(
        "load-model",
        f"Load {body.name}",
        model_runner.load_body(cfg, body.name),
        sub="loading",
    )
    return {"job_id": job.id}


@router.post("/unload")
def unload_model() -> dict:
    """Enqueue a job unloading the current model; return its job id."""
    job = manager.submit(
        "unload-model", "Unload model", model_runner.unload_body()
    )
    return {"job_id": job.id}
