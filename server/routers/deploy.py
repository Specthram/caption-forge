"""Deploy routes: differential deploy / undeploy and zip download."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from server.jobs import manager
from server.schemas import DeployBody, DeployMediaBody
from src import deploy as deploy_engine
from src import sqlite_store as store
from src import storage, wm_compose
from src.zip import create_zip_archive

router = APIRouter(prefix="/api/deploy", tags=["deploy"])


def _deploy_items(dataset_id: int, caption_type: str) -> list[dict]:
    """Return the per-media items the deploy engine expects.

    Adapts :func:`src.storage.list_media` dicts to the deploy shape (the
    active ``caption_type`` resolved as ``ext``, ``file_extension`` mapped to
    the on-disk deployed extension as ``file_ext``). The dataset's deploy
    ``resolution`` (shortest-side resize target, ``0`` when off) rides along on
    every item, and images deployed under a resolution take a ``png``
    extension since they are re-encoded losslessly.

    Watermark patches are applied here too: a media with patched zones deploys
    its *composited* image (patches pasted over the original — the source file
    itself is untouched).
    """
    resolution = deploy_engine.dataset_deploy_resolution(dataset_id)
    media_list = storage.list_media(dataset_id)
    ids = [int(media["key"]) for media in media_list]
    zones_by = store.zones_bulk(ids)
    items = []
    for media in media_list:
        media_id = int(media["key"])
        path = media["path"]
        file_ext = deploy_engine.deployed_ext(
            media["file_extension"], media["is_video"], resolution
        )
        zones = zones_by.get(media_id, [])
        if path and zones:
            composed = wm_compose.ensure_composed(path, media["sha256"], zones)
            if composed is not None:
                path = str(composed)
                if not resolution:
                    file_ext = "png"
        items.append(
            {
                "key": media["key"],
                "path": path,
                "ext": caption_type,
                "hidden": media["hidden"],
                "sha256": media["sha256"],
                "file_ext": file_ext,
                "is_video": media["is_video"],
                "resolution": resolution,
                "missing": media["missing"],
                "repeats": media["repeats"],
            }
        )
    return items


@router.get("/status")
def deploy_status(dataset_id: int, caption_type: str) -> dict:
    """Return the dataset's aggregate deploy state and folder path."""
    items = _deploy_items(dataset_id, caption_type)
    folder = deploy_engine.dataset_deploy_dir(dataset_id)
    return {
        "status": deploy_engine.dataset_status(dataset_id, items),
        "folder": str(folder) if folder else None,
    }


@router.post("/dataset")
def deploy_dataset(body: DeployBody, caption_type: str) -> dict:
    """Enqueue a differential deploy of a dataset; return its job id."""
    items = _deploy_items(body.dataset_id, caption_type)

    def run(_progress) -> dict:
        return deploy_engine.deploy_dataset(body.dataset_id, items)

    job = manager.submit("deploy", f"Deploy dataset {body.dataset_id}", run)
    return {"job_id": job.id}


@router.post("/media")
def deploy_media(body: DeployMediaBody) -> dict:
    """Deploy a selection of media into the dataset folder (differential)."""
    items = {
        item["key"]: item
        for item in _deploy_items(body.dataset_id, body.caption_type)
    }
    written = 0
    for key in body.keys:
        item = items.get(key)
        if item is not None:
            result = deploy_engine.deploy_media(body.dataset_id, item)
            written += result.get("written", 0)
    return {"written": written, "count": len(body.keys)}


@router.post("/undeploy")
def undeploy_dataset(body: DeployBody) -> dict:
    """Remove a dataset's deployed files (synchronous, no model)."""
    return deploy_engine.undeploy_dataset(body.dataset_id)


@router.get("/zip")
def download_zip(dataset_id: int) -> FileResponse:
    """Zip the dataset's current deploy folder and stream it back."""
    folder = deploy_engine.dataset_deploy_dir(dataset_id)
    if folder is None or not folder.is_dir():
        raise HTTPException(status_code=400, detail="nothing deployed")
    archive = create_zip_archive(str(folder))
    return FileResponse(
        archive,
        media_type="application/zip",
        filename="collection.zip",
    )
