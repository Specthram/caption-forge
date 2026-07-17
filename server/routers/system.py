"""System routes: DB info/backup, runtime, maintenance, SQL explorer.

Explorer mirrors :mod:`src.db_explorer`: read-only queries on a read-only
connection, plus the single-row delete on a writable one.
"""

import gc
import os
import sys
import threading
from contextlib import closing

from fastapi import APIRouter, HTTPException

from server.jobs import manager
from server.schemas import DeleteRowBody, RestoreBody, SqlBody
from src import (
    db,
    db_backup,
    db_explorer,
    fs_browse,
    maintenance,
    quality,
    thumbnails,
)
from src import sqlite_store as store
from src.constants import RESTART_EXIT_CODE

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/counts")
def nav_counts() -> dict:
    """Return the sidebar's per-section totals.

    Three cheap ``COUNT`` queries — no media dict, no file stat. Caption
    badge is absent on purpose: it counts the active dataset's media, which
    the front-end already holds.
    """
    return {
        "media": store.count_library_media(),
        "tags": store.count_tags(),
        "libraries": len(store.list_libraries()),
    }


@router.get("/browse")
def browse(path: str = "") -> dict:
    """List a folder's sub-folders for the picker (drives at the root)."""
    try:
        return fs_browse.browse(path)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/browse-files")
def browse_files(path: str = "", exts: str = "") -> dict:
    """List a folder's sub-folders and matching files, for a file picker.

    ``exts`` is a comma-separated suffix allow-list (e.g. ``safetensors``);
    empty lists every file. Backs the Watermark Lab's local model/encoder
    pickers.
    """
    wanted = [part for part in exts.split(",") if part.strip()]
    try:
        return fs_browse.browse_files(path, wanted)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _db_size() -> int:
    """Return the database file size in bytes (0 when not a real file)."""
    path = db.get_db_path()
    try:
        return os.path.getsize(path)
    except (OSError, TypeError, ValueError):
        return 0


@router.get("/database")
def database() -> dict:
    """Return the database path, size, per-table counts and backups."""
    counts = {}
    with closing(db.connect_readonly()) as conn:
        for table in db_explorer.list_tables():
            row = conn.execute(
                f'SELECT COUNT(*) AS n FROM "{table}"'
            ).fetchone()
            counts[table] = row["n"] if row else 0
    return {
        "path": str(db.get_db_path()),
        "size_bytes": _db_size(),
        "counts": counts,
        "backups": [
            {"filename": path.name, "size_bytes": path.stat().st_size}
            for path in db_backup.list_backups()
        ],
    }


@router.post("/backup")
def backup() -> dict:
    """Create a database backup snapshot; return its file name."""
    path = db_backup.create_backup()
    return {"filename": path.name}


@router.post("/restore")
def restore(body: RestoreBody) -> dict:
    """Restore a backup over the live database."""
    try:
        db_backup.restore_backup(body.filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True}


@router.get("/runtime")
def runtime() -> dict:
    """Return best-effort runtime info (python / CUDA / GPU / caches)."""
    info = {
        "python": sys.version.split()[0],
        "vram_total_gb": quality.detect_vram_gb(),
        "vram_used_gb": None,
        "cuda": None,
        "gpu": None,
        "thumbnail_cache_bytes": _thumbnail_cache_bytes(),
    }
    try:
        import torch  # pylint: disable=import-outside-toplevel

        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_used_gb"] = round(
                torch.cuda.memory_allocated(0) / 1024**3, 2
            )
    except Exception:  # pylint: disable=broad-except
        pass
    return info


def _thumbnail_cache_bytes() -> int:
    """Return the total size of the thumbnail cache directory."""
    root = thumbnails.get_thumbnails_dir()
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def _free_models() -> None:
    """Unload every model and free CUDA + host caches.

    Shared by the RAM/VRAM purge action and the safe-shutdown teardown.
    """
    # pylint: disable=import-outside-toplevel
    from src import embeddings, loader, tagger

    for status in loader.unload_model():
        _ = status
    quality.unload_metric()
    embeddings.unload_model()
    tagger.release()
    gc.collect()
    try:
        import torch  # pylint: disable=import-outside-toplevel

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pylint: disable=broad-except
        pass


@router.post("/purge")
def purge() -> dict:
    """Unload every model and free CUDA + host caches."""
    _free_models()
    return {"ok": True}


@router.get("/cleanup")
def cleanup() -> dict:
    """Return the live orphan counts for the Database cleanup block."""
    return maintenance.cleanup_report()


@router.get("/cleanup/{category}")
def cleanup_category(category: str) -> dict:
    """Return one category's ``{count, bytes}`` (per-row lazy loader)."""
    try:
        return maintenance.category_report(category)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/cleanup/{category}")
def cleanup_purge(category: str) -> dict:
    """Purge one cleanup category; return what was removed and reclaimed."""
    try:
        return maintenance.run_cleanup(category)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/restart")
def restart() -> dict:
    """Exit with the restart code so run.bat rebuilds + relaunches."""
    threading.Timer(0.4, lambda: os._exit(RESTART_EXIT_CODE)).start()
    return {"ok": True}


@router.post("/shutdown")
def shutdown() -> dict:
    """Tear down cleanly then exit with code 0 (run.bat does not relaunch).

    Mirror of :func:`restart` for a deliberate power-off. Before exiting it
    drains the job worker, unloads every model + frees VRAM, and checkpoints
    the SQLite WAL, so the process leaves nothing loaded or pending. The
    short timer lets the HTTP response flush before ``os._exit(0)``.
    """
    for job in manager.list_jobs():
        if job["state"] in {"queued", "running"}:
            manager.request_stop(job["id"])
    _free_models()
    db.checkpoint_wal()
    threading.Timer(0.4, lambda: os._exit(0)).start()
    return {"ok": True}


@router.get("/db/tables")
def db_tables() -> dict:
    """Return the database table names."""
    return {"tables": db_explorer.list_tables()}


@router.post("/db/query")
def db_query(body: SqlBody) -> dict:
    """Run a read-only query; return headers + rows (guarded)."""
    headers, data, error = db_explorer.run_query(body.sql)
    if error is not None:
        raise HTTPException(status_code=400, detail=error)
    return {"headers": headers, "rows": data}


@router.post("/db/delete-row")
def db_delete_row(body: DeleteRowBody) -> dict:
    """Delete one row by id (cascades via foreign keys)."""
    message = db_explorer.delete_row(body.table, body.row_id)
    return {"message": message}
