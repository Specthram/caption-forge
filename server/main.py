"""Caption Forge FastAPI application — the React front-end's backend.

Assembles the JSON routers, the media file routes and the job WebSocket,
serves the built Vite bundle in production, and owns the single-worker job
queue lifecycle. Launch with ``python -m uvicorn server.main:app``.

This module never imports the Gradio UI; ``app.py`` (the legacy Gradio app)
remains runnable in parallel until the React port reaches parity.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

# torch (Intel OpenMP) and llama-cpp (LLVM OpenMP) each link an OpenMP
# runtime; loading both aborts with "OMP Error #15" unless duplicates are
# allowed. Must be set before importing torch (via the engines below).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# pylint: disable=wrong-import-position
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server import env, media, ws
from server.jobs import manager
from server.routers import (
    autobuild,
    caption_score,
    captions,
    crops,
    datasets,
    deploy,
    grounding,
    jobs,
    libraries,
    medias,
    models,
    prompts,
    review,
    settings,
    system,
    tagger,
    tags,
    watermarks,
)
from src import db
from src import settings as cf_settings
from src.settings import get_caption_extensions

# pylint: enable=wrong-import-position

env.apply_redirects()

WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start the job worker and ensure the database on startup."""
    db.ensure_database()
    db.seed_caption_types(get_caption_extensions())
    cf_settings.apply_hf_token()
    await manager.start()
    yield
    await manager.stop()


app = FastAPI(title="Caption Forge API", version="2.0", lifespan=lifespan)

# Dev only: the Vite dev server (5173) calls the API cross-origin. In
# production the bundle is same-origin (served below), so this is a no-op.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (
    models.router,
    prompts.router,
    datasets.router,
    autobuild.router,
    captions.router,
    crops.router,
    medias.router,
    libraries.router,
    tags.router,
    tagger.router,
    review.router,
    grounding.router,
    caption_score.router,
    deploy.router,
    settings.router,
    system.router,
    jobs.router,
    watermarks.router,
):
    app.include_router(router)
app.include_router(media.router)
app.include_router(ws.router)


@app.get("/api/health")
def health() -> dict:
    """Return a trivial liveness payload."""
    return {"status": "ok"}


def _mount_frontend() -> None:
    """Serve the built React bundle with SPA fallback, if present.

    Absent during backend-only development (the Vite dev server serves the
    front-end then); present after ``npm run build`` in production.
    """
    if not WEB_DIST.is_dir():
        return
    app.mount(
        "/assets",
        StaticFiles(directory=WEB_DIST / "assets"),
        name="assets",
    )

    @app.get("/{full_path:path}")
    def spa(full_path: str) -> FileResponse:
        """Return a built asset when it exists, else the SPA index."""
        candidate = WEB_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(WEB_DIST / "index.html"))


_mount_frontend()
