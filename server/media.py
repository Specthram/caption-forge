"""Media file serving: cached thumbnails and full-resolution originals.

Replaces Gradio's ``set_static_paths`` indirection. Thumbnails back every
grid; originals back the detail-panel preview and the zoom lightbox.
Starlette's :class:`~starlette.responses.FileResponse` handles HTTP Range
requests, so video scrubbing and large-image zoom stream correctly.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src import index_steps, settings, thumbnails, watermark
from src.media import is_video_file
from src.sqlite_store import media as media_store

router = APIRouter(prefix="/api/media", tags=["media"])


@router.get("/{media_id}/thumb")
def get_thumbnail(media_id: int) -> FileResponse:
    """Return the media's cached 512px thumbnail, generating it if needed.

    A watermark-patched media is thumbnailed from its *composited* image
    (patches over the original) and cached under the composite's own key, so
    the erased watermark shows in the grid without ever touching the source.
    When the "thumbs" index step is off, an unpatched image is served straight
    from disk; a composited one still gets a thumbnail so the patch is
    visible. A video always gets its cached first frame.
    """
    row = media_store.get_media(media_id)
    if row is None:
        raise HTTPException(status_code=404, detail="media not found")
    source, composed_sha = watermark.display_source(media_id)
    if source is None:
        raise HTTPException(status_code=404, detail="file missing on disk")
    thumbs_off = not settings.is_index_step_enabled(index_steps.THUMBS)
    if thumbs_off and composed_sha is None and not is_video_file(source):
        return FileResponse(str(source))
    path = thumbnails.ensure_thumbnail(source, composed_sha or row["sha256"])
    if path is None:
        raise HTTPException(status_code=404, detail="no thumbnail")
    return FileResponse(str(path))


@router.get("/{media_id}/file")
def get_original(media_id: int) -> FileResponse:
    """Return the media's display file (Range-enabled, for zoom/video).

    Composited (patches applied) when the media carries watermark patches,
    else the plain original — the source file is never modified.
    """
    source, _composed_sha = watermark.display_source(media_id)
    if source is None:
        raise HTTPException(status_code=404, detail="file missing on disk")
    return FileResponse(str(source))


@router.get("/{media_id}/original")
def get_raw_original(media_id: int) -> FileResponse:
    """Return the media's untouched source file (never composited).

    The Watermark Lab's before/after comparators need the raw original — the
    watermark still visible — alongside the composited ``/file`` view.
    """
    source = media_store.effective_file(media_id)
    if source is None:
        raise HTTPException(status_code=404, detail="file missing on disk")
    return FileResponse(str(source))
