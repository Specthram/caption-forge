"""WD14 auto-tagger routes: available models, defaults and a run job."""

from fastapi import APIRouter

from server.jobs import manager
from server.runners import tagger as tagger_runner
from server.schemas import TaggerRunBody
from src import sqlite_store as store
from src import tagger

router = APIRouter(prefix="/api/tagger", tags=["tagger"])


@router.get("/models")
def list_models() -> dict:
    """Return the known WD taggers and whether each is cached on disk."""
    models = [
        {
            "source": source,
            "label": label,
            "available": tagger.is_available(source),
        }
        for source, label in tagger.KNOWN_TAGGERS.items()
    ]
    return {
        "models": models,
        "default_source": tagger.DEFAULT_REPO_ID,
        "general": tagger.DEFAULT_GENERAL_THRESHOLD,
        "character": tagger.DEFAULT_CHARACTER_THRESHOLD,
    }


@router.post("/run")
def run_tagger(body: TaggerRunBody) -> dict:
    """Enqueue a WD14 auto-tag run over the requested scope; return job id."""
    if body.scope == "filtered":
        media_ids = store.library_media_ids(
            body.filter_tag_ids or None,
            body.match,
            exclude_tag_ids=body.exclude_tag_ids or None,
        )
    else:
        media_ids = body.media_ids
    job = manager.submit(
        "wd14",
        f"WD14 tag · {len(media_ids)} media",
        tagger_runner.tag_media_body(
            media_ids,
            body.source,
            body.local_dir,
            body.general,
            body.character,
            replace_underscores=body.replace_underscores,
            ground_after=body.ground_after,
        ),
    )
    return {"job_id": job.id, "count": len(media_ids)}
