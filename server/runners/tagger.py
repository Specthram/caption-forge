"""Gradio-free WD14 auto-tagger batch (mirrors medias_gallery autotag)."""

from server.jobs import Progress
from src import settings, siglip_grounding, storage
from src import sqlite_store as store
from src import tagger
from src.media import is_video_file


def tag_media_body(
    media_ids,
    source,
    local_dir,
    general,
    character,
    replace_underscores=True,
    ground_after=False,
):
    """Return a job body auto-tagging each media.

    Loads the ONNX session lazily (downloading the model first when it is
    missing), runs the tagger on every image (videos / missing files are
    skipped), adds the confident tags to the media and releases the session
    at the end. Existing tags are never removed. New tag names land in the
    "Uncategorized" holding pen; names already known reuse their tag.

    With ``ground_after``, the freshly tagged media are then scored against
    their image by SigLIP — a WD14 tagger is confident, not correct, and
    grounding is what catches the ``horse`` it hallucinated. The ONNX
    session is released first, so the two models never hold VRAM at once.
    """

    def run(progress: Progress) -> dict:
        if not tagger.is_available(source, local_dir):
            progress(sub="downloading tagger model…")
            tagger.download(source)
        total = len(media_ids)
        progress(total=total, done=0, sub=f"0 / {total}")
        added = 0
        tagged = []
        for index, media_id in enumerate(media_ids, start=1):
            path = store.effective_file(media_id)
            if path and not is_video_file(path):
                added += tag_one(
                    media_id,
                    path,
                    source,
                    local_dir,
                    general,
                    character,
                    replace_underscores,
                )
                tagged.append(media_id)
            progress(done=index, sub=f"{index} / {total}")
        tagger.release()
        grounded = _ground_tags(progress, tagged) if ground_after else 0
        return {"added": added, "media": total, "grounded": grounded}

    return run


def _ground_tags(progress: Progress, media_ids) -> int:
    """Score each media's tags with SigLIP; return how many were grounded."""
    if not media_ids:
        return 0
    total = len(media_ids)
    grounded = 0
    try:
        model_id = siglip_grounding.load_model(
            settings.get_grounding_model_size(),
            settings.get_grounding_resolution(),
        )
        for index, media_id in enumerate(media_ids, start=1):
            verdict = storage.ground_tags(str(media_id), model_id)
            grounded += verdict.get("status") == "ok"
            progress(sub=f"grounding {index} / {total}")
    finally:
        siglip_grounding.unload_model()
    return grounded


def tag_one(
    media_id,
    path,
    source,
    local_dir,
    general,
    character,
    replace_underscores,
) -> int:
    """Tag one image; return how many tags were attached.

    Shared with the "Auto-tags" step of the Index chain (see
    :func:`server.runners.library.index_body`), which drives the tagger
    session itself over a whole library.
    """
    suggestions = tagger.tag_image(path, general, character, source, local_dir)
    names = [
        name
        for kind in ("character", "general")
        for name, _ in suggestions[kind]
    ]
    added = 0
    for name in names:
        applied = name.replace("_", " ") if replace_underscores else name
        # Reuse an existing tag by name wherever it lives; a genuinely new
        # name lands in the "Uncategorized" holding pen. The link is a
        # per-media tag (source NULL), so the media counts as tagged.
        tag_id = store.get_or_create_tag_reuse(applied)
        store.add_tag_to_media(media_id, tag_id)
        added += 1
    return added
