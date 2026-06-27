"""Job body for reference-free caption scoring.

One media, one job: the configured encoders (SigLIP2 + CLIP + BLIP) score
the whole caption against the image, each loaded and freed in turn by the
engine. No VLM is needed — the caption is scored as-is, never decomposed —
but a VLM sitting in the loader slot from a captioning run would share VRAM
with the checkpoints about to load, so it is freed first (house rule: the
model queue serialises one heavy model at a time).
"""

from server.jobs import Progress
from src import storage


def _unload_vlm(progress: Progress) -> None:
    """Free any resident VLM before the score encoders load."""
    # pylint: disable=import-outside-toplevel
    from src import loader

    if not loader.is_model_loaded():
        return
    progress(sub="freeing VLM…")
    for status, _loaded in loader.unload_model():
        progress(sub=status)


def score_caption_body(dataset_id, key, caption_type):
    """Return a job body scoring one media's caption with every encoder."""

    def run(progress: Progress) -> dict:
        progress(total=1, done=0, sub="preparing…")
        target = storage.caption_score_target(
            dataset_id, str(key), caption_type
        )
        if target["status"] != "ok":
            progress(done=1, sub=target["reason"])
            return target
        _unload_vlm(progress)
        verdict = storage.score_caption(
            target["path"], target["revision_id"], target["caption"], progress
        )
        progress(done=1, sub=f"scored {verdict['scored']}")
        return verdict

    return run


def score_tags_body(key):
    """Return a job body scoring one media's tags with every encoder.

    Scores the media's tags joined into one comma-separated text (the Media
    tab "Tags Score" card), not a caption.
    """

    def run(progress: Progress) -> dict:
        progress(total=1, done=0, sub="preparing…")
        target = storage.media_tag_score_target(str(key))
        if target["status"] != "ok":
            progress(done=1, sub=target["reason"])
            return target
        _unload_vlm(progress)
        verdict = storage.score_media_tags(str(key), progress)
        progress(done=1, sub=f"scored {verdict['scored']}")
        return verdict

    return run


def score_dataset_body(dataset_id, caption_type):
    """Return a job body scoring every caption of a dataset.

    Each encoder is loaded once for the whole set (see
    :func:`src.caption_score.score_dataset`), so the run pays for the three
    checkpoints once, not once per media.
    """

    def run(progress: Progress) -> dict:
        progress(total=1, done=0, sub="preparing…")
        _unload_vlm(progress)
        verdict = storage.score_dataset_captions(
            dataset_id, caption_type, progress
        )
        progress(sub=f"scored {verdict['scored']} across {verdict['media']}")
        return verdict

    return run
