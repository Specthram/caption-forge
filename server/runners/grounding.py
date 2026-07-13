"""Job bodies for SigLIP grounding — caption claims and media tags.

Every body owns the SigLIP load/unload pair (``finally``, house rule: a
model job never leaves weights resident for the next one). The caption
bodies additionally need the VLM already sitting in the shared loader slot,
because decomposing a caption into claims is an LLM task; the tag bodies
need nothing but SigLIP.

The batch caption body runs in two phases on purpose. Draining every
caption through the VLM first, *then* loading SigLIP once for the whole
set, means the checkpoint is paid for once per job instead of once per
media — at the cost of holding both models at the same time, which is why
the tag path (no VLM) is the cheap one.
"""

from server.jobs import Progress
from src import settings, storage
from src import siglip_grounding as grounding


def _model_id() -> str:
    """Return the configured SigLIP checkpoint and load it."""
    return grounding.load_model(
        settings.get_grounding_model_size(),
        settings.get_grounding_resolution(),
    )


def _ensure_claim_model(progress: Progress) -> bool:
    """Ensure a VLM is loaded to decompose captions; return whether one is.

    The VLM already in the loader slot is used as-is (respecting the user's
    session — any VLM can split a caption). Only when the slot is empty is
    the configured default claim model loaded, so grounding a caption works
    without a manual load. Returns False when neither is available, letting
    the caller skip cleanly instead of erroring inside the LLM call.
    """
    # pylint: disable=import-outside-toplevel
    from src import loader, scanner

    if loader.is_model_loaded():
        return True
    name = settings.get_grounding_claim_model()
    if not name:
        return False
    cfg = scanner.scan_local_models().get(name)
    if cfg is None:
        return False
    progress(sub=f"loading {name}…")
    for status, _loaded in loader.load_model(cfg):
        progress(sub=status)
    return loader.is_model_loaded()


def _unload_vlm(progress: Progress) -> None:
    """Free the VLM before SigLIP loads, so the two never share VRAM.

    Claim decomposition (VLM) and scoring (SigLIP) are consecutive, never
    concurrent; holding both resident is what OOMs a tight GPU. The VLM is
    dropped once its claims are extracted — the user reloads it to caption
    again.
    """
    # pylint: disable=import-outside-toplevel
    from src import loader

    if not loader.is_model_loaded():
        return
    progress(sub="freeing VLM…")
    for status, _loaded in loader.unload_model():
        progress(sub=status)


def ground_caption_body(dataset_id, key, caption_type):
    """Return a job body grounding one media's caption."""

    def run(progress: Progress) -> dict:
        progress(total=1, done=0, sub="preparing…")
        if not _ensure_claim_model(progress):
            progress(done=1, sub="no VLM")
            return {
                "status": "skipped",
                "reason": "no VLM loaded and no claim model set in Settings",
            }
        progress(sub="decomposing caption…")
        extraction = storage.extract_caption_claims(
            dataset_id, str(key), caption_type
        )
        if extraction["status"] != "ok":
            progress(done=1, sub=extraction["reason"])
            return extraction
        # VLM done → free it before SigLIP, never both resident at once.
        _unload_vlm(progress)
        progress(sub="loading SigLIP…")
        try:
            model_id = _model_id()
            verdict = storage.score_caption_claims(
                extraction["path"],
                extraction["revision_id"],
                extraction["claims"],
                model_id,
            )
        finally:
            grounding.unload_model()
        progress(done=1, sub="grounded")
        return verdict

    return run


def ground_dataset_body(dataset_id, caption_type, media_ids=None):
    """Return a job body grounding every caption of a dataset.

    Phase one asks the loaded VLM for each caption's claims; phase two
    loads SigLIP once and scores them all. A media whose caption yields no
    claim (a video, a missing file, an empty caption) is simply skipped.
    """

    def run(progress: Progress) -> dict:
        progress(total=1, done=0, sub="preparing…")
        if not _ensure_claim_model(progress):
            return {
                "status": "skipped",
                "reason": "no VLM loaded and no claim model set in Settings",
            }
        keys = media_ids
        if keys is None:
            keys = [
                int(item["key"]) for item in storage.list_media(dataset_id)
            ]
        total = len(keys)
        progress(total=total * 2, done=0, sub=f"claims 0 / {total}")
        extracted = []
        for index, key in enumerate(keys, start=1):
            extraction = storage.extract_caption_claims(
                dataset_id, str(key), caption_type
            )
            if extraction["status"] == "ok":
                extracted.append(extraction)
            progress(done=index, sub=f"claims {index} / {total}")
        scored = 0
        # Every claim extracted → free the VLM before SigLIP loads.
        _unload_vlm(progress)
        try:
            model_id = _model_id()
            for index, extraction in enumerate(extracted, start=1):
                storage.score_caption_claims(
                    extraction["path"],
                    extraction["revision_id"],
                    extraction["claims"],
                    model_id,
                )
                scored += 1
                progress(
                    done=total + index,
                    sub=f"scoring {index} / {len(extracted)}",
                )
        finally:
            grounding.unload_model()
        return {"grounded": scored, "skipped": total - scored}

    return run


def ground_tags_body(media_ids):
    """Return a job body scoring every tag of each media (no LLM)."""

    def run(progress: Progress) -> dict:
        total = len(media_ids)
        progress(total=total, done=0, sub="preparing…")
        # Tag grounding is SigLIP-only; free any resident VLM so it does not
        # share VRAM with the checkpoint about to load.
        _unload_vlm(progress)
        grounded = 0
        try:
            model_id = _model_id()
            for index, media_id in enumerate(media_ids, start=1):
                verdict = storage.ground_tags(str(media_id), model_id)
                grounded += verdict.get("status") == "ok"
                progress(done=index, sub=f"{index} / {total}")
        finally:
            grounding.unload_model()
        return {"grounded": grounded, "skipped": total - grounded}

    return run


def caption_heat_body(dataset_id, key, caption_type):
    """Return a job body rebuilding a caption's per-claim heat maps.

    The maps are the job's *result*, read back through
    ``GET /api/jobs/{id}/result`` — they are far too bulky for the progress
    WebSocket and are never persisted (one forward pass rebuilds them).
    """

    def run(progress: Progress) -> dict:
        progress(total=1, done=0, sub="loading SigLIP…")
        try:
            model_id = _model_id()
            progress(sub="mapping claims…")
            claims = storage.caption_heatmaps(
                dataset_id, str(key), caption_type
            )
        finally:
            grounding.unload_model()
        progress(done=1, sub=f"{len(claims)} claim(s)")
        return {"model_id": model_id, "elements": claims}

    return run


def tag_heat_body(key):
    """Return a job body rebuilding a media's per-tag heat maps."""

    def run(progress: Progress) -> dict:
        progress(total=1, done=0, sub="loading SigLIP…")
        try:
            model_id = _model_id()
            progress(sub="mapping tags…")
            tags = storage.tag_heatmaps(str(key), model_id)
        finally:
            grounding.unload_model()
        progress(done=1, sub=f"{len(tags)} tag(s)")
        return {"model_id": model_id, "elements": tags}

    return run
