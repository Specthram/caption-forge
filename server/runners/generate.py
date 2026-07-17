"""Job body for the Caption tab's "Generate all captions" batch."""

import random

from server.jobs import Progress
from src import captioner, loader, settings, siglip_grounding, storage
from src.media import is_video_file


def _resolve_seed(seed) -> int:
    """Return a concrete seed (a random one when the request left it out)."""
    if seed is None:
        return random.randint(0, 2**31 - 1)
    return int(seed)


def generate_body(params):
    """Return a job body captioning every requested media in a dataset.

    ``params`` is a :class:`server.schemas.GenerateBody`. Each media is
    captioned with the loaded VLM (video vs image dispatch), the result is
    saved as a new revision, and — when requested — the integrity heuristics
    run on it (``review_after``). With ``ground_after``, a SigLIP grounding
    pass then scores every fresh caption in one go (see :func:`_ground_all`).
    Progress is reported per media.
    """

    def run(progress: Progress) -> dict:
        dataset_ref = params.dataset_id
        caption_type = params.caption_type
        keys = params.media_ids
        if keys is None:
            keys = [
                int(item["key"]) for item in storage.list_media(dataset_ref)
            ]
        excluded = set(params.exclude_ids or [])
        keys = [key for key in keys if int(key) not in excluded]
        if not params.recaption:
            # Only fill the blanks: drop media whose caption already has
            # content (one bulk query, not one read per media).
            texts = storage.read_captions_bulk(
                dataset_ref, [str(key) for key in keys], caption_type
            )
            keys = [key for key in keys if not texts.get(str(key), "").strip()]
        total = len(keys)
        progress(total=total, done=0, sub=f"0 / {total}")
        seed = _resolve_seed(params.seed)
        for index, key in enumerate(keys, start=1):
            key = str(key)
            path = storage.media_path(dataset_ref, key)
            if path:
                text = _caption_one(path, params, seed)
                storage.write_caption(dataset_ref, key, caption_type, text)
                if params.review_after:
                    storage.run_integrity_review(
                        dataset_ref, key, caption_type
                    )
            progress(done=index, sub=f"{index} / {total}")
        grounded = 0
        if params.ground_after:
            grounded = _ground_all(progress, dataset_ref, keys, caption_type)
        findings = 0
        if params.review_after:
            findings = _review_all(progress, params, keys)
        return {"done": total, "grounded": grounded, "findings": findings}

    return run


def _review_all(progress: Progress, params, keys) -> int:
    """Run the rule-based review over the freshly generated captions.

    Chains the Review sub-tab's judge pass at the end of a generation (the
    "Review after generation" toggle): the captioner has been freed by the
    grounding pass (or is swapped out here), the judge is loaded for the run
    and freed after, and every proposal lands as a *pending* finding — nothing
    is applied without the human. Reuses the review job body so the det / text
    / vision passes and the merge rule stay in one place.
    """
    # pylint: disable=import-outside-toplevel
    from server.runners.review_run import review_run_body
    from server.schemas import ReviewRunBody

    body = ReviewRunBody(
        dataset_id=params.dataset_id,
        caption_type=params.caption_type,
        media_ids=[int(key) for key in keys],
        judge_model=params.review_judge_model,
        scope="all",
        seed=params.seed,
    )
    result = review_run_body(body)(progress)
    return result.get("findings", 0)


def _caption_one(path: str, params, seed: int) -> str:
    """Caption one media file (video vs image dispatch).

    Videos read their fps / seconds / frame prompt from Settings inside
    :func:`src.captioner.generate_captions_for_video`, so only the shared
    parameters are passed here.
    """
    generate = (
        captioner.generate_captions_for_video
        if is_video_file(path)
        else captioner.generate_caption
    )
    return generate(
        path,
        params.prompt,
        params.temperature,
        seed,
        think_mode=params.think_mode,
    )


def _ground_all(progress: Progress, dataset_ref, keys, caption_type: str):
    """Ground every freshly written caption; return how many were scored.

    Two phases, like :func:`server.runners.grounding.ground_dataset_body`:
    the VLM still loaded from the captioning pass decomposes each caption
    into claims, then SigLIP is loaded *once* to score the whole batch and
    unloaded in a ``finally``. Doing it per media would pay for the
    checkpoint on every image.
    """
    total = len(keys)
    extracted = []
    for index, key in enumerate(keys, start=1):
        extraction = storage.extract_caption_claims(
            dataset_ref, str(key), caption_type
        )
        if extraction["status"] == "ok":
            extracted.append(extraction)
        progress(sub=f"claims {index} / {total}")
    if not extracted:
        return 0
    # The VLM (still loaded from captioning) has produced every claim; free it
    # before SigLIP loads so the two never share VRAM and OOM the GPU.
    for status, _loaded in loader.unload_model():
        progress(sub=status)
    try:
        model_id = siglip_grounding.load_model(
            settings.get_grounding_model_size(),
            settings.get_grounding_resolution(),
        )
        for index, extraction in enumerate(extracted, start=1):
            storage.score_caption_claims(
                extraction["path"],
                extraction["revision_id"],
                extraction["claims"],
                model_id,
            )
            progress(sub=f"grounding {index} / {len(extracted)}")
    finally:
        siglip_grounding.unload_model()
    return len(extracted)
