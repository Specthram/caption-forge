"""Job body for a rule-based caption review run (the Review sub-tab).

Three passes over the targeted media, unified into one finding queue:

* **det** — deterministic rules (trigger word). No model, so it runs first
  and for free.
* **text** — text-only rules judged by the LLM with no image loaded.
* **vision** — rules that need the image, judged by the VLM.

The judge is a model profile chosen independently from the captioner: it is
swapped into VRAM for the run and freed in a ``finally`` (house rule — a
model job never leaves weights resident). A ``judge_profile_id`` of None
means "use whatever model is already loaded"; when nothing is loaded the
model passes are skipped and only the deterministic pass runs.

The pure judging logic (prompt, JSON parsing, the anti-free-rewrite guard)
lives in :mod:`src.caption_judge`; persistence and the merge rule in
:mod:`src.sqlite_store.review_queue` (via :mod:`src.storage`).
"""

from server.jobs import Progress
from src import caption_judge, storage


def _split_rules(rules: list) -> tuple:
    """Return the enabled rules split into (det, text, vision) lists."""
    det = [r for r in rules if r["kind"] == "det"]
    text = [r for r in rules if r["kind"] == "text"]
    vision = [r for r in rules if r["kind"] == "vlm"]
    return det, text, vision


def _enabled_rules(dataset_ref, rule_ids) -> list:
    """Return the dataset's enabled rules, optionally filtered to an id set."""
    return [
        rule
        for rule in storage.review_rules(dataset_ref)
        if rule["enabled"] and (rule_ids is None or rule["id"] in rule_ids)
    ]


def _load_judge(judge_profile_id, progress: Progress) -> bool:
    """Swap the judge profile into VRAM; return whether we loaded.

    ``None`` keeps the model already resident, as does an unknown profile or
    one with no weights file (skip the swap rather than abort the run). A
    judge already loaded is kept — and reported as not-ours so it survives
    the run (the captioner keeps its slot).
    """
    # pylint: disable=import-outside-toplevel
    from src import loader, model_profiles

    if judge_profile_id is None:
        return False
    if model_profiles.get_loaded_id() == judge_profile_id:
        return False
    profile = model_profiles.get_profile(judge_profile_id)
    cfg = model_profiles.load_cfg(profile) if profile else None
    if cfg is None:
        return False
    if loader.is_model_loaded():
        for status, _loaded in loader.unload_model():
            progress(sub=status)
        model_profiles.set_loaded_id(None)
    progress(sub=f"loading judge {profile['name']}…")
    for status, _loaded in loader.load_model(cfg):
        progress(sub=status)
    if loader.is_model_loaded():
        model_profiles.set_loaded_id(judge_profile_id)
    return True


def _judge_name(judge_profile_id) -> str:
    """Return the judge profile's display name for the run record, or ""."""
    # pylint: disable=import-outside-toplevel
    from src import model_profiles

    if judge_profile_id is None:
        return ""
    profile = model_profiles.get_profile(judge_profile_id)
    return profile["name"] if profile else ""


def _trigger_words(dataset_ref) -> list:
    """Return the dataset's trigger words (for the deterministic rule)."""
    # pylint: disable=import-outside-toplevel
    from src import sqlite_store as store

    dataset_id = storage.sqlite_dataset_id(dataset_ref)
    if dataset_id is None:
        return []
    return [row["name"] for row in store.dataset_triggerwords(dataset_id)]


def _record(run_id, target, rule, note, caption_after, caption_type) -> None:
    """Persist one pending finding for a media / rule pair."""
    storage.record_review_finding(
        run_id,
        target["media_id"],
        caption_type,
        note,
        target["caption"],
        caption_after,
        rule_id=rule["id"],
        rule_kind=rule["kind"],
    )


def _judge_rule(rule, target, seed) -> dict | None:
    """Run one text/vision rule against a media; return a finding or None.

    A text rule is judged with no image; a vision rule loads the image
    (videos are skipped for vision). The raw answer is parsed and passed
    through the anti-free-rewrite guard.
    """
    # pylint: disable=import-outside-toplevel
    from src import captioner

    prompt = caption_judge.build_prompt(rule["text"], target["caption"])
    if rule["kind"] == "vlm":
        if not target["path"] or target["is_video"]:
            return None
        raw = captioner.generate_caption(
            target["path"], prompt, 0.0, seed, think_mode="off"
        )
    else:
        raw = captioner.generate_text(prompt, seed=seed)
    verdict = caption_judge.parse_judgement(raw)
    return caption_judge.judged_finding(target["caption"], verdict)


def review_run_body(params):
    """Return a job body running a rule-based review over a dataset.

    ``params`` is a :class:`server.schemas.ReviewRunBody`. A whole-dataset
    run replaces the queue; a single-media run replaces only that media's
    findings (``scope == 'single'`` with one id in ``media_ids``).
    """

    def run(progress: Progress) -> dict:
        dataset_ref = params.dataset_id
        caption_type = params.caption_type
        seed = params.seed if params.seed is not None else -1
        rules = _enabled_rules(dataset_ref, params.rule_ids)
        det_rules, text_rules, vision_rules = _split_rules(rules)

        keys = params.media_ids
        if keys is None:
            keys = [
                int(item["key"]) for item in storage.list_media(dataset_ref)
            ]
        targets = storage.review_targets(dataset_ref, caption_type, keys)

        if params.scope == "single" and len(keys) == 1:
            storage.reset_review_queue(dataset_ref, int(keys[0]))
        else:
            storage.reset_review_queue(dataset_ref)

        run_id = storage.open_review_run(
            dataset_ref,
            _judge_name(params.judge_profile_id),
            params.scope,
            len(targets),
        )

        state = {"done": 0, "found": 0}
        needs_model = bool(text_rules or vision_rules)
        passes = (1 if det_rules else 0) + (1 if needs_model else 0)
        total = max(len(targets) * passes, 1)
        progress(total=total, done=0, sub=f"0 / {len(targets)}")

        if det_rules:
            _run_det_pass(
                run_id,
                targets,
                det_rules,
                caption_type,
                dataset_ref,
                progress,
                state,
            )
        if needs_model:
            _run_model_pass(
                run_id,
                targets,
                text_rules,
                vision_rules,
                caption_type,
                seed,
                params.judge_profile_id,
                progress,
                state,
            )

        storage.close_review_run(run_id, state["found"])
        return {"reviewed": len(targets), "findings": state["found"]}

    return run


def _run_det_pass(
    run_id, targets, det_rules, caption_type, dataset_ref, progress, state
):
    """Run the deterministic pass (trigger word) over every target."""
    trigger_words = _trigger_words(dataset_ref)
    for target in targets:
        for rule in det_rules:
            hit = caption_judge.check_det_rule(
                rule, target["caption"], trigger_words
            )
            if hit is not None:
                _record(
                    run_id,
                    target,
                    rule,
                    hit["note"],
                    hit["caption_after"],
                    caption_type,
                )
                state["found"] += 1
        state["done"] += 1
        progress(done=state["done"], sub=f"check {state['done']}")


def _run_model_pass(
    run_id,
    targets,
    text_rules,
    vision_rules,
    caption_type,
    seed,
    judge_profile_id,
    progress,
    state,
):
    """Load the judge, run the text + vision passes, then free it."""
    # pylint: disable=import-outside-toplevel
    from src import loader, model_profiles

    loaded = _load_judge(judge_profile_id, progress)
    if not loaded and not loader.is_model_loaded():
        progress.warn("no judge model loaded — model rules skipped")
        return
    model_rules = text_rules + vision_rules
    try:
        _judge_targets(
            run_id, targets, model_rules, seed, caption_type, progress, state
        )
    finally:
        if loaded:
            progress(sub="freeing judge…")
            for status, _done in loader.unload_model():
                progress(sub=status)
            model_profiles.set_loaded_id(None)


def _judge_targets(
    run_id, targets, model_rules, seed, caption_type, progress, state
):
    """Judge every target against the model rules, recording findings."""
    for target in targets:
        for rule in model_rules:
            finding = _judge_rule(rule, target, seed)
            if finding is not None:
                _record(
                    run_id,
                    target,
                    rule,
                    finding["note"],
                    finding["caption_after"],
                    caption_type,
                )
                state["found"] += 1
        state["done"] += 1
        progress(done=state["done"], sub=f"judge {state['done']}")
