"""Libraries batch jobs: scan and the chained "Index" pipeline.

The Index job walks the steps of :mod:`src.index_steps` in order, over one
library or every library, skipping the steps disabled on this machine (see
:func:`src.settings.get_index_steps`) and the media a step already covered
(unless ``force``). Each heavy model is loaded once and unloaded in a
``finally``, so the single-worker queue never leaves two models resident.

Dimensions, perceptual hashes and the model-free image statistics are not a
step: they are cheap CPU probes every other feature depends on
(near-duplicates, the resolution floor, the composer's blur/noise filters,
the quality report), so a full chain always refreshes them — even when the
"thumbs" step is off and no thumbnail JPEG is written.
"""

import logging

from server.jobs import Progress
from server.runners.tagger import tag_one
from src import depth_embeddings, embeddings, image_stats, index_steps
from src import perceptual_hash, quality, settings, siglip_grounding
from src import sqlite_store as store
from src import tagger, thumbnails
from src.media import is_video_file

logger = logging.getLogger(__name__)


def scan_body(library_id: int):
    """Return a job body rescanning one library for new/removed files."""

    def run(progress: Progress) -> dict:
        progress(sub="discovering files…")

        def report(done: int, total: int) -> None:
            progress(done=done, total=total, sub=f"{done} / {total} files")

        return store.scan_library(library_id, progress=report)

    return run


def _thumb_pending(library_id, force: bool) -> dict:
    """Return the media lacking a cached thumbnail, keyed by media id."""
    rows = store.media_pending_index(library_id, force=True)
    if force:
        return {row["id"]: row for row in rows}
    cached = thumbnails.cached_sha256()
    return {row["id"]: row for row in rows if row["sha256"] not in cached}


def _geometry_pending(library_id, force: bool) -> dict:
    """Return the media whose dimensions/hashes must be (re)computed."""
    rows = store.media_pending_index(library_id, force=force)
    return {row["id"]: row for row in rows}


def _index_one(row, path, write_thumb: bool, write_geometry: bool, force):
    """Write one media's thumbnail and/or geometry, hashes and statistics."""
    if write_thumb:
        thumbnails.ensure_thumbnail(path, row["sha256"], force=force)
    if write_geometry:
        dimensions = thumbnails.probe_dimensions(path)
        width, height = dimensions if dimensions else (None, None)
        phash, dhash = perceptual_hash.compute_hashes(path)
        store.set_media_index(row["id"], width, height, phash, dhash)
        if not is_video_file(path):
            store.set_media_stats(row["id"], image_stats.analyze(path))


class _Chain:
    """Cumulative progress across the steps of one Index run."""

    def __init__(self, progress: Progress, total: int):
        self.progress = progress
        self.total = total
        self.done = 0
        progress(total=total, done=0)

    def tick(self, label: str) -> None:
        """Count one processed media and report it under ``label``."""
        self.done += 1
        self.progress(
            done=self.done, sub=f"{label} · {self.done} / {self.total}"
        )

    def warn(self, message: str) -> None:
        """Report a media the step could not process, and carry on."""
        self.progress.warn(message)


def _for_each(chain: _Chain, rows, label: str, work) -> int:
    """Apply ``work(row, path)`` to every media; return how many succeeded.

    One media, one failure: an image a decoder chokes on, or a file that
    left the disk since the last scan, must not cost the rest of the
    library. The media is named on the job (the Jobs drawer lists it) and
    the traceback goes to the server log; the run keeps going.

    A stop request still ends the run, because :meth:`_Chain.tick` raises
    outside the guarded call.
    """
    succeeded = 0
    for row in rows:
        path = row["eff_path"]
        if not path:
            chain.warn(f"{row['name']}: file missing on disk — skipped")
        else:
            try:
                work(row, path)
                succeeded += 1
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("%s failed on %s", label, row["name"])
                chain.warn(f"{row['name']}: {type(exc).__name__}: {exc}")
        chain.tick(label)
    return succeeded


def _thumbs_pass(chain: _Chain, thumbs: dict, geometry: dict, force) -> int:
    """Run the thumbnails/geometry pass; return how many media were read.

    The progress label reflects what is actually written: "thumbnails" only
    when the thumbs step is on (JPEGs are cached), otherwise "dimensions" —
    dimensions and perceptual hashes are a cheap probe that always runs, so
    a disabled thumbs step must not read as if it were generating files.
    """
    label = "thumbnails" if thumbs else "dimensions"
    rows = {**geometry, **thumbs}

    def work(row, path):
        _index_one(
            row, path, row["id"] in thumbs, row["id"] in geometry, force
        )

    _for_each(chain, rows.values(), label, work)
    return len(rows)


def _score_metric(chain: _Chain, rows: list, metric: str) -> int:
    """Score every row with one loaded metric; return how many were scored."""

    def work(row, path):
        store.upsert_media_quality(
            row["id"], metric, quality.score_media(path, metric)
        )

    return _for_each(chain, rows, metric, work)


def _quality_pass(chain: _Chain, pending: dict) -> int:
    """Score each metric over its pending media, one model at a time."""
    scored = 0
    for metric, rows in pending.items():
        try:
            scored += _score_metric(chain, rows, metric)
        finally:
            quality.unload_metric()
    return scored


def _embed_rows(chain: _Chain, rows: list) -> int:
    """Embed every row with the loaded DINOv2 model."""

    def work(row, path):
        store.upsert_media_embedding(
            row["id"],
            embeddings.MODEL_ID,
            embeddings.vector_to_blob(embeddings.embed_image(path)),
        )

    return _for_each(chain, rows, "embeddings", work)


def _embed_pass(chain: _Chain, rows: list) -> int:
    """Embed every pending image with DINOv2; return how many were embedded."""
    embeddings.load_model()
    try:
        return _embed_rows(chain, rows)
    finally:
        embeddings.unload_model()


def _depth_rows(chain: _Chain, rows: list) -> int:
    """Embed every row with the loaded Depth-Anything V2 model."""

    def work(row, path):
        store.upsert_media_embedding(
            row["id"],
            depth_embeddings.MODEL_ID,
            depth_embeddings.vector_to_blob(
                depth_embeddings.embed_image(path)
            ),
        )

    return _for_each(chain, rows, "composition", work)


def _depth_pass(chain: _Chain, rows: list) -> int:
    """Embed every pending image's composition; return how many were done."""
    depth_embeddings.load_model()
    try:
        return _depth_rows(chain, rows)
    finally:
        depth_embeddings.unload_model()


def _siglip_rows(chain: _Chain, rows: list, model_id: str) -> int:
    """Embed every row with the loaded SigLIP 2 checkpoint."""

    def work(row, path):
        store.upsert_media_embedding(
            row["id"],
            model_id,
            embeddings.vector_to_blob(siglip_grounding.embed_image(path)),
        )

    return _for_each(chain, rows, "semantic search", work)


def _siglip_pass(chain: _Chain, rows: list) -> int:
    """Embed every pending image with SigLIP 2; return how many were done.

    The vectors are keyed by the checkpoint's repository id, so switching
    the grounding size/resolution in Settings does not silently compare a
    query against vectors from another model — the next Index refills them.
    """
    model_id = siglip_grounding.load_model(
        settings.get_grounding_model_size(),
        settings.get_grounding_resolution(),
    )
    try:
        return _siglip_rows(chain, rows, model_id)
    finally:
        siglip_grounding.unload_model()


def _tag_rows(chain: _Chain, rows: list, model) -> int:
    """Auto-tag every row with the loaded tagger; return the tags attached."""
    source, local_dir, general, character = model
    added = [0]

    def work(row, path):
        added[0] += tag_one(
            row["id"], path, source, local_dir, general, character, True
        )

    _for_each(chain, rows, "auto-tags", work)
    return added[0]


def _wd14_pass(chain: _Chain, rows: list) -> int:
    """Auto-tag every pending image; return how many tags were attached."""
    model = (
        settings.get_autotag_source(),
        settings.get_autotag_local_dir(),
        settings.get_autotag_general(),
        settings.get_autotag_character(),
    )
    if not tagger.is_available(model[0], model[1]):
        chain.progress(sub="downloading tagger model…")
        tagger.download(model[0])
    try:
        return _tag_rows(chain, rows, model)
    finally:
        tagger.release()


def _plan(library_id, steps, force: bool) -> dict:
    """Return the pending work of every step to run, keyed by step key."""
    enabled = settings.get_index_steps()
    plan = index_steps.normalize_steps(steps, enabled)
    work = {}
    if index_steps.THUMBS in plan:
        work[index_steps.THUMBS] = _thumb_pending(library_id, force)
    if steps is None or index_steps.THUMBS in steps:
        work["geometry"] = _geometry_pending(library_id, force)
    if index_steps.QUALITY in plan:
        work[index_steps.QUALITY] = {
            metric: store.media_pending_score(metric, library_id, force=force)
            for metric in settings.get_index_quality_metrics()
        }
    if index_steps.EMBED in plan:
        work[index_steps.EMBED] = store.media_pending_embedding(
            embeddings.MODEL_ID, library_id, force=force
        )
    if index_steps.DEPTH in plan:
        work[index_steps.DEPTH] = store.media_pending_embedding(
            depth_embeddings.MODEL_ID, library_id, force=force
        )
    if index_steps.SIGLIP in plan:
        work[index_steps.SIGLIP] = store.media_pending_embedding(
            settings.get_grounding_model_id(), library_id, force=force
        )
    if index_steps.WD14 in plan:
        work[index_steps.WD14] = store.media_pending_autotag(
            library_id, force=force
        )
    return work


def _plan_total(work: dict) -> int:
    """Return how many media reads the planned work adds up to."""
    thumbs = work.get(index_steps.THUMBS, {})
    geometry = work.get("geometry", {})
    total = len(set(thumbs) | set(geometry))
    total += sum(
        len(rows) for rows in work.get(index_steps.QUALITY, {}).values()
    )
    total += len(work.get(index_steps.EMBED, []))
    total += len(work.get(index_steps.DEPTH, []))
    total += len(work.get(index_steps.SIGLIP, []))
    total += len(work.get(index_steps.WD14, []))
    return total


def _run_index(library_id, steps, force, progress: Progress) -> dict:
    """Plan then chain the enabled index steps over an existing progress.

    The plan is built here (at run time), so a caller that scanned first —
    see :func:`full_reindex_body` — indexes the media it just ingested.
    """
    work = _plan(library_id, steps, force)
    chain = _Chain(progress, _plan_total(work))
    thumbs = work.get(index_steps.THUMBS, {})
    geometry = work.get("geometry", {})
    result = {
        "indexed": 0,
        "scored": 0,
        "embedded": 0,
        "depth": 0,
        "semantic": 0,
        "tagged": 0,
    }
    if thumbs or geometry:
        result["indexed"] = _thumbs_pass(chain, thumbs, geometry, force)
    if work.get(index_steps.QUALITY):
        result["scored"] = _quality_pass(chain, work[index_steps.QUALITY])
    if work.get(index_steps.EMBED):
        result["embedded"] = _embed_pass(chain, work[index_steps.EMBED])
    if work.get(index_steps.DEPTH):
        result["depth"] = _depth_pass(chain, work[index_steps.DEPTH])
    if work.get(index_steps.SIGLIP):
        result["semantic"] = _siglip_pass(chain, work[index_steps.SIGLIP])
    if work.get(index_steps.WD14):
        result["tagged"] = _wd14_pass(chain, work[index_steps.WD14])
    return result


def index_body(library_id, steps=None, force: bool = False):
    """Return a job body chaining the enabled index steps.

    Parameters
    ----------
    library_id : int or None
        The library to index; None spans every library ("Index everything
        missing").
    steps : iterable of str, optional
        The steps to run (:data:`src.index_steps.STEP_KEYS`); None (the
        default) runs the whole chain. A step disabled on this machine is
        never run, whatever is requested.
    force : bool, optional
        Re-run each step over media it already covered.
    """

    def run(progress: Progress) -> dict:
        return _run_index(library_id, steps, force, progress)

    return run


def _scan_all(progress: Progress) -> dict:
    """Rescan every top-level library over ``progress``; return the summary.

    Sub-libraries are skipped: their files are re-resolved and re-routed by
    the walk of their parent's tree (``store.scan_library`` re-resolves the
    whole mapping), so scanning them again would only re-walk the parent.
    """
    libraries = [
        row
        for row in store.list_libraries()
        if row["parent_library_id"] is None
    ]
    progress(total=len(libraries), done=0, sub="scan · all libraries…")
    summary = {"new_media": 0, "existing": 0, "libraries": len(libraries)}
    for done, library in enumerate(libraries, start=1):
        name = library["name"]

        def report(scanned, files, name=name):
            progress(sub=f"scan · {name} · {scanned} / {files} files")

        result = store.scan_library(library["id"], progress=report)
        summary["new_media"] += result["new_media"]
        summary["existing"] += result["existing"]
        progress(done=done, sub=f"scan · {name} · done")
    return summary


def scan_all_body():
    """Return a job body rescanning every library, one after another.

    The progress bar advances one notch per library (the count is the
    denominator); the subtitle streams the live file count of the library
    currently being walked. The per-library summaries are added up so the
    caller can report how many fresh media the whole sweep ingested.
    """

    def run(progress: Progress) -> dict:
        return _scan_all(progress)

    return run


def full_reindex_body():
    """Return a job body: rescan every library, then index everything.

    One chained job so the phases run back to back on the single worker.
    Phase 1 rescans every folder (ingesting new/recovered files); phase 2
    runs the whole enabled index chain — thumbnails, quality, embeddings
    and the WD14 auto-tags — over every library. Because the index plan is
    built after the scan, the media just ingested are indexed and tagged
    in the same run.
    """

    def run(progress: Progress) -> dict:
        scan = _scan_all(progress)
        index = _run_index(None, None, False, progress)
        return {"scan": scan, "index": index}

    return run
