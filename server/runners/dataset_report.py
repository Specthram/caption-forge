"""Dataset quality-report job: score, embed, analyze, compile, persist.

The Datasets → Quality report tab's "Re-run evaluation" enqueues this body.
It follows the same VRAM rule as every other heavy job: the IQA metrics are
chained one at a time and unloaded in a ``finally``, then DINOv2 is loaded
on its own — never two models resident at once.

A run only computes what is missing (a media × metric pair with no stored
score, an image with no stored vector) unless ``force`` is set, so the
second evaluation of a dataset is nearly instant. The finished report is
stored as a JSON blob (``dataset_report``) and the stale issue resolutions
are dropped (see :func:`src.sqlite_store.reconcile_resolutions`).
"""

import logging
import time
from dataclasses import dataclass

from server.jobs import Progress
from src import config, dataset_quality, depth_embeddings, embeddings
from src import hue_bucket, image_stats, quality
from src import sqlite_store as store
from src import storage
from src.media import is_video_file

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSpec:
    """What one evaluation run is asked to do.

    Attributes
    ----------
    dataset_id : int
        The dataset to evaluate.
    scorers : tuple of str
        The enabled scorer ids: IQA metric keys, plus
        :data:`src.dataset_quality.EMBEDDING_SCORER` to enable DINOv2.
    caption_type : str
        The caption type the hygiene pillar and the caption issues read.
    target_type : str
        The auto-build target type driving the framing and size advice.
    force : bool
        Recompute every score and embedding instead of only the missing.
    """

    dataset_id: int
    scorers: tuple = dataset_quality.DEFAULT_SCORERS
    caption_type: str = "txt"
    target_type: str = "character"
    force: bool = False


def _is_video(item: dict) -> bool:
    """Return whether a dataset media dict is a video."""
    return is_video_file(f"x.{item['file_extension']}")


def _score_stage(spec, progress: Progress, total: int, done: int) -> int:
    """Score every missing (media, metric) pair, one metric at a time."""
    metrics = [
        metric_id
        for metric_id in spec.scorers
        if metric_id in quality.QUALITY_METRICS
    ]
    if not metrics:
        return done
    stage = "Scoring quality — " + " · ".join(
        dataset_quality.scorer_label(metric_id) for metric_id in metrics
    )
    for metric_id in metrics:
        rows = store.dataset_media_pending_score(
            spec.dataset_id, metric_id, force=spec.force
        )
        try:
            for row in rows:
                if row["eff_path"]:
                    _try(
                        lambda r=row, m=metric_id: store.upsert_media_quality(
                            r["id"],
                            m,
                            quality.score_media(r["eff_path"], m),
                        ),
                        f"quality {metric_id}",
                        row,
                    )
                done += 1
                progress(done=done, total=total, sub=stage)
        finally:
            quality.unload_metric()
    return done


def _try(work, label: str, row: dict) -> None:
    """Run one per-media analysis, skipping (not crashing) on any failure.

    The DB readers already keep videos out of the report (every analysis is
    image-only), but a single unreadable file — a truncated download, a
    corrupt image, a media whose extension lies — must not sink the whole
    evaluation. The offending media is logged and skipped; the run goes on.
    """
    _try_value(work, label, row)


def _embed_stage(spec, progress: Progress, total: int, done: int) -> int:
    """Embed every dataset image DINOv2 has not seen yet."""
    if dataset_quality.EMBEDDING_SCORER not in spec.scorers:
        return done
    rows = store.dataset_media_pending_embedding(
        spec.dataset_id, embeddings.MODEL_ID, force=spec.force
    )
    if not rows:
        return done
    stage = f"Embedding {len(rows)} media with DINOv2…"
    embeddings.load_model()
    try:
        for row in rows:
            if row["eff_path"]:
                _try(
                    lambda r=row: store.upsert_media_embedding(
                        r["id"],
                        embeddings.MODEL_ID,
                        embeddings.vector_to_blob(
                            embeddings.embed_image(r["eff_path"])
                        ),
                    ),
                    "embedding",
                    row,
                )
            done += 1
            progress(done=done, total=total, sub=stage)
    finally:
        embeddings.unload_model()
    return done


def _stats_stage(images, progress: Progress, total: int, done: int) -> tuple:
    """Compute the model-free per-image statistics; return them and done."""
    stage = "Clustering, duplicate & outlier detection…"
    stats = {}
    for item in images:
        if item["eff_path"]:
            found = _try_value(
                lambda it=item: image_stats.analyze(it["eff_path"]),
                "image stats",
                item,
            )
            if found:
                stats[item["id"]] = found
        done += 1
        progress(done=done, total=total, sub=stage)
    return stats, done


def _try_value(work, label: str, row: dict):
    """Like :func:`_try` but returns the work's value, or None on failure."""
    try:
        return work()
    except Exception:  # pylint: disable=broad-except
        log.warning(
            "%s failed on media %s (%s); skipping",
            label,
            row["id"],
            row["eff_path"],
            exc_info=True,
        )
        return None


def _styles(images, depth_vectors: dict) -> dict:
    """Return the composition-map style bucket of each depth-embedded image.

    Model-free hue bucketing (:func:`src.hue_bucket.classify`), computed only
    for the media carrying a depth signature — so a dataset whose depth index
    never ran pays nothing here. An undecodable file is skipped, not fatal.
    """
    styles = {}
    for item in images:
        if item["id"] not in depth_vectors or not item["eff_path"]:
            continue
        bucket = _try_value(
            lambda it=item: hue_bucket.classify(it["eff_path"]),
            "style bucket",
            item,
        )
        if bucket:
            styles[item["id"]] = bucket
    return styles


def _snapshot(spec, images, videos, stats) -> dataset_quality.Snapshot:
    """Gather every repository read the engine needs into one snapshot."""
    ids = [item["id"] for item in images]
    hashes = store.media_hashes(ids)
    # The dataset-media projection is narrow (no index columns), so the
    # dimensions and the favorite flag are read in one extra query.
    index_info = store.media_index_info(ids)
    for item in images:
        item["phash"], item["dhash"] = hashes.get(item["id"], (None, None))
        item.update(index_info.get(item["id"], {}))
    blobs = store.media_embeddings(embeddings.MODEL_ID, ids)
    vectors = {
        media_id: embeddings.blob_to_vector(blob)
        for media_id, blob in blobs.items()
    }
    # The composition pillar and map read the depth signatures the (opt-in)
    # depth index step stored — never computed here. Empty when it never ran;
    # the pillar and map then stay diagnostic-only.
    depth_blobs = store.media_embeddings(depth_embeddings.MODEL_ID, ids)
    depth_vectors = {
        media_id: depth_embeddings.blob_to_vector(blob)
        for media_id, blob in depth_blobs.items()
    }
    styles = _styles(images, depth_vectors)
    captions = storage.read_captions_bulk(
        spec.dataset_id, [str(media_id) for media_id in ids], spec.caption_type
    )
    tags = store.tags_for_media_bulk(ids)
    autobuild = config.load_autobuild_config()
    quality_config = config.load_dataset_quality_config()
    target = (autobuild.get("target_types") or {}).get(spec.target_type, {})
    return dataset_quality.Snapshot(
        images=tuple(images),
        video_count=len(videos),
        missing_count=sum(1 for item in images + videos if item["missing"]),
        scorers=tuple(spec.scorers),
        vectors_by_id=vectors,
        depth_vectors_by_id=depth_vectors,
        styles_by_id=styles,
        stats_by_id=stats,
        captions_by_id={
            media_id: captions.get(str(media_id), "") for media_id in ids
        },
        tags_by_id={
            media_id: [row["name"] for row in tags.get(media_id, [])]
            for media_id in ids
        },
        buckets=autobuild.get("framing_buckets") or {},
        target_ratios=target.get("ratios") or {},
        size_range=dataset_quality.recommended_size_range(
            quality_config, spec.target_type
        ),
        settings=quality_config,
    )


def _dataset_media(dataset_id: int) -> tuple:
    """Return the dataset's visible ``(images, videos)`` media dicts.

    A media hidden inside the dataset is never deployed, so it is never
    trained on: the report leaves it out entirely, exactly like the deploy.
    """
    media = [
        item
        for item in store.media_in_dataset(dataset_id)
        if not item["hidden"]
    ]
    images = [item for item in media if not _is_video(item)]
    videos = [item for item in media if _is_video(item)]
    return images, videos


def _stage_total(spec, image_count: int) -> int:
    """Return the job's progress total: every unit of work it will do."""
    total = image_count + 1  # the stats pass, then the compile step
    for metric_id in spec.scorers:
        if metric_id in quality.QUALITY_METRICS:
            total += len(
                store.dataset_media_pending_score(
                    spec.dataset_id, metric_id, force=spec.force
                )
            )
    if dataset_quality.EMBEDDING_SCORER in spec.scorers:
        total += len(
            store.dataset_media_pending_embedding(
                spec.dataset_id, embeddings.MODEL_ID, force=spec.force
            )
        )
    return total


def report_body(spec):
    """Return a job body evaluating one dataset and storing its report.

    Parameters
    ----------
    spec : RunSpec
        The dataset, the enabled scorers, the caption type the hygiene
        pillar reads, the target type driving the framing/size advice and
        the ``force`` flag.

    Returns
    -------
    callable
        The ``run(progress) -> dict`` body the job manager executes.
    """

    def run(progress: Progress) -> dict:
        started = time.time()
        images, videos = _dataset_media(spec.dataset_id)
        total = _stage_total(spec, len(images))
        progress(done=0, total=total, sub="Preparing…")
        done = _score_stage(spec, progress, total, 0)
        done = _embed_stage(spec, progress, total, done)
        stats, done = _stats_stage(images, progress, total, done)
        progress(done=done, total=total, sub="Compiling report…")
        # Re-read: the scoring stage above just wrote new quality rows.
        images, videos = _dataset_media(spec.dataset_id)
        report = dataset_quality.evaluate(
            _snapshot(spec, images, videos, stats)
        )
        store.reconcile_resolutions(spec.dataset_id, report.issues)
        store.save_dataset_report(
            spec.dataset_id,
            dataset_quality.to_dict(report),
            spec.scorers,
            spec.caption_type,
            time.time() - started,
        )
        progress(done=total, total=total, sub="Done")
        return {
            "overall": report.overall,
            "grade": report.grade,
            "issues": len(report.issues),
        }

    return run
