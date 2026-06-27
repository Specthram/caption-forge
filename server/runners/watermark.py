"""Job bodies for the Watermark Lab (v2): scan, patch, scan+patch, flatten.

Detection and editing are decoupled. A *scan* only locates boxes (OWLv2 the
default, or the fine-tuned YOLO) and leaves them ``detected``; a *patch* runs
FLUX.2 klein over already-detected zones; *scan & patch* chains the two behind
one model load at a time (the house rule — the detector unloads before FLUX
loads). A *flatten* bakes a media's patches into its source file. Any resident
VLM is freed first, as in the grounding runner, so a scan never shares VRAM.
"""

from server.jobs import Progress
from src import settings
from src import sqlite_store as store
from src import watermark
from src import watermark_detect as detect
from src import watermark_flux as flux
from src.media import is_video_file


def _free_vlm(progress: Progress) -> None:
    """Free any resident VLM so it never shares VRAM with the scan models."""
    from src import loader  # pylint: disable=import-outside-toplevel

    if not loader.is_model_loaded():
        return
    progress(sub="freeing VLM…")
    for status, _loaded in loader.unload_model():
        progress(sub=status)


def _scan_candidates(media_ids) -> list:
    """Return the image media in the set that carry no zone yet.

    A media already holding zones is skipped so a re-scan never duplicates or
    clobbers reviewed work; videos are excluded (a clip is not patched here).
    ``media_ids`` is the explicit selection (or the whole filtered set) the
    router resolved from the Medias tab.
    """
    ids = [int(mid) for mid in media_ids]
    have_zones = set(store.zones_bulk(ids).keys())
    keep = []
    for media_id in ids:
        if media_id in have_zones:
            continue
        row = store.get_media(media_id)
        if row is None or is_video_file(f"x.{row['file_extension']}"):
            continue
        if store.effective_file(media_id) is not None:
            keep.append(media_id)
    return keep


def _detector_label(prefs) -> str:
    """Return a human label for the selected detector (progress lines)."""
    if prefs["detector"] == "owlv2":
        return prefs["owlv2_model"].split("/")[-1]
    return prefs["yolo_model"] or "YOLO"


def _detect_source(source, prefs) -> list:
    """Return the detections for one source with the selected detector."""
    if prefs["detector"] == "owlv2":
        return detect.detect_owlv2(
            source, prefs["owlv2_queries"], prefs["owlv2_confidence"]
        )
    return detect.detect_yolo(source, prefs["confidence_min"])


def _load_detector(prefs) -> None:
    """Load the selected detector (OWLv2 eager; YOLO loads lazily on use)."""
    if prefs["detector"] == "owlv2":
        detect.load_owlv2(prefs["owlv2_model"])


def _unload_detector(prefs) -> None:
    """Free the selected detector's VRAM once the detect phase ends."""
    if prefs["detector"] == "owlv2":
        detect.unload_owlv2()
    else:
        detect.unload_yolo()


def _detect(progress: Progress, prefs, keep) -> dict:
    """Detect boxes over every kept media and create their zones.

    Returns ``{media_id: [zone ids]}``. Loads the detector once, unloads it in
    a ``finally`` so an interrupted scan never holds VRAM.
    """
    progress(total=len(keep), done=0, sub=f"loading {_detector_label(prefs)}…")
    _load_detector(prefs)
    try:
        return _detect_all(progress, prefs, keep)
    finally:
        _unload_detector(prefs)


def _detect_all(progress: Progress, prefs, keep) -> dict:
    """Detect and create zones for every kept media (detector loaded)."""
    total = len(keep)
    dilate = prefs["dilate_px"]
    zones_by_media = {}
    for index, media_id in enumerate(keep, start=1):
        source = store.effective_file(media_id)
        boxes = _detect_source(source, prefs) if source else []
        if boxes:
            zones_by_media[media_id] = watermark.create_zones(
                media_id, boxes, dilate
            )
        progress(done=index, sub=f"detect {index} / {total}")
    return zones_by_media


def _edit(progress: Progress, prefs, zones_by_media, tags) -> int:
    """FLUX-edit a patch for every created zone; return how many patched."""
    zone_ids = [zid for ids in zones_by_media.values() for zid in ids]
    total = len(zone_ids)
    if total == 0:
        return 0
    progress(
        total=total, done=0, sub=f"loading FLUX.2 {flux.model_label(prefs)}…"
    )
    flux.load_model(prefs)
    try:
        return _edit_all(progress, prefs, zones_by_media, tags)
    finally:
        flux.unload_model()


def _edit_all(progress, prefs, zones_by_media, tags) -> int:
    """Patch every zone (pipeline already loaded); return the patched count."""
    total = sum(len(ids) for ids in zones_by_media.values())
    done = 0
    for media_id, ids in zones_by_media.items():
        for zone_id in ids:
            watermark.generate_patch(zone_id, prefs)
            done += 1
            progress(done=done, sub=f"edit {done} / {total}")
        if tags is not None:
            watermark.maybe_cleanup_tags(media_id, tags)
    return done


def _cleanup_tags(prefs):
    """Return the configured tags to strip on full patch, or None when off."""
    return prefs["tags_to_remove"] if prefs["tag_cleanup"] else None


def scan_body(media_ids):
    """Return a job body detecting watermarks over the set (no patching)."""

    def run(progress: Progress) -> dict:
        prefs = settings.get_watermark_prefs()
        progress(total=1, done=0, sub="preparing…")
        _free_vlm(progress)
        candidates = _scan_candidates(media_ids)
        if not candidates:
            progress(done=1, sub="nothing to scan")
            return {"scanned": 0, "media": 0, "detected": 0, "patched": 0}
        zones_by = _detect(progress, prefs, candidates)
        return {
            "scanned": len(candidates),
            "media": len(zones_by),
            "detected": sum(len(v) for v in zones_by.values()),
            "patched": 0,
        }

    return run


def scan_and_patch_body(media_ids):
    """Return a job body detecting then FLUX-patching over the set."""

    def run(progress: Progress) -> dict:
        prefs = settings.get_watermark_prefs()
        progress(total=1, done=0, sub="preparing…")
        _free_vlm(progress)
        candidates = _scan_candidates(media_ids)
        if not candidates:
            progress(done=1, sub="nothing to scan")
            return {"scanned": 0, "media": 0, "detected": 0, "patched": 0}
        zones_by = _detect(progress, prefs, candidates)
        patched = _edit(progress, prefs, zones_by, _cleanup_tags(prefs))
        return {
            "scanned": len(candidates),
            "media": len(zones_by),
            "detected": sum(len(v) for v in zones_by.values()),
            "patched": patched,
        }

    return run


def patch_media_body(media_id: int, prefs, tags):
    """Return a job body FLUX-patching every ``detected`` zone of one media."""

    def run(progress: Progress) -> dict:
        return _patch_media_ids(progress, prefs, [media_id], tags)

    return run


def patch_body(media_ids, prefs, tags):
    """Return a job body patching every ``detected`` zone of many media.

    ``media_ids`` empty means "every media that still has a detected zone";
    the job chains them behind one FLUX load. Zones already patched are left
    alone (patch only ever touches detected zones).
    """

    def run(progress: Progress) -> dict:
        ids = (
            [int(mid) for mid in media_ids]
            if media_ids
            else store.media_ids_with_detected_zones()
        )
        return _patch_media_ids(progress, prefs, ids, tags)

    return run


def _patch_media_ids(progress: Progress, prefs, media_ids, tags) -> dict:
    """Patch the detected zones of every given media behind one FLUX load."""
    zones_by_media = {}
    for media_id in media_ids:
        ids = [
            zone["id"]
            for zone in store.list_zones(media_id)
            if zone["status"] == store.STATUS_DETECTED
        ]
        if ids:
            zones_by_media[media_id] = ids
    total = sum(len(ids) for ids in zones_by_media.values())
    if total == 0:
        progress(total=1, done=1, sub="nothing to patch")
        return {"media": 0, "patched": 0}
    progress(
        total=total, done=0, sub=f"loading FLUX.2 {flux.model_label(prefs)}…"
    )
    _free_vlm(progress)
    flux.load_model(prefs)
    try:
        patched = _edit_all(progress, prefs, zones_by_media, tags)
    finally:
        flux.unload_model()
    return {"media": len(zones_by_media), "patched": patched}


def patch_zone_body(zone_id: int, prefs, prompt, seed, tags):
    """Return a job body FLUX-editing one zone (loads/unloads the pipeline)."""

    def run(progress: Progress) -> dict:
        progress(
            total=1, done=0, sub=f"loading FLUX.2 {flux.model_label(prefs)}…"
        )
        flux.load_model(prefs)
        try:
            _patch_one(zone_id, prefs, prompt, seed, tags)
        finally:
            flux.unload_model()
        progress(done=1, sub="patched")
        return {"zone_id": zone_id}

    return run


def _patch_one(zone_id, prefs, prompt, seed, tags) -> None:
    """FLUX-edit one zone (pipeline loaded) and run its tag cleanup."""
    watermark.generate_patch(zone_id, prefs, prompt=prompt, seed=seed)
    zone = store.get_zone(zone_id)
    if tags is not None and zone is not None:
        watermark.maybe_cleanup_tags(zone["media_id"], tags)


def flatten_body(media_ids):
    """Return a job body baking each media's patches into its source file."""

    def run(progress: Progress) -> dict:
        ids = [int(mid) for mid in media_ids]
        total = len(ids)
        progress(total=total or 1, done=0, sub="flattening to disk…")
        flattened = 0
        for done, media_id in enumerate(ids, start=1):
            try:
                if watermark.flatten_media(media_id):
                    flattened += 1
            except ValueError:
                pass
            progress(done=done, sub=f"flatten {done} / {total}")
        return {"flattened": flattened}

    return run
