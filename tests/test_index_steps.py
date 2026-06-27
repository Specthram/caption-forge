"""The Index pipeline: catalogue, machine toggles and coverage counters."""

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src import dataset_quality, embeddings, index_steps, settings, thumbnails
from src import sqlite_store as store


@pytest.fixture(name="library")
def _library(store_db, thumb_cache_dir, tmp_path):
    """Seed one library holding two images, none of them indexed."""
    # pylint: disable=unused-argument
    for name, color in (("a.png", (200, 30, 30)), ("b.png", (30, 30, 200))):
        Image.new("RGB", (48, 48), color).save(tmp_path / name)
    library_id = store.create_library("fixtures", str(tmp_path))
    store.scan_library(library_id)
    media = store.list_library_media()
    return {"id": library_id, "media": [row["id"] for row in media]}


class TestCatalogue:
    """The step catalogue and the scorer mapping it feeds."""

    def test_quality_step_runs_the_report_default_scorers(self):
        """The chained metrics are the ones the quality report defaults to."""
        assert (
            index_steps.QUALITY_METRIC_IDS == dataset_quality.DEFAULT_SCORERS
        )

    def test_iqa_scorers_map_to_the_quality_step(self):
        """Any IQA chip depends on the quality scan."""
        assert index_steps.scorer_step("musiq") == index_steps.QUALITY
        assert index_steps.scorer_step("qalign_4bit") == index_steps.QUALITY

    def test_the_embedding_scorer_maps_to_the_embed_step(self):
        """The DINOv2 chip depends on the embeddings scan."""
        assert (
            index_steps.scorer_step(dataset_quality.EMBEDDING_SCORER)
            == index_steps.EMBED
        )


class TestNormalizeSteps:
    """``normalize_steps`` never returns a step disabled on this machine."""

    def test_none_runs_every_enabled_step_in_chain_order(self):
        """A full chain keeps the catalogue order."""
        enabled = dict.fromkeys(index_steps.STEP_KEYS, True)
        assert index_steps.normalize_steps(None, enabled) == list(
            index_steps.STEP_KEYS
        )

    def test_a_disabled_step_is_dropped_even_when_requested(self):
        """Asking for an off step runs nothing."""
        enabled = dict.fromkeys(index_steps.STEP_KEYS, True)
        enabled[index_steps.WD14] = False
        assert index_steps.normalize_steps([index_steps.WD14], enabled) == []

    def test_a_request_narrows_the_chain(self):
        """A per-step button runs only that step."""
        enabled = dict.fromkeys(index_steps.STEP_KEYS, True)
        assert index_steps.normalize_steps([index_steps.EMBED], enabled) == [
            index_steps.EMBED
        ]


class TestMachineToggles:
    """The ``index_steps`` setting is machine-local and self-healing."""

    def test_every_step_is_enabled_by_default(self):
        """A fresh install runs the whole chain."""
        assert settings.get_index_steps() == dict.fromkeys(
            index_steps.STEP_KEYS, True
        )

    def test_a_saved_toggle_is_read_back(self):
        """Saving the settings persists the machine toggles."""
        settings.save_settings({"index_steps": {index_steps.EMBED: False}})
        assert settings.is_index_step_enabled(index_steps.EMBED) is False
        assert settings.is_index_step_enabled(index_steps.THUMBS) is True

    def test_unknown_keys_are_dropped_and_missing_ones_default_on(self):
        """A step added later is on out of the box."""
        assert settings.clamp_index_steps({"nope": False}) == dict.fromkeys(
            index_steps.STEP_KEYS, True
        )

    def test_thresholds_are_clamped_to_a_probability(self):
        """A nonsense confidence floor falls back to the factory default."""
        assert settings.clamp_threshold("2.5", 0.35) == 1.0
        assert settings.clamp_threshold("nope", 0.35) == 0.35

    def test_index_quality_metrics_default_and_clamp(self):
        """The metric selection keeps known keys, deduped, else the trio."""
        default = list(index_steps.QUALITY_METRIC_IDS)
        assert settings.get_index_quality_metrics() == default
        settings.save_settings(
            {"index_quality_metrics": ["musiq", "nope", "musiq", "qalign"]}
        )
        assert settings.get_index_quality_metrics() == ["musiq", "qalign"]
        settings.save_settings({"index_quality_metrics": []})
        assert settings.get_index_quality_metrics() == default


class TestCoverageCounters:
    """The repository counters behind the ``done / total`` of each step."""

    def test_live_media_are_the_denominator(self, library):
        """Every scanned media counts, and a discarded one no longer does."""
        assert store.count_live_media(library["id"]) == 2
        store.set_media_discarded(library["media"][0])
        assert store.count_live_media(library["id"]) == 1

    def test_a_media_counts_as_scored_only_once_every_metric_ran(
        self, library
    ):
        """A partial score set leaves the quality step incomplete."""
        metrics = index_steps.QUALITY_METRIC_IDS
        media_id = library["media"][0]
        store.upsert_media_quality(media_id, metrics[0], 50.0)
        assert store.count_media_scored(metrics, library["id"]) == 0
        for metric in metrics[1:]:
            store.upsert_media_quality(media_id, metric, 50.0)
        assert store.count_media_scored(metrics, library["id"]) == 1

    def test_embeddings_are_counted_per_model(self, library):
        """Only vectors of the queried model count."""
        store.upsert_media_embedding(library["media"][0], "other", b"x")
        assert (
            store.count_media_with_embedding(
                embeddings.MODEL_ID, library["id"]
            )
            == 0
        )
        store.upsert_media_embedding(
            library["media"][0], embeddings.MODEL_ID, b"x"
        )
        assert (
            store.count_media_with_embedding(
                embeddings.MODEL_ID, library["id"]
            )
            == 1
        )

    def test_tagged_counts_per_media_tags_not_library_bulk(self, library):
        """A per-media tag marks it tagged; a library-bulk tag never does."""
        cat = store.create_tag_category("color")
        red = store.get_or_create_tag("red", cat)
        blue = store.get_or_create_tag("blue", cat)
        lib = library["id"]
        # A library-wide bulk tag on every media: still 0 tagged, 2 pending.
        store.add_tags_to_library(lib, [red])
        assert store.count_media_tagged(lib) == 0
        assert len(store.media_pending_autotag(lib)) == 2
        # A genuine per-media tag (WD14 / manual, source NULL) counts.
        store.add_tag_to_media(library["media"][0], blue)
        assert store.count_media_tagged(lib) == 1
        assert len(store.media_pending_autotag(lib)) == 1

    def test_missing_media_detected_then_purged(self, library, tmp_path):
        """A file deleted off disk is listed missing, then hard-purged."""
        (tmp_path / "a.png").unlink()
        missing = store.missing_media(library["id"])
        assert len(missing) == 1
        assert store.count_live_media(library["id"]) == 2
        assert store.purge_media([missing[0]["id"]]) == 1
        assert store.count_live_media(library["id"]) == 1
        assert store.missing_media(library["id"]) == []

    def test_per_media_tag_promotes_a_library_link(self, library):
        """Re-adding a library tag per-media promotes it to counting."""
        cat = store.create_tag_category("color")
        red = store.get_or_create_tag("red", cat)
        lib = library["id"]
        store.add_tags_to_library(lib, [red])
        assert store.count_media_tagged(lib) == 0
        store.add_tag_to_media(library["media"][0], red)  # source -> NULL
        assert store.count_media_tagged(lib) == 1

    def test_disabled_thumbs_step_writes_no_jpeg(self, library):
        """Thumbs off: a full index probes geometry but writes no thumbnail."""
        from server.runners import library as runner  # pylint: disable=C0415

        settings.save_settings(
            {
                "index_steps": {
                    index_steps.THUMBS: False,
                    index_steps.QUALITY: False,
                    index_steps.EMBED: False,
                    index_steps.WD14: False,
                }
            }
        )
        assert thumbnails.cached_sha256() == set()
        runner.index_body(None, None, False)(lambda **_: None)
        assert thumbnails.cached_sha256() == set()
        assert store.count_media_without_hash() == 0

    def test_reindex_all_processes_newly_scanned_media(
        self, library, tmp_path
    ):
        """A file appearing between runs is scanned then indexed at once.

        The chained rescan + re-index job must index the media its own scan
        phase just ingested — the plan is built after the scan, not before.
        """
        from server.runners import (
            library as runner,
        )  # noqa: E501 pylint: disable=C0415

        settings.save_settings(
            {
                "index_steps": {
                    index_steps.QUALITY: False,
                    index_steps.EMBED: False,
                    index_steps.WD14: False,
                }
            }
        )
        Image.new("RGB", (48, 48), (30, 200, 30)).save(tmp_path / "c.png")
        assert store.count_live_media(library["id"]) == 2
        runner.full_reindex_body()(lambda **_: None)
        assert store.count_live_media(library["id"]) == 3
        assert store.count_media_without_hash() == 0

    def test_thumbnails_are_counted_from_the_cache(self, library, tmp_path):
        """The thumbs step reads the cache, not a database column."""
        assert thumbnails.cached_sha256() == set()
        media = store.get_media(library["media"][0])
        thumbnails.ensure_thumbnail(
            store.effective_file(media["id"]), media["sha256"]
        )
        cached = thumbnails.cached_sha256()
        assert cached == {media["sha256"]}
        hashes = store.live_media_sha256(library["id"])
        assert len(hashes) == 2
        assert sum(1 for sha in hashes if sha in cached) == 1


class TestIndexStatusRoute:
    """``GET /api/libraries/index-status`` feeds the Libraries panels."""

    @pytest.fixture(name="client")
    def _client(self, library):
        """Return a TestClient bound to the seeded library."""
        from server.main import app  # pylint: disable=C0415

        with TestClient(app) as test_client:
            yield test_client, library

    def test_it_reports_every_step_with_its_machine_toggle(self, client):
        """A step disabled in Settings is reported off, not missing."""
        test_client, _ = client
        settings.save_settings({"index_steps": {index_steps.WD14: False}})
        body = test_client.get("/api/libraries/index-status").json()
        steps = {step["key"]: step for step in body["steps"]}
        assert list(steps) == list(index_steps.STEP_KEYS)
        assert steps[index_steps.WD14]["enabled"] is False
        assert steps[index_steps.EMBED]["enabled"] is True

    def test_it_counts_each_step_per_library_and_overall(self, client):
        """Nothing is indexed yet: every step misses every media."""
        test_client, library = client
        body = test_client.get("/api/libraries/index-status").json()
        assert body["totals"][index_steps.THUMBS] == {"done": 0, "total": 2}
        library_row = next(
            row for row in body["libraries"] if row["id"] == library["id"]
        )
        assert library_row["steps"][index_steps.QUALITY]["total"] == 2
        assert library_row["steps"][index_steps.EMBED]["done"] == 0

    def test_a_run_queues_one_chained_job(self, client):
        """The Index route returns a single job id, scoped to the request."""
        test_client, library = client
        body = test_client.post(
            "/api/libraries/index",
            json={
                "library_id": library["id"],
                "steps": [index_steps.THUMBS],
                "force": False,
            },
        ).json()
        assert body["job_id"]

    def test_rescan_all_queues_one_job(self, client):
        """The rescan-all route sweeps every library under one job id."""
        test_client, _ = client
        body = test_client.post("/api/libraries/scan-all").json()
        assert body["job_id"]

    def test_reindex_all_queues_one_job(self, client):
        """The reindex-all route chains scan + index under one job id."""
        test_client, _ = client
        body = test_client.post("/api/libraries/reindex-all").json()
        assert body["job_id"]
