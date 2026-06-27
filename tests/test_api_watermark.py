"""End-to-end tests for the Watermark Lab v2: scan, patch, flatten, deploy.

Real repository, real routes and real composition against the shared in-memory
database and throwaway caches. A watermark travels from an OWLv2-detected box
to a patch PNG composed over the original — and, on demand, baked into the
source file — without the source ever being touched by default. No model is
downloaded: the OWLv2 detector is stubbed to a fixed box and the FLUX.2 klein
edit is faked at its seam (:func:`src.watermark_flux.edit`), so the whole
plumbing runs without weights.
"""

import hashlib
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src import deploy, settings, watermark_flux, wm_compose
from src import sqlite_store as store
from src import storage, watermark
from src import watermark_detect as detect
from server.runners import watermark as wm_runner


class _Progress:
    """A no-op progress reporter (the queue is bypassed in these tests)."""

    def __call__(self, done=None, total=None, sub=None):
        # pylint: disable=unused-argument
        return None


@pytest.fixture(name="wm_cache", autouse=True)
def _wm_cache(monkeypatch, tmp_path):
    """Redirect the patch, composite and flatten-backup caches to tmp."""
    monkeypatch.setattr(
        wm_compose, "get_patches_dir", lambda: tmp_path / "patches"
    )
    monkeypatch.setattr(
        wm_compose, "get_composed_dir", lambda: tmp_path / "composed"
    )
    monkeypatch.setattr(
        watermark, "WATERMARK_BACKUPS_DIR", tmp_path / "wm_backups"
    )
    return tmp_path


@pytest.fixture(name="fake_models", autouse=True)
def _fake_models(monkeypatch):
    """Stub OWLv2 detection and the FLUX pipeline so no weights are needed."""

    def _detect_owlv2(source_path, queries, confidence_min=70.0):
        # pylint: disable=unused-argument
        return [
            {
                "box": {"x": 0.25, "y": 0.80, "w": 0.5, "h": 0.12},
                "score": 88.0,
                "detector": "owlv2",
                "query": (queries or ["watermark"])[0],
            }
        ]

    monkeypatch.setattr(detect, "load_owlv2", lambda *a, **k: (None, None))
    monkeypatch.setattr(detect, "unload_owlv2", lambda: None)
    monkeypatch.setattr(detect, "detect_owlv2", _detect_owlv2)

    def _edit(source_path, box, **_kwargs):
        # pylint: disable=unused-argument
        return Image.new("RGB", (64, 64), (0, 200, 0))

    monkeypatch.setattr(watermark_flux, "edit", _edit)
    monkeypatch.setattr(watermark_flux, "load_model", lambda prefs: None)
    monkeypatch.setattr(watermark_flux, "unload_model", lambda: None)


@pytest.fixture(name="scenario")
def _scenario(store_db, thumb_cache_dir, monkeypatch, tmp_path):
    """Seed one 400x300 image with a 'watermark' tag, in a library+dataset."""
    # pylint: disable=unused-argument
    monkeypatch.setattr(deploy, "get_deploy_dir", lambda: tmp_path / "out")
    images = tmp_path / "pics"
    images.mkdir()
    Image.new("RGB", (400, 300), (219, 68, 55)).save(images / "photo.png")
    library_id = store.create_library("fixtures", str(images))
    store.scan_library(library_id)
    media_id = store.list_library_media()[0]["id"]
    dataset_id = store.create_dataset("shapes")
    store.add_media_to_dataset(dataset_id, media_id)
    storage.write_caption(dataset_id, str(media_id), "txt", "a red square")
    category_id = store.get_or_create_uncategorized_category()
    tag_id = store.get_or_create_tag("watermark", category_id)
    store.add_tag_to_media(media_id, tag_id)
    settings.set_watermark_prefs({"detector": "owlv2", "tag_cleanup": True})
    return {"dataset_id": dataset_id, "media_id": media_id}


@pytest.fixture(name="client")
def _client(scenario):
    """Return a TestClient bound to the seeded scenario."""
    from server.main import app  # pylint: disable=import-outside-toplevel

    with TestClient(app) as test_client:
        yield test_client, scenario


def _run_job(test_client, job_id):
    """Poll a job route until the single worker finishes it; return result."""
    for _ in range(400):
        body = test_client.get(f"/api/jobs/{job_id}/result").json()
        if body["state"] in ("done", "stopped", "error"):
            return body
    raise AssertionError("job never finished")


class TestOwlv2Nms:
    """OWLv2 non-max suppression merges overlapping boxes on one mark."""

    def test_nms_keeps_top_and_drops_nested(self):
        """Top box survives; a nested sub-box and a twin are dropped."""
        big = {"box": {"x": 0.0, "y": 0.9, "w": 0.16, "h": 0.1}, "score": 40}
        nested = {
            "box": {"x": 0.05, "y": 0.92, "w": 0.05, "h": 0.05},
            "score": 23,
        }
        twin = {
            "box": {"x": 0.005, "y": 0.9, "w": 0.16, "h": 0.1},
            "score": 10,
        }
        far = {"box": {"x": 0.7, "y": 0.0, "w": 0.1, "h": 0.1}, "score": 15}
        kept = detect._nms([nested, big, twin, far])
        scores = sorted(d["score"] for d in kept)
        assert scores == [15, 40]


class TestScan:
    """Detection is decoupled from patching: a scan only locates boxes."""

    def test_scan_detects_without_patching(self, scenario):
        """The scan creates a detected OWLv2 zone and patches nothing."""
        media_id = scenario["media_id"]
        summary = wm_runner.scan_body([media_id])(_Progress())
        assert summary["detected"] == 1
        assert summary["patched"] == 0
        zones = store.list_zones(media_id)
        assert len(zones) == 1
        assert zones[0]["status"] == store.STATUS_DETECTED
        assert zones[0]["detector"] == "owlv2"
        assert zones[0]["query"] == "watermark"

    def test_scan_is_idempotent(self, scenario):
        """A second scan never duplicates zones on an already-touched media."""
        wm_runner.scan_body([scenario["media_id"]])(_Progress())
        wm_runner.scan_body([scenario["media_id"]])(_Progress())
        assert len(store.list_zones(scenario["media_id"])) == 1

    def test_scan_and_patch_cleans_tags(self, scenario):
        """Scan+patch erases the zone and strips the watermark tag."""
        media_id = scenario["media_id"]
        summary = wm_runner.scan_and_patch_body([media_id])(_Progress())
        assert summary["patched"] == 1
        zones = store.list_zones(media_id)
        assert zones[0]["status"] == store.STATUS_PATCHED
        assert "watermark" not in store.media_tag_names(media_id)


class TestScanJobAndInventory:
    """The scan through the real queue, then the tabbed inventory."""

    def test_scan_job_and_tabs(self, client):
        """POST /scan runs, then the media shows under the Watermarked tab."""
        test_client, scenario = client
        job_id = test_client.post(
            "/api/watermarks/scan",
            json={"select_all": True, "tab": "media"},
        ).json()["job_id"]
        body = _run_job(test_client, job_id)
        assert body["state"] == "done"
        assert body["result"]["detected"] == 1
        inv = test_client.get("/api/watermarks?tab=watermarked").json()
        assert inv["counts"]["watermarked"] == 1
        assert scenario["media_id"] in [i["media_id"] for i in inv["items"]]
        # Nothing patched yet: the Patched tab is empty.
        assert inv["counts"]["patched"] == 0


class TestComposition:
    """Non-destructive composition: patch over the original, source intact."""

    def test_display_source_is_a_composite(self, scenario):
        """After patching, the display file is a composite, not the source."""
        media_id = scenario["media_id"]
        wm_runner.scan_and_patch_body([media_id])(_Progress())
        original = store.effective_file(media_id)
        path, composed_sha = watermark.display_source(media_id)
        assert composed_sha is not None
        assert path != original
        assert os.path.exists(path)
        assert os.path.exists(original)


class TestReviewActions:
    """The synchronous review actions the Lab drives."""

    def test_manual_zone_then_regenerate(self, client):
        """A manual zone starts patch-less, then a FLUX job fills it."""
        test_client, scenario = client
        media_id = scenario["media_id"]
        created = test_client.post(
            f"/api/watermarks/{media_id}/zones",
            json={"box": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.2}},
        )
        assert created.status_code == 200
        zone_id = created.json()["zone_id"]
        assert created.json()["media"]["status"] == "detected"
        regen = test_client.post(
            f"/api/watermarks/{zone_id}/regenerate",
            json={"prompt": "erase the signature"},
        )
        body = _run_job(test_client, regen.json()["job_id"])
        assert body["state"] == "done"
        media = test_client.get(f"/api/watermarks/{media_id}").json()
        assert media["status"] == "patched"
        assert media["zones"][0]["prompt"] == "erase the signature"

    def test_dismiss_makes_media_clean_again(self, client):
        """Dismiss deletes every zone; the media leaves the inventory."""
        test_client, scenario = client
        media_id = scenario["media_id"]
        wm_runner.scan_body([media_id])(_Progress())
        response = test_client.delete(f"/api/watermarks/media/{media_id}")
        assert response.json()["media"]["status"] is None
        assert store.list_zones(media_id) == []

    def test_revert_patch_back_to_watermarked(self, client):
        """Reverting a patched media drops its patches (back to detected)."""
        test_client, scenario = client
        media_id = scenario["media_id"]
        wm_runner.scan_and_patch_body([media_id])(_Progress())
        reverted = test_client.post(f"/api/watermarks/{media_id}/revert")
        assert reverted.json()["media"]["status"] == "detected"

    def test_config_round_trips_queries(self, client):
        """A prefs PATCH of the OWLv2 queries persists and reads back."""
        test_client, _ = client
        patched = test_client.patch(
            "/api/watermarks/config",
            json={"owlv2_queries": ["logo", "signature"]},
        )
        assert patched.json()["prefs"]["owlv2_queries"] == [
            "logo",
            "signature",
        ]
        got = test_client.get("/api/watermarks/config").json()
        assert got["prefs"]["owlv2_queries"] == ["logo", "signature"]


class TestFlatten:
    """Flatten bakes the patches into the source file, reversibly."""

    def test_flatten_and_unflatten(self, scenario):
        """Flatten rewrites the source, swaps its sha; unflatten restores."""
        media_id = scenario["media_id"]
        wm_runner.scan_and_patch_body([media_id])(_Progress())
        source = Path(store.effective_file(media_id))
        original_bytes = source.read_bytes()
        original_sha = store.get_media(media_id)["sha256"]

        assert watermark.flatten_media(media_id) is True
        assert store.is_flattened(media_id) is True
        baked_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        assert baked_sha != original_sha
        assert store.get_media(media_id)["sha256"] == baked_sha
        # A flattened media serves its (baked) source directly, not a compose.
        path, composed_sha = watermark.display_source(media_id)
        assert composed_sha is None
        assert path == str(source)

        assert watermark.unflatten_media(media_id) is True
        assert store.is_flattened(media_id) is False
        assert source.read_bytes() == original_bytes
        assert store.get_media(media_id)["sha256"] == original_sha


class TestDeploy:
    """The deploy adapter composites patched media (no exclusion anymore)."""

    def test_deploy_items_composite(self, scenario):
        """A patched media deploys a composite; it is never hidden by us."""
        # pylint: disable=import-outside-toplevel
        from server.routers.deploy import _deploy_items

        media_id = scenario["media_id"]
        wm_runner.scan_and_patch_body([media_id])(_Progress())
        items = _deploy_items(scenario["dataset_id"], "txt")
        item = next(i for i in items if int(i["key"]) == media_id)
        assert item["path"].endswith(".png")
        assert "composed" in item["path"]
        assert item["hidden"] is False
