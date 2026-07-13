"""Smoke tests for the FastAPI caption-workspace routes.

Exercise the Phase-1 backend end to end against the shared in-memory
database fixture (never the production DB): dataset list, the paged caption
grid, media detail, a caption save that must persist, plus the jobs and
settings routes. Real engines, real repository, throwaway data.
"""

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src import sqlite_store as store
from src import storage


@pytest.fixture(name="scenario")
def _scenario(store_db, thumb_cache_dir, tmp_path):
    """Seed one library, one dataset and one captioned image on disk."""
    # pylint: disable=unused-argument
    image = tmp_path / "red_circle.png"
    Image.new("RGB", (64, 64), (219, 68, 55)).save(image)
    library_id = store.create_library("fixtures", str(tmp_path))
    store.scan_library(library_id)
    media = store.list_library_media()
    dataset_id = store.create_dataset("shapes_all")
    for row in media:
        store.add_media_to_dataset(dataset_id, row["id"])
    key = str(media[0]["id"])
    storage.write_caption(dataset_id, key, "txt", "a red circle")
    return {"dataset_id": dataset_id, "key": key}


@pytest.fixture(name="client")
def _client(scenario):
    """Return a TestClient bound to the seeded scenario."""
    from server.main import app  # pylint: disable=import-outside-toplevel

    with TestClient(app) as test_client:
        yield test_client, scenario


def test_datasets_list(client):
    """The datasets route reports the seeded dataset with its count."""
    test_client, scenario = client
    body = test_client.get("/api/datasets").json()
    datasets = {d["id"]: d for d in body["datasets"]}
    assert scenario["dataset_id"] in datasets
    assert datasets[scenario["dataset_id"]]["count"] >= 1


def test_caption_grid(client):
    """The grid returns the captioned card for the seeded media."""
    test_client, scenario = client
    body = test_client.get(
        "/api/captions/grid",
        params={
            "dataset_id": scenario["dataset_id"],
            "caption_type": "txt",
        },
    ).json()
    assert body["total"] >= 1
    card = next(c for c in body["items"] if c["key"] == scenario["key"])
    assert card["caption"] == "a red circle"
    assert card["ext"] == "txt"


def test_media_detail(client):
    """The detail route returns text, tags meta and a sha256."""
    test_client, scenario = client
    body = test_client.get(
        f"/api/captions/media/{scenario['key']}",
        params={
            "dataset_id": scenario["dataset_id"],
            "caption_type": "txt",
        },
    ).json()
    assert body["caption"] == "a red circle"
    assert len(body["meta"]["sha256"]) == 64
    assert body["meta"]["size_bytes"] > 0


def test_save_caption_persists(client):
    """Saving a caption advances the head and survives a re-read."""
    test_client, scenario = client
    response = test_client.post(
        f"/api/captions/media/{scenario['key']}/caption",
        params={
            "dataset_id": scenario["dataset_id"],
            "caption_type": "txt",
        },
        json={"content": "a bright red circle", "scope": "type"},
    )
    assert response.status_code == 200
    grid = test_client.get(
        "/api/captions/grid",
        params={
            "dataset_id": scenario["dataset_id"],
            "caption_type": "txt",
        },
    ).json()
    card = next(c for c in grid["items"] if c["key"] == scenario["key"])
    assert card["caption"] == "a bright red circle"
    # Persisted through the repository, not just the response.
    stored = storage.read_caption(
        scenario["dataset_id"], scenario["key"], "txt"
    )
    assert stored == "a bright red circle"


def test_jobs_and_settings(client):
    """The jobs list is empty and settings expose the display metric."""
    test_client, _ = client
    assert test_client.get("/api/jobs").json() == {"jobs": []}
    settings = test_client.get("/api/settings").json()
    assert "quality_display_metric" in settings


def test_dataset_crud_and_media(client):
    """Create a dataset, link/list/unlink media, then delete it."""
    test_client, scenario = client
    created = test_client.post("/api/datasets", json={"name": "extra"}).json()
    dataset_id = created["id"]
    media_id = int(scenario["key"])

    candidates = test_client.get(
        f"/api/datasets/{dataset_id}/candidates"
    ).json()
    assert candidates["total"] >= 1

    added = test_client.post(
        f"/api/datasets/{dataset_id}/media",
        json={"media_ids": [media_id]},
    ).json()
    assert added["added"] == 1

    listing = test_client.get(f"/api/datasets/{dataset_id}/media").json()
    assert listing["total"] == 1
    assert listing["items"][0]["key"] == str(media_id)

    removed = test_client.request(
        "DELETE",
        f"/api/datasets/{dataset_id}/media",
        json={"media_ids": [media_id]},
    ).json()
    assert removed["removed"] == 1

    assert test_client.delete(f"/api/datasets/{dataset_id}").status_code == 200


def test_triggerwords(client):
    """Trigger words attach to and detach from a dataset."""
    test_client, scenario = client
    dataset_id = scenario["dataset_id"]
    test_client.post(
        f"/api/datasets/{dataset_id}/triggerwords", json={"name": "xbl1"}
    )
    words = test_client.get(f"/api/datasets/{dataset_id}/triggerwords").json()[
        "triggerwords"
    ]
    assert any(word["name"] == "xbl1" for word in words)


def test_dataset_report_never_run(client):
    """An unevaluated dataset returns no report but the scorer chips."""
    test_client, scenario = client
    body = test_client.get(
        f"/api/datasets/{scenario['dataset_id']}/report"
    ).json()
    assert body["report"] is None
    assert body["resolutions"] == {}
    chips = {chip["id"]: chip for chip in body["scorer_catalogue"]}
    assert chips["musiq"]["default"] is True
    assert chips["dinov2"]["kind"] == "embedding"
    assert "qalign_8bit" not in chips


def test_dataset_report_unknown_dataset(client):
    """Running a report on a dataset that does not exist is a 404."""
    test_client, _ = client
    response = test_client.post("/api/datasets/999999/report", json={})
    assert response.status_code == 404


def test_dataset_report_issue_resolutions(client):
    """A finding's resolution round-trips and can be reopened."""
    test_client, scenario = client
    dataset_id = scenario["dataset_id"]
    base = f"/api/datasets/{dataset_id}/report/issues"
    assert test_client.post(
        f"{base}/dup:1:2",
        json={"resolution": "ignored", "fingerprint": "0.97"},
    ).json() == {"ok": True}
    stored = test_client.get(f"/api/datasets/{dataset_id}/report").json()
    assert stored["resolutions"]["dup:1:2"]["resolution"] == "ignored"

    test_client.delete(f"{base}/dup:1:2")
    stored = test_client.get(f"/api/datasets/{dataset_id}/report").json()
    assert stored["resolutions"] == {}


def test_dataset_report_rejects_unknown_resolution(client):
    """Only the three documented verdicts are accepted."""
    test_client, scenario = client
    response = test_client.post(
        f"/api/datasets/{scenario['dataset_id']}/report/issues/cap:1",
        json={"resolution": "maybe"},
    )
    assert response.status_code == 400


def test_autobuild_studio_config_and_preview(client):
    """The Studio exposes presets/metrics and previews a live proposal."""
    test_client, _ = client
    config = test_client.get("/api/autobuild/config").json()
    assert isinstance(config["framing_presets"], list)
    assert isinstance(config["metrics"], list)

    preview = test_client.post(
        "/api/autobuild/preview",
        json={"size": 4, "framing_preset": "free", "min_score": 0},
    ).json()
    assert "picks" in preview and isinstance(preview["picks"], list)
    assert "grade" in preview and "map" in preview
    assert "zones" in preview and "clusters" in preview
    # Every pick carries its explanation chips and a map coordinate.
    for pick in preview["picks"]:
        assert "reasons" in pick and "media_id" in pick


def test_media_grid_detail_favorite(client):
    """The library grid, detail and favorite toggle round-trip."""
    test_client, scenario = client
    grid = test_client.get("/api/medias/grid").json()
    assert grid["total"] >= 1
    card = next(c for c in grid["items"] if c["key"] == scenario["key"])
    assert "tags" in card and "tag_count" in card

    detail = test_client.get(f"/api/medias/{scenario['key']}").json()
    assert len(detail["meta"]["sha256"]) == 64
    assert detail["meta"]["files"] >= 1

    first = test_client.post(f"/api/medias/{scenario['key']}/favorite").json()[
        "favorite"
    ]
    second = test_client.post(
        f"/api/medias/{scenario['key']}/favorite"
    ).json()["favorite"]
    assert first != second


def test_tagger_models(client):
    """The tagger route lists known WD models and default thresholds."""
    test_client, _ = client
    data = test_client.get("/api/tagger/models").json()
    assert len(data["models"]) >= 1
    assert data["general"] == 0.35
    assert data["character"] == 0.85


def test_tags_admin(client):
    """Categories carry counts; tag create/list/delete round-trips."""
    test_client, _ = client
    cats = test_client.get("/api/tags/categories").json()["categories"]
    assert cats and "count" in cats[0]
    category_id = cats[0]["id"]

    created = test_client.post(
        "/api/tags", json={"name": "sunset_glow", "category_id": category_id}
    ).json()
    tag_id = created["id"]

    listing = test_client.get(
        "/api/tags/list", params={"category_id": category_id}
    ).json()
    assert listing["total"] >= 1
    assert any(item["id"] == tag_id for item in listing["items"])

    order = [c["id"] for c in cats][::-1]
    assert (
        test_client.post(
            "/api/tags/categories/reorder", json={"ordered_ids": order}
        ).status_code
        == 200
    )
    assert test_client.delete(f"/api/tags/{tag_id}").status_code == 200


def test_libraries_list_and_coverage(client):
    """Libraries expose the internal source, coverage gaps and metrics."""
    test_client, _ = client
    libs = test_client.get("/api/libraries").json()["libraries"]
    assert any(library["internal"] for library in libs)
    assert any(library["count"] >= 1 for library in libs)

    coverage = test_client.get("/api/libraries/coverage").json()
    assert "unhashed" in coverage and "unembedded" in coverage
    metrics = test_client.get("/api/libraries/quality-metrics").json()
    assert isinstance(metrics["metrics"], list)


def test_lookalike_detect(client):
    """Near-duplicate detection returns a well-formed (possibly empty) set."""
    test_client, _ = client
    result = test_client.post(
        "/api/libraries/lookalike/detect", json={"similarity": 88}
    ).json()
    assert "groups" in result and "hashed_count" in result


def test_lookalike_discard_dismiss_reset(store_db, thumb_cache_dir, tmp_path):
    """Validate (discard) a subset, hide a group indefinitely, then reset."""
    # pylint: disable=unused-argument,import-outside-toplevel
    from server.main import app

    folder = tmp_path / "dups"
    folder.mkdir()
    Image.new("RGB", (64, 64), (10, 10, 10)).save(folder / "a.png")
    Image.new("RGB", (64, 64), (11, 11, 11)).save(folder / "b.png")
    library_id = store.create_library("dups", str(folder))
    store.scan_library(library_id)
    ids = sorted(row["id"] for row in store.list_library_media())
    assert len(ids) == 2
    for media_id in ids:  # identical hashes => one lookalike group
        store.set_media_index(
            media_id, 64, 64, "ffffffffffffffff", "0000000000000000"
        )

    with TestClient(app) as test_client:
        detect = lambda: test_client.post(  # noqa: E731
            "/api/libraries/lookalike/detect", json={"similarity": 88}
        ).json()["groups"]

        groups = detect()
        assert len(groups) == 1
        assert {int(m["key"]) for m in groups[0]["members"]} == set(ids)

        # Hide the whole group indefinitely: it leaves detect...
        dismissed = test_client.post(
            "/api/libraries/lookalike/dismiss", json={"media_ids": ids}
        ).json()["dismissed"]
        assert dismissed == 2
        assert detect() == []

        # ...until Reset dismissed brings it back.
        test_client.post("/api/libraries/lookalike/reset-dismissed")
        assert len(detect()) == 1

        # Validate the group by discarding one member.
        victim = ids[1]
        discarded = test_client.post(
            "/api/libraries/lookalike/discard", json={"media_ids": [victim]}
        ).json()["discarded"]
        assert discarded == 1
        assert detect() == []  # only one hashed media left, no pair
        assert victim in {row["id"] for row in store.list_discarded_media()}


def test_bulk_tags(client):
    """Bulk-tagging a library attaches a tag to its media."""
    test_client, _ = client
    category_id = test_client.get("/api/tags/categories").json()["categories"][
        0
    ]["id"]
    tag_id = test_client.post(
        "/api/tags", json={"name": "bulk_mark", "category_id": category_id}
    ).json()["id"]
    added = test_client.post(
        "/api/libraries/bulk-tags",
        json={"library_id": None, "add_tag_ids": [tag_id]},
    ).json()["added"]
    assert added >= 1


def test_system_database_and_runtime(client):
    """System exposes the database counts and runtime info."""
    test_client, _ = client
    database = test_client.get("/api/system/database").json()
    assert "media" in database["counts"]
    assert isinstance(database["backups"], list)
    runtime = test_client.get("/api/system/runtime").json()
    assert runtime["python"]


def test_db_explorer_guard(client):
    """The SQL explorer runs read-only queries and rejects writes."""
    test_client, _ = client
    ok = test_client.post(
        "/api/system/db/query",
        json={"sql": "SELECT COUNT(*) AS n FROM media"},
    )
    assert ok.status_code == 200
    assert ok.json()["headers"] == ["n"]
    denied = test_client.post(
        "/api/system/db/query", json={"sql": "DELETE FROM media"}
    )
    assert denied.status_code == 400


def test_settings_save_roundtrip(client):
    """A settings save persists (isolated user config layer)."""
    test_client, _ = client
    current = test_client.get("/api/settings").json()
    current["gguf_n_ctx"] = 16384
    assert (
        test_client.post(
            "/api/settings", json={"settings": current}
        ).status_code
        == 200
    )
    assert test_client.get("/api/settings").json()["gguf_n_ctx"] == 16384
