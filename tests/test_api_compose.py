"""Smoke tests for the dataset composer routes.

Real engines, real repository, throwaway database: one library of three
images, one dataset holding the first, and the composer asked for its
candidates and for the live preview of a selection. SigLIP is never
loaded — no image carries a semantic vector, so the search box reports
itself unavailable and the query is skipped.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src import embeddings
from src import sqlite_store as store

_COLORS = ((219, 68, 55), (60, 180, 75), (66, 133, 244))


def _vector(index: int) -> bytes:
    """Return a distinct unit vector as the stored float32 blob."""
    vector = np.zeros(embeddings.VECTOR_DIM, dtype=np.float32)
    vector[index] = 1.0
    return embeddings.vector_to_blob(vector)


@pytest.fixture(name="scenario")
def _scenario(store_db, thumb_cache_dir, tmp_path):
    """Seed three scored, embedded images; the first one is in a dataset."""
    # pylint: disable=unused-argument
    for index, color in enumerate(_COLORS):
        image = tmp_path / f"shape_{index}.png"
        Image.new("RGB", (1600, 2400), color).save(image)
    library_id = store.create_library("fixtures", str(tmp_path))
    store.scan_library(library_id)
    media = sorted(store.list_library_media(), key=lambda row: row["name"])
    for index, row in enumerate(media):
        store.set_media_index(row["id"], 1600, 2400, f"{index}" * 16, "a" * 16)
        store.set_media_stats(
            row["id"],
            {"sharpness": 80.0, "clipping": 1.0, "cleanliness": 90.0},
        )
        store.upsert_media_quality(row["id"], "musiq", 40.0 + index * 25.0)
        store.upsert_media_embedding(
            row["id"], embeddings.MODEL_ID, _vector(index)
        )
    dataset_id = store.create_dataset("shapes")
    store.add_media_to_dataset(dataset_id, media[0]["id"])
    return {
        "dataset_id": dataset_id,
        "ids": [row["id"] for row in media],
    }


@pytest.fixture(name="client")
def _client(scenario):
    """Return a TestClient bound to the seeded scenario."""
    from server.main import app  # pylint: disable=import-outside-toplevel

    with TestClient(app) as test_client:
        yield test_client, scenario


def _candidates(client, dataset_id, **params):
    """GET the composer's candidates and return the decoded payload."""
    response = client.get(
        f"/api/datasets/{dataset_id}/candidates", params=params
    )
    assert response.status_code == 200
    return response.json()


def test_candidates_annotate_every_card(client):
    """Each candidate carries its score, gain, map point and dup alert."""
    test_client, scenario = client
    body = _candidates(test_client, scenario["dataset_id"], metric="musiq")
    assert body["total"] == 2
    assert body["pool"] == 2
    assert len(body["pool_points"]) == 2
    assert body["semantic_available"] is False
    card = body["items"][0]
    assert card["score"] is not None
    assert 0.0 <= card["gain"] <= 1.0
    assert card["near_dup"] is None
    assert len(card["xy"]) == 2
    assert card["metric"] == "musiq"


def test_candidates_sort_by_quality_by_default(client):
    """The best-scored candidate leads the grid."""
    test_client, scenario = client
    body = _candidates(test_client, scenario["dataset_id"], metric="musiq")
    scores = [card["score"] for card in body["items"]]
    assert scores == sorted(scores, reverse=True)


def test_candidates_honor_the_score_floor(client):
    """A floor above the weakest candidate drops it."""
    test_client, scenario = client
    body = _candidates(
        test_client, scenario["dataset_id"], metric="musiq", min_score=80
    )
    assert body["total"] == 1


def test_candidates_keep_a_selected_media_under_the_floor(client):
    """A picked candidate survives a floor it no longer passes."""
    test_client, scenario = client
    weakest = scenario["ids"][1]
    body = _candidates(
        test_client,
        scenario["dataset_id"],
        metric="musiq",
        min_score=95,
        selected_ids=[weakest],
    )
    assert weakest in [int(card["key"]) for card in body["items"]]


def test_candidates_page_the_filtered_set(client):
    """Offset and limit slice the ranked rows, total counts them all."""
    test_client, scenario = client
    body = _candidates(test_client, scenario["dataset_id"], limit=1)
    assert body["total"] == 2
    assert len(body["items"]) == 1


def test_a_semantic_query_is_skipped_without_vectors(client):
    """No SigLIP vector means no filtering, and no model load."""
    test_client, scenario = client
    body = _candidates(
        test_client, scenario["dataset_id"], semantic_q="a red circle"
    )
    assert body["total"] == 2
    assert body["semantic_available"] is False


def test_compose_preview_projects_the_selection(client):
    """Adding a stronger image lifts the projected score."""
    test_client, scenario = client
    dataset_id = scenario["dataset_id"]
    response = test_client.post(
        f"/api/datasets/{dataset_id}/compose/preview",
        json={"selected_media_ids": [scenario["ids"][2]], "metric": "musiq"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["size"] == {
        "base": 1,
        "picked": 1,
        "total": 2,
        "min": body["size"]["min"],
        "max": body["size"]["max"],
        "percent": body["size"]["percent"],
        "over": False,
    }
    assert body["delta"] > 0
    assert body["grade"] in ("A", "B+", "B", "C", "D")
    assert body["pillars"]["duplicates"] == 0
    assert body["map"]["dataset"] and body["map"]["selected"]
    assert body["advice"]


def test_compose_preview_without_a_selection_has_no_delta(client):
    """An empty selection projects the dataset exactly as it is."""
    test_client, scenario = client
    response = test_client.post(
        f"/api/datasets/{scenario['dataset_id']}/compose/preview",
        json={"selected_media_ids": [], "metric": "musiq"},
    )
    body = response.json()
    assert body["delta"] == pytest.approx(0.0)
    assert body["size"]["picked"] == 0


def test_compose_release_is_safe_without_a_loaded_model(client):
    """Closing the composer never fails, loaded checkpoint or not."""
    test_client, _scenario = client
    assert test_client.post("/api/datasets/compose/release").json() == {
        "ok": True
    }
