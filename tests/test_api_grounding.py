"""Smoke tests for the FastAPI SigLIP-grounding routes.

Exercised end to end against the shared in-memory database (never the
production DB) with both models mocked at their engine seams — the LLM at
:func:`src.caption_claims.extract_claims`, SigLIP at
:func:`src.siglip_grounding.ground_image`. Real repository, real routes,
real job queue; no weights.
"""

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src import caption_claims, settings, siglip_grounding
from src import sqlite_store as store
from src import storage


@pytest.fixture(name="scenario")
def _scenario(store_db, thumb_cache_dir, tmp_path):
    """Seed a dataset with one captioned, tagged image on disk."""
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
    storage.write_caption(dataset_id, key, "txt", "a red circle on grass")
    category_id = store.create_tag_category("general", "#fff")
    tag_id = store.get_or_create_tag("horse", category_id)
    store.add_tag_to_media(int(key), tag_id)
    return {"dataset_id": dataset_id, "key": key, "tag_id": tag_id}


@pytest.fixture(name="mocked_models")
def _mocked_models(monkeypatch):
    """Replace the VLM and SigLIP with deterministic stand-ins.

    ``load_model`` answers with the *configured* checkpoint id, exactly as
    the real one does — the readers filter their scores on it, so a fake id
    here would make every grounding read back as belonging to another model.
    """
    monkeypatch.setattr(
        caption_claims,
        "extract_claims",
        lambda *a: [
            {"text": "a red circle", "kind": "object"},
            {"text": "two dogs", "kind": "count"},
        ],
    )
    scores = {"a red circle": 91.0, "two dogs": 12.0}

    def _ground(_path, texts, with_heat=True):
        return [
            {
                "text": text,
                "score": scores.get(text, 44.0),
                "heat": "AAAA" if with_heat else None,
                "side": 2,
            }
            for text in texts
        ]

    # The caption grounding job first ensures a VLM is loaded to decompose
    # the caption; pretend one already is, so it uses the mocked
    # extract_claims above instead of trying to auto-load a real model. It
    # then frees the VLM before SigLIP — a no-op generator stands in for the
    # real unload so no weights are touched.
    monkeypatch.setattr(
        "src.loader.is_model_loaded", lambda: True, raising=False
    )
    monkeypatch.setattr(
        "src.loader.unload_model",
        lambda: iter([("unloaded", False)]),
        raising=False,
    )

    monkeypatch.setattr(siglip_grounding, "ground_image", _ground)
    monkeypatch.setattr(
        siglip_grounding,
        "load_model",
        lambda *a: settings.get_grounding_model_id(),
    )
    monkeypatch.setattr(siglip_grounding, "unload_model", lambda: None)


@pytest.fixture(name="client")
def _client(scenario, mocked_models):
    """Return a TestClient bound to the seeded scenario."""
    # pylint: disable=unused-argument
    from server.main import app  # pylint: disable=import-outside-toplevel

    with TestClient(app) as test_client:
        yield test_client, scenario


def _run(test_client, method, url, **kwargs):
    """Submit a job route and block until the single worker finishes it."""
    job_id = getattr(test_client, method)(url, **kwargs).json()["job_id"]
    for _ in range(200):
        body = test_client.get(f"/api/jobs/{job_id}/result").json()
        if body["state"] in ("done", "stopped"):
            return body["result"]
    raise AssertionError("job never finished")


def test_grounding_config_exposes_the_checkpoint(client):
    """The modal reads the configured checkpoint, never a hard-coded one."""
    test_client, _ = client

    body = test_client.get("/api/grounding/config").json()

    assert body["model_id"].startswith("google/siglip2-")
    assert body["tag_prompt"] == "a photo that contains {tag}"
    assert 0 <= body["threshold_caption"] <= 100
    # The claim-splitting model default (empty = use the loaded VLM).
    assert body["claim_model"] == ""


def test_ground_caption_job_persists_scored_claims(client):
    """The caption job decomposes, scores, and stores its claims in order."""
    test_client, scenario = client

    result = _run(
        test_client,
        "post",
        "/api/grounding/caption",
        json={
            "dataset_id": scenario["dataset_id"],
            "key": scenario["key"],
            "caption_type": "txt",
        },
    )
    assert result["status"] == "ok"

    body = test_client.get(
        "/api/grounding/caption",
        params={
            "dataset_id": scenario["dataset_id"],
            "key": scenario["key"],
            "caption_type": "txt",
        },
    ).json()
    claims = body["grounding"]["claims"]
    assert [(c["text"], c["score"], c["kind"]) for c in claims] == [
        ("a red circle", 91.0, "object"),
        ("two dogs", 12.0, "count"),
    ]

    revision_id = storage.effective_revision_id(
        scenario["dataset_id"], scenario["key"], "txt"
    )
    stored = store.get_caption_grounding(revision_id)
    assert stored["model_id"] == settings.get_grounding_model_id()


def test_caption_card_summarises_the_grounding(client):
    """The card counts validated / flagged claims against the threshold."""
    test_client, scenario = client
    _run(
        test_client,
        "post",
        "/api/grounding/caption",
        json={
            "dataset_id": scenario["dataset_id"],
            "key": scenario["key"],
            "caption_type": "txt",
        },
    )

    card = test_client.get(
        "/api/captions/grid",
        params={"dataset_id": scenario["dataset_id"], "caption_type": "txt"},
    ).json()["items"][0]

    # 91 clears the default 55 threshold, 12 does not.
    assert card["grounding"]["validated"] == 1
    assert card["grounding"]["flagged"] == 1
    assert card["grounding"]["coverage"] == 50
    assert card["grounding"]["stale"] is False


def test_card_marks_another_checkpoints_scores_stale(client):
    """Another checkpoint's scores are flagged, never shown as current."""
    test_client, scenario = client
    revision_id = storage.effective_revision_id(
        scenario["dataset_id"], scenario["key"], "txt"
    )
    store.upsert_caption_grounding(
        revision_id,
        "google/siglip2-base-patch16-256",
        [{"text": "a red circle", "kind": "object", "score": 91.0}],
    )

    card = test_client.get(
        "/api/captions/grid",
        params={"dataset_id": scenario["dataset_id"], "caption_type": "txt"},
    ).json()["items"][0]

    assert card["grounding"]["stale"] is True


def test_ungrounded_card_carries_a_null_summary(client):
    """A caption nobody grounded shows no tiles rather than empty ones."""
    test_client, scenario = client

    card = test_client.get(
        "/api/captions/grid",
        params={"dataset_id": scenario["dataset_id"], "caption_type": "txt"},
    ).json()["items"][0]

    assert card["grounding"] is None


def test_reject_claim_toggles_the_flag(client):
    """Marking a claim non-validated persists and drops it from the count."""
    test_client, scenario = client
    _run(
        test_client,
        "post",
        "/api/grounding/caption",
        json={
            "dataset_id": scenario["dataset_id"],
            "key": scenario["key"],
            "caption_type": "txt",
        },
    )
    params = {
        "dataset_id": scenario["dataset_id"],
        "key": scenario["key"],
        "caption_type": "txt",
    }
    claims = test_client.get("/api/grounding/caption", params=params).json()
    weak = next(c for c in claims["grounding"]["claims"] if c["score"] == 12.0)

    test_client.post(
        "/api/grounding/claim/reject",
        json={"claim_id": weak["id"], "rejected": True},
    )

    card = test_client.get(
        "/api/captions/grid",
        params={"dataset_id": scenario["dataset_id"], "caption_type": "txt"},
    ).json()["items"][0]
    assert card["grounding"]["flagged"] == 0
    assert card["grounding"]["total"] == 1
    assert card["grounding"]["coverage"] == 100


def test_ungrounded_filter_pages_weak_captions(client):
    """The grid's "weak grounding" filter is resolved in SQL, not in Python."""
    test_client, scenario = client
    _run(
        test_client,
        "post",
        "/api/grounding/caption",
        json={
            "dataset_id": scenario["dataset_id"],
            "key": scenario["key"],
            "caption_type": "txt",
        },
    )
    params = {
        "dataset_id": scenario["dataset_id"],
        "caption_type": "txt",
        "review_filter": "ungrounded",
    }

    # "two dogs" scored 12, under the default threshold of 55.
    body = test_client.get("/api/captions/grid", params=params).json()
    assert body["total"] == 1
    assert body["items"][0]["key"] == scenario["key"]


def test_ground_tags_job_scores_every_attached_tag(client):
    """Each tag is scored through the pre-prompt; no LLM is involved."""
    test_client, scenario = client

    result = _run(
        test_client,
        "post",
        "/api/grounding/tags",
        json={"media_ids": [int(scenario["key"])]},
    )
    assert result == {"grounded": 1, "skipped": 0}

    body = test_client.get(
        "/api/grounding/tags", params={"key": scenario["key"]}
    ).json()
    assert body["tags"][0]["name"] == "horse"
    assert body["tags"][0]["score"] == 44.0


def test_remove_tag_detaches_the_hallucination(client):
    """ "Retirer le tag" really deletes the media_tag row and its scores."""
    test_client, scenario = client
    _run(
        test_client,
        "post",
        "/api/grounding/tags",
        json={"media_ids": [int(scenario["key"])]},
    )

    body = test_client.post(
        "/api/grounding/tags/remove",
        json={"key": scenario["key"], "tag_id": scenario["tag_id"]},
    ).json()

    assert body["tags"] == []
    assert store.tags_for_media(int(scenario["key"])) == []


def test_caption_heat_job_returns_a_grid_per_claim(client):
    """Heat maps travel as a job result, never through the progress socket."""
    test_client, scenario = client
    target = {
        "dataset_id": scenario["dataset_id"],
        "key": scenario["key"],
        "caption_type": "txt",
    }
    _run(test_client, "post", "/api/grounding/caption", json=target)

    result = _run(
        test_client, "post", "/api/grounding/caption/heat", json=target
    )

    assert result["model_id"] == settings.get_grounding_model_id()
    assert [element["heat"] for element in result["elements"]] == [
        "AAAA",
        "AAAA",
    ]
    assert result["elements"][0]["side"] == 2


def test_tag_heat_job_returns_a_grid_per_tag(client):
    """The Media modal's maps are rebuilt the same way, without the LLM."""
    test_client, scenario = client

    result = _run(
        test_client,
        "post",
        "/api/grounding/tags/heat",
        json={"media_ids": [int(scenario["key"])]},
    )

    assert result["elements"][0]["heat"] == "AAAA"
    # A tag the batch never scored still earns a live score from the pass.
    assert result["elements"][0]["score"] == 44.0


def test_job_result_of_a_missing_job_is_404(client):
    """A stale job id must not read back as an empty result."""
    test_client, _ = client

    assert test_client.get("/api/jobs/nope/result").status_code == 404
