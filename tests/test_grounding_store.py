"""Tests for the grounding repository and the storage grounding facade.

Both models are mocked: :func:`src.caption_claims.extract_claims` stands in
for the VLM, :func:`src.siglip_grounding.ground_image` for SigLIP. No
weights are ever loaded.
"""

import pytest

from src import caption_claims, siglip_grounding
from src import sqlite_store as store
from src import storage


@pytest.fixture(name="ground_ref")
def _ground_ref(tmp_path, store_db):
    # pylint: disable=unused-argument
    """Dataset with two captioned media; return (dataset_id, keys, rev_ids)."""
    dataset_id = store.create_dataset("d")
    keys, rev_ids = [], []
    for name in ("a.png", "b.png"):
        image = tmp_path / name
        image.write_bytes(name.encode())
        media_id, _ = store.ingest_file(str(image))
        store.add_media_to_dataset(dataset_id, media_id)
        key = str(media_id)
        storage.write_caption(dataset_id, key, "txt", f"caption {name}.")
        keys.append(key)
        rev_ids.append(storage.effective_revision_id(dataset_id, key, "txt"))
    return dataset_id, keys, rev_ids


def _fake_scores(scores):
    """Return a ground_image stand-in yielding ``scores`` in text order."""

    def _ground(_path, texts, with_heat=True):
        return [
            {
                "text": text,
                "score": scores[index],
                "heat": "AAAA" if with_heat else None,
                "side": 2,
            }
            for index, text in enumerate(texts)
        ]

    return _ground


class TestCaptionGroundingRepo:
    """upsert / get / bulk on caption_grounding + its claims."""

    def test_upsert_then_get(self, ground_ref):
        """A run reads back with its claims in position order."""
        _, _, rev_ids = ground_ref
        store.upsert_caption_grounding(
            rev_ids[0],
            "model-x",
            [
                {"text": "a ball", "kind": "object", "score": 80.0},
                {"text": "two dogs", "kind": "count", "score": 20.0},
            ],
        )

        grounding = store.get_caption_grounding(rev_ids[0])

        assert grounding["model_id"] == "model-x"
        assert [claim["text"] for claim in grounding["claims"]] == [
            "a ball",
            "two dogs",
        ]
        assert grounding["claims"][1]["kind"] == "count"
        assert grounding["claims"][0]["rejected"] is False

    def test_get_of_ungrounded_revision_is_none(self, ground_ref):
        """A revision nobody grounded reads back as None, not an empty run."""
        _, _, rev_ids = ground_ref

        assert store.get_caption_grounding(rev_ids[0]) is None

    def test_rerun_replaces_claims_and_clears_rejections(self, ground_ref):
        """A fresh measurement never carries an old rejection onto new text."""
        _, _, rev_ids = ground_ref
        store.upsert_caption_grounding(
            rev_ids[0],
            "model-x",
            [{"text": "old", "kind": "object", "score": 9}],
        )
        claim_id = store.get_caption_grounding(rev_ids[0])["claims"][0]["id"]
        store.set_claim_rejected(claim_id, True)

        store.upsert_caption_grounding(
            rev_ids[0],
            "model-x",
            [{"text": "new", "kind": "object", "score": 9}],
        )

        claims = store.get_caption_grounding(rev_ids[0])["claims"]
        assert [claim["text"] for claim in claims] == ["new"]
        assert claims[0]["rejected"] is False

    def test_set_claim_rejected_round_trips(self, ground_ref):
        """The user's "marked non-validated" flag persists and reverts."""
        _, _, rev_ids = ground_ref
        store.upsert_caption_grounding(
            rev_ids[0],
            "m",
            [{"text": "a ball", "kind": "object", "score": 80}],
        )
        claim_id = store.get_caption_grounding(rev_ids[0])["claims"][0]["id"]

        store.set_claim_rejected(claim_id, True)
        assert store.get_caption_grounding(rev_ids[0])["claims"][0]["rejected"]

        store.set_claim_rejected(claim_id, False)
        claims = store.get_caption_grounding(rev_ids[0])["claims"]
        assert claims[0]["rejected"] is False

    def test_bulk_skips_ungrounded_revisions(self, ground_ref):
        """A page reads in two queries; absent means never grounded."""
        _, _, rev_ids = ground_ref
        store.upsert_caption_grounding(
            rev_ids[0],
            "m",
            [{"text": "a ball", "kind": "object", "score": 80}],
        )

        bulk = store.caption_groundings_bulk(rev_ids)

        assert set(bulk) == {rev_ids[0]}
        assert bulk[rev_ids[0]]["claims"][0]["text"] == "a ball"

    def test_delete_drops_the_claims_too(self, ground_ref):
        """The claim rows cascade with their grounding."""
        _, _, rev_ids = ground_ref
        store.upsert_caption_grounding(
            rev_ids[0],
            "m",
            [{"text": "a ball", "kind": "object", "score": 80}],
        )

        store.delete_caption_grounding(rev_ids[0])

        assert store.get_caption_grounding(rev_ids[0]) is None
        assert store.caption_groundings_bulk(rev_ids) == {}


class TestGroundingFilters:
    """The DB-side media filters backing the grid."""

    def test_grounded_media_ids_is_scoped_to_the_checkpoint(self, ground_ref):
        """Scores from another checkpoint do not count as grounded."""
        dataset_id, keys, rev_ids = ground_ref
        store.upsert_caption_grounding(
            rev_ids[0],
            "model-x",
            [{"text": "a", "kind": "object", "score": 80}],
        )
        type_id = store.get_or_create_caption_type("txt")

        assert store.grounded_media_ids(dataset_id, type_id, "model-x") == {
            int(keys[0])
        }
        assert (
            store.grounded_media_ids(dataset_id, type_id, "model-y") == set()
        )

    def test_low_grounding_media_ids_flags_below_threshold(self, ground_ref):
        """A media with any claim under the threshold is flagged, once."""
        dataset_id, keys, rev_ids = ground_ref
        store.upsert_caption_grounding(
            rev_ids[0],
            "m",
            [
                {"text": "a", "kind": "object", "score": 90.0},
                {"text": "b", "kind": "object", "score": 10.0},
            ],
        )
        store.upsert_caption_grounding(
            rev_ids[1], "m", [{"text": "c", "kind": "object", "score": 90.0}]
        )
        type_id = store.get_or_create_caption_type("txt")

        flagged = store.low_grounding_media_ids(dataset_id, type_id, "m", 55.0)

        assert flagged == {int(keys[0])}

    def test_low_grounding_ignores_rejected_claims(self, ground_ref):
        """A claim the user already handled never re-flags its media."""
        dataset_id, _, rev_ids = ground_ref
        store.upsert_caption_grounding(
            rev_ids[0], "m", [{"text": "b", "kind": "object", "score": 10.0}]
        )
        claim_id = store.get_caption_grounding(rev_ids[0])["claims"][0]["id"]
        store.set_claim_rejected(claim_id, True)
        type_id = store.get_or_create_caption_type("txt")

        assert (
            store.low_grounding_media_ids(dataset_id, type_id, "m", 55.0)
            == set()
        )


class TestTagGroundingRepo:
    """media_tag_grounding, keyed on (media, tag, checkpoint)."""

    def test_upsert_then_read(self, ground_ref):
        """An attached tag's score reads back under its checkpoint."""
        _, keys, _ = ground_ref
        media_id = int(keys[0])
        category_id = store.create_tag_category("general", "#fff")
        tag_id = store.get_or_create_tag("horse", category_id)
        store.add_tag_to_media(media_id, tag_id)

        store.upsert_tag_grounding(media_id, tag_id, "m", 12.5)

        assert store.tag_grounding_for_media(media_id, "m") == {tag_id: 12.5}
        assert store.tag_grounding_for_media(media_id, "other") == {}

    def test_upsert_refreshes_an_existing_score(self, ground_ref):
        """Re-scoring the same triple updates rather than duplicates."""
        _, keys, _ = ground_ref
        media_id = int(keys[0])
        category_id = store.create_tag_category("general", "#fff")
        tag_id = store.get_or_create_tag("horse", category_id)
        store.add_tag_to_media(media_id, tag_id)

        store.upsert_tag_grounding(media_id, tag_id, "m", 12.5)
        store.upsert_tag_grounding(media_id, tag_id, "m", 88.0)

        assert store.tag_grounding_for_media(media_id, "m") == {tag_id: 88.0}

    def test_detached_tag_is_hidden_by_the_join(self, ground_ref):
        """A tag removed from the media leaves an orphan row nobody reads."""
        _, keys, _ = ground_ref
        media_id = int(keys[0])
        category_id = store.create_tag_category("general", "#fff")
        tag_id = store.get_or_create_tag("horse", category_id)
        store.add_tag_to_media(media_id, tag_id)
        store.upsert_tag_grounding(media_id, tag_id, "m", 12.5)

        store.remove_tag_from_media(media_id, tag_id)

        assert store.tag_grounding_for_media(media_id, "m") == {}

    def test_bulk_groups_by_media(self, ground_ref):
        """One query fills every media's ``{tag_id: score}`` map."""
        _, keys, _ = ground_ref
        category_id = store.create_tag_category("general", "#fff")
        tag_id = store.get_or_create_tag("horse", category_id)
        for key in keys:
            store.add_tag_to_media(int(key), tag_id)
        store.upsert_tag_grounding(int(keys[0]), tag_id, "m", 12.5)

        bulk = store.tag_groundings_bulk([int(key) for key in keys], "m")

        assert bulk == {int(keys[0]): {tag_id: 12.5}, int(keys[1]): {}}


class TestStorageGroundingFacade:
    """storage.ground_caption / ground_tags, both models mocked."""

    def test_ground_caption_scores_and_persists(self, ground_ref, monkeypatch):
        """Claims come from the LLM, scores from SigLIP; both persist."""
        dataset_id, keys, rev_ids = ground_ref
        monkeypatch.setattr(
            caption_claims,
            "extract_claims",
            lambda *a: [
                {"text": "a ball", "kind": "object"},
                {"text": "two dogs", "kind": "count"},
            ],
        )
        monkeypatch.setattr(
            siglip_grounding, "ground_image", _fake_scores([80.0, 20.0])
        )

        verdict = storage.ground_caption(dataset_id, keys[0], "txt", "model-x")

        assert verdict["status"] == "ok"
        claims = store.get_caption_grounding(rev_ids[0])["claims"]
        assert [claim["score"] for claim in claims] == [80.0, 20.0]
        assert [claim["kind"] for claim in claims] == ["object", "count"]

    def test_ground_caption_of_a_video_is_skipped(
        self, ground_ref, monkeypatch
    ):
        """SigLIP scores pictures; a clip's first frame is not the clip."""
        dataset_id, keys, _ = ground_ref
        monkeypatch.setattr(storage, "media_path", lambda *a: "/tmp/clip.mp4")

        verdict = storage.ground_caption(dataset_id, keys[0], "txt", "m")

        assert verdict["status"] == "skipped"

    def test_ground_caption_with_no_claim_drops_a_stale_run(
        self, ground_ref, monkeypatch
    ):
        """A caption the LLM cannot decompose invalidates its old scores."""
        dataset_id, keys, rev_ids = ground_ref
        store.upsert_caption_grounding(
            rev_ids[0], "m", [{"text": "old", "kind": "object", "score": 9}]
        )
        monkeypatch.setattr(caption_claims, "extract_claims", lambda *a: [])

        verdict = storage.ground_caption(dataset_id, keys[0], "txt", "m")

        assert verdict["status"] == "skipped"
        assert store.get_caption_grounding(rev_ids[0]) is None

    def test_extract_and_score_are_separable(self, ground_ref, monkeypatch):
        """The batch path decomposes everything before loading SigLIP once."""
        dataset_id, keys, rev_ids = ground_ref
        monkeypatch.setattr(
            caption_claims,
            "extract_claims",
            lambda *a: [{"text": "a ball", "kind": "object"}],
        )
        monkeypatch.setattr(
            siglip_grounding, "ground_image", _fake_scores([77.0])
        )

        extraction = storage.extract_caption_claims(dataset_id, keys[0], "txt")
        assert extraction["revision_id"] == rev_ids[0]
        assert store.get_caption_grounding(rev_ids[0]) is None

        storage.score_caption_claims(
            extraction["path"],
            extraction["revision_id"],
            extraction["claims"],
            "model-x",
        )
        grounding = store.get_caption_grounding(rev_ids[0])
        assert grounding["claims"][0]["score"] == 77.0
        assert grounding["model_id"] == "model-x"

    def test_ground_tags_wraps_each_tag_in_the_pre_prompt(
        self, ground_ref, monkeypatch
    ):
        """Every tag is scored through TAG_PROMPT, no LLM in sight."""
        _, keys, _ = ground_ref
        media_id = int(keys[0])
        category_id = store.create_tag_category("general", "#fff")
        for name in ("horse", "grass"):
            store.add_tag_to_media(
                media_id, store.get_or_create_tag(name, category_id)
            )
        seen = {}

        def _ground(_path, texts, with_heat=True):
            seen["texts"] = texts
            return _fake_scores([5.0, 95.0])(_path, texts, with_heat)

        monkeypatch.setattr(siglip_grounding, "ground_image", _ground)

        verdict = storage.ground_tags(keys[0], "model-x")

        assert verdict["status"] == "ok"
        assert seen["texts"] == [
            "a photo that contains grass",
            "a photo that contains horse",
        ]
        scores = {tag["name"]: tag["score"] for tag in verdict["tags"]}
        assert scores == {"grass": 5.0, "horse": 95.0}

    def test_ground_tags_of_an_untagged_media_is_skipped(self, ground_ref):
        """No tag, no forward pass."""
        _, keys, _ = ground_ref

        assert storage.ground_tags(keys[0], "m")["status"] == "skipped"

    def test_tag_grounding_lists_ungrounded_tags_as_none(self, ground_ref):
        """A never-scored tag is visible with a null score, not missing."""
        _, keys, _ = ground_ref
        media_id = int(keys[0])
        category_id = store.create_tag_category("general", "#fff")
        store.add_tag_to_media(
            media_id, store.get_or_create_tag("horse", category_id)
        )

        tags = storage.tag_grounding(keys[0], "m")

        assert [(tag["name"], tag["score"]) for tag in tags] == [
            ("horse", None)
        ]

    def test_remove_grounded_tag_detaches_and_drops_scores(self, ground_ref):
        """Retiring a hallucinated tag takes its scores with it."""
        _, keys, _ = ground_ref
        media_id = int(keys[0])
        category_id = store.create_tag_category("general", "#fff")
        tag_id = store.get_or_create_tag("horse", category_id)
        store.add_tag_to_media(media_id, tag_id)
        store.upsert_tag_grounding(media_id, tag_id, "m", 5.0)

        storage.remove_grounded_tag(keys[0], tag_id)

        assert storage.tag_grounding(keys[0], "m") == []
        assert store.tag_grounding_for_media(media_id, "m") == {}

    def test_caption_heatmaps_attach_a_grid_per_claim(
        self, ground_ref, monkeypatch
    ):
        """The modal's maps are rebuilt from the stored claims, in one pass."""
        dataset_id, keys, rev_ids = ground_ref
        store.upsert_caption_grounding(
            rev_ids[0],
            "m",
            [{"text": "a ball", "kind": "object", "score": 80}],
        )
        monkeypatch.setattr(
            siglip_grounding, "ground_image", _fake_scores([80.0])
        )

        claims = storage.caption_heatmaps(dataset_id, keys[0], "txt")

        assert claims[0]["heat"] == "AAAA"
        assert claims[0]["side"] == 2

    def test_caption_heatmaps_of_an_ungrounded_caption_is_empty(
        self, ground_ref
    ):
        """Nothing to map before the caption has been grounded."""
        dataset_id, keys, _ = ground_ref

        assert storage.caption_heatmaps(dataset_id, keys[0], "txt") == []
