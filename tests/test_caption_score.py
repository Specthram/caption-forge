"""Tests for reference-free caption scoring (engine, repo, storage)."""

import pytest

from src import caption_score
from src import sqlite_store as store
from src import storage


class TestCalibration:
    """The per-encoder cosine window maps a raw cosine to 0-100."""

    def test_rescale_spans_the_window(self):
        """A cosine at the window bounds maps to 0 and 100, mid to ~50."""
        low, high = caption_score.COSINE_BOUNDS["clip"]
        assert caption_score._rescale(low, "clip") == 0.0
        assert caption_score._rescale(high, "clip") == 100.0
        mid = caption_score._rescale((low + high) / 2, "clip")
        assert 49.0 <= mid <= 51.0

    def test_rescale_clamps_out_of_band(self):
        """Cosines outside the window clamp to 0 / 100, never overflow."""
        assert caption_score._rescale(-1.0, "blip") == 0.0
        assert caption_score._rescale(1.0, "blip") == 100.0


def _fake_sessions(monkeypatch, values):
    """Patch _SESSIONS so each kind yields a scorer returning a fixed value.

    ``values`` maps a kind to a float (a working encoder) or an Exception
    instance (a session whose load raises).
    """
    from contextlib import contextmanager

    def make(kind):
        @contextmanager
        def session(_spec):
            outcome = values[kind]
            if isinstance(outcome, Exception):
                raise outcome
            yield lambda _path, _caption: outcome

        return session

    monkeypatch.setattr(
        caption_score,
        "_SESSIONS",
        {kind: make(kind) for kind in values},
    )


_SPECS = [
    {"kind": "siglip2", "label": "S", "model_id": "s"},
    {"kind": "clip", "label": "C", "model_id": "c"},
    {"kind": "blip", "label": "B", "model_id": "b"},
]


class TestScoreOrchestration:
    """score_caption / score_dataset run every encoder, survive one failing."""

    def test_one_failure_does_not_sink_the_run(self, monkeypatch):
        """A raising session yields a None line; the others still score."""
        _fake_sessions(
            monkeypatch,
            {
                "siglip2": 88.0,
                "clip": RuntimeError("no weights"),
                "blip": 88.0,
            },
        )
        results = caption_score.score_caption("x.png", "a caption", _SPECS)
        by_kind = {r["kind"]: r for r in results}
        assert by_kind["siglip2"]["score"] == 88.0
        assert by_kind["blip"]["score"] == 88.0
        assert by_kind["clip"]["score"] is None
        assert "no weights" in by_kind["clip"]["error"]

    def test_score_dataset_loads_each_encoder_once(self, monkeypatch):
        """A failed family is skipped for every item; others score all."""
        _fake_sessions(
            monkeypatch,
            {"siglip2": 70.0, "clip": RuntimeError("boom"), "blip": 33.0},
        )
        items = [
            {"revision_id": 1, "path": "a.png", "caption": "a"},
            {"revision_id": 2, "path": "b.png", "caption": "b"},
        ]
        scores = caption_score.score_dataset(items, _SPECS)
        assert scores[1] == {"siglip2": 70.0, "blip": 33.0}
        assert scores[2] == {"siglip2": 70.0, "blip": 33.0}


@pytest.fixture(name="score_ref")
def _score_ref(tmp_path, store_db):
    # pylint: disable=unused-argument
    """Dataset with one captioned image; return (dataset_id, key, rev_id)."""
    dataset_id = store.create_dataset("d")
    image = tmp_path / "a.png"
    image.write_bytes(b"a.png")
    media_id, _ = store.ingest_file(str(image))
    store.add_media_to_dataset(dataset_id, media_id)
    key = str(media_id)
    storage.write_caption(dataset_id, key, "txt", "a red ball on grass.")
    rev_id = storage.effective_revision_id(dataset_id, key, "txt")
    return dataset_id, key, rev_id


class TestScoreRepo:
    """upsert / get / delete on caption_score."""

    def test_upsert_then_get(self, score_ref):
        """A stored score reads back keyed by encoder family."""
        _, _, rev_id = score_ref
        store.upsert_caption_score(rev_id, "clip", "openai/x", 61.0)
        scores = store.get_caption_scores(rev_id)
        assert scores["clip"] == {"model_id": "openai/x", "score": 61.0}

    def test_upsert_replaces_same_kind(self, score_ref):
        """A second upsert on the same family overwrites the first."""
        _, _, rev_id = score_ref
        store.upsert_caption_score(rev_id, "clip", "openai/x", 61.0)
        store.upsert_caption_score(rev_id, "clip", "openai/y", 12.0)
        scores = store.get_caption_scores(rev_id)
        assert scores["clip"] == {"model_id": "openai/y", "score": 12.0}

    def test_delete(self, score_ref):
        """Deleting drops every family's score for the revision."""
        _, _, rev_id = score_ref
        store.upsert_caption_score(rev_id, "clip", "openai/x", 61.0)
        store.upsert_caption_score(rev_id, "blip", "sf/x", 40.0)
        store.delete_caption_scores(rev_id)
        assert store.get_caption_scores(rev_id) == {}


class TestStorageScore:
    """storage-level target resolution, scoring and read-back."""

    def test_target_ok(self, score_ref):
        """A captioned image resolves to a scorable target."""
        dataset_id, key, rev_id = score_ref
        target = storage.caption_score_target(dataset_id, key, "txt")
        assert target["status"] == "ok"
        assert target["revision_id"] == rev_id
        assert target["caption"].startswith("a red ball")

    def test_target_empty_caption_skips(self, score_ref):
        """An empty caption is skipped and any stored scores dropped."""
        dataset_id, key, rev_id = score_ref
        store.upsert_caption_score(rev_id, "clip", "openai/x", 61.0)
        storage.write_caption(dataset_id, key, "txt", "   ")
        target = storage.caption_score_target(dataset_id, key, "txt")
        assert target["status"] == "skipped"

    def test_target_tags_skips(self, score_ref):
        """The virtual tags type has no revision, so it is skipped."""
        dataset_id, key, _ = score_ref
        target = storage.caption_score_target(
            dataset_id, key, storage.TAGS_TYPE
        )
        assert target["status"] == "skipped"

    def test_score_persists_successful_lines(self, score_ref, monkeypatch):
        """Only successful lines persist; failed encoders are skipped."""
        dataset_id, key, rev_id = score_ref

        def fake_engine(_path, _caption, specs, progress=None):
            # pylint: disable=unused-argument
            return [
                {
                    "kind": "siglip2",
                    "model_id": "s",
                    "score": 70.0,
                    "error": None,
                },
                {
                    "kind": "clip",
                    "model_id": "c",
                    "score": None,
                    "error": "boom",
                },
                {
                    "kind": "blip",
                    "model_id": "b",
                    "score": 33.0,
                    "error": None,
                },
            ]

        monkeypatch.setattr(caption_score, "score_caption", fake_engine)
        verdict = storage.score_caption(
            "a.png", rev_id, "a red ball on grass."
        )
        assert verdict["scored"] == 2
        scores = storage.caption_scores(dataset_id, key, "txt")
        assert set(scores) == {"siglip2", "blip"}
        assert scores["blip"]["score"] == 33.0


class TestMediaTagScore:
    """The Media tab "Tags Score": tags scored as one comma-joined text."""

    def _tag(self, key):
        """Attach one tag to the media, returning the joined tag text."""
        category = store.list_tag_categories()[0]["id"]
        tag_id = store.get_or_create_tag("blue", category)
        store.add_tag_to_media(int(key), tag_id)
        return storage.media_tags_text(key)

    def test_target_ok_uses_joined_tags(self, score_ref):
        """A tagged media resolves to its comma-joined tag text."""
        _, key, _ = score_ref
        text = self._tag(key)
        target = storage.media_tag_score_target(key)
        assert target["status"] == "ok"
        assert target["text"] == text == "blue"

    def test_no_tags_skips(self, score_ref):
        """A media with no tags is skipped and stored scores dropped."""
        _, key, _ = score_ref
        store.upsert_media_tag_score(int(key), "clip", "x", 5.0, "old")
        target = storage.media_tag_score_target(key)
        assert target["status"] == "skipped"
        assert store.get_media_tag_scores(int(key)) == {}

    def test_score_persists_with_scored_text(self, score_ref, monkeypatch):
        """Scoring stores each line with the exact tag text it scored."""
        _, key, _ = score_ref
        self._tag(key)

        def fake_engine(_path, text, _specs, progress=None):
            # pylint: disable=unused-argument
            return [
                {
                    "kind": "siglip2",
                    "model_id": "s",
                    "score": 60.0,
                    "error": None,
                },
            ]

        monkeypatch.setattr(caption_score, "score_caption", fake_engine)
        # A real path so effective_file resolves; reuse the seeded image.
        monkeypatch.setattr(store, "effective_file", lambda _mid: "a.png")
        verdict = storage.score_media_tags(key)
        assert verdict["scored"] == 1
        stored = storage.media_tag_scores(key)
        assert stored["text"] == "blue"
        assert stored["scores"]["siglip2"]["scored_text"] == "blue"
        assert stored["scores"]["siglip2"]["score"] == 60.0
