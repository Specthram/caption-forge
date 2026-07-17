"""Unit tests for the Auto-build runner's pure helpers."""

# pylint: disable=protected-access

import json

from server.runners import autobuild as runner
from src import sqlite_store as store
from src.autobuild_studio import Candidate, Recipe


def _media(tmp_path, name, data) -> int:
    """Ingest a small file with distinct ``data`` bytes; return its id."""
    path = tmp_path / name
    path.write_bytes(data)
    return store.ingest_file(str(path))[0]


def _cand(media_id, **over):
    """Return a minimal candidate for the reason helper."""
    base = {
        "id": media_id,
        "name": f"m{media_id}",
        "favorite": False,
        "is_video": False,
        "width": 1,
        "height": 1,
        "quality": None,
        "bucket": "face",
        "tags": (),
        "subject": 0.0,
        "signals": {},
        "eligible": True,
        "excluded": "",
        "has_vector": True,
    }
    base.update(over)
    return Candidate(**base)


def test_drop_excluded_tags_removes_matching_media():
    """A media carrying an excluded tag is cut from the pool."""
    recipe = Recipe(exclude_tags=("watermark",))
    pool = [{"id": 1}, {"id": 2}, {"id": 3}]
    tags = {
        1: [{"name": "portrait"}],
        2: [{"name": "watermark"}],
        3: [{"name": "Watermark"}],  # normalised, still matches
    }
    kept = runner._drop_excluded_tags(recipe, pool, tags)
    assert [item["id"] for item in kept] == [1]


def test_drop_excluded_tags_noop_without_excludes():
    """No excluded tags leaves the pool untouched."""
    recipe = Recipe()
    pool = [{"id": 1}, {"id": 2}]
    assert runner._drop_excluded_tags(recipe, pool, {}) == pool


def test_keep_required_tags_keeps_only_media_with_every_tag():
    """Locked tags are a hard filter: a media missing one is cut."""
    recipe = Recipe(locked_tags=("red_hair", "1girl"))
    pool = [{"id": 1}, {"id": 2}, {"id": 3}]
    tags = {
        1: [{"name": "red_hair"}, {"name": "1girl"}],  # both -> kept
        2: [{"name": "red_hair"}],  # missing 1girl -> cut
        3: [{"name": "Red_Hair"}, {"name": "1girl"}],  # normalised -> kept
    }
    kept = runner._keep_required_tags(recipe, pool, tags)
    assert [item["id"] for item in kept] == [1, 3]


def test_keep_required_tags_noop_without_locked():
    """No locked tags leaves the pool untouched."""
    recipe = Recipe()
    pool = [{"id": 1}, {"id": 2}]
    assert runner._keep_required_tags(recipe, pool, {}) == pool


def test_live_tags_drops_deleted_tags(store_db):
    """A locked/excluded tag absent from the vocabulary is dropped, reported.

    Without this a saved recipe naming a since-deleted *locked* tag would
    empty the whole build when the dataset is re-edited.
    """
    category = store.create_tag_category("general")
    store.get_or_create_tag("red_hair", category)
    recipe = Recipe(
        locked_tags=("red_hair", "ghost_tag"), exclude_tags=("blur",)
    )
    clean, stale = runner._live_tags(recipe)
    assert clean.locked_tags == ("red_hair",)
    assert clean.exclude_tags == ()
    assert stale == ["blur", "ghost_tag"]


def test_live_tags_noop_when_all_live(store_db):
    """A tag that still exists (matched normalized) is kept, nothing stale."""
    category = store.create_tag_category("general")
    store.get_or_create_tag("red_hair", category)
    recipe = Recipe(locked_tags=("Red Hair",))  # normalises to red_hair
    clean, stale = runner._live_tags(recipe)
    assert clean is recipe
    assert stale == []


def test_run_preview_events_streams_stages_then_result(monkeypatch):
    """The event stream names each stage in order, then the payload."""

    def fake_stages(_recipe, _reuse=False):
        yield "pool"
        yield "vectors"
        yield "semantic"
        yield "select"
        return ("corpus", "cands", "picks", "meta", "tags", (0, 0))

    monkeypatch.setattr(runner, "_select_stages", fake_stages)
    monkeypatch.setattr(runner.config, "load_autobuild_config", lambda: {})
    monkeypatch.setattr(runner, "_recipe_of", lambda _p, _b: Recipe(size=50))
    monkeypatch.setattr(runner, "_assemble", lambda *_a: {"picks": ["ok"]})

    events = list(runner.run_preview_events(object()))
    stages = [e["stage"] for e in events if "stage" in e]
    assert stages == ["pool", "vectors", "semantic", "select", "assemble"]
    assert [e["index"] for e in events if "stage" in e] == [0, 1, 2, 3, 4]
    assert all(e["total"] == 5 for e in events if "stage" in e)
    assert events[-1] == {"result": {"picks": ["ok"]}}


def test_select_reuse_skips_pool_and_corpus_reads(monkeypatch):
    """A reuse run reuses the memo instead of rereading pool and geometry.

    The pool load, the corpus geometry and the SigLIP relevance are the
    slow work; a pick edit (``reuse=True``) with an unchanged recipe scope
    must not run any of them again — only ``prepare``/``select`` rerun.
    """
    runner._POOL_MEMO.clear()
    calls = {"pool": 0, "corpus": 0, "relevance": 0}

    class _Corpus:
        vectors = {}

    def load_pool(_recipe):
        calls["pool"] += 1
        return [{"id": 1}], {}, (1, 1)

    def corpus_of(_pool):
        calls["corpus"] += 1
        return _Corpus()

    def relevance(_query, _ids):
        calls["relevance"] += 1
        return None

    monkeypatch.setattr(runner, "_load_pool", load_pool)
    monkeypatch.setattr(runner, "_corpus_of_pool", corpus_of)
    monkeypatch.setattr(runner, "_semantic_relevance", relevance)
    monkeypatch.setattr(
        runner.autobuild_studio, "prepare", lambda *_a: ["cand"]
    )
    monkeypatch.setattr(
        runner.autobuild_studio, "select", lambda *_a: (["p"], {})
    )

    recipe = Recipe(semantic_q="cat", size=50)
    runner._select_for(recipe)  # cold Build fills the memo
    assert calls == {"pool": 1, "corpus": 1, "relevance": 1}

    # A pick edit: same scope, reuse the memo — no heavy read repeats.
    edited = Recipe(semantic_q="cat", size=50, dropped=(1,))
    runner._select_for(edited, reuse=True)
    assert calls == {"pool": 1, "corpus": 1, "relevance": 1}

    # Without reuse (a fresh Build) the heavy work runs again.
    runner._select_for(recipe)
    assert calls == {"pool": 2, "corpus": 2, "relevance": 2}
    runner._POOL_MEMO.clear()


def test_get_recipe_returns_stored_blob(store_db):
    """A Studio-built dataset returns its recipe and live flag."""
    dataset_id = store.create_dataset("d")
    store.save_autobuild_recipe(
        dataset_id, json.dumps({"size": 30, "live": False}), False
    )
    out = runner.get_recipe(dataset_id)
    assert out["recipe"]["size"] == 30
    assert out["live"] is False


def test_get_recipe_none_for_handmade_dataset(store_db):
    """A dataset with no stored recipe yields a null recipe."""
    dataset_id = store.create_dataset("d")
    assert runner.get_recipe(dataset_id) == {"recipe": None, "live": False}


def test_update_overwrites_media_and_recipe(store_db, tmp_path):
    """An overwrite unlinks stale members, links the selection, re-stores."""
    first = _media(tmp_path, "a.png", b"a")
    second = _media(tmp_path, "b.png", b"b")
    third = _media(tmp_path, "c.png", b"c")
    dataset_id = store.create_dataset("d")
    store.add_media_ids_to_dataset(dataset_id, [first, second])
    store.save_autobuild_recipe(dataset_id, json.dumps({"size": 1}), True)

    runner.update(dataset_id, [second, third], {"size": 2, "live": False})

    assert store.media_ids_in_dataset(dataset_id) == {second, third}
    stored = store.get_autobuild_recipe(dataset_id)
    assert stored["recipe"]["size"] == 2
    assert stored["live"] is False


def test_run_preview_events_empty_recipe_yields_only_result(
    monkeypatch, store_db
):
    """A size-0 recipe skips the stages and streams the empty payload.

    ``store_db`` supplies a schema'd in-memory database: the empty payload
    still probes ``_semantic_available`` (a media/embedding read), which must
    not fall through to the developer's real database.
    """
    monkeypatch.setattr(runner.config, "load_autobuild_config", lambda: {})
    monkeypatch.setattr(runner, "_recipe_of", lambda _p, _b: Recipe(size=0))

    events = list(runner.run_preview_events(object()))
    assert len(events) == 1
    assert "result" in events[0]
    assert events[0]["result"]["picks"] == []


def test_upgrade_reason_prefers_subject_then_quality():
    """The reason names the dominant driver of an incoming upgrade."""
    subject = _cand(1, signals={"semantic": 0.8})
    assert runner._upgrade_reason(subject, 0.0, None) == (
        "stronger subject match"
    )
    better = _cand(2, quality=90.0)
    assert runner._upgrade_reason(better, 0.0, 50.0) == "higher quality"
    diverse = _cand(3, quality=None)
    assert runner._upgrade_reason(diverse, 0.7, None) == "adds visual variety"
    neutral = _cand(4, quality=None)
    assert "prefers" in runner._upgrade_reason(neutral, 0.0, None)
