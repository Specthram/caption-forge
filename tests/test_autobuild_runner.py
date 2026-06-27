"""Unit tests for the Auto-build runner's pure helpers."""

# pylint: disable=protected-access

from server.runners import autobuild as runner
from src.autobuild_studio import Candidate, Recipe


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


def test_run_preview_events_streams_stages_then_result(monkeypatch):
    """The event stream names each stage in order, then the payload."""

    def fake_stages(_recipe):
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


def test_run_preview_events_empty_recipe_yields_only_result(monkeypatch):
    """A size-0 recipe skips the stages and streams the empty payload."""
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
