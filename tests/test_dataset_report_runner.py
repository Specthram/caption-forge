"""End-to-end test of the dataset quality-report job body.

Drives :func:`server.runners.dataset_report.report_body` against the shared
in-memory database and real image files on disk: the run must score the
media it is missing, compile a report, store the blob and reconcile the
stale resolutions. The IQA metric itself is stubbed — loading MUSIQ would
pull model weights, which a unit test must never do — but every other
layer (repository, engine, serialisation) is real.
"""

import os

import pytest
from PIL import Image

from server.runners import dataset_report
from src import quality
from src import sqlite_store as store
from src import storage


def _progress(**_kwargs) -> None:
    """Swallow the job's progress updates."""


@pytest.fixture(name="scenario")
def _scenario(store_db, tmp_path, monkeypatch):
    """Seed a library, a dataset of three images and a stubbed scorer."""
    # pylint: disable=unused-argument
    for index, color in enumerate([(220, 60, 50), (60, 220, 50), (0, 0, 0)]):
        Image.new("RGB", (1536, 1536), color).save(tmp_path / f"{index}.png")
    library_id = store.create_library("fixtures", str(tmp_path))
    store.scan_library(library_id)
    media = sorted(store.list_library_media(), key=lambda row: row["name"])
    dataset_id = store.create_dataset("shapes_all")
    for row in media:
        store.add_media_to_dataset(dataset_id, row["id"])
    storage.write_caption(dataset_id, str(media[0]["id"]), "txt", "A red one.")
    storage.write_caption(
        dataset_id, str(media[1]["id"]), "txt", "a red " * 12
    )
    # The stub scores the third (black) image under the low-quality floor.
    scores = {"0.png": 90.0, "1.png": 85.0, "2.png": 30.0}

    def _score(path, _metric):
        return scores[os.path.basename(str(path))]

    monkeypatch.setattr(quality, "score_media", _score)
    monkeypatch.setattr(quality, "unload_metric", lambda: None)
    return {"dataset_id": dataset_id, "media": media}


def _body(dataset_id, **kwargs) -> dataset_report.RunSpec:
    """Return a run spec scoring with MUSIQ only (no DINOv2 weights)."""
    kwargs.setdefault("scorers", ("musiq",))
    kwargs.setdefault("caption_type", "txt")
    return dataset_report.RunSpec(dataset_id=dataset_id, **kwargs)


def test_run_scores_and_stores_a_report(scenario):
    """The job scores the missing media and persists the blob."""
    dataset_id = scenario["dataset_id"]
    result = dataset_report.report_body(_body(dataset_id))(_progress)
    assert result["issues"] >= 1
    stored = store.get_dataset_report(dataset_id)
    assert stored["scorers"] == ["musiq"]
    assert stored["caption_type"] == "txt"
    assert stored["report"]["images"] == 3
    assert stored["report"]["grade"] == result["grade"]


def test_run_persists_the_quality_scores(scenario):
    """Every dataset image ends up with a stored MUSIQ score."""
    dataset_report.report_body(_body(scenario["dataset_id"]))(_progress)
    for row in scenario["media"]:
        assert store.get_media(row["id"]) is not None
    assert dict(store.available_quality_metrics())["musiq"] == 3


def test_one_unreadable_media_does_not_sink_the_run(scenario, monkeypatch):
    """A scorer crashing on one image is skipped; the others still score."""
    dataset_id = scenario["dataset_id"]

    def _score(path, _metric):
        if os.path.basename(str(path)) == "1.png":
            raise OSError("truncated file")
        return 80.0

    monkeypatch.setattr(quality, "score_media", _score)
    result = dataset_report.report_body(_body(dataset_id))(_progress)
    assert "grade" in result
    # The two readable images got a score; the crashing one did not.
    assert dict(store.available_quality_metrics())["musiq"] == 2


def test_second_run_scores_nothing_new(scenario, monkeypatch):
    """A repeat evaluation only computes the pairs it is missing."""
    dataset_id = scenario["dataset_id"]
    dataset_report.report_body(_body(dataset_id))(_progress)
    calls = []
    monkeypatch.setattr(
        quality, "score_media", lambda path, metric: calls.append(path) or 1.0
    )
    dataset_report.report_body(_body(dataset_id))(_progress)
    assert calls == []


def test_force_rescores_everything(scenario, monkeypatch):
    """``force`` recomputes every (media, metric) pair."""
    dataset_id = scenario["dataset_id"]
    dataset_report.report_body(_body(dataset_id))(_progress)
    calls = []
    monkeypatch.setattr(
        quality, "score_media", lambda path, metric: calls.append(path) or 50.0
    )
    dataset_report.report_body(_body(dataset_id, force=True))(_progress)
    assert len(calls) == 3


def test_low_quality_and_caption_issues_are_reported(scenario):
    """The black image is flagged, and so is the looping caption."""
    dataset_id = scenario["dataset_id"]
    dataset_report.report_body(_body(dataset_id))(_progress)
    issues = store.get_dataset_report(dataset_id)["report"]["issues"]
    kinds = {issue["kind"] for issue in issues}
    assert "low_quality" in kinds
    assert "caption" in kinds


def test_hidden_media_are_left_out(scenario):
    """A media hidden in the dataset is never deployed, never evaluated."""
    dataset_id = scenario["dataset_id"]
    store.set_media_hidden(dataset_id, scenario["media"][0]["id"], True)
    dataset_report.report_body(_body(dataset_id))(_progress)
    assert store.get_dataset_report(dataset_id)["report"]["images"] == 2


def test_a_run_drops_the_stale_resolutions(scenario):
    """A resolution whose finding is gone does not survive the re-run."""
    dataset_id = scenario["dataset_id"]
    store.set_issue_resolution(dataset_id, "lowq:999", "ignored", "40")
    dataset_report.report_body(_body(dataset_id))(_progress)
    assert "lowq:999" not in store.dataset_resolutions(dataset_id)


def test_an_unchanged_finding_keeps_its_resolution(scenario):
    """Ignoring the black image survives a second, identical run."""
    dataset_id = scenario["dataset_id"]
    dataset_report.report_body(_body(dataset_id))(_progress)
    issues = store.get_dataset_report(dataset_id)["report"]["issues"]
    low = next(issue for issue in issues if issue["kind"] == "low_quality")
    store.set_issue_resolution(
        dataset_id, low["key"], "ignored", low["fingerprint"]
    )
    dataset_report.report_body(_body(dataset_id))(_progress)
    assert low["key"] in store.dataset_resolutions(dataset_id)
