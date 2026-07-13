"""Tests for the Index chain's per-media resilience.

One unreadable image must not cost the rest of the library: the runner
names it on the job and carries on. These tests drive the private
``_for_each`` guard directly with a fake progress reporter, so no model is
ever loaded.
"""

import pytest

from server.jobs import MAX_WARNINGS, Job, JobStopped, Progress
from server.runners import library


class _FakeManager:  # pylint: disable=too-few-public-methods
    """A job manager that records broadcasts instead of sending them."""

    def __init__(self):
        self.broadcasts = 0

    def broadcast(self, job) -> None:
        """Count one broadcast."""
        _ = job
        self.broadcasts += 1


@pytest.fixture(name="progress")
def _progress():
    """Return a live Progress bound to a throwaway job."""
    job = Job(id="j", type="index", name="Index")
    return Progress(_FakeManager(), job), job


def _rows(*names):
    """Return media rows shaped like the repository's dicts."""
    return [
        {"id": index, "name": name, "eff_path": f"/tmp/{name}"}
        for index, name in enumerate(names, start=1)
    ]


def test_a_failing_media_does_not_stop_the_pass(progress):
    """The other media are still processed, and the run stays alive."""
    reporter, job = progress
    chain = library._Chain(reporter, 3)  # pylint: disable=protected-access
    seen = []

    def work(row, path):
        _ = path
        if row["name"] == "bad.png":
            raise ValueError("cannot identify image file")
        seen.append(row["name"])

    done = library._for_each(  # pylint: disable=protected-access
        chain, _rows("a.png", "bad.png", "b.png"), "embeddings", work
    )
    assert done == 2
    assert seen == ["a.png", "b.png"]
    assert chain.done == 3
    assert job.warning_count == 1
    assert job.warnings == ["bad.png: ValueError: cannot identify image file"]


def test_a_media_missing_from_disk_is_named_and_skipped(progress):
    """A row whose file left the disk warns instead of raising."""
    reporter, job = progress
    chain = library._Chain(reporter, 1)  # pylint: disable=protected-access
    rows = [{"id": 1, "name": "gone.png", "eff_path": None}]

    def work(row, path):
        raise AssertionError(f"never called: {row} {path}")

    done = library._for_each(  # pylint: disable=protected-access
        chain, rows, "embeddings", work
    )
    assert done == 0
    assert job.warnings == ["gone.png: file missing on disk — skipped"]


def test_warnings_are_capped_but_still_counted(progress):
    """A broken library streams a bounded list and an honest total."""
    reporter, job = progress
    count = MAX_WARNINGS + 5
    chain = library._Chain(reporter, count)  # pylint: disable=W0212

    def work(row, path):
        _ = path
        raise OSError(f"broken {row['id']}")

    library._for_each(  # pylint: disable=protected-access
        chain, _rows(*[f"i{n}.png" for n in range(count)]), "quality", work
    )
    assert job.warning_count == count
    assert len(job.warnings) == MAX_WARNINGS


def test_a_stop_request_still_ends_the_run(progress):
    """The guard never swallows a stop: tick raises outside the try."""
    reporter, job = progress
    chain = library._Chain(reporter, 2)  # pylint: disable=protected-access
    job.stop_requested = True

    with pytest.raises(JobStopped):
        library._for_each(  # pylint: disable=protected-access
            chain, _rows("a.png"), "embeddings", lambda row, path: None
        )
