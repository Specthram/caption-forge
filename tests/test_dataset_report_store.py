"""Tests for :mod:`src.sqlite_store.reports` (report blob + resolutions).

The interesting behavior is :func:`reconcile_resolutions`: a resolution
must survive a re-run while its finding is unchanged, and must be dropped
the moment the finding disappears or its measurement moves.
"""

import pytest
from PIL import Image

from src import dataset_issues
from src import sqlite_store as store


def _issue(key: str, fingerprint: str) -> dataset_issues.Issue:
    """Return a minimal issue carrying only the reconciliation fields."""
    return dataset_issues.Issue(
        key=key,
        kind="caption",
        media_ids=(1,),
        reason="",
        metric="",
        value=0.0,
        impact=0.0,
        fingerprint=fingerprint,
    )


@pytest.fixture(name="dataset_id")
def _dataset_id(store_db):  # pylint: disable=unused-argument
    """Return the id of a throwaway dataset."""
    return store.create_dataset("report-set")


class TestReportBlob:
    """Tests for the stored report payload."""

    def test_missing_report_is_none(self, dataset_id):
        """A dataset never evaluated carries no report."""
        assert store.get_dataset_report(dataset_id) is None

    def test_round_trip(self, dataset_id):
        """The blob, scorers, caption type and duration all come back."""
        store.save_dataset_report(
            dataset_id, {"overall": 82.5}, ["musiq", "dinov2"], "txt", 41.5
        )
        stored = store.get_dataset_report(dataset_id)
        assert stored["report"] == {"overall": 82.5}
        assert stored["scorers"] == ["musiq", "dinov2"]
        assert stored["caption_type"] == "txt"
        assert stored["duration_s"] == 41.5
        assert stored["created_at"]

    def test_a_second_run_replaces_the_first(self, dataset_id):
        """One report per dataset: the latest run wins."""
        store.save_dataset_report(dataset_id, {"overall": 1}, [], "txt", 0)
        store.save_dataset_report(dataset_id, {"overall": 2}, [], "txt", 0)
        assert store.get_dataset_report(dataset_id)["report"] == {"overall": 2}

    def test_delete_clears_it(self, dataset_id):
        """Clearing a report returns the tab to "never run"."""
        store.save_dataset_report(dataset_id, {}, [], "txt", 0)
        store.delete_dataset_report(dataset_id)
        assert store.get_dataset_report(dataset_id) is None


class TestResolutions:
    """Tests for the per-finding resolutions."""

    def test_set_and_read(self, dataset_id):
        """A resolution reads back with its fingerprint."""
        store.set_issue_resolution(dataset_id, "cap:1", "ignored", "abc")
        found = store.dataset_resolutions(dataset_id)
        assert found["cap:1"]["resolution"] == "ignored"
        assert found["cap:1"]["fingerprint"] == "abc"

    def test_set_twice_updates(self, dataset_id):
        """Re-resolving a finding overwrites the previous verdict."""
        store.set_issue_resolution(dataset_id, "cap:1", "ignored", "abc")
        store.set_issue_resolution(dataset_id, "cap:1", "removed", "abc")
        found = store.dataset_resolutions(dataset_id)
        assert found["cap:1"]["resolution"] == "removed"

    def test_unknown_resolution_is_rejected(self, dataset_id):
        """Only the three documented verdicts are storable."""
        with pytest.raises(ValueError):
            store.set_issue_resolution(dataset_id, "cap:1", "maybe")

    def test_clear_reopens_the_finding(self, dataset_id):
        """Clearing a resolution makes the row open again."""
        store.set_issue_resolution(dataset_id, "cap:1", "ignored")
        store.clear_issue_resolution(dataset_id, "cap:1")
        assert store.dataset_resolutions(dataset_id) == {}

    def test_resolutions_are_per_dataset(self, dataset_id):
        """Two datasets never share a resolution."""
        other = store.create_dataset("other")
        store.set_issue_resolution(dataset_id, "cap:1", "ignored")
        assert store.dataset_resolutions(other) == {}


class TestReconcile:
    """Tests for the re-run reconciliation rules."""

    def test_unchanged_finding_keeps_its_resolution(self, dataset_id):
        """Same key, same fingerprint: the "ignored" marker survives."""
        store.set_issue_resolution(dataset_id, "cap:1", "ignored", "abc")
        dropped = store.reconcile_resolutions(
            dataset_id, [_issue("cap:1", "abc")]
        )
        assert dropped == 0
        assert "cap:1" in store.dataset_resolutions(dataset_id)

    def test_changed_measurement_drops_the_resolution(self, dataset_id):
        """A recaptioned image is a new finding: the marker is cleared."""
        store.set_issue_resolution(dataset_id, "cap:1", "ignored", "abc")
        dropped = store.reconcile_resolutions(
            dataset_id, [_issue("cap:1", "xyz")]
        )
        assert dropped == 1
        assert store.dataset_resolutions(dataset_id) == {}

    def test_vanished_finding_drops_the_resolution(self, dataset_id):
        """The removed media's finding is gone; so is its marker."""
        store.set_issue_resolution(dataset_id, "lowq:9", "removed", "40")
        dropped = store.reconcile_resolutions(dataset_id, [])
        assert dropped == 1
        assert store.dataset_resolutions(dataset_id) == {}

    def test_other_resolutions_are_untouched(self, dataset_id):
        """Reconciliation only drops what the fresh report invalidates."""
        store.set_issue_resolution(dataset_id, "cap:1", "ignored", "abc")
        store.set_issue_resolution(dataset_id, "cap:2", "ignored", "def")
        store.reconcile_resolutions(dataset_id, [_issue("cap:1", "abc")])
        assert list(store.dataset_resolutions(dataset_id)) == ["cap:1"]


class TestMediaIndexInfo:
    """The dataset-media projection is narrow; this reader fills the gap."""

    def test_returns_dimensions_and_favorite(self, store_db, tmp_path):
        """The index columns come back keyed by media id."""
        # pylint: disable=unused-argument
        image = tmp_path / "x.png"
        Image.new("RGB", (8, 8), (1, 2, 3)).save(image)
        library_id = store.create_library("fixtures", str(tmp_path))
        store.scan_library(library_id)
        media_id = store.list_library_media()[0]["id"]
        store.set_media_index(media_id, 1536, 2048)
        store.set_media_favorite(media_id, True)

        info = store.media_index_info([media_id])
        assert info[media_id] == {
            "width": 1536,
            "height": 2048,
            "favorite": True,
        }

    def test_unindexed_media_has_no_dimensions(self, store_db, tmp_path):
        """A media the Index action never saw carries None dimensions."""
        # pylint: disable=unused-argument
        Image.new("RGB", (8, 8), (1, 2, 3)).save(tmp_path / "x.png")
        library_id = store.create_library("fixtures", str(tmp_path))
        store.scan_library(library_id)
        media_id = store.list_library_media()[0]["id"]
        info = store.media_index_info([media_id])
        assert info[media_id]["width"] is None

    def test_no_ids_is_no_query(self):
        """An empty request short-circuits."""
        assert store.media_index_info([]) == {}
