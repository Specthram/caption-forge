"""Tests for the caption-review repository and the storage filter helpers."""

import pytest

from src import caption_review
from src import sqlite_store as store
from src import storage


@pytest.fixture(name="review_ref")
def _review_ref(tmp_path, store_db):
    # pylint: disable=unused-argument
    """Dataset with two captioned media; return (dataset_id, keys, rev_ids).

    ``keys`` are the two media ids (strings); ``rev_ids`` their effective
    "txt" revisions.
    """
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


class TestReviewRepo:
    """upsert / get / bulk / delete on caption_review."""

    def test_upsert_then_get(self, review_ref):
        """A stored review reads back with its issues decoded."""
        _, _, rev_ids = review_ref
        store.upsert_review(
            rev_ids[0], "integrity", [{"code": "empty", "detail": "x"}]
        )
        review = store.get_review(rev_ids[0])
        assert review["status"] == "integrity"
        assert review["issues"] == [{"code": "empty", "detail": "x"}]

    def test_upsert_replaces(self, review_ref):
        """A second upsert on the same revision overwrites the first."""
        _, _, rev_ids = review_ref
        store.upsert_review(rev_ids[0], "integrity", [{"code": "empty"}])
        store.upsert_review(rev_ids[0], "ok", [])
        review = store.get_review(rev_ids[0])
        assert review["status"] == "ok"
        assert review["issues"] == []

    def test_reviews_bulk_omits_unreviewed(self, review_ref):
        """Bulk returns only revisions that carry a review."""
        _, _, rev_ids = review_ref
        store.upsert_review(rev_ids[0], "ok", [])
        result = store.reviews_bulk(rev_ids)
        assert set(result) == {rev_ids[0]}

    def test_delete_review(self, review_ref):
        """A deleted review is gone."""
        _, _, rev_ids = review_ref
        store.upsert_review(rev_ids[0], "ok", [])
        store.delete_review(rev_ids[0])
        assert store.get_review(rev_ids[0]) is None

    def test_new_revision_supersedes(self, review_ref):
        """A new revision leaves the old review untouched but unreferenced."""
        dataset_id, keys, rev_ids = review_ref
        store.upsert_review(rev_ids[0], "integrity", [{"code": "empty"}])
        storage.write_caption(dataset_id, keys[0], "txt", "a fresh caption.")
        new_rev = storage.effective_revision_id(dataset_id, keys[0], "txt")
        assert new_rev != rev_ids[0]
        assert store.get_review(new_rev) is None


class TestStorageReviewHelpers:
    """storage-level review helpers used by the gallery."""

    def test_run_integrity_review_flags(self, review_ref):
        """Running the integrity review on a truncated caption flags it."""
        dataset_id, keys, _ = review_ref
        storage.write_caption(dataset_id, keys[0], "txt", "cut off here")
        status, issues = storage.run_integrity_review(
            dataset_id, keys[0], "txt"
        )
        assert status == "integrity"
        assert any(i["code"] == "truncated" for i in issues)

    def test_run_integrity_review_ok(self, review_ref):
        """A clean caption reviews as ok and persists the verdict."""
        dataset_id, keys, rev_ids = review_ref
        status, _ = storage.run_integrity_review(dataset_id, keys[0], "txt")
        assert status == "ok"
        assert store.get_review(rev_ids[0])["status"] == "ok"

    def test_run_integrity_review_tags_noop(self, review_ref):
        """The virtual tags type is never reviewed."""
        dataset_id, keys, _ = review_ref
        status, issues = storage.run_integrity_review(
            dataset_id, keys[0], storage.TAGS_TYPE
        )
        assert status is None
        assert issues == []

    def test_reviews_bulk_maps_keys(self, review_ref):
        """The keyed bulk read returns a review per media key."""
        dataset_id, keys, _ = review_ref
        storage.run_integrity_review(dataset_id, keys[0], "txt")
        result = storage.reviews_bulk(dataset_id, keys, "txt")
        assert result[keys[0]]["status"] == "ok"
        assert result[keys[1]] is None


class TestReviewFilter:
    """The "to review" filter pages through the DB."""

    def test_flagged_media_ids(self, review_ref):
        """Only media with a flagged review are returned."""
        dataset_id, keys, _ = review_ref
        storage.write_caption(dataset_id, keys[0], "txt", "cut off here")
        storage.run_integrity_review(dataset_id, keys[0], "txt")
        storage.run_integrity_review(dataset_id, keys[1], "txt")
        flagged = storage.flagged_media_ids(
            dataset_id, "txt", storage.REVIEW_FILTER_TO_REVIEW
        )
        assert flagged == {int(keys[0])}

    def test_all_filter_returns_none(self, review_ref):
        """The ALL filter means no restriction."""
        dataset_id, _, _ = review_ref
        assert (
            storage.flagged_media_ids(
                dataset_id, "txt", storage.REVIEW_FILTER_ALL
            )
            is None
        )

    def test_filter_pages_count_and_page(self, review_ref):
        """count_media and list_media_page honor the flagged id set."""
        dataset_id, keys, _ = review_ref
        storage.write_caption(dataset_id, keys[0], "txt", "cut off here")
        storage.run_integrity_review(dataset_id, keys[0], "txt")
        storage.run_integrity_review(dataset_id, keys[1], "txt")
        flagged = storage.flagged_media_ids(
            dataset_id, "txt", storage.REVIEW_FILTER_TO_REVIEW
        )
        assert storage.count_media(dataset_id, media_id_filter=flagged) == 1
        page = storage.list_media_page(
            dataset_id, 0, 50, media_id_filter=flagged
        )
        assert [item["key"] for item in page] == [keys[0]]

    def test_empty_filter_yields_nothing(self, review_ref):
        """An empty flagged set pages to zero rows (not 'all')."""
        dataset_id, _, _ = review_ref
        assert storage.count_media(dataset_id, media_id_filter=set()) == 0
        assert (
            storage.list_media_page(dataset_id, 0, 50, media_id_filter=set())
            == []
        )
