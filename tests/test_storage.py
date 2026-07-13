"""Tests for :mod:`src.storage` (the Caption-tab storage operations)."""

import os

import pytest

from src import sqlite_store as store
from src import storage


@pytest.fixture(name="sqlite_ref")
def _sqlite_ref(tmp_path, store_db):
    # pylint: disable=unused-argument
    """Dataset with one linked media; return (dataset_id, key, path).

    ``key`` is the media's database id (the opaque storage key), ``path`` the
    file it was ingested from.
    """
    image = tmp_path / "pics" / "a.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"x")
    dataset_id = store.create_dataset("d")
    media_id, _ = store.ingest_file(str(image))
    store.add_media_to_dataset(dataset_id, media_id)
    return dataset_id, str(media_id), str(image)


class TestSqliteMode:
    """The facade delegates to the revision repository."""

    def test_list_media_keyed_by_id(self, sqlite_ref):
        """Media are keyed by their database id, path is the effective file."""
        dataset_id, key, image = sqlite_ref
        items = storage.list_media(dataset_id)
        assert len(items) == 1
        assert items[0]["key"] == key
        assert items[0]["path"] == image

    def test_write_then_read_roundtrip(self, sqlite_ref):
        """A written caption is read back through the revision model."""
        dataset_id, key, _ = sqlite_ref
        storage.write_caption(dataset_id, key, "txt", "a caption")
        assert storage.read_caption(dataset_id, key, "txt") == "a caption"
        assert storage.present_types(dataset_id, key, ["txt", "booru"]) == [
            "txt"
        ]

    def test_dataset_scope_is_isolated(self, sqlite_ref):
        """A dataset-scoped edit does not leak to another dataset."""
        dataset_id, key, _ = sqlite_ref
        other = store.create_dataset("other")
        store.add_media_to_dataset(other, int(key))
        storage.write_caption(dataset_id, key, "txt", "shared", "type")
        storage.write_caption(dataset_id, key, "txt", "only-d", "dataset")
        assert storage.read_caption(dataset_id, key, "txt") == "only-d"
        assert storage.read_caption(other, key, "txt") == "shared"

    def test_caption_types_lists_db_types(self, sqlite_ref):
        """Caption types come from the database once one exists."""
        dataset_id, key, _ = sqlite_ref
        storage.write_caption(dataset_id, key, "txt", "x")
        assert "txt" in storage.caption_types()

    def test_revision_options_and_pin_follow(self, sqlite_ref):
        """List revisions; selecting one pins it, FOLLOW restores head."""
        dataset_id, key, _ = sqlite_ref
        storage.write_caption(dataset_id, key, "txt", "v1", "type")
        storage.write_caption(dataset_id, key, "txt", "v2", "type")

        choices, value = storage.revision_options(dataset_id, key, "txt")
        assert len(choices) == 3  # FOLLOW + two revisions
        assert value == storage.FOLLOW

        oldest_id = choices[-1][1]
        assert (
            storage.select_revision(dataset_id, key, "txt", oldest_id) == "v1"
        )
        _, pinned_value = storage.revision_options(dataset_id, key, "txt")
        assert pinned_value == oldest_id

        assert (
            storage.select_revision(dataset_id, key, "txt", storage.FOLLOW)
            == "v2"
        )

    def test_delete_unlinks_but_keeps_file(self, sqlite_ref):
        """Deleting only unlinks the media; its file stays on disk."""
        dataset_id, key, image = sqlite_ref
        storage.delete_media(dataset_id, key, ["txt"])
        assert storage.list_media(dataset_id) == []
        assert os.path.exists(image)


class TestHasDataset:
    """Tests for :func:`storage.has_dataset`."""

    def test_true_for_existing_id(self, sqlite_ref):
        """An existing dataset id is valid."""
        dataset_id, _, _ = sqlite_ref
        assert storage.has_dataset(dataset_id) is True

    def test_false_for_unknown_id(self, sqlite_ref):
        """An unknown dataset id is not valid."""
        assert storage.has_dataset(99999) is False

    def test_false_for_empty(self, sqlite_ref):
        """An empty reference is not a dataset."""
        # pylint: disable=unused-argument
        assert storage.has_dataset("") is False


class TestTagsType:
    """The virtual "tags" caption type reflects the media's gallery tags."""

    def test_caption_types_includes_tags(self, sqlite_ref):
        """The Caption tab's type list offers the virtual "tags" type."""
        assert storage.TAGS_TYPE in storage.caption_types()

    def test_read_joins_tags_in_category_order(self, sqlite_ref):
        """The tags caption is the media's tags, comma-joined by category."""
        dataset_id, key, _ = sqlite_ref
        media_id = int(key)
        cat_a = store.create_tag_category("acat")
        cat_b = store.create_tag_category("bcat")
        store.add_tag_to_media(media_id, store.get_or_create_tag("z", cat_a))
        store.add_tag_to_media(media_id, store.get_or_create_tag("y", cat_b))
        # Default positions are 0, so category name breaks the tie: acat first.
        assert (
            storage.read_caption(dataset_id, key, storage.TAGS_TYPE) == "z, y"
        )
        # Reordering the categories reorders the deployed tags.
        store.reorder_tag_categories([cat_b, cat_a])
        assert (
            storage.read_caption(dataset_id, key, storage.TAGS_TYPE) == "y, z"
        )

    def test_write_is_a_noop(self, sqlite_ref):
        """Writing the tags caption changes nothing (edited via the editor)."""
        dataset_id, key, _ = sqlite_ref
        storage.write_caption(dataset_id, key, storage.TAGS_TYPE, "a, b")
        assert storage.read_caption(dataset_id, key, storage.TAGS_TYPE) == ""
        names = {row["name"] for row in store.list_caption_types()}
        assert storage.TAGS_TYPE not in names  # never created a real type

    def test_present_when_tagged(self, sqlite_ref):
        """The tags type counts as present once the media has any tag."""
        dataset_id, key, _ = sqlite_ref
        cat = store.create_tag_category("c")
        store.add_tag_to_media(int(key), store.get_or_create_tag("blue", cat))
        present = storage.present_types(
            dataset_id, key, ["txt", storage.TAGS_TYPE]
        )
        assert storage.TAGS_TYPE in present and "txt" not in present

    def test_no_revisions(self, sqlite_ref):
        """The tags type has no revision history."""
        dataset_id, key, _ = sqlite_ref
        assert storage.revision_options(
            dataset_id, key, storage.TAGS_TYPE
        ) == ([], None)
