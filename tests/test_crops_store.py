"""Tests for the crop repository (:mod:`src.sqlite_store.crops`)."""

import pytest
from PIL import Image

from src import crops
from src import sqlite_store as store
from src import storage


@pytest.fixture(name="crop_ref")
def _crop_ref(tmp_path, store_db):
    # pylint: disable=unused-argument
    """Return ``(dataset_id, media_id)`` for one 800x600 image in a dataset."""
    image = tmp_path / "photo.png"
    Image.new("RGB", (800, 600), "red").save(image)
    media_id, _ = store.ingest_file(str(image))
    dataset_id = store.create_dataset("d")
    store.add_media_to_dataset(dataset_id, media_id)
    return dataset_id, media_id


RECT = {"x": 10, "y": 20, "w": 50, "h": 40}


class TestCreateCrop:
    """Creating a crop row and rendering its pixels."""

    def test_creates_a_virtual_media_with_the_rendered_size(self, crop_ref):
        """The crop carries its parent, its rectangle and its pixel size."""
        _, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT, "3:2")

        crop = store.get_crop(crop_id)
        assert crop["parent_media_id"] == media_id
        assert crop["rect"] == {"x": 10.0, "y": 20.0, "w": 50.0, "h": 40.0}
        assert crop["ratio"] == "3:2"
        assert (crop["width"], crop["height"]) == (400, 240)

    def test_effective_file_is_the_rendered_png(self, crop_ref):
        """A crop owns no media_file: its pixels resolve from the cache."""
        _, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT)

        rendered = store.effective_file(crop_id)
        assert rendered is not None
        with Image.open(rendered) as img:
            assert img.size == (400, 240)

    def test_an_identical_rectangle_reuses_the_same_crop(self, crop_ref):
        """The synthetic hash collides, so no duplicate row is created."""
        _, media_id = crop_ref
        first = store.create_crop(media_id, RECT)
        assert store.create_crop(media_id, dict(RECT)) == first
        assert len(store.list_crops(media_id)) == 1

    def test_a_crop_cannot_be_cropped_again(self, crop_ref):
        """Crops do not nest: the rectangle is always of a real image."""
        _, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT)
        with pytest.raises(ValueError, match="cannot be cropped again"):
            store.create_crop(crop_id, RECT)

    def test_an_unknown_parent_is_refused(self, store_db):
        # pylint: disable=unused-argument
        """A crop of nothing is a programming error, not an empty row."""
        with pytest.raises(ValueError, match="does not exist"):
            store.create_crop(999, RECT)


class TestListings:
    """Where a crop shows up, and where it must not."""

    def test_the_crop_is_excluded_from_the_library_grid(self, crop_ref):
        """A crop is not a library media: it has no file on disk."""
        _, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT)

        page = store.library_media_page(
            None, "any", store.SORT_DATE_DESC, offset=0, limit=10
        )
        assert [item["id"] for item in page] == [media_id]
        assert crop_id not in [item["id"] for item in page]
        assert store.count_library_media() == 1

    def test_the_crop_is_excluded_from_the_index_scope(self, crop_ref):
        """Index steps describe scanned files; a crop is not one of them."""
        _, media_id = crop_ref
        store.create_crop(media_id, RECT)

        assert store.count_live_media() == 1
        assert [row["id"] for row in store.media_pending_index()] == [media_id]

    def test_the_crop_appears_in_its_dataset(self, crop_ref):
        """The dataset grid shows the crop as a card of its own."""
        dataset_id, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT)
        store.place_crop(dataset_id, crop_id, "beside")

        page = store.media_in_dataset_page(dataset_id, 0, 10)
        assert [item["id"] for item in page] == [media_id, crop_id]
        assert page[1]["name"] == "photo.png · crop"
        assert page[1]["missing"] is False

    def test_list_crops_reports_the_datasets_it_stands_in(self, crop_ref):
        """The panel needs to know whether a crop is already placed."""
        dataset_id, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT)
        assert store.list_crops(media_id)[0]["dataset_ids"] == []

        store.place_crop(dataset_id, crop_id, "beside")
        assert store.list_crops(media_id)[0]["dataset_ids"] == [dataset_id]


class TestPlaceCrop:
    """Replacing the parent's entry, or standing beside it."""

    def test_replace_swaps_the_parent_entry(self, crop_ref):
        """The dataset keeps one sample, framed differently."""
        dataset_id, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT)

        assert store.place_crop(dataset_id, crop_id)["replaced"] is True
        assert store.media_ids_in_dataset(dataset_id) == {crop_id}

    def test_replace_inherits_the_parent_hidden_and_repeats(self, crop_ref):
        """The crop takes the parent's place, deploy settings included."""
        dataset_id, media_id = crop_ref
        store.set_media_repeats(dataset_id, media_id, 3)
        store.set_media_hidden(dataset_id, media_id, True)
        crop_id = store.create_crop(media_id, RECT)

        store.place_crop(dataset_id, crop_id)
        assert store.get_media_repeats(dataset_id, crop_id) == 3
        assert store.is_media_hidden(dataset_id, crop_id) is True

    def test_beside_keeps_both_entries(self, crop_ref):
        """The dataset holds two samples of the same image."""
        dataset_id, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT)

        placed = store.place_crop(dataset_id, crop_id, "beside")
        assert placed["replaced"] is False
        assert store.media_ids_in_dataset(dataset_id) == {media_id, crop_id}

    def test_placing_a_plain_media_is_refused(self, crop_ref):
        """Only crops are placed this way; media use add_media_to_dataset."""
        dataset_id, media_id = crop_ref
        with pytest.raises(ValueError, match="is not a crop"):
            store.place_crop(dataset_id, media_id)

    def test_replace_needs_the_parent_to_be_an_entry(self, crop_ref):
        """With nothing to replace, deleting it would *add* the parent."""
        _, media_id = crop_ref
        other = store.create_dataset("other")
        crop_id = store.create_crop(media_id, RECT)

        with pytest.raises(ValueError, match="is not in dataset"):
            store.place_crop(other, crop_id)
        assert store.media_ids_in_dataset(other) == set()

    def test_placing_an_already_placed_crop_is_a_no_op(self, crop_ref):
        """A replay must not swallow the parent's entry a second time."""
        dataset_id, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT)
        store.place_crop(dataset_id, crop_id, "beside")

        assert store.place_crop(dataset_id, crop_id)["replaced"] is False
        assert store.media_ids_in_dataset(dataset_id) == {media_id, crop_id}


class TestUpdateCrop:
    """Re-framing a crop re-hashes it and invalidates its measurements."""

    def test_the_rectangle_and_the_rendered_size_follow(self, crop_ref):
        """A new frame yields new pixels and a new size."""
        _, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT)

        crop = store.update_crop(crop_id, {"x": 0, "y": 0, "w": 25, "h": 50})
        assert (crop["width"], crop["height"]) == (200, 300)
        with Image.open(store.effective_file(crop_id)) as img:
            assert img.size == (200, 300)

    def test_the_stale_render_is_dropped_from_the_cache(self, crop_ref):
        """The old PNG is named after the old hash; nothing may reuse it."""
        _, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT)
        stale = crops.crop_path(store.get_crop(crop_id)["sha256"])
        assert stale.is_file()

        store.update_crop(crop_id, {"x": 0, "y": 0, "w": 25, "h": 50})
        assert not stale.is_file()

    def test_quality_and_tag_scores_are_invalidated(self, crop_ref):
        """Numbers measured on the old pixels describe nothing now."""
        _, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT)
        store.upsert_media_quality(crop_id, "musiq", 72.0)

        store.update_crop(crop_id, {"x": 0, "y": 0, "w": 25, "h": 50})
        assert store.get_media_display(crop_id)["quality_scores"] == {}

    def test_the_caption_survives_the_reframe(self, crop_ref):
        """Only measurements are dropped; the text the user wrote stays."""
        dataset_id, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT)
        store.place_crop(dataset_id, crop_id, "beside")
        storage.write_caption(dataset_id, str(crop_id), "txt", "a red square.")

        store.update_crop(crop_id, {"x": 0, "y": 0, "w": 25, "h": 50})
        assert (
            storage.read_caption(dataset_id, str(crop_id), "txt")
            == "a red square."
        )

    def test_reframing_onto_a_sibling_is_refused(self, crop_ref):
        """Two identical rectangles of one parent are one crop, not two."""
        _, media_id = crop_ref
        first = store.create_crop(media_id, RECT)
        second = store.create_crop(
            media_id, {"x": 0, "y": 0, "w": 25, "h": 25}
        )

        with pytest.raises(ValueError, match="already exists"):
            store.update_crop(second, RECT)
        assert store.get_crop(first)["rect"]["x"] == 10.0

    def test_updating_a_plain_media_is_refused(self, crop_ref):
        """An ordinary media has no rectangle to move."""
        _, media_id = crop_ref
        with pytest.raises(ValueError, match="is not a crop"):
            store.update_crop(media_id, RECT)


class TestDeleteCrop:
    """Deleting a crop never silently shrinks a dataset."""

    def test_the_parent_takes_its_place_back(self, crop_ref):
        """A crop that had replaced its parent hands the entry back."""
        dataset_id, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT)
        store.place_crop(dataset_id, crop_id)

        assert store.delete_crop(crop_id) == {
            "deleted": True,
            "restored": [dataset_id],
        }
        assert store.media_ids_in_dataset(dataset_id) == {media_id}

    def test_a_crop_placed_beside_just_goes(self, crop_ref):
        """The parent was already an entry: nothing to restore."""
        dataset_id, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT)
        store.place_crop(dataset_id, crop_id, "beside")

        assert store.delete_crop(crop_id)["restored"] == []
        assert store.media_ids_in_dataset(dataset_id) == {media_id}

    def test_the_cached_png_is_removed(self, crop_ref):
        """No orphan pixels are left behind on disk."""
        _, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT)
        rendered = crops.crop_path(store.get_crop(crop_id)["sha256"])

        store.delete_crop(crop_id)
        assert not rendered.is_file()

    def test_deleting_a_plain_media_is_a_no_op(self, crop_ref):
        """The route answers 404 rather than purging a real image."""
        _, media_id = crop_ref
        assert store.delete_crop(media_id) == {
            "deleted": False,
            "restored": [],
        }
        assert store.get_media(media_id) is not None

    def test_deleting_the_parent_cascades_onto_its_crops(self, crop_ref):
        """A crop cannot outlive the pixels it frames."""
        _, media_id = crop_ref
        crop_id = store.create_crop(media_id, RECT)

        store.purge_media([media_id])
        assert store.get_media(crop_id) is None
