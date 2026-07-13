"""Tests for the tag/category/media-tag layer in sqlite_store."""

from contextlib import closing

import pytest

from src import db
from src import sqlite_store as store


def test_existing_tag_names_returns_the_known_subset(store_db):
    # pylint: disable=unused-argument
    """Only names that already exist come back; unknowns and blanks drop."""
    category = store.get_or_create_uncategorized_category()
    store.get_or_create_tag("robe", category)
    store.get_or_create_tag("ete", category)
    result = store.existing_tag_names(["robe", "ete", "bijoux", "  ", "robe"])
    assert set(result) == {"robe", "ete"}


def _insert_media(image_path):
    """Insert a media (keyed by a fake unique hash) and a file; return its id.

    The path need not exist on disk — these tests only exercise tag and filter
    logic over media rows.
    """
    with closing(db.connect()) as conn:
        with conn:
            cur = conn.execute(
                "INSERT INTO media (sha256, file_extension) "
                "VALUES (?, 'png')",
                (image_path,),
            )
            media_id = cur.lastrowid
            conn.execute(
                "INSERT INTO media_file (media_id, path) VALUES (?, ?)",
                (media_id, image_path),
            )
        return media_id


class TestSeededCategories:
    """Tests for the default categories seeded by ensure_database."""

    def test_defaults_present(self, store_db):
        """The factory categories are seeded into a fresh database."""
        names = {row["name"] for row in store.list_tag_categories()}
        assert {"character", "style", "artist", "content"} <= names

    def test_not_resurrected_after_delete(self, store_db):
        """Re-running ensure_database does not recreate a deleted default."""
        character = store.get_tag_category(
            next(
                row["id"]
                for row in store.list_tag_categories()
                if row["name"] == "character"
            )
        )
        store.delete_tag_category(character["id"])
        db.ensure_database()
        names = {row["name"] for row in store.list_tag_categories()}
        assert "character" not in names


class TestCategories:
    """Tests for tag-category CRUD."""

    def test_create_and_color(self, store_db):
        """A category keeps the color it was created with."""
        cat_id = store.create_tag_category("mood", "#123456")
        assert store.get_tag_category(cat_id)["color"] == "#123456"

    def test_duplicate_raises(self, store_db):
        """Two categories cannot share a name."""
        store.create_tag_category("mood")
        with pytest.raises(ValueError):
            store.create_tag_category("mood")

    def test_empty_name_raises(self, store_db):
        """An empty category name is rejected."""
        with pytest.raises(ValueError):
            store.create_tag_category("   ")

    def test_update_color(self, store_db):
        """A category's color can be updated."""
        cat_id = store.create_tag_category("mood", "#000000")
        store.update_tag_category(cat_id, color="#ffffff")
        assert store.get_tag_category(cat_id)["color"] == "#ffffff"

    def test_delete_cascades_to_tags(self, store_db):
        """Deleting a category removes its tags."""
        cat_id = store.create_tag_category("mood")
        store.get_or_create_tag("happy", cat_id)
        store.delete_tag_category(cat_id)
        assert store.list_tags(cat_id) == []


class TestTags:
    """Tests for tags and media tagging."""

    def test_get_or_create_idempotent(self, store_db):
        """Creating the same tag twice in a category reuses the row."""
        cat_id = store.create_tag_category("mood")
        first = store.get_or_create_tag("happy", cat_id)
        assert store.get_or_create_tag("happy", cat_id) == first

    def test_reuse_across_categories_no_duplicate(self, store_db):
        """A name already in another category is reused, not cloned.

        The auto-tagger writer: ``blue`` living under ``color`` must be
        returned as-is, never re-created elsewhere just because WD14 hit it.
        """
        color = store.create_tag_category("color")
        existing = store.get_or_create_tag("blue", color)
        assert store.get_or_create_tag_reuse("blue") == existing

    def test_reuse_creates_new_name_in_uncategorized(self, store_db):
        """A genuinely new name lands in the "Uncategorized" holding pen."""
        tag_id = store.get_or_create_tag_reuse("teal")
        category = store.get_tag(tag_id)["category_id"]
        assert category == store.uncategorized_category_id()
        assert store.get_tag_category(category)["name"] == "Uncategorized"


class TestMoveTag:
    """Drag-and-drop: moving a tag between categories."""

    def test_plain_move(self, store_db):
        """Moving to a free category just re-points it, id unchanged."""
        color = store.create_tag_category("color")
        mood = store.create_tag_category("mood")
        tag = store.get_or_create_tag("blue", color)
        assert store.move_tag(tag, mood) == tag
        assert store.get_tag(tag)["category_id"] == mood

    def test_move_merges_on_name_collision(self, store_db):
        """A name already in the target merges; media are re-pointed."""
        color = store.create_tag_category("color")
        mood = store.create_tag_category("mood")
        target = store.get_or_create_tag("blue", mood)
        moved = store.get_or_create_tag("blue", color)
        media = _insert_media("m.png")
        store.add_tag_to_media(media, moved)
        assert store.move_tag(moved, mood) == target
        assert store.get_tag(moved) is None
        assert [t["id"] for t in store.tags_for_media(media)] == [target]

    def test_move_to_same_category_is_noop(self, store_db):
        """Moving to the current category returns it untouched."""
        color = store.create_tag_category("color")
        tag = store.get_or_create_tag("blue", color)
        assert store.move_tag(tag, color) == tag
        assert store.get_tag(tag)["category_id"] == color

    def test_move_unknown_category_raises(self, store_db):
        """Moving to a category that no longer exists is reported."""
        color = store.create_tag_category("color")
        tag = store.get_or_create_tag("blue", color)
        with pytest.raises(ValueError):
            store.move_tag(tag, 9999)


class TestDedupeTags:
    """Merging tags the auto-tagger cloned into its fallback category."""

    def test_keeps_real_category_and_repoints_media(self, store_db):
        """A WD14 duplicate is merged into the real tag; media re-pointed."""
        color = store.create_tag_category("color")
        wd14 = store.create_tag_category("wd14")
        real = store.get_or_create_tag("blue", color)
        clone = store.get_or_create_tag("blue", wd14)
        media = _insert_media("m.png")
        store.add_tag_to_media(media, clone)

        report = store.dedupe_tags(reserved_category_id=wd14)

        assert store.get_tag(clone) is None
        assert store.get_tag(real)["category_id"] == color
        assert [t["id"] for t in store.tags_for_media(media)] == [real]
        assert report == [
            {
                "name": "blue",
                "kept_tag_id": real,
                "kept_category_id": color,
                "merged": 1,
            }
        ]

    def test_prefers_real_even_when_clone_is_older(self, store_db):
        """The kept copy is the one outside WD14, regardless of age."""
        wd14 = store.create_tag_category("wd14")
        color = store.create_tag_category("color")
        clone = store.get_or_create_tag("red", wd14)  # older (lower id)
        real = store.get_or_create_tag("red", color)
        store.dedupe_tags(reserved_category_id=wd14)
        assert store.get_tag(clone) is None
        assert store.get_tag(real) is not None

    def test_repoint_skips_existing_link_no_double(self, store_db):
        """A media already tagged with both copies keeps a single link."""
        color = store.create_tag_category("color")
        wd14 = store.create_tag_category("wd14")
        real = store.get_or_create_tag("blue", color)
        clone = store.get_or_create_tag("blue", wd14)
        media = _insert_media("m.png")
        store.add_tag_to_media(media, real)
        store.add_tag_to_media(media, clone)
        store.dedupe_tags(reserved_category_id=wd14)
        assert [t["id"] for t in store.tags_for_media(media)] == [real]

    def test_unique_name_untouched_and_idempotent(self, store_db):
        """A name with no duplicate is left alone; a second run is a no-op."""
        color = store.create_tag_category("color")
        wd14 = store.create_tag_category("wd14")
        solo = store.get_or_create_tag("solo", wd14)
        store.get_or_create_tag("blue", color)
        store.get_or_create_tag("blue", wd14)
        assert len(store.dedupe_tags(reserved_category_id=wd14)) == 1
        assert store.dedupe_tags(reserved_category_id=wd14) == []
        assert store.get_tag(solo) is not None

    def test_deleted_category_raises_value_error(self, store_db):
        """A stale (deleted) category id is reported, not a raw FK crash.

        Regression: a category deleted from another tab/click while this one
        still held it selected used to hit an unhandled
        ``sqlite3.IntegrityError`` on the ``category_id`` foreign key.
        """
        cat_id = store.create_tag_category("mood")
        store.delete_tag_category(cat_id)
        with pytest.raises(ValueError):
            store.get_or_create_tag("happy", cat_id)

    def test_list_tags_carries_color(self, store_db):
        """A listed tag carries its category name and color."""
        cat_id = store.create_tag_category("mood", "#abcdef")
        store.get_or_create_tag("happy", cat_id)
        row = store.list_tags(cat_id)[0]
        assert row["color"] == "#abcdef"
        assert row["category_name"] == "mood"

    def test_rename_tag(self, store_db):
        """A tag can be renamed in place."""
        cat_id = store.create_tag_category("mood")
        tag_id = store.get_or_create_tag("happy", cat_id)
        store.rename_tag(tag_id, "cheerful")
        assert store.get_tag(tag_id)["name"] == "cheerful"

    def test_rename_tag_empty_name_raises(self, store_db):
        """Renaming to an empty name is rejected."""
        cat_id = store.create_tag_category("mood")
        tag_id = store.get_or_create_tag("happy", cat_id)
        with pytest.raises(ValueError):
            store.rename_tag(tag_id, "   ")

    def test_rename_tag_missing_raises(self, store_db):
        """Renaming a deleted (stale) tag id is reported, not a crash."""
        cat_id = store.create_tag_category("mood")
        tag_id = store.get_or_create_tag("happy", cat_id)
        store.delete_tag(tag_id)
        with pytest.raises(ValueError):
            store.rename_tag(tag_id, "cheerful")

    def test_rename_tag_duplicate_in_category_raises(self, store_db):
        """Renaming to a name already used in the same category is refused."""
        cat_id = store.create_tag_category("mood")
        store.get_or_create_tag("happy", cat_id)
        tag_id = store.get_or_create_tag("sad", cat_id)
        with pytest.raises(ValueError):
            store.rename_tag(tag_id, "happy")

    def test_rename_tag_same_name_in_other_category_is_fine(self, store_db):
        """The per-category uniqueness does not block cross-category reuse."""
        mood_id = store.create_tag_category("mood")
        other_id = store.create_tag_category("other")
        store.get_or_create_tag("happy", mood_id)
        tag_id = store.get_or_create_tag("cheerful", other_id)
        store.rename_tag(tag_id, "happy")
        assert store.get_tag(tag_id)["name"] == "happy"

    def test_tag_media_idempotent(self, store_db):
        """Tagging the same media twice with one tag links it once."""
        cat_id = store.create_tag_category("mood")
        tag_id = store.get_or_create_tag("happy", cat_id)
        media_id = _insert_media("ab/x.png")
        store.add_tag_to_media(media_id, tag_id)
        store.add_tag_to_media(media_id, tag_id)
        tags = store.tags_for_media(media_id)
        assert len(tags) == 1
        assert tags[0]["name"] == "happy"

    def test_remove_tag_from_media(self, store_db):
        """A tag can be detached from a media."""
        cat_id = store.create_tag_category("mood")
        tag_id = store.get_or_create_tag("happy", cat_id)
        media_id = _insert_media("ab/x.png")
        store.add_tag_to_media(media_id, tag_id)
        store.remove_tag_from_media(media_id, tag_id)
        assert store.tags_for_media(media_id) == []


class TestLibraryMediaFilter:
    """Tests for the library listing and tag filter."""

    def test_lists_all_media_whatever_storage(self, store_db):
        """Every non-deleted media is listed, whatever its source."""
        m1 = _insert_media("ab/x.png")
        m2 = _insert_media("/ext/y.png")
        ids = {row["id"] for row in store.list_library_media()}
        assert ids == {m1, m2}

    def test_filter_any(self, store_db):
        """``any`` keeps media carrying at least one of the tags."""
        cat_id = store.create_tag_category("mood")
        t1 = store.get_or_create_tag("a", cat_id)
        t2 = store.get_or_create_tag("b", cat_id)
        m1 = _insert_media("ab/1.png")
        m2 = _insert_media("ab/2.png")
        store.add_tag_to_media(m1, t1)
        store.add_tag_to_media(m2, t2)
        ids = {r["id"] for r in store.library_media_filtered([t1, t2], "any")}
        assert ids == {m1, m2}

    def test_filter_all(self, store_db):
        """``all`` keeps only media carrying every one of the tags."""
        cat_id = store.create_tag_category("mood")
        t1 = store.get_or_create_tag("a", cat_id)
        t2 = store.get_or_create_tag("b", cat_id)
        m1 = _insert_media("ab/1.png")
        m2 = _insert_media("ab/2.png")
        store.add_tag_to_media(m1, t1)
        store.add_tag_to_media(m1, t2)
        store.add_tag_to_media(m2, t1)
        ids = {r["id"] for r in store.library_media_filtered([t1, t2], "all")}
        assert ids == {m1}

    def test_empty_filter_returns_all(self, store_db):
        """An empty tag filter returns the whole library."""
        _insert_media("ab/1.png")
        _insert_media("ab/2.png")
        assert len(store.library_media_filtered([], "any")) == 2


class TestCountAndSearchTags:
    """Tests for the bounded tag queries a large catalogue relies on."""

    def test_count_tags_all_and_by_category(self, store_db):
        """The count reflects the whole catalogue or one category."""
        mood_id = store.create_tag_category("mood")
        other_id = store.create_tag_category("other")
        store.get_or_create_tag("happy", mood_id)
        store.get_or_create_tag("sad", mood_id)
        store.get_or_create_tag("cheerful", other_id)
        assert store.count_tags(mood_id) == 2
        assert store.count_tags(other_id) == 1
        assert store.count_tags() == 3

    def test_count_tags_with_query(self, store_db):
        """A name search narrows the count."""
        cat_id = store.create_tag_category("mood")
        store.get_or_create_tag("happy", cat_id)
        store.get_or_create_tag("sad", cat_id)
        assert store.count_tags(cat_id, "app") == 1
        assert store.count_tags(cat_id, "zzz") == 0

    def test_list_tags_page_is_bounded_and_ordered(self, store_db):
        """A page returns at most ``limit`` tags, ordered by name."""
        cat_id = store.create_tag_category("mood")
        for name in ("delta", "alpha", "charlie", "bravo"):
            store.get_or_create_tag(name, cat_id)
        page = store.list_tags_page(cat_id, limit=2, offset=0)
        assert [r["name"] for r in page] == ["alpha", "bravo"]
        page2 = store.list_tags_page(cat_id, limit=2, offset=2)
        assert [r["name"] for r in page2] == ["charlie", "delta"]

    def test_list_tags_page_with_query(self, store_db):
        """A name search filters the page."""
        cat_id = store.create_tag_category("mood")
        store.get_or_create_tag("long_hair", cat_id)
        store.get_or_create_tag("short_hair", cat_id)
        store.get_or_create_tag("blue_eyes", cat_id)
        page = store.list_tags_page(cat_id, "hair", limit=10, offset=0)
        assert {r["name"] for r in page} == {"long_hair", "short_hair"}

    def test_search_tags_caps_matches(self, store_db):
        """Matches beyond ``limit`` are not returned."""
        cat_id = store.create_tag_category("mood")
        for i in range(10):
            store.get_or_create_tag(f"tag_{i}", cat_id)
        rows = store.search_tags("tag_", limit=3)
        assert len(rows) == 3

    def test_search_tags_always_includes_given_ids(self, store_db):
        """A selected tag stays in the results even if it does not match."""
        cat_id = store.create_tag_category("mood")
        happy_id = store.get_or_create_tag("happy", cat_id)
        store.get_or_create_tag("sad", cat_id)
        rows = store.search_tags(
            "zzz-no-match", limit=5, include_ids=[happy_id]
        )
        assert [r["id"] for r in rows] == [happy_id]

    def test_search_tags_no_query_returns_a_browse_list(self, store_db):
        """An empty query still returns a bounded default list."""
        cat_id = store.create_tag_category("mood")
        for i in range(5):
            store.get_or_create_tag(f"tag_{i}", cat_id)
        rows = store.search_tags(limit=3)
        assert len(rows) == 3

    def test_search_tags_ranks_exact_match_first(self, store_db):
        """An exact match outranks longer tags that merely contain it.

        Regression: results were ordered by category then name, so an exact
        match like "horse" could land in the middle of the list behind
        "drafthorse" and "laika horse" just because of category grouping.
        """
        cat_id = store.create_tag_category("mood")
        store.get_or_create_tag("laika horse", cat_id)
        store.get_or_create_tag("drafthorse", cat_id)
        store.get_or_create_tag("horse", cat_id)
        rows = store.search_tags("horse", limit=10)
        assert [r["name"] for r in rows][0] == "horse"

    def test_search_tags_ranks_prefix_before_suffix_before_contains(
        self, store_db
    ):
        """Prefix matches rank above suffix matches, above mid-string ones."""
        cat_id = store.create_tag_category("mood")
        store.get_or_create_tag("seahorse_rider", cat_id)  # contains only
        store.get_or_create_tag("drafthorse", cat_id)  # ends with
        store.get_or_create_tag("horseback", cat_id)  # starts with
        rows = store.search_tags("horse", limit=10)
        names = [r["name"] for r in rows]
        assert names == ["horseback", "drafthorse", "seahorse_rider"]

    def test_list_tags_page_ranks_exact_match_first(self, store_db):
        """The paginated Tags-tab search ranks the same way as the dropdown."""
        cat_id = store.create_tag_category("mood")
        store.get_or_create_tag("laika horse", cat_id)
        store.get_or_create_tag("drafthorse", cat_id)
        store.get_or_create_tag("horse", cat_id)
        page = store.list_tags_page(cat_id, "horse", limit=10, offset=0)
        assert [r["name"] for r in page][0] == "horse"

    def test_search_tags_reports_usage_count(self, store_db):
        """``usage_count`` reflects how many media currently carry the tag."""
        cat_id = store.create_tag_category("mood")
        happy_id = store.get_or_create_tag("happy", cat_id)
        media_a = _insert_media("a.png")
        media_b = _insert_media("b.png")
        store.add_tag_to_media(media_a, happy_id)
        store.add_tag_to_media(media_b, happy_id)
        rows = {r["id"]: r for r in store.search_tags("happy")}
        assert rows[happy_id]["usage_count"] == 2

    def test_search_tags_boosts_already_used_tags(self, store_db):
        """A used tag ranks before an unused one in the same match tier.

        Both names only *contain* "hair" (neither starts nor ends with it),
        so they land in the same relevance tier; alphabetically "aaa_hair"
        would normally sort first, but tagging "bbb_hair" to a media should
        boost it ahead regardless.
        """
        cat_id = store.create_tag_category("mood")
        unused_id = store.get_or_create_tag("aaa_hairstyle", cat_id)
        used_id = store.get_or_create_tag("bbb_hairstyle", cat_id)
        media_id = _insert_media("a.png")
        store.add_tag_to_media(media_id, used_id)
        rows = store.search_tags("hair", limit=10)
        assert [r["id"] for r in rows] == [used_id, unused_id]

    def test_search_tags_exact_match_still_wins_over_usage(self, store_db):
        """An exact name match outranks a used tag from a weaker tier."""
        cat_id = store.create_tag_category("mood")
        store.get_or_create_tag("horse", cat_id)
        used_id = store.get_or_create_tag("drafthorse", cat_id)
        media_id = _insert_media("a.png")
        store.add_tag_to_media(media_id, used_id)
        rows = store.search_tags("horse", limit=10)
        assert [r["name"] for r in rows][0] == "horse"

    def test_search_tags_no_query_boosts_used_tags(self, store_db):
        """The default (no search text) browse list also favors used tags."""
        cat_id = store.create_tag_category("mood")
        unused_id = store.get_or_create_tag("aaa_unused", cat_id)
        used_id = store.get_or_create_tag("zzz_used", cat_id)
        media_id = _insert_media("a.png")
        store.add_tag_to_media(media_id, used_id)
        rows = store.search_tags(limit=10)
        assert [r["id"] for r in rows] == [used_id, unused_id]


class TestBulkCreateTags:
    """Tests for the batched tag-import path."""

    def test_creates_new_tags_in_one_transaction(self, store_db):
        """Every (name, category) pair is created."""
        cat_id = store.create_tag_category("mood")
        created = store.bulk_create_tags(
            [("happy", cat_id), ("sad", cat_id), ("gloomy", cat_id)]
        )
        assert created == 3
        assert {r["name"] for r in store.list_tags(cat_id)} == {
            "happy",
            "sad",
            "gloomy",
        }

    def test_skips_names_already_used_in_the_category(self, store_db):
        """A duplicate (name, category) pair is ignored, not an error."""
        cat_id = store.create_tag_category("mood")
        store.get_or_create_tag("happy", cat_id)
        created = store.bulk_create_tags([("happy", cat_id), ("sad", cat_id)])
        assert created == 1
        assert store.count_tags(cat_id) == 2

    def test_missing_category_raises_value_error(self, store_db):
        """A nonexistent category id is reported, not a raw FK crash."""
        with pytest.raises(ValueError):
            store.bulk_create_tags([("happy", 999999)])

    def test_same_name_in_different_categories_both_created(self, store_db):
        """Per-category uniqueness allows the same name in two categories."""
        mood_id = store.create_tag_category("mood")
        other_id = store.create_tag_category("other")
        created = store.bulk_create_tags(
            [("happy", mood_id), ("happy", other_id)]
        )
        assert created == 2


class TestCategoryOrder:
    """Category ``position`` drives the list and per-media tag order."""

    def test_reorder_changes_relative_order(self, store_db):
        """Reordering swaps two categories' relative position."""
        cat_a = store.create_tag_category("zzz_alpha")
        cat_b = store.create_tag_category("zzz_beta")

        def relative():
            return [
                row["id"]
                for row in store.list_tag_categories()
                if row["id"] in (cat_a, cat_b)
            ]

        assert relative() == [cat_a, cat_b]  # position 0 -> tie broken by name
        store.reorder_tag_categories([cat_b, cat_a])
        assert relative() == [cat_b, cat_a]

    def test_media_tag_names_follow_category_order(self, store_db):
        """A media's tag names are grouped in category display order."""
        media_id = _insert_media("m1")
        cat_a = store.create_tag_category("zzz_alpha")
        cat_b = store.create_tag_category("zzz_beta")
        store.add_tag_to_media(media_id, store.get_or_create_tag("x", cat_b))
        store.add_tag_to_media(media_id, store.get_or_create_tag("y", cat_a))
        assert store.media_tag_names(media_id) == ["y", "x"]
        store.reorder_tag_categories([cat_b, cat_a])
        assert store.media_tag_names(media_id) == ["x", "y"]
