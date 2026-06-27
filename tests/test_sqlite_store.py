"""Tests for :mod:`src.sqlite_store` (the SQLite repository)."""

from contextlib import closing

import pytest

from src import db
from src import sqlite_store as store


@pytest.fixture(name="media_file")
def _media_file(tmp_path):
    """Return a small on-disk image file path (bytes are arbitrary)."""
    path = tmp_path / "pics" / "img.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image-bytes")
    return path


class TestDatasets:
    """Tests for dataset CRUD."""

    def test_create_and_list(self, store_db):
        """A created dataset shows up in the listing, description kept."""
        store.create_dataset("alpha")
        store.create_dataset("beta", "second one")
        rows = store.list_datasets()
        assert [d["name"] for d in rows] == ["alpha", "beta"]
        assert rows[1]["description"] == "second one"

    def test_duplicate_name_raises(self, store_db):
        """Two datasets cannot share a name."""
        store.create_dataset("alpha")
        with pytest.raises(ValueError):
            store.create_dataset("alpha")

    def test_empty_name_raises(self, store_db):
        """An empty name is rejected."""
        with pytest.raises(ValueError):
            store.create_dataset("   ")

    def test_delete(self, store_db):
        """A deleted dataset is gone from the listing."""
        dataset_id = store.create_dataset("alpha")
        store.delete_dataset(dataset_id)
        assert store.list_datasets() == []


class TestMedia:
    """Tests for media creation and de-duplication."""

    def test_create_stores_hash_and_extension(self, store_db, media_file):
        """A new media stores its extension and content hash."""
        media_id = store.ingest_file(media_file)[0]
        row = store.get_media(media_id)
        assert row["file_extension"] == "png"
        assert row["sha256"] == store.compute_sha256(media_file)

    def test_same_path_reuses_row(self, store_db, media_file):
        """Adding the same path twice reuses one media row."""
        first = store.ingest_file(media_file)[0]
        second = store.ingest_file(media_file)[0]
        assert first == second


def _media(tmp_path, name, data) -> int:
    """Ingest a small file with distinct ``data`` bytes; return its id."""
    path = tmp_path / name
    path.write_bytes(data)
    return store.ingest_file(str(path))[0]


class TestMediaSort:
    """Tests for the date/quality/dimension sort keys (the Index action)."""

    def test_quality_desc_puts_best_first_unindexed_last(
        self, store_db, tmp_path
    ):
        """Higher quality first; a media with no score sorts to the bottom."""
        low = _media(tmp_path, "a.png", b"a")
        high = _media(tmp_path, "b.png", b"b")
        unscored = _media(tmp_path, "c.png", b"c")
        store.upsert_media_quality(low, "musiq", 20.0)
        store.upsert_media_quality(high, "musiq", 90.0)
        ids = [
            r["id"]
            for r in store.list_library_media(
                store.SORT_QUALITY_DESC, quality_metric_selected="musiq"
            )
        ]
        assert ids == [high, low, unscored]

    def test_quality_asc_puts_worst_first_unscored_still_last(
        self, store_db, tmp_path
    ):
        """Ascending ranks worst first but still pushes an unscored media
        last (NULL last in both directions)."""
        low = _media(tmp_path, "a.png", b"a")
        high = _media(tmp_path, "b.png", b"b")
        unscored = _media(tmp_path, "c.png", b"c")
        store.upsert_media_quality(low, "musiq", 20.0)
        store.upsert_media_quality(high, "musiq", 90.0)
        ids = [
            r["id"]
            for r in store.list_library_media(
                store.SORT_QUALITY_ASC, quality_metric_selected="musiq"
            )
        ]
        assert ids == [low, high, unscored]

    def test_quality_sort_by_average_across_metrics(self, store_db, tmp_path):
        """The average sort ranks by each media's normalized mean score."""
        low = _media(tmp_path, "a.png", b"a")
        high = _media(tmp_path, "b.png", b"b")
        # high: TOPIQ 0.90 (->90) + Q-Align 4.5 (->87.5) -> mean 88.75.
        store.upsert_media_quality(high, "topiq_nr", 0.90)
        store.upsert_media_quality(high, "qalign", 4.5)
        # low: MUSIQ 20 (->20).
        store.upsert_media_quality(low, "musiq", 20.0)
        ids = [
            r["id"]
            for r in store.list_library_media(
                store.SORT_QUALITY_DESC, quality_metric_selected="average"
            )
        ]
        assert ids == [high, low]

    def test_dimension_desc_puts_largest_area_first(self, store_db, tmp_path):
        """Sorted by width*height, largest first."""
        small = _media(tmp_path, "a.png", b"a")
        big = _media(tmp_path, "b.png", b"b")
        store.set_media_index(small, 10, 10)
        store.set_media_index(big, 100, 100)
        ids = [
            r["id"]
            for r in store.list_library_media(store.SORT_DIMENSION_DESC)
        ]
        assert ids[:2] == [big, small]

    def test_default_sort_is_newest_first(self, store_db, tmp_path):
        """With no sort given, the newest media (by id) comes first."""
        first = _media(tmp_path, "a.png", b"a")
        second = _media(tmp_path, "b.png", b"b")
        ids = [r["id"] for r in store.list_library_media()]
        assert ids == [second, first]

    def test_filtered_listing_honors_the_same_sort(self, store_db, tmp_path):
        """A tag-filtered listing sorts identically to the unfiltered one."""
        category_id = store.create_tag_category("c")
        tag_id = store.get_or_create_tag("t", category_id)
        low = _media(tmp_path, "a.png", b"a")
        high = _media(tmp_path, "b.png", b"b")
        store.add_tag_to_media(low, tag_id)
        store.add_tag_to_media(high, tag_id)
        store.upsert_media_quality(low, "musiq", 20.0)
        store.upsert_media_quality(high, "musiq", 90.0)
        ids = [
            r["id"]
            for r in store.library_media_filtered(
                [tag_id],
                "any",
                store.SORT_QUALITY_DESC,
                quality_metric_selected="musiq",
            )
        ]
        assert ids == [high, low]


class TestTagsForMediaBulk:
    """The bulk tag reader chunks its ids under SQLite's variable limit."""

    def test_handles_id_list_over_the_variable_limit(self, store_db, tmp_path):
        """A huge id list is chunked, not passed as one oversized IN ()."""
        category_id = store.create_tag_category("c")
        tag_id = store.get_or_create_tag("t", category_id)
        media_id = _media(tmp_path, "a.png", b"a")
        store.add_tag_to_media(media_id, tag_id)
        # Far more ids than SQLite's bound-variable limit (autobuild's pool):
        # unchunked this raised "too many SQL variables".
        ids = [media_id] + list(range(10_000, 12_500))
        result = store.tags_for_media_bulk(ids)
        assert [row["name"] for row in result[media_id]] == ["t"]
        assert set(result) == {media_id}


class TestExcludeTagsFilter:
    """Tests for the exclude-tags filter on the library listings."""

    def test_excludes_media_carrying_the_tag(self, store_db, tmp_path):
        """A media with an excluded tag is dropped from page and count."""
        cat = store.create_tag_category("c")
        excl = store.get_or_create_tag("nsfw", cat)
        tagged = _media(tmp_path, "a.png", b"a")
        plain = _media(tmp_path, "b.png", b"b")
        store.add_tag_to_media(tagged, excl)
        assert store.count_library_media(exclude_tag_ids=[excl]) == 1
        rows = store.library_media_page(
            None, "any", store.SORT_DATE_DESC, 0, 50, exclude_tag_ids=[excl]
        )
        assert [r["id"] for r in rows] == [plain]

    def test_exclude_combines_with_include(self, store_db, tmp_path):
        """Include and exclude filters both apply (include minus excluded)."""
        cat = store.create_tag_category("c")
        keep = store.get_or_create_tag("char", cat)
        excl = store.get_or_create_tag("nsfw", cat)
        both = _media(tmp_path, "a.png", b"a")
        only_keep = _media(tmp_path, "b.png", b"b")
        store.add_tag_to_media(both, keep)
        store.add_tag_to_media(both, excl)
        store.add_tag_to_media(only_keep, keep)
        ids = store.library_media_ids([keep], "any", exclude_tag_ids=[excl])
        assert ids == [only_keep]


class TestSetLibraryPath:
    """Tests for repointing a folder library at a new folder."""

    def test_repoints_a_folder_library(self, store_db, tmp_path):
        """The folder path is updated; media keep their recorded paths."""
        (tmp_path / "old").mkdir()
        (tmp_path / "new").mkdir()
        (tmp_path / "old" / "1.png").write_bytes(b"a")
        lib = store.create_library("L", str(tmp_path / "old"))
        media_id = store.ingest_file(str(tmp_path / "old" / "1.png"), lib)[0]
        store.set_library_path(lib, str(tmp_path / "new"))
        assert store.get_library(lib)["path"] == str(tmp_path / "new")
        # The existing file row keeps its original recorded path.
        assert media_id in store.library_media_ids()

    def test_rejects_internal_library(self, store_db, tmp_path):
        """The internal upload library cannot be repointed this way."""
        internal = store.get_internal_library()
        with pytest.raises(ValueError):
            store.set_library_path(internal["id"], str(tmp_path))

    def test_rejects_a_folder_already_taken(self, store_db, tmp_path):
        """Two libraries cannot share a folder."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        store.create_library("A", str(tmp_path / "a"))
        lib_b = store.create_library("B", str(tmp_path / "b"))
        with pytest.raises(ValueError):
            store.set_library_path(lib_b, str(tmp_path / "a"))


class TestLibraryMediaPaging:
    """Tests for the database-side library paging (the grid hot path)."""

    def _tagged_media(self, tmp_path, count=5):
        """Ingest ``count`` media, tag the even ones; return (ids, tag)."""
        category_id = store.create_tag_category("c")
        tag_id = store.get_or_create_tag("t", category_id)
        ids = []
        for i in range(count):
            media_id = _media(tmp_path, f"m{i}.png", bytes([i]))
            if i % 2 == 0:
                store.add_tag_to_media(media_id, tag_id)
            ids.append(media_id)
        return ids, tag_id

    def test_count_matches_the_materialized_listing(self, store_db, tmp_path):
        """The cheap count agrees with the full listing, filtered or not."""
        _, tag_id = self._tagged_media(tmp_path)
        assert store.count_library_media() == 5
        assert store.count_library_media([tag_id], "any") == 3

    def test_count_with_match_all(self, store_db, tmp_path):
        """The "all" match counts only media carrying every tag."""
        ids, tag_a = self._tagged_media(tmp_path, count=2)
        category_id = store.create_tag_category("d")
        tag_b = store.get_or_create_tag("u", category_id)
        store.add_tag_to_media(ids[0], tag_b)
        assert store.count_library_media([tag_a, tag_b], "all") == 1

    def test_page_slices_the_sorted_listing(self, store_db, tmp_path):
        """A page equals the same slice of the materialized listing."""
        self._tagged_media(tmp_path)
        full = [r["id"] for r in store.list_library_media()]
        page = [
            r["id"]
            for r in store.library_media_page(
                None, "any", store.SORT_DATE_DESC, 2, 2
            )
        ]
        assert page == full[2:4]

    def test_page_honors_the_tag_filter(self, store_db, tmp_path):
        """A filtered page only contains matching media."""
        _, tag_id = self._tagged_media(tmp_path)
        rows = store.library_media_page(
            [tag_id], "any", store.SORT_DATE_DESC, 0, 2
        )
        filtered = [r["id"] for r in store.library_media_filtered([tag_id])]
        assert [r["id"] for r in rows] == filtered[:2]

    def test_library_media_ids_matches_the_filter(self, store_db, tmp_path):
        """The id listing returns exactly the filtered identities."""
        _, tag_id = self._tagged_media(tmp_path)
        expected = {r["id"] for r in store.library_media_filtered([tag_id])}
        assert set(store.library_media_ids([tag_id], "any")) == expected


class TestBulkDatasetLinks:
    """Tests for the one-transaction bulk link/unlink operations."""

    def test_add_reports_only_new_links(self, store_db, tmp_path):
        """Already-linked media are ignored and not counted."""
        dataset_id = store.create_dataset("d")
        first = _media(tmp_path, "a.png", b"a")
        second = _media(tmp_path, "b.png", b"b")
        store.add_media_to_dataset(dataset_id, first)
        added = store.add_media_ids_to_dataset(dataset_id, [first, second])
        assert added == 1
        assert store.media_ids_in_dataset(dataset_id) == {first, second}

    def test_remove_reports_only_actual_unlinks(self, store_db, tmp_path):
        """Unknown ids are ignored and not counted."""
        dataset_id = store.create_dataset("d")
        first = _media(tmp_path, "a.png", b"a")
        second = _media(tmp_path, "b.png", b"b")
        store.add_media_ids_to_dataset(dataset_id, [first, second])
        removed = store.remove_media_ids_from_dataset(
            dataset_id, [first, 99999]
        )
        assert removed == 1
        assert store.media_ids_in_dataset(dataset_id) == {second}


# A fully indexed image also carries the model-free statistics of
# src.image_stats; without them it stays pending for the backfill pass.
_STATS = {"sharpness": 80.0, "clipping": 1.0, "cleanliness": 90.0}


class TestMediaIndex:
    """Tests for :func:`set_media_index` / :func:`media_pending_index`."""

    def test_set_media_index_persists_and_stamps_indexed_at(
        self, store_db, tmp_path
    ):
        """Dimensions persist and ``indexed_at`` gets stamped."""
        media_id = _media(tmp_path, "a.png", b"a")
        store.set_media_index(media_id, 640, 480)
        row = store.get_media(media_id)
        assert (row["width"], row["height"]) == (640, 480)
        assert row["indexed_at"] is not None

    def test_pending_index_defaults_to_never_indexed(self, store_db, tmp_path):
        """Without force, only media with no ``indexed_at`` are returned."""
        indexed = _media(tmp_path, "a.png", b"a")
        pending = _media(tmp_path, "b.png", b"b")
        store.set_media_index(indexed, 10, 10, phash="0" * 16, dhash="1" * 16)
        store.set_media_stats(indexed, _STATS)
        ids = {r["id"] for r in store.media_pending_index()}
        assert ids == {pending}

    def test_pending_index_force_returns_everything(self, store_db, tmp_path):
        """With force, an already-indexed media is returned too."""
        indexed = _media(tmp_path, "a.png", b"a")
        pending = _media(tmp_path, "b.png", b"b")
        store.set_media_index(indexed, 10, 10, phash="0" * 16, dhash="1" * 16)
        ids = {r["id"] for r in store.media_pending_index(force=True)}
        assert ids == {indexed, pending}

    def test_pending_index_scoped_to_one_library(self, store_db, tmp_path):
        """A library id restricts the pending list to its own media."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        lib_a = store.create_library("A", str(tmp_path / "a"))
        lib_b = store.create_library("B", str(tmp_path / "b"))
        path_a = tmp_path / "a" / "1.png"
        path_a.write_bytes(b"a")
        path_b = tmp_path / "b" / "1.png"
        path_b.write_bytes(b"b")
        media_a = store.ingest_file(str(path_a), lib_a)[0]
        media_b = store.ingest_file(str(path_b), lib_b)[0]
        ids = {r["id"] for r in store.media_pending_index(lib_a)}
        assert ids == {media_a}
        assert media_b not in ids

    def test_pending_index_ignores_metric_when_not_given(
        self, store_db, tmp_path
    ):
        """No ``metric`` argument keeps the metric-blind behavior."""
        indexed = _media(tmp_path, "a.png", b"a")
        pending = _media(tmp_path, "b.png", b"b")
        store.set_media_index(indexed, 10, 10, phash="0" * 16, dhash="1" * 16)
        store.set_media_stats(indexed, _STATS)
        store.upsert_media_quality(indexed, "musiq", 50.0)
        ids = {r["id"] for r in store.media_pending_index()}
        assert ids == {pending}

    def test_pending_index_flags_a_media_missing_the_metric(
        self, store_db, tmp_path
    ):
        """A metric switch resurfaces an indexed media lacking that score."""
        has_musiq = _media(tmp_path, "a.png", b"a")
        has_topiq = _media(tmp_path, "b.png", b"b")
        never_indexed = _media(tmp_path, "c.png", b"c")
        store.set_media_index(
            has_musiq, 10, 10, phash="0" * 16, dhash="1" * 16
        )
        store.set_media_stats(has_musiq, _STATS)
        store.upsert_media_quality(has_musiq, "musiq", 50.0)
        store.set_media_index(
            has_topiq, 10, 10, phash="2" * 16, dhash="3" * 16
        )
        store.set_media_stats(has_topiq, _STATS)
        store.upsert_media_quality(has_topiq, "topiq_nr", 0.8)
        ids = {r["id"] for r in store.media_pending_index(metric="musiq")}
        assert ids == {has_topiq, never_indexed}
        assert has_musiq not in ids

    def test_set_media_index_persists_perceptual_hashes(
        self, store_db, tmp_path
    ):
        """The ``phash``/``dhash`` hex strings round-trip through the DB."""
        media_id = _media(tmp_path, "a.png", b"a")
        store.set_media_index(
            media_id, 10, 10, phash="dead" * 4, dhash="beef" * 4
        )
        row = store.get_media(media_id)
        assert (row["phash"], row["dhash"]) == ("dead" * 4, "beef" * 4)

    def test_pending_index_backfills_a_media_missing_phash(
        self, store_db, tmp_path
    ):
        """A media indexed before hashes existed resurfaces for backfill."""
        no_hash = _media(tmp_path, "a.png", b"a")
        hashed = _media(tmp_path, "b.png", b"b")
        # ``no_hash`` was indexed (dimensions + score) before perceptual
        # hashing existed, so its phash stays NULL.
        store.set_media_index(no_hash, 10, 10)
        store.set_media_stats(no_hash, _STATS)
        store.upsert_media_quality(no_hash, "musiq", 50.0)
        store.set_media_index(hashed, 10, 10, phash="0" * 16, dhash="1" * 16)
        store.set_media_stats(hashed, _STATS)
        ids = {r["id"] for r in store.media_pending_index()}
        assert ids == {no_hash}

    def test_pending_index_backfills_an_image_missing_stats(
        self, store_db, tmp_path
    ):
        """An image indexed before image_stats existed resurfaces."""
        no_stats = _media(tmp_path, "a.png", b"a")
        complete = _media(tmp_path, "b.png", b"b")
        store.set_media_index(no_stats, 10, 10, phash="0" * 16, dhash="1" * 16)
        store.set_media_index(complete, 10, 10, phash="2" * 16, dhash="3" * 16)
        store.set_media_stats(complete, _STATS)
        ids = {r["id"] for r in store.media_pending_index()}
        assert ids == {no_stats}

    def test_pending_index_never_backfills_a_video(self, store_db, tmp_path):
        """A video carries no statistics, so it must not read as pending."""
        video = _media(tmp_path, "a.mp4", b"a")
        store.set_media_index(video, 10, 10, phash="0" * 16, dhash="1" * 16)
        assert store.media_pending_index() == []

    def test_set_media_stats_round_trips(self, store_db, tmp_path):
        """The three percentages come back on the media dict."""
        media_id = _media(tmp_path, "a.png", b"a")
        store.set_media_stats(media_id, _STATS)
        row = store.get_media_display(media_id)
        assert row["stats"] == _STATS


class TestMediaQuality:
    """Tests for per-metric quality scores and their display/average."""

    def test_upsert_is_idempotent_and_replaces(self, store_db, tmp_path):
        """A second upsert for the same metric replaces, not duplicates."""
        media_id = _media(tmp_path, "a.png", b"a")
        store.upsert_media_quality(media_id, "musiq", 50.0)
        store.upsert_media_quality(media_id, "musiq", 80.0)
        assert store.available_quality_metrics() == [("musiq", 1)]
        rows = store.library_media_page(
            None,
            "any",
            store.SORT_DATE_DESC,
            0,
            10,
            quality_metric_selected="musiq",
        )
        assert rows[0]["quality_score"] == 80.0

    def test_available_quality_metrics_counts_per_metric(
        self, store_db, tmp_path
    ):
        """Each stored metric is reported once with its media count."""
        first = _media(tmp_path, "a.png", b"a")
        second = _media(tmp_path, "b.png", b"b")
        store.upsert_media_quality(first, "musiq", 50.0)
        store.upsert_media_quality(second, "musiq", 60.0)
        store.upsert_media_quality(first, "topiq_nr", 0.7)
        assert dict(store.available_quality_metrics()) == {
            "musiq": 2,
            "topiq_nr": 1,
        }

    def test_selected_metric_absent_yields_no_badge(self, store_db, tmp_path):
        """A media without a score for the selected metric shows none."""
        media_id = _media(tmp_path, "a.png", b"a")
        store.upsert_media_quality(media_id, "musiq", 50.0)
        rows = store.library_media_page(
            None,
            "any",
            store.SORT_DATE_DESC,
            0,
            10,
            quality_metric_selected="topiq_nr",
        )
        assert rows[0]["quality_score"] is None
        assert rows[0]["quality_metric"] == "topiq_nr"

    def test_normalized_average_across_three_metrics(self, store_db, tmp_path):
        """0.90 TOPIQ + 4.5 Q-Align + 90 MUSIQ average to about 89."""
        media_id = _media(tmp_path, "a.png", b"a")
        store.upsert_media_quality(media_id, "topiq_nr", 0.90)
        store.upsert_media_quality(media_id, "qalign", 4.5)
        store.upsert_media_quality(media_id, "musiq", 90.0)
        rows = store.library_media_page(
            None,
            "any",
            store.SORT_DATE_DESC,
            0,
            10,
            quality_metric_selected="average",
        )
        assert rows[0]["quality_metric"] == "average"
        assert round(rows[0]["quality_score"]) == 89
        assert rows[0]["quality_scores"] == {
            "topiq_nr": 0.90,
            "qalign": 4.5,
            "musiq": 90.0,
        }


class TestMediaPendingScore:
    """Tests for :func:`media_pending_score` (the quality-run queue)."""

    def test_returns_media_missing_the_metric(self, store_db, tmp_path):
        """Only media with no score for the metric are pending by default."""
        scored = _media(tmp_path, "a.png", b"a")
        unscored = _media(tmp_path, "b.png", b"b")
        store.upsert_media_quality(scored, "musiq", 50.0)
        ids = {r["id"] for r in store.media_pending_score("musiq")}
        assert ids == {unscored}

    def test_force_returns_everything(self, store_db, tmp_path):
        """With force, an already-scored media is pending again."""
        scored = _media(tmp_path, "a.png", b"a")
        other = _media(tmp_path, "b.png", b"b")
        store.upsert_media_quality(scored, "musiq", 50.0)
        ids = {r["id"] for r in store.media_pending_score("musiq", force=True)}
        assert ids == {scored, other}

    def test_scoped_to_one_library(self, store_db, tmp_path):
        """A library id restricts the pending list to its own media."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        lib_a = store.create_library("A", str(tmp_path / "a"))
        lib_b = store.create_library("B", str(tmp_path / "b"))
        (tmp_path / "a" / "1.png").write_bytes(b"a")
        (tmp_path / "b" / "1.png").write_bytes(b"b")
        media_a = store.ingest_file(str(tmp_path / "a" / "1.png"), lib_a)[0]
        store.ingest_file(str(tmp_path / "b" / "1.png"), lib_b)
        ids = {r["id"] for r in store.media_pending_score("musiq", lib_a)}
        assert ids == {media_a}


class TestDeleteMediaQuality:
    """Tests for :func:`delete_media_quality`."""

    def test_deletes_the_selected_metrics(self, store_db, tmp_path):
        """Only the named metrics' rows are removed; others stay."""
        media_id = _media(tmp_path, "a.png", b"a")
        store.upsert_media_quality(media_id, "musiq", 50.0)
        store.upsert_media_quality(media_id, "topiq_nr", 0.7)
        removed = store.delete_media_quality(["musiq"])
        assert removed == 1
        assert dict(store.available_quality_metrics()) == {"topiq_nr": 1}

    def test_scoped_to_one_library(self, store_db, tmp_path):
        """Deleting for one library leaves another library's scores."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        lib_a = store.create_library("A", str(tmp_path / "a"))
        lib_b = store.create_library("B", str(tmp_path / "b"))
        (tmp_path / "a" / "1.png").write_bytes(b"a")
        (tmp_path / "b" / "1.png").write_bytes(b"b")
        media_a = store.ingest_file(str(tmp_path / "a" / "1.png"), lib_a)[0]
        media_b = store.ingest_file(str(tmp_path / "b" / "1.png"), lib_b)[0]
        store.upsert_media_quality(media_a, "musiq", 50.0)
        store.upsert_media_quality(media_b, "musiq", 60.0)
        store.delete_media_quality(["musiq"], library_id=lib_a)
        assert dict(store.available_quality_metrics()) == {"musiq": 1}

    def test_empty_metric_list_is_a_noop(self, store_db, tmp_path):
        """No metrics named deletes nothing."""
        media_id = _media(tmp_path, "a.png", b"a")
        store.upsert_media_quality(media_id, "musiq", 50.0)
        assert store.delete_media_quality([]) == 0
        assert dict(store.available_quality_metrics()) == {"musiq": 1}


class TestFavorite:
    """Tests for the media favorite flag and its listing filter."""

    def test_toggle_flips_and_reports(self, store_db, tmp_path):
        """Toggling returns the new state and persists it."""
        media_id = _media(tmp_path, "a.png", b"a")
        assert store.toggle_media_favorite(media_id) is True
        assert store.get_media(media_id)["favorite"] == 1
        assert store.toggle_media_favorite(media_id) is False
        assert store.get_media(media_id)["favorite"] == 0

    def test_set_favorite_explicit(self, store_db, tmp_path):
        """set_media_favorite writes the given boolean state."""
        media_id = _media(tmp_path, "a.png", b"a")
        store.set_media_favorite(media_id, True)
        assert store.get_media(media_id)["favorite"] == 1

    def test_media_dict_carries_favorite(self, store_db, tmp_path):
        """A listing dict exposes the favorite flag as a bool."""
        media_id = _media(tmp_path, "a.png", b"a")
        store.set_media_favorite(media_id, True)
        row = next(
            r for r in store.list_library_media() if r["id"] == media_id
        )
        assert row["favorite"] is True

    def test_favorites_only_filters_page_and_count(self, store_db, tmp_path):
        """The favorites_only flag restricts the paged listing and count."""
        fav = _media(tmp_path, "a.png", b"a")
        _media(tmp_path, "b.png", b"b")
        store.set_media_favorite(fav, True)
        assert store.count_library_media(favorites_only=True) == 1
        rows = store.library_media_page(
            None, "any", store.SORT_DATE_DESC, 0, 50, favorites_only=True
        )
        assert [r["id"] for r in rows] == [fav]


def _soft_delete(media_id: int) -> None:
    """Mark a media as deleted (a soft delete via ``deleted_at``)."""
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "UPDATE media SET deleted_at = datetime('now') WHERE id = ?",
                (media_id,),
            )


class TestMediaWithHashes:
    """Tests for the hashed-media listing and the unhashed count."""

    def test_returns_only_hashed_live_media(self, store_db, tmp_path):
        """Hashed live media are returned; NULL-phash and deleted excluded."""
        hashed = _media(tmp_path, "a.png", b"a")
        _media(tmp_path, "b.png", b"b")  # never hashed
        deleted = _media(tmp_path, "c.png", b"c")
        store.set_media_index(hashed, 640, 480, phash="a" * 16, dhash="b" * 16)
        store.upsert_media_quality(hashed, "musiq", 50.0)
        store.set_media_index(deleted, 10, 10, phash="c" * 16, dhash="d" * 16)
        _soft_delete(deleted)
        rows = store.media_with_hashes(quality_metric_selected="musiq")
        assert [r["id"] for r in rows] == [hashed]
        row = rows[0]
        assert (row["phash"], row["dhash"]) == ("a" * 16, "b" * 16)
        assert row["quality_score"] == 50.0
        assert row["eff_path"] is not None

    def test_count_without_hash_excludes_deleted(self, store_db, tmp_path):
        """The pending count sees only live, never-hashed media."""
        hashed = _media(tmp_path, "a.png", b"a")
        _media(tmp_path, "b.png", b"b")  # never hashed -> counted
        deleted = _media(tmp_path, "c.png", b"c")  # never hashed but deleted
        store.set_media_index(hashed, 10, 10, phash="a" * 16, dhash="b" * 16)
        _soft_delete(deleted)
        assert store.count_media_without_hash() == 1

    def test_empty_when_nothing_hashed(self, store_db, tmp_path):
        """No hashed media -> an empty listing."""
        _media(tmp_path, "a.png", b"a")
        assert store.media_with_hashes() == []

    def test_videos_excluded_from_detection(self, store_db, tmp_path):
        """A hashed video never enters the (image-only) lookalike set."""
        image = _media(tmp_path, "a.png", b"a")
        clip = _media(tmp_path, "b.mp4", b"b")
        store.set_media_index(image, 10, 10, phash="a" * 16, dhash="b" * 16)
        store.set_media_index(clip, 10, 10, phash="c" * 16, dhash="d" * 16)
        assert [r["id"] for r in store.media_with_hashes()] == [image]


class TestDiscard:
    """Tests for the discarded state (Lookalikes resolution)."""

    def test_discard_restore_roundtrip(self, store_db, tmp_path):
        """Discarding stamps ``discarded_at``; restoring clears it."""
        media_id = _media(tmp_path, "a.png", b"a")
        store.set_media_discarded(media_id)
        assert store.get_media(media_id)["discarded_at"] is not None
        store.restore_media(media_id)
        assert store.get_media(media_id)["discarded_at"] is None

    def test_list_discarded_media(self, store_db, tmp_path):
        """Only discarded live media appear, carrying ``discarded_at``."""
        kept = _media(tmp_path, "a.png", b"a")
        gone = _media(tmp_path, "b.png", b"b")
        store.set_media_discarded(gone)
        rows = store.list_discarded_media()
        assert [r["id"] for r in rows] == [gone]
        assert kept not in {r["id"] for r in rows}
        assert rows[0]["discarded_at"] is not None

    def test_library_listings_exclude_discarded(self, store_db, tmp_path):
        """A discarded media drops out of the library media listings."""
        keep = _media(tmp_path, "a.png", b"a")
        drop = _media(tmp_path, "b.png", b"b")
        store.set_media_discarded(drop)
        assert store.count_library_media() == 1
        ids = [
            r["id"]
            for r in store.library_media_page(
                None, "any", store.SORT_DATE_DESC, 0, 10
            )
        ]
        assert ids == [keep]

    def test_with_hashes_excludes_discarded(self, store_db, tmp_path):
        """A discarded media never feeds detection again."""
        media_id = _media(tmp_path, "a.png", b"a")
        store.set_media_index(media_id, 10, 10, phash="a" * 16, dhash="b" * 16)
        store.set_media_discarded(media_id)
        assert store.media_with_hashes() == []

    def test_pending_index_excludes_discarded(self, store_db, tmp_path):
        """A discarded media is not offered for (re)indexing."""
        media_id = _media(tmp_path, "a.png", b"a")
        store.set_media_discarded(media_id)
        assert store.media_pending_index() == []

    def test_ingest_does_not_restore_a_discarded_media(
        self, store_db, tmp_path
    ):
        """The key guarantee: a re-scan re-sees the file but keeps it aside.

        ``ingest_file`` re-attaches the file to its media (identity, tags
        and captions survive) but must leave ``discarded_at`` untouched — a
        discarded lookalike whose source file stays in the folder must not
        silently come back on the next scan.
        """
        path = tmp_path / "a.png"
        path.write_bytes(b"a")
        media_id = store.ingest_file(str(path))[0]
        category_id = store.create_tag_category("c")
        tag_id = store.get_or_create_tag("t", category_id)
        store.add_tag_to_media(media_id, tag_id)
        store.set_media_discarded(media_id)

        again, _fresh = store.ingest_file(str(path))

        assert again == media_id  # same media, not a new row
        assert store.get_media(media_id)["discarded_at"] is not None
        assert tag_id in {t["id"] for t in store.tags_for_media(media_id)}


class TestComposition:
    """Tests for composing a dataset from media."""

    def test_add_is_idempotent(self, store_db, media_file):
        """Linking the same media twice keeps a single link."""
        dataset_id = store.create_dataset("alpha")
        media_id = store.ingest_file(media_file)[0]
        store.add_media_to_dataset(dataset_id, media_id)
        store.add_media_to_dataset(dataset_id, media_id)
        assert store.media_ids_in_dataset(dataset_id) == {media_id}

    def test_remove(self, store_db, media_file):
        """Removing a media unlinks it from the dataset."""
        dataset_id = store.create_dataset("alpha")
        media_id = store.ingest_file(media_file)[0]
        store.add_media_to_dataset(dataset_id, media_id)
        store.remove_media_from_dataset(dataset_id, media_id)
        assert not store.is_media_in_dataset(dataset_id, media_id)

    def test_hidden_flag(self, store_db, media_file):
        """The hidden flag is stored and reported per dataset media."""
        dataset_id = store.create_dataset("alpha")
        media_id = store.ingest_file(media_file)[0]
        store.add_media_to_dataset(dataset_id, media_id)
        store.set_media_hidden(dataset_id, media_id, True)
        rows = store.media_in_dataset(dataset_id)
        assert rows[0]["hidden"] == 1


class TestRevisions:
    """Tests for the caption revision model (follow vs pinned)."""

    def _setup(self, media_file):
        """Return (d1, d2, media_id, txt_type_id) for a caption scenario."""
        d1 = store.create_dataset("pony6")
        d2 = store.create_dataset("illustrious")
        media_id = store.ingest_file(media_file)[0]
        txt = store.get_or_create_caption_type("txt")
        return d1, d2, media_id, txt

    def test_type_edit_is_shared_by_followers(self, store_db, media_file):
        """A type-scoped edit is seen by every following dataset."""
        d1, d2, media_id, txt = self._setup(media_file)
        store.save_caption(d1, media_id, txt, "v1", scope="type")
        assert store.read_caption(d1, media_id, txt) == "v1"
        assert store.read_caption(d2, media_id, txt) == "v1"

    def test_dataset_edit_is_isolated(self, store_db, media_file):
        """A dataset-scoped edit pins only that dataset; others unaffected."""
        d1, d2, media_id, txt = self._setup(media_file)
        store.save_caption(d1, media_id, txt, "v1", scope="type")
        store.save_caption(d1, media_id, txt, "v2-pony", scope="dataset")
        assert store.read_caption(d1, media_id, txt) == "v2-pony"
        assert store.read_caption(d2, media_id, txt) == "v1"

    def test_later_type_edit_skips_pinned_dataset(self, store_db, media_file):
        """A new head reaches followers but not a pinned dataset."""
        d1, d2, media_id, txt = self._setup(media_file)
        store.save_caption(d1, media_id, txt, "v1", scope="type")
        store.save_caption(d1, media_id, txt, "v2-pony", scope="dataset")
        store.save_caption(d2, media_id, txt, "v3-type", scope="type")
        assert store.read_caption(d1, media_id, txt) == "v2-pony"
        assert store.read_caption(d2, media_id, txt) == "v3-type"

    def test_history_lists_all_revisions(self, store_db, media_file):
        """Every saved revision appears in the history, newest first."""
        d1, _, media_id, txt = self._setup(media_file)
        store.save_caption(d1, media_id, txt, "v1", scope="type")
        store.save_caption(d1, media_id, txt, "v2", scope="type")
        caption = store.get_caption(media_id, txt)
        contents = [r["content"] for r in store.list_revisions(caption["id"])]
        assert contents == ["v2", "v1"]

    def test_pin_then_follow_restores_head(self, store_db, media_file):
        """Switching a pinned dataset back to follow shows the head again."""
        d1, _, media_id, txt = self._setup(media_file)
        store.save_caption(d1, media_id, txt, "v1", scope="type")
        store.save_caption(d1, media_id, txt, "v2-pony", scope="dataset")
        caption = store.get_caption(media_id, txt)
        store.set_dataset_caption(d1, caption["id"], "follow")
        assert store.read_caption(d1, media_id, txt) == "v1"

    def test_read_without_caption_is_empty(self, store_db, media_file):
        """A media with no caption of that type reads as empty."""
        d1, _, media_id, txt = self._setup(media_file)
        assert store.read_caption(d1, media_id, txt) == ""

    def test_amend_overwrites_in_place_no_new_revision(
        self, store_db, media_file
    ):
        """Autosave amends the head; repeated amends keep one revision."""
        d1, d2, media_id, txt = self._setup(media_file)
        store.amend_caption(d1, media_id, txt, "draft-1")
        store.amend_caption(d1, media_id, txt, "draft-2")
        store.amend_caption(d1, media_id, txt, "draft-3")
        caption = store.get_caption(media_id, txt)
        revisions = store.list_revisions(caption["id"])
        assert len(revisions) == 1  # never floods the history
        assert store.read_caption(d1, media_id, txt) == "draft-3"
        assert store.read_caption(d2, media_id, txt) == "draft-3"  # follower

    def test_save_after_amend_snapshots_a_version(self, store_db, media_file):
        """An explicit Save appends a new revision over the amended head."""
        d1, _, media_id, txt = self._setup(media_file)
        store.amend_caption(d1, media_id, txt, "working")
        store.save_caption(d1, media_id, txt, "committed", scope="type")
        caption = store.get_caption(media_id, txt)
        contents = [r["content"] for r in store.list_revisions(caption["id"])]
        assert contents == ["committed", "working"]  # snapshot on top

    def test_amend_pinned_dataset_edits_its_revision(
        self, store_db, media_file
    ):
        """Amending a pinned dataset rewrites its revision, not the head."""
        d1, d2, media_id, txt = self._setup(media_file)
        store.save_caption(d1, media_id, txt, "v1", scope="type")
        store.save_caption(d1, media_id, txt, "v2-pin", scope="dataset")
        store.amend_caption(d1, media_id, txt, "v2-edited")
        assert store.read_caption(d1, media_id, txt) == "v2-edited"
        assert store.read_caption(d2, media_id, txt) == "v1"  # head intact


class TestTriggerwords:
    """Tests for dataset trigger words and the export prefix."""

    def test_add_and_list_in_order(self, store_db):
        """Trigger words are listed in attachment order."""
        dataset_id = store.create_dataset("alpha")
        store.add_triggerword_to_dataset(dataset_id, "xbl1")
        store.add_triggerword_to_dataset(dataset_id, "xbl2")
        names = [w["name"] for w in store.dataset_triggerwords(dataset_id)]
        assert names == ["xbl1", "xbl2"]

    def test_add_is_idempotent(self, store_db):
        """Attaching the same word twice keeps a single link."""
        dataset_id = store.create_dataset("alpha")
        store.add_triggerword_to_dataset(dataset_id, "xbl1")
        store.add_triggerword_to_dataset(dataset_id, "xbl1")
        assert len(store.dataset_triggerwords(dataset_id)) == 1

    def test_prefix_format(self, store_db):
        """The export prefix is each word followed by '. ', in order."""
        dataset_id = store.create_dataset("alpha")
        store.add_triggerword_to_dataset(dataset_id, "xbl1")
        store.add_triggerword_to_dataset(dataset_id, "xbl2")
        assert store.triggerword_prefix(dataset_id) == "xbl1. xbl2. "

    def test_remove(self, store_db):
        """A removed trigger word leaves the dataset's prefix."""
        dataset_id = store.create_dataset("alpha")
        store.add_triggerword_to_dataset(dataset_id, "xbl1")
        word = store.dataset_triggerwords(dataset_id)[0]
        store.remove_triggerword_from_dataset(
            dataset_id, word["triggerword_id"]
        )
        assert store.triggerword_prefix(dataset_id) == ""

    def test_empty_prefix(self, store_db):
        """A dataset with no trigger words has an empty prefix."""
        dataset_id = store.create_dataset("alpha")
        assert store.triggerword_prefix(dataset_id) == ""


class TestUnusedCaptions:
    """Tests for cleaning up captions of media in no dataset."""

    def test_counts_and_deletes_only_orphans(self, store_db, tmp_path):
        """Captions of an orphaned media go; an in-dataset caption stays."""
        used = tmp_path / "used.png"
        used.write_bytes(b"a")
        orphan = tmp_path / "orphan.png"
        orphan.write_bytes(b"b")
        dataset_id = store.create_dataset("d")
        txt = store.get_or_create_caption_type("txt")
        used_id = store.ingest_file(str(used))[0]
        orphan_id = store.ingest_file(str(orphan))[0]
        store.add_media_to_dataset(dataset_id, used_id)
        store.add_media_to_dataset(dataset_id, orphan_id)
        store.save_caption(dataset_id, used_id, txt, "keep")
        store.save_caption(dataset_id, orphan_id, txt, "drop")
        # Orphan the second media: its caption is now used nowhere.
        store.remove_media_from_dataset(dataset_id, orphan_id)

        assert store.unused_caption_count() == 1
        assert store.delete_unused_captions() == 1
        assert store.unused_caption_count() == 0
        # The used caption survives; the orphan's caption is gone but the
        # media row itself is kept (it stays in the library).
        assert store.read_caption(dataset_id, used_id, txt) == "keep"
        assert store.get_caption(orphan_id, txt) is None
        assert store.get_media(orphan_id) is not None

    def test_nothing_to_clean(self, store_db, media_file):
        """A caption whose media is in a dataset is never counted/removed."""
        dataset_id = store.create_dataset("d")
        txt = store.get_or_create_caption_type("txt")
        media_id = store.ingest_file(media_file)[0]
        store.add_media_to_dataset(dataset_id, media_id)
        store.save_caption(dataset_id, media_id, txt, "x")
        assert store.unused_caption_count() == 0
        assert store.delete_unused_captions() == 0
        assert store.read_caption(dataset_id, media_id, txt) == "x"


class TestBulkVariableLimit:
    """Bulk readers chunk their IN () lookups under SQLite's variable cap.

    A whole-library read (a forced index over tens of thousands of media) once
    raised ``sqlite3.OperationalError: too many SQL variables``; the chunked
    readers must swallow any id count without a raise.
    """

    def test_bulk_readers_accept_a_huge_id_list(self, store_db, media_file):
        # pylint: disable=unused-argument
        real_id = store.ingest_file(media_file)[0]
        # Far beyond any SQLITE_MAX_VARIABLE_NUMBER (999 or 32766).
        ids = list(range(1, 60000)) + [real_id]
        assert real_id in store.media_files_bulk(ids)
        assert store.zones_bulk(ids) == {}
        assert store.media_hashes(ids)[real_id] == (None, None)
        assert real_id in store.media_index_info(ids)
        assert store.media_embeddings("m", ids) == {}
