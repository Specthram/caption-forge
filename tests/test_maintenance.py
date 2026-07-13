"""Tests for the System view's database/cache cleanup sweeps.

Exercises :mod:`src.maintenance` and the store helpers it drives against a
fresh in-memory database and throwaway cache folders, so nothing touches the
developer's real data.
"""

import pytest

from src import maintenance, wm_compose
from src import sqlite_store as store


@pytest.fixture(name="patch_cache_dirs")
def _patch_cache_dirs(monkeypatch, tmp_path):
    """Redirect the patch and composite caches to throwaway folders."""
    patches = tmp_path / "patches"
    composed = tmp_path / "composed"
    patches.mkdir()
    composed.mkdir()
    monkeypatch.setattr(wm_compose, "get_patches_dir", lambda: patches)
    monkeypatch.setattr(wm_compose, "get_composed_dir", lambda: composed)
    return patches, composed


# -- orphan media -----------------------------------------------------------


def test_orphan_media_only_counts_unreferenced_rows(store_db):
    """A media is orphan only with no file row and no dataset membership."""
    orphan = store.get_or_create_media("sha_orphan", "png")

    with_file = store.get_or_create_media("sha_file", "png")
    store.add_media_file(with_file, "/some/where.png", library_id=None)

    in_dataset = store.get_or_create_media("sha_dataset", "png")
    dataset_id = store.create_dataset("d")
    store.add_media_to_dataset(dataset_id, in_dataset)

    assert store.count_orphan_media() == 1
    assert store.orphan_media_ids() == [orphan]


def test_purge_orphan_media_removes_only_the_orphan(store_db):
    """Purging drops the orphan row and leaves the referenced ones intact."""
    orphan = store.get_or_create_media("sha_orphan", "png")
    kept = store.get_or_create_media("sha_file", "png")
    store.add_media_file(kept, "/some/where.png")

    result = maintenance.run_cleanup("media")

    assert result == {"purged": 1, "bytes": 0, "vacuumed": True}
    assert store.get_media(orphan) is None
    assert store.get_media(kept) is not None


# -- unused caption versions ------------------------------------------------


def _build_caption_history(media_id):
    """Save three type-scoped revisions; return ``(caption_id, rev_ids)``."""
    dataset_id = store.create_dataset("history")
    store.add_media_to_dataset(dataset_id, media_id)
    caption_type = store.get_or_create_caption_type("caption")
    revs = [
        store.save_caption(dataset_id, media_id, caption_type, text)
        for text in ("v1", "v2", "v3")
    ]
    caption_id = store.get_caption(media_id, caption_type)["id"]
    return dataset_id, caption_id, revs


def test_unused_revision_count_excludes_head_and_pinned(store_db):
    """Head and a dataset-pinned revision are never counted as unused."""
    media_id = store.get_or_create_media("sha", "png")
    dataset_id, caption_id, (rev1, _rev2, rev3) = _build_caption_history(
        media_id
    )

    # Head is rev3; rev1 and rev2 are superseded history.
    assert store.head_revision_id(caption_id) == rev3
    assert store.unused_revision_count() == 2

    # Pinning rev1 rescues it from the sweep.
    store.set_dataset_caption(dataset_id, caption_id, "pinned", rev1)
    assert store.unused_revision_count() == 1


def test_prune_unused_revisions_keeps_head_and_pinned(store_db):
    """Pruning drops only the superseded revisions and unbreaks the chain."""
    media_id = store.get_or_create_media("sha", "png")
    dataset_id, caption_id, (rev1, rev2, rev3) = _build_caption_history(
        media_id
    )
    store.set_dataset_caption(dataset_id, caption_id, "pinned", rev1)

    result = maintenance.run_cleanup("captions")

    assert result == {"purged": 1, "bytes": 0, "vacuumed": True}
    assert store.get_revision(rev2) is None
    assert store.get_revision(rev1) is not None
    assert store.head_revision_id(caption_id) == rev3
    # The head no longer points at the deleted middle revision.
    assert store.get_revision(rev3)["parent_revision_id"] is None


# -- orphan patch / composite cache files -----------------------------------


def test_orphan_patches_spare_the_referenced_patch(store_db, patch_cache_dirs):
    """Only patch/composite files with no live owner are swept."""
    patches, composed = patch_cache_dirs
    media_id = store.get_or_create_media("sha", "png")
    zone_id = store.create_zone(
        media_id, {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    )

    live_patch = patches / f"{zone_id}.png"
    live_patch.write_bytes(b"live")
    orphan_patch = patches / "999999.png"
    orphan_patch.write_bytes(b"orphan")

    shard = composed / "ab"
    shard.mkdir()
    orphan_composite = shard / "abdeadbeef.png"
    orphan_composite.write_bytes(b"stale")

    report = maintenance.orphan_patch_report()
    assert report["count"] == 2

    result = maintenance.run_cleanup("patches")
    assert result["purged"] == 2
    assert result["vacuumed"] is False
    assert live_patch.exists()
    assert not orphan_patch.exists()
    assert not orphan_composite.exists()


# -- thumbnail cache --------------------------------------------------------


def test_purge_thumbnail_cache_empties_the_tree(store_db, thumb_cache_dir):
    """Every cached thumbnail is removed; the sweep reports what it freed."""
    shard = thumb_cache_dir / "ab"
    shard.mkdir(parents=True)
    (shard / "abcd.jpg").write_bytes(b"x" * 10)
    (shard / "abef.jpg").write_bytes(b"y" * 20)

    assert maintenance.thumbnail_report() == {"count": 2, "bytes": 30}

    result = maintenance.run_cleanup("thumbs")
    assert result == {"purged": 2, "bytes": 30, "vacuumed": False}
    assert maintenance.thumbnail_report() == {"count": 0, "bytes": 0}


def test_run_cleanup_rejects_unknown_category(store_db):
    """An unknown category is a hard error, not a silent no-op."""
    with pytest.raises(ValueError):
        maintenance.run_cleanup("bogus")
