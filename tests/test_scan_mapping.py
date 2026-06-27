"""Tests for the mapping-aware library scan (routing + folder tags)."""

from src import db
from src import sqlite_store as store


def _media_id(path) -> int:
    """Return the media id ingested for a file path."""
    with db.connect() as conn:
        row = conn.execute(
            "SELECT media_id FROM media_file WHERE path = ?", (str(path),)
        ).fetchone()
    return row["media_id"]


def _names(path) -> set:
    """Return the tag names attached to the media at ``path``."""
    return set(store.media_tag_names(_media_id(path)))


def _build(tmp_path):
    """Seed a nested folder tree and return its paths."""
    root = tmp_path / "vet"
    (root / "robe" / "ete").mkdir(parents=True)
    (root / "sys").mkdir()
    files = {
        "top": root / "top.png",
        "a": root / "robe" / "a.png",
        "b": root / "robe" / "b.png",
        "c": root / "robe" / "ete" / "c.png",
        "junk": root / "sys" / "junk.png",
    }
    for path in files.values():
        path.write_bytes(path.name.encode())
    return root, files


class TestScanRoutingAndTags:
    """A mapped scan skips, routes and tags files by their folder chain."""

    def test_exclude_and_effective_tags(self, tmp_path, store_db):
        # pylint: disable=unused-argument
        """Excluded files are skipped; kept files get chain-effective tags."""
        root, files = _build(tmp_path)
        parent = store.create_library("vet", str(root), recursive=True)
        store.apply_folder_mapping(
            parent,
            "all",
            [
                {"rel_path": "robe", "mode": "keep", "tags": ["fabric"]},
                {"rel_path": "sys", "mode": "exclude"},
            ],
        )
        summary = store.scan_library(parent)

        assert summary["skipped"] == 1
        # top + a + b + c ingested into the parent; sys/junk skipped.
        assert store.count_media_in_library(parent) == 4
        assert _names(files["top"]) == set()
        assert _names(files["a"]) == {"robe", "fabric"}
        assert _names(files["c"]) == {"robe", "fabric", "ete"}

    def test_sublib_routes_media_to_its_library(self, tmp_path, store_db):
        # pylint: disable=unused-argument
        """A sublib folder routes its whole subtree to the sub-library."""
        root, files = _build(tmp_path)
        parent = store.create_library("vet", str(root), recursive=True)
        store.apply_folder_mapping(
            parent,
            "all",
            [
                {
                    "rel_path": "robe",
                    "mode": "sublib",
                    "sub_name": "Robe",
                },
                {"rel_path": "sys", "mode": "exclude"},
            ],
        )
        store.scan_library(parent)

        subs = store.list_sub_libraries(parent)
        assert [row["name"] for row in subs] == ["Robe"]
        sub_id = subs[0]["id"]
        # robe/{a,b} and robe/ete/c belong to the sub-library, not the parent.
        assert store.count_media_in_library(sub_id) == 3
        assert store.count_media_in_library(parent) == 1
        assert _media_id(files["a"]) is not None

    def test_deep_nesting_routes_and_tags(self, tmp_path, store_db):
        # pylint: disable=unused-argument
        """A 4-level tree routes to a mid-chain sublib and tags the chain."""
        deep = tmp_path / "bijoux" / "rollex" / "daytona" / "macro"
        deep.mkdir(parents=True)
        leaf = deep / "watch.png"
        leaf.write_bytes(b"watch")
        parent = store.create_library("bijoux", str(tmp_path), recursive=True)
        store.apply_folder_mapping(
            parent,
            "all",
            [
                {
                    "rel_path": "bijoux/rollex",
                    "mode": "sublib",
                    "sub_name": "Rolex",
                }
            ],
        )
        store.scan_library(parent)

        subs = store.list_sub_libraries(parent)
        assert [row["name"] for row in subs] == ["Rolex"]
        # The deepest file belongs to the mid-chain sub-library, not parent.
        assert store.count_media_in_library(subs[0]["id"]) == 1
        assert store.count_media_in_library(parent) == 0
        # It carries every ancestor's auto tag, down four levels.
        assert _names(leaf) == {"bijoux", "rollex", "daytona", "macro"}

    def test_demote_sublib_merges_media_back(self, tmp_path, store_db):
        # pylint: disable=unused-argument
        """Demoting a sublib to keep prunes it and merges media to parent."""
        root, files = _build(tmp_path)
        parent = store.create_library("vet", str(root), recursive=True)
        store.apply_folder_mapping(
            parent,
            "0",
            [{"rel_path": "robe", "mode": "sublib", "sub_name": "Robe"}],
        )
        store.scan_library(parent)
        assert len(store.list_sub_libraries(parent)) == 1

        store.apply_folder_mapping(
            parent, "0", [{"rel_path": "robe", "mode": "keep"}]
        )
        store.scan_library(parent)
        assert store.list_sub_libraries(parent) == []
        # robe/{a,b,c} merged back: all five media now in the parent.
        assert store.count_media_in_library(parent) == 5
        assert _media_id(files["c"]) is not None

    def test_reapply_reroutes_existing_media(self, tmp_path, store_db):
        # pylint: disable=unused-argument
        """Promoting a folder to a sublib re-points already-scanned files."""
        root, files = _build(tmp_path)
        parent = store.create_library("vet", str(root), recursive=True)
        store.scan_library(parent)  # flat first scan: all in the parent
        assert store.count_media_in_library(parent) == 5

        store.apply_folder_mapping(
            parent,
            "0",
            [{"rel_path": "robe", "mode": "sublib", "sub_name": "Robe"}],
        )
        store.scan_library(parent)
        sub_id = store.list_sub_libraries(parent)[0]["id"]
        assert store.count_media_in_library(sub_id) == 3
        # top + sys/junk stay in the parent (sys is not excluded here).
        assert store.count_media_in_library(parent) == 2
        assert _media_id(files["c"]) is not None
