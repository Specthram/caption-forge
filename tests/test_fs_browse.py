"""Tests for the server-side folder browser (Libraries picker)."""

from src import fs_browse


def test_lists_only_subfolders_sorted(tmp_path):
    """A directory lists its sub-folders (not files), sorted, with paths."""
    (tmp_path / "beta").mkdir()
    (tmp_path / "Alpha").mkdir()
    (tmp_path / "note.txt").write_text("x")
    result = fs_browse.browse(str(tmp_path))
    assert result["is_root"] is False
    assert [entry["name"] for entry in result["entries"]] == ["Alpha", "beta"]
    assert result["entries"][0]["path"] == str(tmp_path / "Alpha")
    assert result["path"] == str(tmp_path)
    assert result["parent"] == str(tmp_path.parent)


def test_hidden_folders_skipped(tmp_path):
    """A dot-prefixed folder is left out of the listing."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "data").mkdir()
    names = [e["name"] for e in fs_browse.browse(str(tmp_path))["entries"]]
    assert names == ["data"]


def test_missing_path_climbs_to_nearest_existing_ancestor(tmp_path):
    """A gone folder (moved/deleted library) opens on its nearest parent."""
    (tmp_path / "data").mkdir()
    result = fs_browse.browse(str(tmp_path / "data" / "gone" / "deeper"))
    assert result["path"] == str(tmp_path / "data")
    assert result["is_root"] is False


def test_empty_path_is_the_root_level():
    """An empty path returns the top level (drives, or "/"), never errors."""
    result = fs_browse.browse("")
    if fs_browse.list_drives():
        assert result["is_root"] is True
        assert result["parent"] is None
        assert result["entries"]
    else:
        assert result["path"] == "/"


def test_browse_files_lists_dirs_then_matching_files(tmp_path):
    """A file browse lists sub-folders first, then only matching files."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "model.safetensors").write_bytes(b"x")
    (tmp_path / "weights.gguf").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    result = fs_browse.browse_files(str(tmp_path), ["safetensors", "gguf"])
    entries = result["entries"]
    assert [(e["name"], e["kind"]) for e in entries] == [
        ("sub", "dir"),
        ("model.safetensors", "file"),
        ("weights.gguf", "file"),
    ]
    assert result["parent"] == str(tmp_path.parent)


def test_browse_files_empty_exts_lists_every_file(tmp_path):
    """No extension filter lists every file alongside the folders."""
    (tmp_path / "a.bin").write_bytes(b"x")
    names = [
        e["name"] for e in fs_browse.browse_files(str(tmp_path))["entries"]
    ]
    assert names == ["a.bin"]


def _make_tree(root):
    """Seed a nested folder tree of media/non-media files under ``root``."""
    (root / "a.png").write_bytes(b"x")
    (root / "note.txt").write_text("x")
    robe = root / "robe"
    robe.mkdir()
    (robe / "b.jpg").write_bytes(b"x")
    (robe / "c.jpg").write_bytes(b"x")
    ete = robe / "ete"
    ete.mkdir()
    (ete / "d.webp").write_bytes(b"x")
    (root / ".git").mkdir()
    (root / ".git" / "e.png").write_bytes(b"x")


def test_folder_tree_counts_own_and_recursive_total(tmp_path):
    """Each folder reports its direct media and its recursive total."""
    _make_tree(tmp_path)
    tree = fs_browse.folder_tree(str(tmp_path))
    # Root: one direct media (a.png), four in total (b, c, d counted; the
    # .git folder and note.txt are skipped).
    assert tree["own"] == 1
    assert tree["total"] == 4
    assert [child["name"] for child in tree["children"]] == ["robe"]
    robe = tree["children"][0]
    assert robe["rel_path"] == "robe"
    assert robe["own"] == 2
    assert robe["total"] == 3
    ete = robe["children"][0]
    assert ete["rel_path"] == "robe/ete"
    assert ete["own"] == 1
    assert ete["total"] == 1


def test_folder_tree_samples_capped_at_three(tmp_path):
    """A folder keeps at most three sample media paths."""
    for index in range(5):
        (tmp_path / f"{index}.png").write_bytes(b"x")
    tree = fs_browse.folder_tree(str(tmp_path))
    assert len(tree["samples"]) == 3


def test_folder_tree_missing_path_is_empty(tmp_path):
    """A non-existent folder returns an empty tree, never raises."""
    tree = fs_browse.folder_tree(str(tmp_path / "gone"))
    assert tree["total"] == 0
    assert tree["children"] == []
