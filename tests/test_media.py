"""Tests for the media file helpers in :mod:`src.media`."""

from src.media import (
    caption_path,
    get_media_files,
    read_caption,
)


class TestGetMediaFiles:
    """Tests for :func:`get_media_files`."""

    def _make_tree(self, tmp_path):
        """Create a small media tree (top-level + nested) and a non-media."""
        (tmp_path / "a.png").write_bytes(b"x")
        (tmp_path / "notes.txt").write_bytes(b"x")  # not a media file
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.jpg").write_bytes(b"y")

    def test_non_recursive_lists_top_level_only(self, tmp_path):
        """Without recursion only the directory's own media are returned."""
        self._make_tree(tmp_path)
        assert get_media_files(tmp_path) == [str(tmp_path / "a.png")]

    def test_recursive_descends_into_subfolders(self, tmp_path):
        """With recursion nested media are included, sorted, non-media out."""
        self._make_tree(tmp_path)
        assert get_media_files(tmp_path, recursive=True) == [
            str(tmp_path / "a.png"),
            str(tmp_path / "sub" / "b.jpg"),
        ]


class TestCaptionPath:
    """Tests for :func:`caption_path`."""

    def test_simple_name(self):
        """A plain media name gets the caption extension."""
        assert caption_path("/data/img.png").name == "img.txt"

    def test_dotted_stem_is_preserved(self):
        """A dotted name keeps its full stem (with_name, not suffix)."""
        assert caption_path("/data/a.b.png").name == "a.b.txt"

    def test_custom_extension(self):
        """A custom extension is honored."""
        assert caption_path("/data/img.png", "booru").name == "img.booru"

    def test_leading_dot_in_extension_is_stripped(self):
        """A leading dot in the extension argument is ignored."""
        assert caption_path("/data/img.png", ".caption").name == "img.caption"


class TestReadCaption:
    """Tests for reading sidecar captions from disk."""

    def test_read_existing_is_trimmed(self, tmp_path):
        """An existing caption is read back, trimmed of whitespace."""
        (tmp_path / "img.txt").write_text("a cat\n", encoding="utf-8")
        assert read_caption(str(tmp_path / "img.png")) == "a cat"

    def test_read_missing_returns_empty(self, tmp_path):
        """Reading a non-existent caption yields an empty string."""
        assert read_caption(str(tmp_path / "img.png")) == ""

    def test_custom_extension(self, tmp_path):
        """The caption is read from the requested extension's file."""
        (tmp_path / "img.booru").write_text("tag", encoding="utf-8")
        assert read_caption(str(tmp_path / "img.png"), "booru") == "tag"
