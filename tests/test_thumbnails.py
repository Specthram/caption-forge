"""Tests for the thumbnail cache in :mod:`src.thumbnails`."""

import cv2
import numpy as np
from PIL import Image

from src import thumbnails


def _write_image(path, size=(800, 600)):
    """Write a plain RGB image to ``path`` and return it."""
    Image.new("RGB", size, (10, 20, 30)).save(path)
    return path


def _write_video(path, size=(64, 48)):
    """Write a tiny two-frame mp4 to ``path`` and return it."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, size
    )
    for _ in range(2):
        writer.write(np.full((size[1], size[0], 3), 128, dtype=np.uint8))
    writer.release()
    return path


class TestEnsureThumbnail:
    """Generating and caching a thumbnail from a source media file."""

    def test_generates_a_resized_image_thumbnail(
        self, thumb_cache_dir, tmp_path
    ):
        """A large image is cached, resized under the thumbnail bound."""
        source = _write_image(tmp_path / "a.png")
        dest = thumbnails.ensure_thumbnail(source, "sha-a")
        assert dest == thumbnails.thumbnail_path("sha-a")
        assert dest.exists()
        with Image.open(dest) as cached:
            assert max(cached.size) <= thumbnails.THUMBNAIL_SIZE

    def test_generates_a_frame_thumbnail_for_video(
        self, thumb_cache_dir, tmp_path
    ):
        """A video's first frame is cached as its thumbnail."""
        source = _write_video(tmp_path / "clip.mp4")
        dest = thumbnails.ensure_thumbnail(source, "sha-video")
        assert dest is not None
        assert dest.exists()

    def test_second_call_reuses_the_cached_file(
        self, thumb_cache_dir, tmp_path, monkeypatch
    ):
        """An already-cached thumbnail is not regenerated."""
        source = _write_image(tmp_path / "a.png")
        thumbnails.ensure_thumbnail(source, "sha-a")
        calls = []
        monkeypatch.setattr(
            thumbnails,
            "_generate_image",
            lambda *a, **k: calls.append(1) or True,
        )
        thumbnails.ensure_thumbnail(source, "sha-a")
        assert calls == []

    def test_force_regenerates_an_already_cached_thumbnail(
        self, thumb_cache_dir, tmp_path
    ):
        """A forced refresh overwrites an existing cached file."""
        source = _write_image(tmp_path / "a.png")
        dest = thumbnails.ensure_thumbnail(source, "sha-a")
        dest.write_bytes(b"stale placeholder, not a real jpeg")
        refreshed = thumbnails.ensure_thumbnail(source, "sha-a", force=True)
        assert refreshed == dest
        with Image.open(dest) as cached:
            assert max(cached.size) <= thumbnails.THUMBNAIL_SIZE

    def test_without_force_a_cached_file_is_left_untouched(
        self, thumb_cache_dir, tmp_path
    ):
        """Without force, an already-cached file is left untouched."""
        source = _write_image(tmp_path / "a.png")
        dest = thumbnails.ensure_thumbnail(source, "sha-a")
        dest.write_bytes(b"stale placeholder, not a real jpeg")
        thumbnails.ensure_thumbnail(source, "sha-a")
        assert dest.read_bytes() == b"stale placeholder, not a real jpeg"

    def test_different_content_gets_its_own_cache_file(
        self, thumb_cache_dir, tmp_path
    ):
        """Two distinct sha256 keys never collide on one cache file."""
        source_a = _write_image(tmp_path / "a.png")
        source_b = _write_image(tmp_path / "b.png", size=(400, 300))
        dest_a = thumbnails.ensure_thumbnail(source_a, "sha-a")
        dest_b = thumbnails.ensure_thumbnail(source_b, "sha-b")
        assert dest_a != dest_b

    def test_unreadable_file_returns_none(self, thumb_cache_dir, tmp_path):
        """A corrupt/unsupported file fails gracefully instead of raising."""
        bogus = tmp_path / "bogus.png"
        bogus.write_bytes(b"not an image")
        assert thumbnails.ensure_thumbnail(bogus, "sha-bogus") is None
        assert not thumbnails.thumbnail_path("sha-bogus").exists()


class TestThumbnailPath:
    """The cache path is sharded by sha256 prefix (see the Index feature)."""

    def test_sharded_by_first_two_hex_chars(self, thumb_cache_dir):
        """A media's cache file lives in a 2-hex-char prefix sub-folder."""
        path = thumbnails.thumbnail_path("abcd1234")
        assert path.parent.name == "ab"
        assert path.name == "abcd1234.jpg"

    def test_two_media_sharing_a_prefix_land_in_the_same_folder(
        self, thumb_cache_dir
    ):
        """Distinct media with the same 2-char prefix share the sub-folder."""
        first = thumbnails.thumbnail_path("ab111111")
        second = thumbnails.thumbnail_path("ab222222")
        assert first.parent == second.parent
        assert first != second


class TestProbeDimensions:
    """Tests for :func:`thumbnails.probe_dimensions`."""

    def test_image_dimensions(self, tmp_path):
        """An image's own (unscaled) size is returned."""
        source = _write_image(tmp_path / "a.png", size=(800, 600))
        assert thumbnails.probe_dimensions(source) == (800, 600)

    def test_video_dimensions(self, tmp_path):
        """A video's frame size is read from its container metadata."""
        source = _write_video(tmp_path / "clip.mp4", size=(64, 48))
        assert thumbnails.probe_dimensions(source) == (64, 48)

    def test_unreadable_file_returns_none(self, tmp_path):
        """A corrupt/unsupported file fails gracefully instead of raising."""
        bogus = tmp_path / "bogus.png"
        bogus.write_bytes(b"not an image")
        assert thumbnails.probe_dimensions(bogus) is None
