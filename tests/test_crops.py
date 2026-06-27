"""Tests for the virtual-crop engine (:mod:`src.crops`)."""

import pytest
from PIL import Image

from src import crops


@pytest.fixture(name="source_image")
def _source_image(tmp_path):
    """Return an 800x600 PNG on disk."""
    path = tmp_path / "source.png"
    Image.new("RGB", (800, 600), "red").save(path)
    return path


def test_normalize_rect_clamps_inside_the_source():
    """A rectangle hanging off the image is pulled back inside it."""
    assert crops.normalize_rect({"x": 80, "y": 90, "w": 50, "h": 40}) == {
        "x": 50.0,
        "y": 60.0,
        "w": 50.0,
        "h": 40.0,
    }


def test_normalize_rect_falls_back_to_the_full_frame():
    """Unparsable values yield the whole image rather than an error."""
    assert crops.normalize_rect({"x": "nope"}) == {
        "x": 0.0,
        "y": 0.0,
        "w": 100.0,
        "h": 100.0,
    }
    assert crops.is_full_frame(crops.normalize_rect(None))


def test_crop_sha256_is_stable_and_rect_sensitive():
    """The synthetic hash keys the crop's identity on parent + rectangle."""
    rect = {"x": 10, "y": 10, "w": 50, "h": 50}
    assert crops.crop_sha256("abc", rect) == crops.crop_sha256("abc", rect)
    assert crops.crop_sha256("abc", rect) != crops.crop_sha256("def", rect)
    moved = {"x": 11, "y": 10, "w": 50, "h": 50}
    assert crops.crop_sha256("abc", rect) != crops.crop_sha256("abc", moved)


def test_pixel_rect_resolves_percentages():
    """Percentages of the source become an inclusive-exclusive pixel box."""
    assert crops.pixel_rect(
        800, 600, {"x": 10, "y": 20, "w": 50, "h": 40}
    ) == (
        80,
        120,
        480,
        360,
    )


def test_pixel_rect_never_collapses_to_zero():
    """A degenerate rectangle still yields at least one pixel per side."""
    left, top, right, bottom = crops.pixel_rect(
        10, 10, {"x": 99, "y": 99, "w": 1, "h": 1}
    )
    assert right > left and bottom > top


def test_crop_size_reports_the_rendered_dimensions(source_image):
    """The rendered size is the pixel box of the rectangle."""
    assert crops.crop_size(
        source_image, {"x": 0, "y": 0, "w": 50, "h": 50}
    ) == (
        400,
        300,
    )


def test_ensure_render_writes_once_and_reuses_the_cache(
    source_image, crop_cache_dir
):
    """The PNG is rendered on first ask and served from cache afterwards."""
    rect = {"x": 10, "y": 20, "w": 50, "h": 40}
    sha = crops.crop_sha256("parent", rect)
    rendered = crops.ensure_render(source_image, sha, rect)

    assert rendered == crop_cache_dir / sha[:2] / f"{sha}.png"
    with Image.open(rendered) as img:
        assert img.size == (400, 240)

    stamp = rendered.stat().st_mtime_ns
    assert crops.ensure_render(source_image, sha, rect) == rendered
    assert rendered.stat().st_mtime_ns == stamp


def test_ensure_render_returns_none_when_the_source_is_gone(tmp_path):
    """A crop of a vanished parent is missing, not a crash."""
    rect = {"x": 0, "y": 0, "w": 50, "h": 50}
    sha = crops.crop_sha256("parent", rect)
    assert crops.ensure_render(tmp_path / "nope.png", sha, rect) is None


def test_delete_render_drops_the_cached_png(source_image):
    """Deleting a render is idempotent and reports whether it hit a file."""
    rect = {"x": 0, "y": 0, "w": 50, "h": 50}
    sha = crops.crop_sha256("parent", rect)
    crops.ensure_render(source_image, sha, rect)

    assert crops.delete_render(sha) is True
    assert crops.delete_render(sha) is False


def test_render_bakes_in_the_exif_orientation(tmp_path, crop_cache_dir):
    """The rectangle was framed on the oriented image, so the crop is too."""
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (800, 600), "blue")
    exif = image.getexif()
    exif[0x0112] = 6  # 90 degrees: the oriented image is 600x800.
    image.save(path, exif=exif)

    assert crops.source_size(path) == (600, 800)
    sha = crops.crop_sha256("parent", {"x": 0, "y": 0, "w": 100, "h": 50})
    rendered = crops.ensure_render(
        path, sha, {"x": 0, "y": 0, "w": 100, "h": 50}
    )
    with Image.open(rendered) as img:
        assert img.size == (600, 400)
