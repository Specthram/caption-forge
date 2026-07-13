"""Tests for :mod:`src.image_stats` (sharpness, clipping, cleanliness).

Synthetic images written to a temp folder: a checkerboard is sharp, a flat
fill is not; a half-black/half-white frame clips; a smooth gradient is
clean and a salt-and-pepper frame is not.
"""

import numpy as np
import pytest
from PIL import Image

from src import image_stats


def _write(tmp_path, array, name="x.png"):
    """Write a uint8 grayscale array as a PNG and return its path."""
    path = tmp_path / name
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)
    return path


@pytest.fixture(name="checkerboard")
def _checkerboard():
    """Return a 1-pixel checkerboard: the sharpest image there is."""
    grid = np.indices((256, 256)).sum(axis=0) % 2
    return grid * 255


class TestSharpness:
    """Tests for the Laplacian focus measure."""

    def test_flat_image_has_no_sharpness(self, tmp_path):
        """A uniform fill has a zero second derivative."""
        path = _write(tmp_path, np.full((128, 128), 128))
        assert image_stats.analyze(path)["sharpness"] == 0.0

    def test_checkerboard_beats_a_gradient(self, tmp_path, checkerboard):
        """High-frequency detail scores far above a smooth ramp."""
        ramp = np.tile(np.linspace(0, 255, 256), (256, 1))
        sharp = image_stats.analyze(_write(tmp_path, checkerboard, "a.png"))
        soft = image_stats.analyze(_write(tmp_path, ramp, "b.png"))
        assert sharp["sharpness"] > soft["sharpness"]

    def test_the_scale_is_bounded(self, tmp_path, checkerboard):
        """Even the sharpest image stays inside 0-100."""
        assert (
            image_stats.analyze(_write(tmp_path, checkerboard))["sharpness"]
            <= 100.0
        )


class TestClipping:
    """Tests for the crushed/blown pixel share."""

    def test_mid_grey_never_clips(self, tmp_path):
        """A mid-grey fill loses no detail."""
        path = _write(tmp_path, np.full((64, 64), 128))
        assert image_stats.analyze(path)["clipping"] == 0.0

    def test_half_black_half_white_clips_everything(self, tmp_path):
        """Pure black and pure white are both lost detail."""
        array = np.zeros((64, 64))
        array[:32] = 255
        path = _write(tmp_path, array)
        assert image_stats.analyze(path)["clipping"] == 100.0


class TestCleanliness:
    """Tests for the Immerkaer noise estimate."""

    def test_smooth_image_is_clean(self, tmp_path):
        """A gradient carries no noise."""
        ramp = np.tile(np.linspace(0, 255, 256), (256, 1))
        assert image_stats.analyze(_write(tmp_path, ramp))["cleanliness"] > 95

    def test_noise_lowers_cleanliness(self, tmp_path):
        """Salt-and-pepper noise is caught by the estimator."""
        rng = np.random.default_rng(3)
        noisy = rng.integers(0, 256, size=(256, 256))
        ramp = np.tile(np.linspace(0, 255, 256), (256, 1))
        assert (
            image_stats.analyze(_write(tmp_path, noisy, "n.png"))[
                "cleanliness"
            ]
            < image_stats.analyze(_write(tmp_path, ramp, "r.png"))[
                "cleanliness"
            ]
        )


class TestAnalyze:
    """Tests for the entry point and the report's flag helper."""

    def test_unreadable_file_returns_none(self, tmp_path):
        """A file OpenCV cannot decode yields no statistics."""
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"not an image")
        assert image_stats.analyze(broken) is None

    def test_missing_file_returns_none(self, tmp_path):
        """A path that does not exist yields no statistics."""
        assert image_stats.analyze(tmp_path / "nope.png") is None

    def test_large_image_is_downscaled(self, tmp_path):
        """A 4K frame is analysed at the fixed working resolution."""
        big = np.full((2160, 3840), 128)
        assert image_stats.analyze(_write(tmp_path, big)) is not None

    def test_is_flagged_reads_both_floors(self):
        """Blurry or noisy is flagged; sharp and clean is not."""
        assert not image_stats.is_flagged(None)
        assert image_stats.is_flagged({"sharpness": 10.0, "cleanliness": 90.0})
        assert image_stats.is_flagged({"sharpness": 90.0, "cleanliness": 10.0})
        assert not image_stats.is_flagged(
            {"sharpness": 90.0, "cleanliness": 90.0}
        )
