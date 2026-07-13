"""Geometry of the FLUX patch: the paste-cover margin around a box.

The edit engine sends a wide crop (box + context + cover) to FLUX but returns
only the box grown by :data:`~src.wm_compose.PATCH_COVER_PX`, so tight
detection boxes still hide anti-aliased letter edges. No weights: the pipeline
is faked at its ``_run_pipe`` seam.
"""

import pytest
from PIL import Image

from src import watermark_flux as flux
from src import wm_compose


@pytest.fixture(name="identity_pipe", autouse=True)
def _identity_pipe(monkeypatch):
    """Pretend a pipe is loaded; the FLUX call returns its input unchanged."""
    monkeypatch.setattr(flux, "_pipe", object())
    monkeypatch.setattr(
        flux, "_run_pipe", lambda pipe, image, prompt, seed: image
    )


def test_patch_is_box_grown_by_cover(tmp_path):
    """The returned patch is the box grown by PATCH_COVER_PX on every side."""
    src = tmp_path / "img.png"
    Image.new("RGB", (1000, 800), (10, 20, 30)).save(src)
    # 200x160 px box, away from the edges so the dilation never clamps.
    box = {"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2}
    patch = flux.edit(str(src), box, dilate_px=8)
    cover = wm_compose.PATCH_COVER_PX
    assert patch.size == (200 + 2 * cover, 160 + 2 * cover)
    # Strictly larger than the bare detection box.
    assert patch.size != (200, 160)


def test_target_size_lifts_thin_strip_to_min():
    """A wide, short crop is scaled up so both sides clear the 64px floor."""
    width, height = flux._target_size(336, 48, 1024, "long")
    assert width >= 64 and height >= 64
    assert width % 16 == 0 and height % 16 == 0


def test_target_size_downscales_large_crop():
    """A large crop still fits the cap while keeping both sides >= 64."""
    width, height = flux._target_size(4000, 3000, 1024, "long")
    assert max(width, height) <= 1024
    assert min(width, height) >= 64


def test_edit_survives_thin_watermark(tmp_path):
    """A thin strip box no longer trips FLUX's minimum-size check."""
    src = tmp_path / "wide.png"
    Image.new("RGB", (2000, 500), (10, 20, 30)).save(src)
    # 1200x40 px strip (short side well under 64px before scaling).
    box = {"x": 0.1, "y": 0.5, "w": 0.6, "h": 0.08}
    patch = flux.edit(str(src), box, dilate_px=8)
    cover = wm_compose.PATCH_COVER_PX
    assert patch.size == (1200 + 2 * cover, 40 + 2 * cover)
