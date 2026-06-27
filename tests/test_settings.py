"""Tests for the pure validation helpers in :mod:`src.settings`."""

import pytest

from src.constants import (
    DEFAULT_CAPTION_EXTENSIONS,
    DEFAULT_GGUF_N_CTX,
    MAX_IMAGE_SIZE,
)
from src import config
from src import settings
from src.settings import (
    MAX_ALLOWED_IMAGE_SIZE,
    MAX_GGUF_N_CTX,
    MAX_VIDEO_FPS,
    MAX_VIDEO_SECONDS,
    MIN_GGUF_N_CTX,
    MIN_IMAGE_SIZE,
    MIN_VIDEO_FPS,
    clamp_caption_extensions,
    clamp_gguf_n_ctx,
    clamp_image_size,
    clamp_video_fps,
    estimate_video_frames,
)


class TestClampImageSize:
    """Tests for :func:`clamp_image_size`."""

    def test_value_in_range_is_kept(self):
        """A value within bounds is returned unchanged."""
        assert clamp_image_size("768") == 768

    def test_below_minimum_is_raised(self):
        """A value under the minimum is clamped up to it."""
        assert clamp_image_size("10") == MIN_IMAGE_SIZE

    def test_above_maximum_is_lowered(self):
        """A value over the maximum is clamped down to it."""
        assert clamp_image_size("99999") == MAX_ALLOWED_IMAGE_SIZE

    def test_invalid_value_falls_back_to_default(self):
        """A non-numeric value yields the default size."""
        assert clamp_image_size("abc") == MAX_IMAGE_SIZE


class TestClampVideoFps:
    """Tests for :func:`clamp_video_fps`."""

    def test_value_in_range_is_kept(self):
        """A float within bounds is returned unchanged."""
        assert clamp_video_fps("4") == 4.0

    def test_below_minimum_is_raised(self):
        """A value under the minimum is clamped up to it."""
        assert clamp_video_fps("0") == MIN_VIDEO_FPS

    def test_above_maximum_is_lowered(self):
        """A value over the maximum is clamped down to it."""
        assert clamp_video_fps("100") == MAX_VIDEO_FPS


class TestEstimateVideoFrames:
    """Tests for :func:`estimate_video_frames`."""

    def test_includes_the_first_frame(self):
        """The count is fps * seconds + 1 (the frame at t=0 counts)."""
        assert estimate_video_frames(16, 5) == 81
        assert estimate_video_frames(2, 5) == 11

    def test_accepts_string_inputs(self):
        """Dropdown values arrive as strings and are still parsed."""
        assert estimate_video_frames("4", "10") == 41

    def test_seconds_are_clamped_before_counting(self):
        """An out-of-range duration is clamped, not taken literally."""
        assert estimate_video_frames(1, "999999") == (
            int(MAX_VIDEO_SECONDS) + 1
        )

    def test_invalid_input_falls_back_to_defaults(self):
        """Non-numeric inputs fall back to the default fps/seconds."""
        assert estimate_video_frames("abc", "xyz") == 11


class TestClampGgufNCtx:
    """Tests for :func:`clamp_gguf_n_ctx`."""

    def test_value_in_range_is_kept(self):
        """A context size within bounds is returned unchanged."""
        assert clamp_gguf_n_ctx("16384") == 16384

    def test_below_minimum_is_raised(self):
        """A size under the minimum is clamped up to it."""
        assert clamp_gguf_n_ctx("512") == MIN_GGUF_N_CTX

    def test_above_maximum_is_lowered(self):
        """A size over the maximum is clamped down to it."""
        assert clamp_gguf_n_ctx("999999") == MAX_GGUF_N_CTX

    def test_invalid_value_falls_back_to_default(self):
        """A non-numeric value yields the default context size."""
        assert clamp_gguf_n_ctx("abc") == DEFAULT_GGUF_N_CTX


class TestClampCaptionExtensions:
    """Tests for :func:`clamp_caption_extensions`."""

    def test_strips_dots_and_lowercases(self):
        """Leading dots are removed and entries lowercased."""
        assert clamp_caption_extensions([".TXT", "Caption"]) == [
            "txt",
            "caption",
        ]

    def test_deduplicates_preserving_order(self):
        """Duplicates are dropped, first occurrence order preserved."""
        assert clamp_caption_extensions(["txt", "txt", "booru"]) == [
            "txt",
            "booru",
        ]

    def test_media_extensions_are_rejected(self):
        """A media suffix (png) is never a valid caption extension."""
        assert clamp_caption_extensions(["txt", "png"]) == ["txt"]

    def test_invalid_characters_are_rejected(self):
        """Entries with non-alphanumeric characters are dropped."""
        assert clamp_caption_extensions(["txt", "a b", "c/d"]) == ["txt"]

    def test_empty_result_falls_back_to_default(self):
        """When nothing valid remains, the default list is returned."""
        assert (
            clamp_caption_extensions(["png", "jpg"])
            == DEFAULT_CAPTION_EXTENSIONS
        )

    def test_string_input_is_split(self):
        """A comma/newline string is accepted and split."""
        assert clamp_caption_extensions("txt, caption\nbooru") == [
            "txt",
            "caption",
            "booru",
        ]


@pytest.fixture(name="user_config")
def _user_config(tmp_path, monkeypatch):
    """Redirect the user config dir to a temp tree; clear the cache."""
    monkeypatch.setattr(config, "USER_CONFIG_DIR", tmp_path / "user")
    settings._invalidate_cache()
    yield
    settings._invalidate_cache()


class TestLastCaptionType:
    """The remembered gallery-wide caption type round-trips through config."""

    def test_default_is_empty(self, user_config):
        """With nothing saved the remembered type is empty."""
        assert settings.get_last_caption_type() == ""

    def test_set_then_get(self, user_config):
        """A saved type is read back."""
        settings.set_last_caption_type("booru")
        assert settings.get_last_caption_type() == "booru"

    def test_empty_value_clears_it(self, user_config):
        """Saving an empty value forgets the remembered type."""
        settings.set_last_caption_type("tags")
        settings.set_last_caption_type("")
        assert settings.get_last_caption_type() == ""


class TestInternalMediaDir:
    """The internal media directory setting."""

    def test_defaults_to_storage_dir(self, user_config):
        """With nothing configured it falls back to the shipped storage/."""
        from src.constants import STORAGE_DIR  # noqa: local import for clarity

        assert settings.get_internal_media_dir() == STORAGE_DIR

    def test_empty_value_is_unset(self, user_config):
        """An explicitly empty path reads back as None (unset)."""
        config.save_user_settings({"internal_media_dir": ""})
        settings._invalidate_cache()
        assert settings.get_internal_media_dir() is None
