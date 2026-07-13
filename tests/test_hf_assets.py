"""Tests for :mod:`src.hf_assets` (HF config path + presence check)."""

import pytest

from src import hf_assets
from src.hf_assets import REQUIRED_HF_FILES


@pytest.fixture(name="hf_dir")
def _hf_dir(tmp_path, monkeypatch):
    """Redirect the HF config root to a temp directory."""
    monkeypatch.setattr(hf_assets, "HF_CONFIG_DIR", tmp_path)
    return tmp_path


class TestHfConfigDir:
    """Tests for :func:`hf_assets.hf_config_dir`."""

    def test_slashes_are_sanitized(self, hf_dir):
        """A repo id's ``/`` becomes ``--`` in the folder name."""
        folder = hf_assets.hf_config_dir("Qwen/Qwen3-VL-4B-Instruct")
        assert folder == hf_dir / "Qwen--Qwen3-VL-4B-Instruct"


class TestMissingHfFiles:
    """Tests for :func:`hf_assets.missing_hf_files`."""

    def test_empty_folder_lists_all_required(self, hf_dir):
        """With nothing present, every required file is reported missing."""
        assert hf_assets.missing_hf_files("Org/Repo") == list(
            REQUIRED_HF_FILES
        )

    def test_partial_folder_lists_the_rest(self, hf_dir):
        """Only the absent required files are reported."""
        folder = hf_dir / "Org--Repo"
        folder.mkdir()
        (folder / "config.json").write_text("{}", encoding="utf-8")
        missing = hf_assets.missing_hf_files("Org/Repo")
        assert "config.json" not in missing
        assert set(missing) == set(REQUIRED_HF_FILES) - {"config.json"}

    def test_complete_folder_is_empty(self, hf_dir):
        """A folder with all required files reports nothing missing."""
        folder = hf_dir / "Org--Repo"
        folder.mkdir()
        for name in REQUIRED_HF_FILES:
            (folder / name).write_text("{}", encoding="utf-8")
        assert hf_assets.missing_hf_files("Org/Repo") == []
