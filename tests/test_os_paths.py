"""Tests for the native OS folder/file helpers in :mod:`src.os_paths`."""

from src import os_paths


class TestRevealFile:
    """Tests for :func:`reveal_file`.

    ``subprocess.run`` is monkeypatched throughout so a test never actually
    pops open a file manager window.
    """

    def test_empty_path_is_a_noop(self, monkeypatch):
        """An empty path does nothing and reports no reveal happened."""
        calls = []
        monkeypatch.setattr(
            os_paths.subprocess, "run", lambda *a, **k: calls.append(a)
        )
        assert os_paths.reveal_file("") is False
        assert calls == []

    def test_missing_path_is_a_noop(self, tmp_path, monkeypatch):
        """A path that does not exist on disk is left untouched."""
        calls = []
        monkeypatch.setattr(
            os_paths.subprocess, "run", lambda *a, **k: calls.append(a)
        )
        target = tmp_path / "missing.png"
        assert os_paths.reveal_file(str(target)) is False
        assert calls == []

    def test_existing_file_triggers_the_platform_reveal_command(
        self, tmp_path, monkeypatch
    ):
        """An existing file runs the platform's reveal-in-place command."""
        target = tmp_path / "a.png"
        target.write_bytes(b"x")
        calls = []
        monkeypatch.setattr(
            os_paths.subprocess,
            "run",
            lambda *a, **k: calls.append((a, k)),
        )
        assert os_paths.reveal_file(str(target)) is True
        assert len(calls) == 1
