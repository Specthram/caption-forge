"""Loader: Hugging Face download plumbing and cancel propagation.

No weights are ever fetched — the hub metadata and download callbacks are
monkeypatched, so these cover the orchestration (quant pick, byte
accounting, cooperative cancel) without touching the network.
"""

import pytest

from src import loader


class TestPickGguf:
    """Choosing the preferred quant from a repo's gguf files."""

    def test_prefers_q4_k_m(self):
        """Q4_K_M wins over other quants when present."""
        names = ["m-Q8_0.gguf", "m-Q4_K_M.gguf", "m-Q5_K_M.gguf"]
        assert loader._pick_gguf(names) == "m-Q4_K_M.gguf"

    def test_falls_back_to_sorted_first(self):
        """With no preferred quant, the alphabetically first file is used."""
        assert loader._pick_gguf(["b.gguf", "a.gguf"]) == "a.gguf"


class TestByteAccounting:
    """The progress callbacks accumulate bytes against the repo total."""

    def test_report_accumulates_to_total(self, monkeypatch):
        """on_step sums increments and forwards done/total/label."""
        monkeypatch.setattr(loader, "_hf_sizes", lambda repo: {"a": 100})
        seen = []
        on_step, total = loader._hf_report(
            lambda done, tot, label: seen.append((done, tot, label)), "o/r"
        )
        assert total == 100
        on_step(30)
        on_step(70)
        assert seen[-1] == (100, 100, "downloading o/r")

    def test_report_without_sink_is_noop(self):
        """A None sink yields a callback that neither reports nor raises."""
        on_step, total = loader._hf_report(None, "o/r")
        on_step(5)
        assert total == 0

    def test_gguf_report_sums_chosen_files(self, monkeypatch):
        """Only the picked weight + mmproj sizes count toward the total."""
        monkeypatch.setattr(
            loader,
            "_hf_sizes",
            lambda repo: {"w.gguf": 40, "mmproj.gguf": 10, "other.gguf": 999},
        )
        seen = []
        on_step = loader._hf_gguf_report(
            lambda done, tot, label: seen.append((done, tot)),
            "o/r",
            ["w.gguf", "mmproj.gguf"],
        )
        on_step(25)
        assert seen[-1] == (25, 50)


class TestHfLoadFlow:
    """The download+load generator's terminal states."""

    def test_cancel_propagates(self, monkeypatch):
        """A JobStopped raised through the download reaches the caller."""

        class JobStopped(Exception):
            """Named to match the loader's by-name re-raise."""

        def boom(*_args, **_kwargs):
            raise JobStopped()

        monkeypatch.setattr(loader, "_load_hf_safetensors", boom)
        cfg = {"source": "hf", "repo": "o/r", "format": "safetensors"}
        with pytest.raises(JobStopped):
            list(loader._load_hf_model(cfg, "safetensors", "qwen3", None))

    def test_error_surfaces_as_status(self, monkeypatch):
        """A plain failure ends the generator with an error status line."""

        def boom(*_args, **_kwargs):
            raise RuntimeError("nope")

        monkeypatch.setattr(loader, "_load_hf_safetensors", boom)
        cfg = {"source": "hf", "repo": "o/r", "format": "safetensors"}
        statuses = [
            status
            for status, _loaded in loader._load_hf_model(
                cfg, "safetensors", "qwen3", None
            )
        ]
        assert any("Error" in status for status in statuses)
        assert loader.loaded_name is None
