"""Tests for :mod:`src.quality` (pluggable quality metrics via pyiqa).

The real metrics (torch checkpoints, downloaded by pyiqa on first use) are
never loaded here: :func:`quality._get_metric` is stubbed with a tiny fake
that reports an image's mean brightness, so these tests stay fast and
offline while still exercising the real image/video/error paths.
"""

import cv2
import numpy as np
import pytest
from PIL import Image

from src import quality


class _FakeMetric:
    """Stand-in for pyiqa's callable metric: returns mean pixel brightness."""

    def __call__(self, image):
        return _FakeTensor(sum(image.convert("L").getdata()) / 255.0)


class _FakeTensor:
    """Stand-in for the torch.Tensor a real pyiqa metric returns."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


@pytest.fixture(autouse=True)
def _fake_metric(monkeypatch):
    """Stub the lazy-loaded MUSIQ metric for every test in this module."""
    monkeypatch.setattr(
        quality, "_get_metric", lambda metric_id=None: _FakeMetric()
    )


def _write_image(path, size=(64, 64), color=(200, 200, 200)):
    """Write a plain RGB image to ``path`` and return it."""
    Image.new("RGB", size, color).save(path)
    return path


def _write_video(path, size=(32, 24)):
    """Write a tiny two-frame mp4 to ``path`` and return it."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, size
    )
    for _ in range(2):
        writer.write(np.full((size[1], size[0], 3), 180, dtype=np.uint8))
    writer.release()
    return path


class TestUnloadMetric:
    """Tests for :func:`quality.unload_metric`."""

    def test_releases_the_cached_singleton(self, monkeypatch):
        """Unloading drops the cached metric and its key."""
        monkeypatch.setattr(quality, "_metric", object())
        monkeypatch.setattr(quality, "_metric_key", ("musiq", "cpu"))

        quality.unload_metric()

        assert quality._metric is None  # pylint: disable=protected-access
        key = quality._metric_key  # pylint: disable=protected-access
        assert key is None

    def test_noop_when_nothing_is_loaded(self, monkeypatch):
        """Unloading with no metric cached does nothing (and not crash)."""
        monkeypatch.setattr(quality, "_metric", None)

        quality.unload_metric()

        assert quality._metric is None  # pylint: disable=protected-access


class TestScoreMedia:
    """Tests for :func:`quality.score_media`."""

    def test_scores_an_image(self, tmp_path):
        """An image is opened and passed straight to the metric."""
        source = _write_image(tmp_path / "a.png")
        score = quality.score_media(source)
        assert isinstance(score, float)
        assert score > 0

    def test_scores_a_video_from_its_first_frame(self, tmp_path):
        """A video is scored from its first frame, not the whole clip."""
        source = _write_video(tmp_path / "clip.mp4")
        score = quality.score_media(source)
        assert isinstance(score, float)
        assert score > 0

    def test_unreadable_file_returns_none(self, tmp_path):
        """A corrupt/unsupported file fails gracefully instead of raising."""
        bogus = tmp_path / "bogus.png"
        bogus.write_bytes(b"not an image")
        assert quality.score_media(bogus) is None

    def test_unreadable_video_returns_none(self, tmp_path):
        """A file with a video extension but no readable frames is None."""
        bogus = tmp_path / "bogus.mp4"
        bogus.write_bytes(b"not a video")
        assert quality.score_media(bogus) is None


class TestBadgeStyle:
    """Tests for :func:`quality.badge_style` (metric-aware badge styling)."""

    def test_musiq_bands_match_the_documented_thresholds(self):
        """MUSIQ: <60 faible, 60-75 moyen, 75-90 bonne, >=90 excellente."""
        assert quality.badge_style(59, "musiq")[1] == "#d93025"
        assert quality.badge_style(60, "musiq")[1] == "#f9a825"
        assert quality.badge_style(75, "musiq")[1] == "#34a853"
        assert quality.badge_style(90, "musiq")[1] == "#4285f4"

    def test_musiq_text_is_a_rounded_percent(self):
        """MUSIQ's badge text is the score rounded to a percentage."""
        text, _ = quality.badge_style(72.6, "musiq")
        assert text == "73%"

    def test_topiq_nr_bands_match_the_documented_thresholds(self):
        """TOPIQ-NR bands: low/fair/good/very good/excellent."""
        assert quality.badge_style(0.59, "topiq_nr")[1] == "#d93025"
        assert quality.badge_style(0.60, "topiq_nr")[1] == "#f9a825"
        assert quality.badge_style(0.80, "topiq_nr")[1] == "#34a853"
        assert quality.badge_style(0.90, "topiq_nr")[1] == "#0b8043"
        assert quality.badge_style(0.95, "topiq_nr")[1] == "#4285f4"

    def test_topiq_nr_text_is_a_normalized_percent(self):
        """TOPIQ-NR's 0-1 native score displays as a 0-100 percentage."""
        text, _ = quality.badge_style(0.837, "topiq_nr")
        assert text == "84%"

    def test_qalign_text_is_a_normalized_percent(self):
        """Q-Align's 1-5 native score displays as a 0-100 percentage."""
        text, color = quality.badge_style(4.2, "qalign")
        assert text == "80%"
        assert color == "#0b8043"

    def test_qalign_variants_share_the_same_bands(self):
        """The 8-bit/4-bit variants score on the same 1-5 scale as fp16."""
        assert quality.badge_style(4.6, "qalign_8bit") == (
            quality.badge_style(4.6, "qalign_4bit")
        )

    def test_laion_aesthetic_text_is_a_normalized_percent(self):
        """LAION Aesthetic's 1-10 native score displays as a percentage."""
        text, _ = quality.badge_style(6.8, "laion_aes")
        assert text == "64%"

    def test_out_of_range_scores_are_clamped(self):
        """A score outside the metric's native range clamps, never errors."""
        assert quality.badge_style(140.0, "musiq")[0] == "100%"
        assert quality.badge_style(-3, "musiq")[0] == "0%"

    def test_unknown_metric_falls_back_to_the_default(self):
        """A stale/unknown metric id falls back to MUSIQ, not an error."""
        assert quality.badge_style(80, "bogus-metric") == (
            quality.badge_style(80, "musiq")
        )

    def test_none_metric_falls_back_to_the_default(self):
        """No metric recorded (legacy row) also falls back to MUSIQ."""
        no_metric = quality.badge_style(80, None)
        musiq = quality.badge_style(80, "musiq")
        assert no_metric == musiq

    def test_average_metric_reads_its_own_bands(self):
        """The average pseudo-metric is 0-100 with its generic tiers."""
        text, color = quality.badge_style(89, quality.AVERAGE_METRIC_ID)
        assert text == "89%"
        assert color == "#0b8043"  # 85-92 -> "very good" (dark green)
        assert quality.badge_style(95, quality.AVERAGE_METRIC_ID)[1] == (
            "#4285f4"
        )


class TestNormalizeScore:
    """Tests for :func:`quality.normalize_score`."""

    def test_normalizes_each_native_range_to_percent(self):
        """Every metric maps its native range onto 0-100."""
        assert quality.normalize_score("musiq", 90) == 90.0
        assert quality.normalize_score("topiq_nr", 0.90) == 90.0
        assert quality.normalize_score("qalign", 4.5) == 87.5

    def test_none_score_returns_none(self):
        """A missing score normalizes to None (never scored)."""
        assert quality.normalize_score("musiq", None) is None

    def test_clamps_out_of_range(self):
        """A score beyond the native range is clamped before scaling."""
        assert quality.normalize_score("musiq", 140) == 100.0


class TestVramAdvice:
    """Tests for the Settings-tab VRAM helper."""

    def test_every_metric_is_mentioned(self):
        """Each registered metric appears once in the advice text."""
        text = quality.vram_advice_markdown()
        for metric in quality.QUALITY_METRICS.values():
            assert metric.label in text

    def test_no_gpu_warns_that_q_align_is_impractical(self, monkeypatch):
        """Without CUDA, the advice steers away from the heavy VLM."""
        monkeypatch.setattr(quality, "detect_vram_gb", lambda: None)
        text = quality.vram_advice_markdown()
        assert "No CUDA GPU detected" in text

    def test_low_vram_flags_q_align_as_unfit(self, monkeypatch):
        """A small GPU gets a warning mark next to full-precision Q-Align."""
        monkeypatch.setattr(quality, "detect_vram_gb", lambda: 8.0)
        label_lines = [
            line
            for line in quality.vram_advice_markdown().splitlines()
            if line.startswith("-")
        ]
        qalign_label = quality.QUALITY_METRICS["qalign"].label
        musiq_label = quality.QUALITY_METRICS["musiq"].label
        qalign_line = next(
            line for line in label_lines if qalign_label in line
        )
        musiq_line = next(line for line in label_lines if musiq_label in line)
        assert "⚠️" in qalign_line
        assert "✅" in musiq_line


class TestWarmMetricStream:
    """Streaming the metric load ahead of an indexing batch."""

    def test_loads_the_metric_and_ends(self, monkeypatch):
        """The generator drives the load once, then finishes."""
        calls = []
        monkeypatch.setattr(
            quality, "_get_metric", lambda metric_id: calls.append(metric_id)
        )
        snapshots = list(
            quality.warm_metric_stream("musiq", poll_seconds=0.01)
        )
        assert calls == ["musiq"]
        assert all({"desc", "n", "total"} <= set(snap) for snap in snapshots)

    def test_reraises_a_load_failure(self, monkeypatch):
        """An exception in the worker surfaces to the caller."""

        def boom(metric_id):
            raise RuntimeError("no weights")

        monkeypatch.setattr(quality, "_get_metric", boom)
        with pytest.raises(RuntimeError, match="no weights"):
            list(quality.warm_metric_stream("musiq", poll_seconds=0.01))

    def test_restores_tqdm_instrumentation(self, monkeypatch):
        """tqdm's methods come back untouched once the stream ends."""
        tqdm_module = pytest.importorskip("tqdm")
        original_init = tqdm_module.tqdm.__init__
        original_update = tqdm_module.tqdm.update
        monkeypatch.setattr(quality, "_get_metric", lambda metric_id: None)
        list(quality.warm_metric_stream("musiq", poll_seconds=0.01))
        assert tqdm_module.tqdm.__init__ is original_init
        assert tqdm_module.tqdm.update is original_update


class TestGpuStatus:
    """quality.gpu_status — the topbar's live GPU memory gauge."""

    def test_off_cuda_is_none(self, monkeypatch):
        """No CUDA device, no gauge."""
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert quality.gpu_status() is None

    def test_reports_name_and_live_memory(self, monkeypatch):
        """Vendor noise is stripped and used = total - free."""
        torch = pytest.importorskip("torch")
        gib = 1024**3
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_name",
            lambda index: "NVIDIA GeForce RTX 4090",
        )
        monkeypatch.setattr(
            torch.cuda,
            "mem_get_info",
            lambda index: (6 * gib, 24 * gib),
        )

        status = quality.gpu_status()

        assert status["name"] == "RTX 4090"
        assert status["total_gb"] == pytest.approx(24.0)
        assert status["free_gb"] == pytest.approx(6.0)
        assert status["used_gb"] == pytest.approx(18.0)
