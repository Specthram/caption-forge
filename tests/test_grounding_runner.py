"""Tests for the grounding runner's claim-model auto-load logic.

``_ensure_claim_model`` decides which VLM decomposes a caption: the one
already loaded, else the default configured in Settings, else none. The
loader and scanner are mocked — no real model is ever loaded.
"""

from server.runners import grounding as runner


class _Progress:
    """A no-op progress reporter capturing the last subtitle."""

    def __init__(self):
        self.subs = []

    def __call__(self, done=None, total=None, sub=None):
        # pylint: disable=unused-argument
        if sub is not None:
            self.subs.append(sub)


def test_caption_grounding_frees_the_vlm_before_loading_siglip(monkeypatch):
    """The VLM is unloaded before SigLIP loads — never both in VRAM at once.

    Guards the OOM fix: decomposition (VLM) and scoring (SigLIP) run one
    after the other, so the runner must drop the VLM between the two phases.
    """
    from src import siglip_grounding, storage

    events = []
    monkeypatch.setattr(
        "src.loader.is_model_loaded", lambda: True, raising=False
    )
    monkeypatch.setattr(
        "src.loader.unload_model",
        lambda: events.append("vlm_unload") or iter([("done", False)]),
        raising=False,
    )
    monkeypatch.setattr(
        storage,
        "extract_caption_claims",
        lambda *a: {
            "status": "ok",
            "path": "img.png",
            "revision_id": 1,
            "claims": [{"text": "a dog", "kind": "object"}],
        },
    )

    def _load(*_a, **_k):
        events.append("siglip_load")
        return "google/siglip2-so400m-patch16-512"

    monkeypatch.setattr(siglip_grounding, "load_model", _load)
    monkeypatch.setattr(siglip_grounding, "unload_model", lambda: None)
    monkeypatch.setattr(
        storage,
        "score_caption_claims",
        lambda *a: {"status": "ok", "claims": []},
    )

    body = runner.ground_caption_body(1, "1", "txt")
    body(_Progress())

    assert events == ["vlm_unload", "siglip_load"]


def test_uses_the_already_loaded_model(monkeypatch):
    """A VLM in the slot is used as-is — no swap, no default lookup."""
    monkeypatch.setattr(
        "src.loader.is_model_loaded", lambda: True, raising=False
    )
    scanned = []
    monkeypatch.setattr(
        "src.scanner.scan_local_models",
        lambda: scanned.append(1) or {},
        raising=False,
    )

    assert runner._ensure_claim_model(_Progress()) is True
    assert scanned == []  # never even scanned for a default


def test_skips_when_nothing_loaded_and_no_default(monkeypatch):
    """No VLM and no configured default → cannot ground, report False."""
    monkeypatch.setattr(
        "src.loader.is_model_loaded", lambda: False, raising=False
    )
    monkeypatch.setattr(
        "src.settings.get_grounding_claim_model", lambda: "", raising=False
    )

    assert runner._ensure_claim_model(_Progress()) is False


def test_skips_when_default_is_not_a_scanned_model(monkeypatch):
    """A stale default that no longer exists on disk skips, never crashes."""
    monkeypatch.setattr(
        "src.loader.is_model_loaded", lambda: False, raising=False
    )
    monkeypatch.setattr(
        "src.settings.get_grounding_claim_model",
        lambda: "ghost-model",
        raising=False,
    )
    monkeypatch.setattr(
        "src.scanner.scan_local_models", lambda: {}, raising=False
    )

    assert runner._ensure_claim_model(_Progress()) is False


def test_loads_the_configured_default_when_slot_is_empty(monkeypatch):
    """With nothing loaded, the configured model is driven to completion."""
    loaded = {"value": False}
    monkeypatch.setattr(
        "src.loader.is_model_loaded",
        lambda: loaded["value"],
        raising=False,
    )
    monkeypatch.setattr(
        "src.settings.get_grounding_claim_model",
        lambda: "my-vlm",
        raising=False,
    )
    monkeypatch.setattr(
        "src.scanner.scan_local_models",
        lambda: {"my-vlm": {"type": "qwen3", "format": "transformers"}},
        raising=False,
    )

    def _fake_load(_cfg):
        yield ("loading weights…", False)
        loaded["value"] = True
        yield ("ready", True)

    monkeypatch.setattr("src.loader.load_model", _fake_load, raising=False)

    progress = _Progress()
    assert runner._ensure_claim_model(progress) is True
    assert "loading my-vlm…" in progress.subs
    assert "ready" in progress.subs
