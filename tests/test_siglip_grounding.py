"""Tests for the SigLIP 2 grounding engine.

The load-bearing claim of :mod:`src.siglip_grounding` is that its dense
patch embeddings are not an approximation of the pooling head but an exact
evaluation of it: pooling a *single* patch collapses the head's attention
to the identity. That is a mathematical identity, so it is asserted here
against a randomly initialized SigLIP vision tower — no checkpoint is ever
downloaded.
"""

import base64

import numpy as np
import pytest

from src import siglip_grounding as grounding


def _tiny_tower():
    """Return a random-init SigLIP vision tower with a 2x2 patch grid."""
    transformers = pytest.importorskip("transformers")
    config = transformers.SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=32,
        patch_size=grounding.PATCH_SIZE,
    )
    return transformers.SiglipVisionModel(config).eval()


def _tiny_model():
    """Return a random-init full SigLIP model (text + vision, 2x2 patches).

    Text and vision share a 32-dim embedding space so their dot product is
    defined — the same constraint the real checkpoints satisfy.
    """
    transformers = pytest.importorskip("transformers")
    config = transformers.SiglipConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "vocab_size": 100,
            "max_position_embeddings": grounding.TEXT_CONTEXT,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 32,
            "patch_size": grounding.PATCH_SIZE,
        },
    )
    return transformers.SiglipModel(config).eval()


class _FakeProcessor:
    """Stand in for the SigLIP processor without a downloaded checkpoint.

    Returns tensors of the shapes the tiny model expects — the pixel content
    is irrelevant to what the test asserts (that the pipeline runs and the
    feature calls are unwrapped correctly), only the shapes are.
    """

    def __init__(self, torch):
        self._torch = torch

    def __call__(self, images=None, text=None, **_kwargs):
        torch = self._torch
        if images is not None:
            return {"pixel_values": torch.randn(1, 3, 32, 32)}
        count = len(text)
        length = grounding.TEXT_CONTEXT
        return {
            "input_ids": torch.randint(0, 100, (count, length)),
            "attention_mask": torch.ones(count, length, dtype=torch.long),
        }


def test_dense_embedding_equals_single_patch_pooling():
    """Each dense patch embedding is the head's output on that patch alone."""
    torch = pytest.importorskip("torch")
    tower = _tiny_tower()
    hidden = torch.randn(1, 4, 32)

    with torch.no_grad():
        dense = grounding._dense_patch_embeddings(hidden, tower.head)
        expected = torch.cat(
            [tower.head(hidden[:, index : index + 1]) for index in range(4)]
        )

    assert dense.shape == (1, 4, 32)
    torch.testing.assert_close(dense[0], expected, rtol=1e-4, atol=1e-4)


def test_dense_embedding_differs_from_raw_patch_tokens():
    """The head's residual MLP is applied, not skipped (a regression guard)."""
    torch = pytest.importorskip("torch")
    tower = _tiny_tower()
    hidden = torch.randn(1, 4, 32)

    with torch.no_grad():
        dense = grounding._dense_patch_embeddings(hidden, tower.head)

    assert not torch.allclose(dense, hidden, atol=1e-3)


def test_clamp_resolution_keeps_offered_value():
    """A resolution the size ships is returned untouched."""
    assert grounding.clamp_resolution("base", 384) == 384
    assert grounding.clamp_resolution("so400m", 512) == 512


def test_clamp_resolution_caps_giant_at_its_largest():
    """Giant ships no 512 checkpoint, so 512 falls back to its 384."""
    assert grounding.clamp_resolution("giant-opt", 512) == 384
    assert grounding.clamp_resolution("giant-opt", 384) == 384


def test_clamp_resolution_rejects_junk():
    """A non-integer resolution falls back to the size's largest."""
    assert grounding.clamp_resolution("large", None) == 512
    assert grounding.clamp_resolution("large", "wide") == 512


def test_clamp_size_falls_back_to_default():
    """An unknown size name resolves to the default tier."""
    assert grounding.clamp_size("huge") == grounding.DEFAULT_SIZE
    assert grounding.clamp_size("base") == "base"


def test_repo_id_matches_the_published_checkpoints():
    """The derived repository ids are the real Hugging Face ones."""
    assert (
        grounding.repo_id("so400m", 512) == "google/siglip2-so400m-patch16-512"
    )
    assert grounding.repo_id("giant-opt", 384) == (
        "google/siglip2-giant-opt-patch16-384"
    )


def test_grid_side_is_the_patch_count_per_side():
    """A patch16 checkpoint grids its input into resolution/16 per side."""
    assert grounding.grid_side(256) == 16
    assert grounding.grid_side(384) == 24
    assert grounding.grid_side(512) == 32


def test_tag_prompt_wraps_the_bare_tag():
    """A booru tag is scored through the fixed pre-prompt sentence."""
    assert grounding.tag_prompt("horse") == "a photo that contains horse"


def test_heat_grid_normalizes_to_the_full_byte_range():
    """The least and most supporting patches land on 0 and 255."""
    similarity = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    grid = grounding._heat_grid(similarity, 2)

    assert grid.shape == (2, 2)
    assert grid.min() == 0
    assert grid.max() == 255


def test_heat_grid_of_a_flat_map_is_all_zero():
    """A uniform cosine map carries no evidence and must not be painted."""
    similarity = np.full(4, 0.25, dtype=np.float32)

    grid = grounding._heat_grid(similarity, 2)

    assert grid.tolist() == [[0, 0], [0, 0]]


def test_heat_grid_is_row_major_over_the_patch_sequence():
    """Patch order maps to rows then columns, as the vision tower emits it."""
    similarity = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

    grid = grounding._heat_grid(similarity, 2)

    assert grid[0, 1] > grid[0, 0]
    assert grid[1, 0] > grid[0, 1]


def test_encode_grid_round_trips_through_base64():
    """The wire format is one raw byte per patch, row-major."""
    grid = np.array([[0, 255], [7, 9]], dtype=np.uint8)

    payload = grounding._encode_grid(grid)

    assert list(base64.b64decode(payload)) == [0, 255, 7, 9]


def test_ground_image_of_no_texts_is_empty():
    """No text, no forward pass — the model is never even touched."""
    assert grounding.ground_image("missing.png", []) == []


def test_calibrate_scores_references_the_negative_bank(monkeypatch):
    """A claim beating the generic negatives scores high; one below, low.

    Guards the calibration that replaced SigLIP's raw sigmoid (which crushed
    real claims to 0). No model is loaded: the negative bank's embeddings are
    mocked so their cosine with the image is a known 0.06.
    """
    torch = pytest.importorskip("torch")
    pooled = torch.tensor([[1.0, 0.0]])
    # Every negative sits at cosine 0.06 with ``pooled`` (its first component).
    neg = torch.tensor([[0.06, 0.998]] * len(grounding._NEG_PROMPTS))
    monkeypatch.setattr(grounding, "_encode_texts", lambda texts: neg)

    scores = grounding._calibrate_scores(
        torch.tensor([0.13, 0.02]), pooled
    )

    assert float(scores[0]) > 90.0  # 0.13 ≫ the 0.06 reference
    assert float(scores[1]) < 10.0  # 0.02 ≪ it


def test_ground_image_runs_end_to_end_on_a_real_model(tmp_path, monkeypatch):
    """The full pipeline runs against a real SigLIP model, no download.

    Guards the transformers-5.x regression where ``get_text_features``
    returns a ``BaseModelOutputWithPooling`` (not a bare tensor), which used
    to crash scoring with ``'BaseModelOutputWithPooling' has no attribute
    'norm'``. Exercises image encoding, dense heat maps, text encoding and
    the sigmoid scoring in one pass.
    """
    torch = pytest.importorskip("torch")
    from PIL import Image  # pylint: disable=import-outside-toplevel

    image_path = tmp_path / "img.png"
    Image.new("RGB", (48, 32), (128, 64, 200)).save(image_path)

    monkeypatch.setattr(grounding, "_model", _tiny_model())
    monkeypatch.setattr(grounding, "_processor", _FakeProcessor(torch))
    monkeypatch.setattr(grounding, "_text_cache", {})

    results = grounding.ground_image(
        str(image_path), ["a purple square", "a horse"], with_heat=True
    )

    assert [r["text"] for r in results] == ["a purple square", "a horse"]
    for result in results:
        assert 0.0 <= result["score"] <= 100.0
        assert result["side"] == 2
        grid = base64.b64decode(result["heat"])
        assert len(grid) == 4  # 2x2 patches, one byte each


def test_ground_image_without_heat_skips_the_grids(tmp_path, monkeypatch):
    """The batch scorer asks for scores only — no heat grids are built."""
    torch = pytest.importorskip("torch")
    from PIL import Image  # pylint: disable=import-outside-toplevel

    image_path = tmp_path / "img.png"
    Image.new("RGB", (32, 32), (10, 200, 90)).save(image_path)

    monkeypatch.setattr(grounding, "_model", _tiny_model())
    monkeypatch.setattr(grounding, "_processor", _FakeProcessor(torch))
    monkeypatch.setattr(grounding, "_text_cache", {})

    results = grounding.ground_image(
        str(image_path), ["a green circle"], with_heat=False
    )

    assert results[0]["heat"] is None
    assert 0.0 <= results[0]["score"] <= 100.0
