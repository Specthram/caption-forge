"""Tests for the caption decomposition (:mod:`src.caption_claims`).

The LLM is mocked at :func:`src.caption_claims._run_llm` — no model weights
are ever loaded.
"""

from src import caption_claims as claims


class TestParsing:
    """Parsing the decomposition output into ``{"text", "kind"}`` dicts."""

    def test_json_objects(self):
        """The prompted shape parses straight through."""
        raw = '[{"text": "a red ball", "kind": "object"}]'

        assert claims.parse_claims(raw) == [
            {"text": "a red ball", "kind": "object"}
        ]

    def test_json_in_prose(self):
        """A model that chatters around the array still gets parsed."""
        raw = 'Sure! [{"text": "two dogs", "kind": "count"}] Hope that helps.'

        assert claims.parse_claims(raw) == [
            {"text": "two dogs", "kind": "count"}
        ]

    def test_json_strings_default_to_object(self):
        """A bare string array is accepted, every claim an object claim."""
        raw = '["a red ball", "the grass is wet"]'

        assert claims.parse_claims(raw) == [
            {"text": "a red ball", "kind": "object"},
            {"text": "the grass is wet", "kind": "object"},
        ]

    def test_unknown_kind_collapses_to_object(self):
        """An invented kind never reaches the UI's badge switch."""
        raw = '[{"text": "a ball", "kind": "vibes"}]'

        assert claims.parse_claims(raw) == [
            {"text": "a ball", "kind": "object"}
        ]

    def test_kind_is_case_insensitive(self):
        """A shouted kind is still a known kind."""
        raw = '[{"text": "left of the tree", "kind": "SPATIAL"}]'

        assert claims.parse_claims(raw)[0]["kind"] == "spatial"

    def test_bullet_fallback(self):
        """A model ignoring the JSON contract falls back to one a line."""
        raw = "- a red ball\n2) the grass is wet\n\n"

        assert claims.parse_claims(raw) == [
            {"text": "a red ball", "kind": "object"},
            {"text": "the grass is wet", "kind": "object"},
        ]

    def test_blank_claims_are_dropped(self):
        """Empty strings never become claims."""
        raw = '[{"text": "  ", "kind": "object"}, {"text": "a ball"}]'

        assert claims.parse_claims(raw) == [
            {"text": "a ball", "kind": "object"}
        ]

    def test_empty_output(self):
        """No reply, no claims."""
        assert claims.parse_claims("") == []
        assert claims.parse_claims(None) == []

    def test_malformed_json_falls_back_to_lines(self):
        """A truncated array is read line by line rather than lost."""
        raw = "[a red ball, the grass"

        assert claims.parse_claims(raw) == [
            {"text": "[a red ball, the grass", "kind": "object"}
        ]


class TestReasoningModels:
    """A thinking model's chain of thought must never become a claim."""

    def test_prose_reasoning_before_json_is_skipped(self):
        """The array is found past the reasoning, which is dropped."""
        raw = (
            "Got it, let's tackle this. The user wants me to break the "
            "caption into atomic visual claims. First, I need to parse each "
            "part.\n"
            '[{"text": "a red ball", "kind": "object"}, '
            '{"text": "the grass is green", "kind": "attribute"}]'
        )

        assert claims.parse_claims(raw) == [
            {"text": "a red ball", "kind": "object"},
            {"text": "the grass is green", "kind": "attribute"},
        ]

    def test_stray_brackets_in_reasoning_do_not_derail(self):
        """A bracketed aside in the prose is skipped for the real array."""
        raw = (
            "I will produce statements [like this one] for each part.\n"
            '[{"text": "a dog", "kind": "object"}]'
        )

        assert claims.parse_claims(raw) == [
            {"text": "a dog", "kind": "object"}
        ]

    def test_incidental_number_list_loses_to_the_real_array(self):
        """A reasoning ``[1, 2, 3]`` never wins against the dict array."""
        raw = (
            "There are 3 parts [1, 2, 3] to handle.\n"
            '[{"text": "a blue car", "kind": "object"}]'
        )

        assert claims.parse_claims(raw) == [
            {"text": "a blue car", "kind": "object"}
        ]

    def test_prose_only_reasoning_yields_no_claims(self):
        """Markerless reasoning with no JSON is filtered, not made a claim."""
        raw = (
            "Got it, let's tackle this. The user wants me to break the "
            "caption into atomic visual claims. First, I need to parse each "
            "part of the caption and convert it into checkable statements."
        )

        assert claims.parse_claims(raw) == []

    def test_line_list_still_works_without_reasoning(self):
        """A plain bullet list from a non-thinking model is preserved."""
        raw = "- a red ball\n- the grass is green"

        assert claims.parse_claims(raw) == [
            {"text": "a red ball", "kind": "object"},
            {"text": "the grass is green", "kind": "object"},
        ]


class TestExtractClaims:
    """The one LLM call, mocked at its single seam."""

    def test_blank_caption_runs_no_inference(self, monkeypatch):
        """An empty caption short-circuits before the model is touched."""
        called = []
        monkeypatch.setattr(
            claims, "_run_llm", lambda *a: called.append(a) or "[]"
        )

        assert claims.extract_claims("img.png", "   ") == []
        assert called == []

    def test_caption_is_injected_without_str_format(self, monkeypatch):
        """The caption reaches the prompt even though it carries JSON braces.

        ``str.format`` would raise on the ``{"text": ...}`` shape the prompt
        shows the model, so the substitution is a plain ``replace``.
        """
        seen = {}

        def _fake(_image_path, prompt):
            seen["prompt"] = prompt
            return '[{"text": "a cat", "kind": "object"}]'

        monkeypatch.setattr(claims, "_run_llm", _fake)

        result = claims.extract_claims("img.png", "A cat sits.")

        assert result == [{"text": "a cat", "kind": "object"}]
        assert "A cat sits." in seen["prompt"]
        assert "{caption}" not in seen["prompt"]
        assert '"kind"' in seen["prompt"]


def test_run_llm_grants_a_wide_budget_for_reasoning(monkeypatch):
    """Decomposition asks for extra tokens so a thinking model can finish."""
    from src import captioner, loader, settings

    captured = {}

    def _fake(image_source, prompt, temperature, seed, think, **kwargs):
        # pylint: disable=unused-argument
        captured["think"] = think
        captured.update(kwargs)
        return '[{"text": "a cat", "kind": "object"}]'

    monkeypatch.setattr(captioner, "generate_caption", _fake)
    monkeypatch.setattr(loader, "current_model_type", "gemma4", raising=False)

    claims.extract_claims("img.png", "A cat.")

    assert captured["max_new_tokens"] == claims._CLAIM_MAX_TOKENS


def test_run_llm_honours_the_loaded_models_think_mode(monkeypatch):
    """The claim call uses the loaded model's configured thinking mode.

    A model the user set to "off" must not reason during decomposition; the
    path no longer hard-codes a mode of its own.
    """
    from src import captioner, loader, settings

    captured = {}

    def _fake(image_source, prompt, temperature, seed, think, **kwargs):
        # pylint: disable=unused-argument
        captured["think"] = think
        return "[]"

    monkeypatch.setattr(captioner, "generate_caption", _fake)
    monkeypatch.setattr(loader, "current_model_type", "gemma4", raising=False)
    monkeypatch.setattr(
        settings, "get_model_think_mode", lambda model_type: "off"
    )

    claims.extract_claims("img.png", "A cat.")
    assert captured["think"] == "off"

    monkeypatch.setattr(
        settings, "get_model_think_mode", lambda model_type: "show"
    )
    claims.extract_claims("img.png", "A cat.")
    assert captured["think"] == "show"


def test_unreliable_kinds_are_a_subset_of_kinds():
    """The amber-badge kinds must be kinds the parser can actually emit."""
    assert set(claims.UNRELIABLE_KINDS) <= set(claims.KINDS)
