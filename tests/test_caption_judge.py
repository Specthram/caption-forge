"""Tests for :mod:`src.caption_judge` — the model-free judging logic."""

from src import caption_judge as cj


class TestBuildPrompt:
    """The judge prompt embeds the rule and caption and asks for JSON."""

    def test_contains_rule_and_caption(self):
        """Both the rule text and the caption appear in the prompt."""
        prompt = cj.build_prompt("no subjective words", "a lovely red ball")
        assert "no subjective words" in prompt
        assert "a lovely red ball" in prompt

    def test_asks_for_json_only(self):
        """The prompt requests JSON with the three expected keys."""
        prompt = cj.build_prompt("rule", "caption")
        assert "JSON" in prompt
        assert "corrected_caption" in prompt


class TestParseJudgement:
    """Recovering the verdict JSON from the judge's raw answer."""

    def test_plain_json(self):
        """A bare JSON object parses to the normalised verdict."""
        raw = '{"violates": true, "note": "n", "corrected_caption": "x"}'
        verdict = cj.parse_judgement(raw)
        assert verdict == {
            "violates": True,
            "note": "n",
            "corrected_caption": "x",
        }

    def test_json_in_code_fence(self):
        """JSON wrapped in a code fence and prose is still recovered."""
        raw = (
            "Here you go:\n```json\n"
            '{"violates": false, "note": "", "corrected_caption": null}\n'
            "```\nhope that helps"
        )
        verdict = cj.parse_judgement(raw)
        assert verdict["violates"] is False
        assert verdict["corrected_caption"] is None

    def test_trailing_sentence_after_object(self):
        """A sentence after the JSON object does not break parsing."""
        raw = (
            '{"violates": true, "note": "n", "corrected_caption": "y"}. Done.'
        )
        assert cj.parse_judgement(raw)["corrected_caption"] == "y"

    def test_no_json_returns_none(self):
        """Text with no JSON object yields None."""
        assert cj.parse_judgement("no json here") is None

    def test_empty_returns_none(self):
        """Empty input yields None."""
        assert cj.parse_judgement("") is None


class TestJudgedFinding:
    """Turning a verdict into a finding, with the anti-rewrite guard."""

    def _verdict(self, violates, corrected):
        """Return a verdict dict with a fixed note."""
        return {
            "violates": violates,
            "note": "note",
            "corrected_caption": corrected,
        }

    def test_no_violation_no_finding(self):
        """A compliant caption produces no finding."""
        assert cj.judged_finding("a", self._verdict(False, None)) is None

    def test_none_correction_no_finding(self):
        """A violation with no correction produces no finding."""
        assert cj.judged_finding("a", self._verdict(True, None)) is None

    def test_identical_correction_no_finding(self):
        """A correction identical to the original produces no finding."""
        before = "a red ball on grass"
        assert cj.judged_finding(before, self._verdict(True, before)) is None

    def test_targeted_fix_kept(self):
        """A close, targeted correction becomes a finding."""
        before = "a red ball on the grass"
        after = "a red ball on the lawn"
        finding = cj.judged_finding(before, self._verdict(True, after))
        assert finding == {"note": "note", "caption_after": after}

    def test_free_rewrite_dropped(self):
        """A correction that rewrites everything is dropped by the guard."""
        before = "a red ball on the grass"
        after = "completely different sentence about a blue truck driving"
        assert cj.judged_finding(before, self._verdict(True, after)) is None


class TestCheckDetRule:
    """The deterministic trigger-word rule."""

    def test_missing_word_flagged_and_prepended(self):
        """A caption missing the quoted word gets it prepended."""
        rule = {"text": 'The caption must contain "ryn".'}
        finding = cj.check_det_rule(rule, "a red ball.", [])
        assert finding is not None
        assert finding["caption_after"] == "ryn, a red ball."
        assert "ryn" in finding["note"]

    def test_present_word_not_flagged(self):
        """A caption already containing the word is not flagged."""
        rule = {"text": 'The caption must contain "ryn".'}
        assert cj.check_det_rule(rule, "ryn, a red ball.", []) is None

    def test_word_boundary(self):
        """The word must appear as a whole token, not inside another."""
        rule = {"text": 'must contain "ryn".'}
        # "rynwood" contains the letters but not the token.
        assert cj.check_det_rule(rule, "a rynwood fence.", []) is not None

    def test_falls_back_to_trigger_words(self):
        """With no quoted word, the dataset trigger word is used."""
        rule = {"text": "The caption must start with the trigger word."}
        finding = cj.check_det_rule(rule, "a red ball.", ["ryn"])
        assert finding["caption_after"].startswith("ryn, ")

    def test_no_word_available_no_finding(self):
        """No quoted word and no trigger word means nothing to check."""
        rule = {"text": "start with the trigger word."}
        assert cj.check_det_rule(rule, "a red ball.", []) is None

    def test_empty_caption_prepends_bare_word(self):
        """An empty caption yields just the trigger word."""
        rule = {"text": 'contain "ryn".'}
        assert cj.check_det_rule(rule, "", [])["caption_after"] == "ryn"


class TestResolveFix:
    """Tests for :func:`cj.resolve_fix` (base-coordinate 3-way merge)."""

    BASE = "a red circle on a green square, watermark bottom right"

    def test_unchanged_current_takes_fix_verbatim(self):
        """When nothing moved since the run, the fix applies as-is."""
        after = self.BASE.replace("watermark bottom right", "clean photo")
        resolved = cj.resolve_fix(self.BASE, self.BASE, after)
        assert resolved == {"text": after, "conflict": False}

    def test_disjoint_fixes_auto_merge(self):
        """Accepting fix B after fix A keeps both (no conflict)."""
        fix_a = self.BASE.replace("red circle", "crimson circle")
        fix_b = self.BASE.replace("watermark bottom right", "no watermark")
        resolved = cj.resolve_fix(self.BASE, fix_a, fix_b)
        assert resolved["conflict"] is False
        assert "crimson circle" in resolved["text"]
        assert "no watermark" in resolved["text"]
        assert "watermark bottom right" not in resolved["text"]

    def test_manual_edit_survives_an_accept(self):
        """A user edit elsewhere in the caption survives the merge."""
        edited = "MYTRIGGER " + self.BASE
        fix = self.BASE.replace("green square", "blue square")
        resolved = cj.resolve_fix(self.BASE, edited, fix)
        assert resolved["conflict"] is False
        assert resolved["text"].startswith("MYTRIGGER ")
        assert "blue square" in resolved["text"]

    def test_same_phrase_is_flagged_and_takes_the_judge_words(self):
        """Both sides rewrote the same phrase: flagged, judge text wins."""
        edited = self.BASE.replace("red circle", "pink disc")
        fix = self.BASE.replace("red circle", "crimson circle")
        resolved = cj.resolve_fix(self.BASE, edited, fix)
        assert resolved["conflict"] is True
        assert "crimson circle" in resolved["text"]
        assert "pink disc" not in resolved["text"]

    def test_conflict_region_never_mixes_texts(self):
        """A whole-sentence rewrite in conflict yields the judge's exact
        rendering of that region, never a word-level chimera."""
        edited = self.BASE.replace(
            "a red circle on a green square",
            "one big pink disc over some grass",
        )
        fix = self.BASE.replace(
            "a red circle on a green square",
            "a crimson circle on a green square",
        )
        resolved = cj.resolve_fix(self.BASE, edited, fix)
        assert resolved["conflict"] is True
        assert "a crimson circle on a green square" in resolved["text"]
        assert "pink" not in resolved["text"]
        assert "grass" not in resolved["text"]

    def test_current_already_equal_to_fix_is_stable(self):
        """Re-accepting an already applied fix changes nothing."""
        fix = self.BASE.replace("red", "crimson")
        resolved = cj.resolve_fix(self.BASE, fix, fix)
        assert resolved == {"text": fix, "conflict": False}

    def test_noop_fix_never_clobbers_current(self):
        """A fix equal to its base leaves the live caption untouched."""
        edited = "MYTRIGGER " + self.BASE
        resolved = cj.resolve_fix(self.BASE, edited, self.BASE)
        assert resolved == {"text": edited, "conflict": False}

    def test_scattered_overlap_takes_one_whole_sentence(self):
        """Two scattered rewrites of one sentence never interleave: the
        judge's whole sentence wins, other sentences still merge."""
        base = "a red ball near the old tree. watermark bottom right."
        # First accept reworded two spots of sentence 1.
        current = "a pink ball near the tall tree. watermark bottom right."
        # This fix rewrites sentence 1 too (collides on "red") and also
        # fixes sentence 2 — which must merge untouched by the conflict.
        incoming = "a crimson sphere near the old tree. no watermark visible."
        resolved = cj.resolve_fix(base, current, incoming)
        assert resolved["conflict"] is True
        assert resolved["text"] == (
            "a crimson sphere near the old tree. no watermark visible."
        )
        # No chimera: nothing of the overridden rewording survives.
        assert "pink" not in resolved["text"]
        assert "tall" not in resolved["text"]

    def test_conflict_in_one_sentence_spares_the_other(self):
        """A conflict resolves per sentence; the untouched sentence keeps
        the current text's changes."""
        base = "a red ball on grass. the sky is blue."
        current = "a pink ball on grass. the sky is very blue."
        incoming = "a crimson ball on grass. the sky is blue."
        resolved = cj.resolve_fix(base, current, incoming)
        assert resolved["conflict"] is True
        assert resolved["text"] == (
            "a crimson ball on grass. the sky is very blue."
        )

    def test_appending_fix_is_not_dropped(self):
        """A fix appending at the very end of the caption still applies."""
        base = "a red ball."
        current = "MYTRIGGER a red ball."
        incoming = "a red ball. high quality photo."
        resolved = cj.resolve_fix(base, current, incoming)
        assert resolved["conflict"] is False
        assert resolved["text"] == (
            "MYTRIGGER a red ball. high quality photo."
        )
