"""Tests for :mod:`src.caption_review` integrity heuristics."""

from src import caption_review as cr


def _codes(text):
    """Return the set of issue codes raised for ``text``."""
    return {issue["code"] for issue in cr.check_integrity(text)}


class TestEmpty:
    """The ``empty`` heuristic and the minimum-length boundary."""

    def test_blank_is_empty(self):
        """Whitespace-only text is flagged empty."""
        assert "empty" in _codes("   \n\t ")

    def test_below_min_chars_is_empty(self):
        """Text shorter than the minimum is empty."""
        short = "a" * (cr.MIN_CAPTION_CHARS - 1)
        assert "empty" in _codes(short)

    def test_at_min_chars_not_empty(self):
        """Text at the minimum length is not empty (boundary)."""
        ok = "a" * cr.MIN_CAPTION_CHARS + "."
        assert "empty" not in _codes(ok)


class TestTruncated:
    """The ``truncated`` heuristic (missing terminal punctuation)."""

    def test_no_terminal_punctuation(self):
        """A caption not ending in .!?… reads as cut off."""
        assert "truncated" in _codes("A red ball rolling across the")

    def test_terminal_punctuation_ok(self):
        """A properly ended sentence is not truncated."""
        assert "truncated" not in _codes("A red ball on the grass.")

    def test_trailing_quote_peeled(self):
        """A closing quote/bracket after the period still counts as ended."""
        assert "truncated" not in _codes('She said "hello".')

    def test_ellipsis_ok(self):
        """An intentional ellipsis is a valid terminator."""
        assert "truncated" not in _codes("A quiet street at dusk…")


class TestRepetition:
    """The ``repetition`` heuristic (looping n-gram)."""

    def test_looping_ngram_flagged(self):
        """A 3-word phrase repeated 3+ times is a repetition loop."""
        text = "the cat sat " * 3 + "on the mat."
        assert "repetition" in _codes(text)

    def test_two_repeats_not_flagged(self):
        """Two occurrences stay under the threshold (boundary)."""
        text = "the cat sat the cat sat on the mat."
        assert "repetition" not in _codes(text)

    def test_normal_caption_not_flagged(self):
        """A varied caption raises no repetition."""
        text = "A woman in a red coat walks a small brown dog downtown."
        assert "repetition" not in _codes(text)


class TestGarbage:
    """The ``garbage`` heuristic (mojibake / non-printable ratio)."""

    def test_replacement_chars_flagged(self):
        """A caption dominated by the replacement char is garbage."""
        assert "garbage" in _codes("a" + "�" * 10 + ".")

    def test_clean_text_not_flagged(self):
        """Clean ASCII prose is never garbage."""
        assert "garbage" not in _codes("A clean, readable caption.")

    def test_accents_not_flagged(self):
        """Legitimate accented characters are printable, not garbage."""
        assert "garbage" not in _codes("The café served a jalapeño soufflé.")


class TestReasoningResidue:
    """The ``reasoning_residue`` heuristic (leftover thinking markers)."""

    def test_think_tag_flagged(self):
        """A leftover <think> tag is residue."""
        assert "reasoning_residue" in _codes("<think>hmm</think> A red ball.")

    def test_channel_marker_flagged(self):
        """A harmony channel marker is residue."""
        assert "reasoning_residue" in _codes(
            "<|channel|>final A red ball on grass."
        )

    def test_clean_caption_no_residue(self):
        """A plain caption carries no residue."""
        assert "reasoning_residue" not in _codes("A red ball on grass.")


class TestVerdict:
    """:func:`review_integrity` maps issues to a status."""

    def test_ok_when_clean(self):
        """A clean caption yields status 'ok' and no issues."""
        verdict = cr.review_integrity("A red ball on the grass.")
        assert verdict == {"status": "ok", "issues": []}

    def test_integrity_when_flagged(self):
        """Any issue yields status 'integrity'."""
        verdict = cr.review_integrity("")
        assert verdict["status"] == "integrity"
        assert verdict["issues"]

    def test_none_input_is_empty(self):
        """None is treated as an empty caption, not an error."""
        assert "empty" in {i["code"] for i in cr.check_integrity(None)}
