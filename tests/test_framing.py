"""Tests for the tag-based framing classification (:mod:`src.framing`)."""

from src import framing

_BUCKETS = {
    "body_part": ["hand_focus", "eye_focus", "lower_body"],
    "face": ["portrait", "close-up"],
    "upper_body": ["upper_body", "cowboy_shot"],
    "full_body": ["full_body", "wide_shot"],
}


class TestNormalizeTag:
    """Tests for :func:`framing.normalize_tag`."""

    def test_lowercases_and_collapses_spaces(self):
        """Case and spacing never break a match against the config."""
        assert framing.normalize_tag("  Upper  Body ") == "upper_body"

    def test_none_and_empty_are_empty(self):
        """A missing name normalizes to the empty string."""
        assert framing.normalize_tag(None) == ""
        assert framing.normalize_tag("   ") == ""


class TestClassify:
    """Tests for :func:`framing.classify`."""

    def test_matches_a_framing_tag(self):
        """A media tagged ``full_body`` lands in the full-body bucket."""
        assert framing.classify(["1girl", "full_body"], _BUCKETS) == (
            "full_body"
        )

    def test_declaration_order_is_the_precedence(self):
        """The first declared bucket owning a tag wins.

        A portrait with an ``eye_focus`` classifies as the body-part
        bucket because ``body_part`` is declared before ``face``.
        """
        assert framing.classify(["portrait", "eye_focus"], _BUCKETS) == (
            "body_part"
        )

    def test_spacing_variants_still_match(self):
        """A tagger storing ``upper body`` matches ``upper_body`` config."""
        assert framing.classify(["Upper Body"], _BUCKETS) == "upper_body"

    def test_no_framing_tag_is_unknown(self):
        """Tags with no framing meaning classify as the unknown bucket."""
        assert framing.classify(["1girl", "smile"], _BUCKETS) == (
            framing.UNKNOWN_BUCKET
        )

    def test_empty_inputs_are_unknown(self):
        """No tags — or no buckets — never raises."""
        assert framing.classify([], _BUCKETS) == framing.UNKNOWN_BUCKET
        assert framing.classify(["portrait"], {}) == framing.UNKNOWN_BUCKET
