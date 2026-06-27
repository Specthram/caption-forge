"""Tests for the reasoning-stripping helper in :mod:`src.captioner`.

Importing :mod:`src.captioner` pulls in torch and OpenCV, so these tests are
heavier than the rest of the suite but exercise the output-cleaning logic.
"""

from src.captioner import (
    _compose_video_prompt,
    _strip_reasoning,
    _video_timing,
)


def test_empty_input_returns_empty():
    """An empty string is returned unchanged."""
    assert _strip_reasoning("") == ""


def test_plain_text_is_untouched():
    """Text without any reasoning markers is returned as-is."""
    assert _strip_reasoning("a red car on a road") == "a red car on a road"


def test_think_block_is_removed():
    """A ``<think>...</think>`` block is stripped, keeping the answer."""
    assert _strip_reasoning("<think>let me see</think>a cat") == "a cat"


def test_think_block_is_case_insensitive_and_multiline():
    """Think blocks are matched across lines and ignoring case."""
    text = "<THINK>\nstep one\nstep two\n</THINK>\nThe final answer"
    assert _strip_reasoning(text) == "The final answer"


def test_harmony_channel_keeps_final_content():
    """With harmony channels, only the final channel's content is kept."""
    text = (
        "<|channel|>analysis<|message|>thinking out loud"
        "<|channel|>final<|message|>the clean caption"
    )
    assert _strip_reasoning(text) == "the clean caption"


def test_unclosed_think_block_is_dropped():
    """A reasoning block cut off before ``</think>`` leaves no answer.

    A thinking model whose reasoning overruns the token budget emits an open
    ``<think>`` and is truncated mid-thought — the dangling block must be
    dropped whole, not leaked as if it were the answer.
    """
    text = "<think>Got it, let's tackle this. First, I need to"
    assert _strip_reasoning(text) == ""


def test_unclosed_think_after_a_closed_one_is_dropped():
    """A closed block is removed and a later dangling one is too."""
    text = "<think>plan</think>partial answer <think>wait, actually"
    assert _strip_reasoning(text) == "partial answer"


def test_video_timing_empty_for_single_frame():
    """A single (or no) timestamp yields no timing sentence."""
    assert _video_timing([]) == ""
    assert _video_timing([1.0]) == ""


def test_video_timing_reports_span_and_pace():
    """Several timestamps produce a span/pace sentence."""
    timing = _video_timing([0.0, 1.0, 2.0])
    assert "2.0s" in timing
    assert "every 1.00s" in timing


def test_compose_video_prompt_native_preamble():
    """The native preamble frames the input as a video and keeps the body."""
    out = _compose_video_prompt("native", 4, [0.0, 1.0], "BODY")
    assert "short video clip" in out
    assert out.endswith("BODY")


def test_compose_video_prompt_frames_preamble_counts_frames():
    """The frames preamble states the still-frame count and keeps the body."""
    out = _compose_video_prompt("frames", 6, [0.0, 1.0], "BODY")
    assert "6 still frames" in out
    assert out.endswith("BODY")
