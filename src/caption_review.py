"""Local caption review — integrity heuristics (gradio-free).

Pure text, no model: a caption is flagged empty, truncated, looping, garbled
or carrying leftover reasoning tags. Each check returns zero or more
``{"code", "detail"}`` dicts; :func:`check_integrity` runs them all and
:func:`review_integrity` turns them into a ``{"status", "issues"}`` verdict.
Applies to images *and* videos (text only). Every threshold is a module
constant, unit-tested at its boundary.

Answers "is the caption well-formed?", never "is it true of the image?" — that
belongs to SigLIP grounding (:mod:`src.caption_claims` +
:mod:`src.siglip_grounding`), which replaced the old VLM veracity judge.
"""

import re
from collections import Counter

# Below this many non-space characters a caption is treated as empty (an
# effectively blank generation, e.g. the model returned nothing usable).
MIN_CAPTION_CHARS = 3

# Characters a well-formed caption may end on (after trailing quotes and
# brackets are peeled off). A caption ending on anything else reads as cut
# off mid-thought — the classic max-tokens truncation.
_TERMINAL_CHARS = ".!?…"
# Trailing wrappers stripped before the terminal-character test, so
# ``the cat sat."`` and ``(a red ball).`` still count as terminated.
_TRAILING_WRAPPERS = " \t\r\n\"')]}»”’"

# A word n-gram of this length repeated at least this many times marks a
# degenerate repetition loop (covers an end-of-text loop too, since the
# looping n-gram simply occurs several times).
REPEAT_NGRAM_WORDS = 3
REPEAT_MIN_TIMES = 3

# Above this share of "bad" characters (the Unicode replacement char or
# non-printable, non-whitespace code points — mojibake and control junk)
# the caption is flagged as garbage.
GARBAGE_RATIO = 0.15
_REPLACEMENT_CHAR = "�"

# Residual reasoning/harmony markers a clean caption must never contain
# (mirrors the shapes stripped by :func:`src.captioner._strip_reasoning`).
_RESIDUE_PATTERNS = (
    re.compile(r"(?i)</?think>"),
    re.compile(r"(?i)<\|?\s*channel\s*\|?>"),
    re.compile(
        r"(?i)<\|?\s*(?:start|end|return|message|assistant|user|system)"
        r"\s*\|?>"
    ),
)


def _check_empty(text: str) -> list[dict]:
    """Flag a caption that is blank or shorter than the minimum length."""
    if len(text.strip()) < MIN_CAPTION_CHARS:
        return [{"code": "empty", "detail": "caption is empty or too short"}]
    return []


def _check_truncated(text: str) -> list[dict]:
    """Flag a caption that ends without terminal punctuation.

    An unterminated caption is the signature of a generation cut off in the
    middle of a sentence (or mid-word), so the missing end punctuation is
    the heuristic. Trailing quotes and brackets are peeled off first.
    """
    core = text.strip().rstrip(_TRAILING_WRAPPERS)
    # Blank / too-short text is already covered by the empty check.
    if len(core) < MIN_CAPTION_CHARS:
        return []
    if core[-1] not in _TERMINAL_CHARS:
        tail = core[-24:]
        return [
            {
                "code": "truncated",
                "detail": f"ends without terminal punctuation: '…{tail}'",
            }
        ]
    return []


def repeated_phrase(text: str) -> tuple | None:
    """Return the looping n-gram of a caption, or None.

    The reader behind :func:`_check_repetition`, exposed so a caller needing
    the phrase itself (the quality report highlights it) skips re-parsing the
    ``detail`` string. Returns ``(phrase, times)`` for the most repeated
    :data:`REPEAT_NGRAM_WORDS`-word n-gram once it hits
    :data:`REPEAT_MIN_TIMES`; None when there's no loop.
    """
    words = text.split()
    if len(words) < REPEAT_NGRAM_WORDS:
        return None
    grams = Counter(
        tuple(word.lower() for word in words[i : i + REPEAT_NGRAM_WORDS])
        for i in range(len(words) - REPEAT_NGRAM_WORDS + 1)
    )
    gram, times = grams.most_common(1)[0]
    if times < REPEAT_MIN_TIMES:
        return None
    return " ".join(gram), times


def _check_repetition(text: str) -> list[dict]:
    """Flag a degenerate repetition loop (an n-gram repeated too often)."""
    found = repeated_phrase(text)
    if found is None:
        return []
    phrase, times = found
    return [
        {
            "code": "repetition",
            "detail": f"phrase repeated {times}x: '{phrase}'",
        }
    ]


def _is_bad_char(char: str) -> bool:
    """Return whether a character counts as garbage (mojibake / control)."""
    if char == _REPLACEMENT_CHAR:
        return True
    if char.isspace():
        return False
    return not char.isprintable()


def _check_garbage(text: str) -> list[dict]:
    """Flag a caption dominated by non-printable / mojibake characters."""
    stripped = text.strip()
    if not stripped:
        return []
    bad = sum(1 for char in stripped if _is_bad_char(char))
    if bad / len(stripped) >= GARBAGE_RATIO:
        return [
            {
                "code": "garbage",
                "detail": "high ratio of unreadable characters",
            }
        ]
    return []


def _check_reasoning_residue(text: str) -> list[dict]:
    """Flag leftover reasoning / harmony-channel markers in the caption."""
    if any(pattern.search(text) for pattern in _RESIDUE_PATTERNS):
        return [
            {
                "code": "reasoning_residue",
                "detail": "leftover reasoning/thinking markers",
            }
        ]
    return []


# Every integrity check, run in order by :func:`check_integrity`.
_CHECKS = (
    _check_empty,
    _check_truncated,
    _check_repetition,
    _check_garbage,
    _check_reasoning_residue,
)


def check_integrity(text: str) -> list[dict]:
    """Run every integrity heuristic and return the issues found.

    One ``{"code", "detail"}`` per issue, in check order. Empty when the
    caption passes every heuristic.
    """
    text = text or ""
    issues: list[dict] = []
    for check in _CHECKS:
        issues.extend(check(text))
    return issues


def review_integrity(text: str) -> dict:
    """Return the integrity verdict for a caption.

    ``{"status": "ok" | "integrity", "issues": [...]}`` — ``"integrity"``
    whenever at least one heuristic fired.
    """
    issues = check_integrity(text)
    return {"status": "ok" if not issues else "integrity", "issues": issues}
