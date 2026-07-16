"""Rule-based caption judging — the pure logic behind the Review sub-tab.

A *judge* model (chosen independently from the captioner) checks each caption
against plain-language *rules* and proposes a minimal correction. This module
holds everything that needs no model, so it is unit-testable on its own:

* :func:`build_prompt` — the strict, JSON-only system prompt for one rule.
* :func:`parse_judgement` — pull the judge's JSON verdict out of its raw text.
* :func:`judged_finding` — turn a verdict into a finding, or ``None`` when the
  caption complies or the judge over-rewrote (the anti-free-rewrite guard).
* :func:`check_det_rule` — the deterministic trigger-word rule: no model, a
  substring test and a template correction, hence a *safe* fix.

The model calls (text pass, vision pass) live in
:mod:`server.runners.review_run`, which feeds this module the raw output. The
merge/persistence rules live in :mod:`src.sqlite_store.review_queue`.
"""

import json
import re
from difflib import SequenceMatcher

# Below this similarity between the original caption and the judge's proposal,
# the proposal is treated as a free rewrite (the judge ignored "only change
# what violates the rule") and dropped. Word-level ratio in [0, 1].
MIN_KEEP_RATIO = 0.5

# Grabs the first `"…"` or `'…'` token of a rule's text — the trigger word a
# deterministic rule is parametrised by.
_QUOTED = re.compile(r"""["']([^"']+)["']""")


def build_prompt(rule_text: str, caption: str) -> str:
    """Return the judge's system+user prompt for one rule and one caption.

    Temperature-0, JSON-only, single-rule: a strict shape is far easier to
    parse back than free prose, and one rule per call keeps each finding
    attributable to exactly one rule (so the wizard can group by rule).
    """
    return (
        "You are a strict caption reviewer. Judge ONE caption against ONE "
        "rule. Only change what violates the rule; keep every other word "
        "identical. Never rewrite the caption freely.\n"
        f"Rule: {rule_text}\n"
        f"Caption: {caption}\n"
        "Return ONLY JSON, no prose, no code fence:\n"
        '{"violates": true or false, "note": "one short sentence", '
        '"corrected_caption": "the minimally fixed caption, or null if it '
        'already complies"}'
    )


def _extract_json(raw: str):
    """Return the first JSON object parsed out of ``raw``, or None.

    The judge may wrap its JSON in prose or a code fence; the widest
    ``{ … }`` span is tried first, then trimmed from the right on failure so
    a trailing sentence after the object still parses.
    """
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return None
    span = raw[start : end + 1]
    while span:
        try:
            return json.loads(span)
        except json.JSONDecodeError:
            cut = span.rfind("}", 0, len(span) - 1)
            if cut <= 0:
                return None
            span = span[: cut + 1]
    return None


def parse_judgement(raw: str) -> dict | None:
    """Return the judge's verdict as a normalised dict, or None.

    ``{"violates": bool, "note": str, "corrected_caption": str | None}``.
    None when no JSON object could be recovered from ``raw``.
    """
    data = _extract_json(raw)
    if not isinstance(data, dict):
        return None
    corrected = data.get("corrected_caption")
    return {
        "violates": bool(data.get("violates")),
        "note": str(data.get("note") or "").strip(),
        "corrected_caption": (
            corrected.strip() if isinstance(corrected, str) else None
        ),
    }


def rewrite_ratio(before: str, after: str) -> float:
    """Return the word-level similarity of two captions, in [0, 1]."""
    return SequenceMatcher(
        None, (before or "").split(), (after or "").split()
    ).ratio()


def judged_finding(before: str, verdict: dict | None) -> dict | None:
    """Return a finding for a judge verdict, or None when there is nothing.

    A finding is produced only when the verdict flags a violation, offers a
    *different* corrected caption, and that correction stays close enough to
    the original (:data:`MIN_KEEP_RATIO`) to be a targeted fix rather than a
    free rewrite. Returns ``{"note", "caption_after"}``.
    """
    if not verdict or not verdict["violates"]:
        return None
    after = verdict["corrected_caption"]
    if not after or after.strip() == (before or "").strip():
        return None
    if rewrite_ratio(before, after) < MIN_KEEP_RATIO:
        return None
    return {"note": verdict["note"], "caption_after": after}


def _target_word(rule_text: str, trigger_words) -> str | None:
    """Return the trigger word a deterministic rule checks for.

    Prefers a word quoted in the rule text (``… "ryn" …``); falls back to the
    dataset's first trigger word so a preset stays correct even if its text
    was edited.
    """
    match = _QUOTED.search(rule_text or "")
    if match:
        return match.group(1).strip()
    words = list(trigger_words or [])
    return words[0] if words else None


def check_det_rule(rule: dict, caption: str, trigger_words) -> dict | None:
    """Return a finding for a deterministic (trigger-word) rule, or None.

    The word must appear as a whole token, case-insensitively. When it is
    missing the correction prepends it in the usual comma-separated form
    (``"ryn, a red ball…"``). No model is involved, so the fix is *safe*.
    """
    word = _target_word(rule.get("text", ""), trigger_words)
    if not word:
        return None
    caption = caption or ""
    present = re.search(
        rf"(?<!\w){re.escape(word)}(?!\w)", caption, flags=re.IGNORECASE
    )
    if present:
        return None
    fixed = f"{word}, {caption}" if caption.strip() else word
    return {
        "note": f'trigger word "{word}" is missing',
        "caption_after": fixed,
    }
