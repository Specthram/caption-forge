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


# Word/whitespace tokens: "".join(_tokens(s)) round-trips s exactly, so a
# merge never mangles spacing or punctuation.
_TOKEN = re.compile(r"\S+|\s+")


def _tokens(text: str) -> list:
    """Split ``text`` into word and whitespace tokens (lossless)."""
    return _TOKEN.findall(text or "")


def _edits(base_tokens: list, side_tokens: list) -> list:
    """Return one side's edits against the base.

    Each edit is ``(start, end, replacement_tokens)`` in base coordinates
    (``start == end`` for a pure insertion); spans are sorted and disjoint.
    """
    matcher = SequenceMatcher(None, base_tokens, side_tokens)
    return [
        (i1, i2, side_tokens[j1:j2])
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]


# A base segment ends on a word carrying sentence/clause punctuation, so a
# conflict resolves over a whole sentence (prose) or a whole tag (booru).
_SEGMENT_END = re.compile(r"[.!?;,]$")


def _segment_bounds(tokens: list) -> list:
    """Return the base's segments as token ranges (sentences / clauses)."""
    bounds, start = [], 0
    for index, token in enumerate(tokens):
        if not token.isspace() and _SEGMENT_END.search(token):
            end = index + 1
            if end < len(tokens) and tokens[end].isspace():
                end += 1
            bounds.append((start, end))
            start = end
    if start < len(tokens) or not bounds:
        bounds.append((start, len(tokens)))
    return bounds


def _overlaps(first: tuple, second: tuple) -> bool:
    """Whether two base-coordinate edits collide (insertions zero-width)."""
    a1, a2, _ = first
    b1, b2, _ = second
    if a1 == a2 and b1 == b2:  # two insertions at the same point collide
        return a1 == b1
    return a1 < b2 and b1 < a2


def _replay(tokens: list, start: int, end: int, edits: list) -> list:
    """Return ``tokens[start:end]`` with the (contained) edits applied."""
    output = []
    position = start
    for span_start, span_end, replacement in sorted(
        edits, key=lambda edit: (edit[0], edit[1])
    ):
        output.extend(tokens[position:span_start])
        output.extend(replacement)
        position = span_end
    output.extend(tokens[position:end])
    return output


def resolve_fix(base: str, current: str, incoming: str) -> dict:
    """Resolve the ``base → incoming`` fix against the live ``current``.

    Git-style three-way merge that never re-locates a diff: both sides'
    edits stay in the coordinates of the ``base`` they were computed
    against, so collision detection is exact interval arithmetic — no fuzzy
    matching, hence no false anchor when a sentence changed.

    Edits that do not collide merge automatically, wherever they sit. When
    the two sides collide, the *whole segment* (sentence / clause) around
    the collision takes the incoming fix's rendering verbatim — one
    coherent, judge-authored sentence, never an interleaving of two
    rewrites — and the result is flagged so the UI shows a ⚠ conflict the
    user settles with accept / reject / inline edit.

    Returns ``{"text": str, "conflict": bool}``.
    """
    if base == incoming:  # no-op fix — never clobber the live caption
        return {"text": current, "conflict": False}
    if current in (base, incoming):
        return {"text": incoming, "conflict": False}
    tokens = _tokens(base)
    ours = _edits(tokens, _tokens(current))
    theirs = _edits(tokens, _tokens(incoming))

    # Group the base's segments so every edit falls entirely inside one
    # group (an edit spanning a boundary fuses the neighbours).
    groups = _segment_bounds(tokens)
    for start, end, _ in ours + theirs:
        end = max(end, min(start + 1, len(tokens)))
        touched = [g for g in groups if g[0] < end and start < g[1]] or [
            groups[-1]
        ]
        fused = (
            min(g[0] for g in touched),
            max(g[1] for g in touched),
        )
        groups = [g for g in groups if g not in touched]
        groups.append(fused)
        groups.sort()

    merged = []
    conflict = False
    for index, (group_start, group_end) in enumerate(groups):
        last = index == len(groups) - 1
        inside_ours = [
            e for e in ours if _inside(e, group_start, group_end, last)
        ]
        inside_theirs = [
            e for e in theirs if _inside(e, group_start, group_end, last)
        ]
        collides = any(
            _overlaps(first, second)
            for first in inside_ours
            for second in inside_theirs
        )
        if collides:
            # The judge's whole sentence, verbatim; our edits there drop.
            conflict = True
            merged.extend(
                _replay(tokens, group_start, group_end, inside_theirs)
            )
        else:
            merged.extend(
                _replay(
                    tokens,
                    group_start,
                    group_end,
                    inside_ours + inside_theirs,
                )
            )
    return {"text": "".join(merged), "conflict": conflict}


def _inside(edit: tuple, start: int, end: int, last: bool) -> bool:
    """Whether an edit falls inside ``[start, end)`` (inserts included).

    An insertion belongs to the group containing its point; one sitting at
    the very end of the text belongs to the last group.
    """
    e1, e2, _ = edit
    if e1 == e2:
        return start <= e1 < end or (last and e1 == end)
    return start <= e1 and e2 <= end


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
