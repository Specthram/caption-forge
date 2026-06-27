"""Caption decomposition — split a caption into atomic visual claims.

The LLM half of caption grounding. SigLIP scores a *short text* against an
image; a whole caption is far too long and too compound for that, so the
model in the shared loader slot first breaks it into atomic, independently
checkable statements ("the dog is running", "the grass is wet"), each
tagged with the :data:`KINDS` category it belongs to.
:mod:`src.siglip_grounding` then scores each claim on its own.

The kind matters because SigLIP is not equally trustworthy across them: it
judges *presence* well, but counting and spatial relations are known weak
spots, so :data:`UNRELIABLE_KINDS` is what the UI paints amber and labels
"indicative".

Every LLM call funnels through :func:`_run_llm`, the single seam tests
monkeypatch so no real weights are touched. Sampling is deterministic
(temperature 0, fixed seed) — the same caption always decomposes the same
way, so a re-run does not shuffle the user's claim list.
"""

import json

from src import config

_LLM_SEED = 0

# Generation ceiling for the decomposition. The JSON is small; a non-reasoning
# model stops at EOS well before this. Mainly caps a *reasoning* model's
# runaway ``<think>`` — kept tight so its KV cache can't OOM the GPU while the
# VLM is resident (4096 did). A reasoning model that needs more truncates and
# yields no claims; use a non-reasoning splitter for those.
_CLAIM_MAX_TOKENS = 1024

# The claim categories the decomposition prompt may emit. Anything else the
# model invents collapses to "object", the safest reading of "this is in
# the picture".
KINDS = ("object", "attribute", "scene", "count", "spatial")
DEFAULT_KIND = "object"

# The kinds SigLIP scores unreliably: it was trained to judge whether a
# text describes an image, not to count instances or resolve "to the left
# of". Their bars are shown, but flagged as indicative.
UNRELIABLE_KINDS = ("count", "spatial")


def _prompt() -> str:
    """Return the decomposition prompt template (config layer, cached)."""
    return config.load_review_prompts()["extract_claims"]


def _run_llm(image_path: str, prompt: str) -> str:
    """Run one deterministic LLM inference on an image, return the text.

    The single seam every decomposition funnels through, so a test can swap in
    a canned answer. Imports the captioner lazily to keep this module (imported
    by :mod:`src.storage`) torch-free. The loaded model's own thinking mode is
    honoured — there's no separate think mode for decomposition.
    """
    # pylint: disable=import-outside-toplevel
    from src import captioner, loader, settings

    think_mode = settings.get_model_think_mode(loader.current_model_type or "")
    return captioner.generate_caption(
        image_path,
        prompt,
        0.0,
        _LLM_SEED,
        think_mode,
        max_new_tokens=_CLAIM_MAX_TOKENS,
    )


def _clean_kind(value) -> str:
    """Return a known claim kind, defaulting to :data:`DEFAULT_KIND`."""
    kind = str(value or "").strip().lower()
    return kind if kind in KINDS else DEFAULT_KIND


def _claim(text, kind=DEFAULT_KIND):
    """Return a cleaned claim dict, or None when the text is not a claim.

    A claim must carry at least one letter: this drops the numeric junk a
    reasoning model's stray list (``[1, 2, 3]``) would otherwise smuggle in.
    """
    text = str(text or "").strip().strip('"')
    if not text or not any(char.isalpha() for char in text):
        return None
    return {"text": text, "kind": _clean_kind(kind)}


# Phrases a reasoning model leaks as plain prose — no <think> tags to strip, so
# they reach the parser and used to become "claims". A line with any of these
# first-person / meta markers is dropped as reasoning.
_REASONING_HINTS = (
    "the user",
    "let's",
    "let me",
    "i need",
    "i'll",
    "i will",
    "i should",
    "i can",
    "we need",
    "first,",
    "okay",
    "ok,",
    "got it",
    "here are",
    "here's",
    "sure,",
    "to do this",
    "step ",
    "the caption",
    "atomic",
    "checkable",
)
# A real visual claim is short; a reasoning sentence runs long. Anything past
# this many words is treated as prose, not a claim.
_MAX_CLAIM_WORDS = 18


def _iter_json_arrays(raw: str):
    """Yield every balanced, parseable JSON list found in the text.

    A reasoning model emits prose before its answer — sometimes with stray
    brackets ("statements [like this]") — so a naive first-``[``-to-last-``]``
    slice would swallow the reasoning and fail. Instead every ``[`` is tried
    as a start and each one that closes into valid JSON is yielded, in order.
    """
    for start, char in enumerate(raw):
        if char != "[":
            continue
        depth = 0
        for end in range(start, len(raw)):
            if raw[end] == "[":
                depth += 1
            elif raw[end] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(raw[start : end + 1])
                    except (ValueError, TypeError):
                        break
                    if isinstance(data, list):
                        yield data
                    break


def _claims_from_array(data) -> list[dict]:
    """Return the cleaned claims of one JSON list (junk items dropped)."""
    claims = []
    for item in data:
        claim = (
            _claim(item.get("text"), item.get("kind"))
            if isinstance(item, dict)
            else _claim(item)
        )
        if claim is not None:
            claims.append(claim)
    return claims


def _parse_json_claims(raw: str):
    """Return the claims of the best JSON array, or None when none yields any.

    Prefers an array of ``{"text", "kind"}`` objects — the prompted shape —
    over any bare list, so a reasoning model's incidental ``[1, 2, 3]`` never
    wins against the real answer that follows it.
    """
    arrays = list(_iter_json_arrays(raw))
    if not arrays:
        return None
    dict_arrays = [a for a in arrays if any(isinstance(x, dict) for x in a)]
    for array in dict_arrays or arrays:
        claims = _claims_from_array(array)
        if claims:
            return claims
    return None


def _looks_like_reasoning(text: str) -> bool:
    """Return whether a line reads as a model's reasoning, not a claim."""
    lowered = text.lower()
    if any(hint in lowered for hint in _REASONING_HINTS):
        return True
    return len(text.split()) > _MAX_CLAIM_WORDS


def _parse_line_claims(raw: str) -> list[dict]:
    """Return one claim per non-empty line, when the model ignored JSON.

    The fallback for a model that answered in plain text. Reasoning prose is
    filtered out (see :func:`_looks_like_reasoning`) so a thinking model's
    chain of thought never lands in the claim list.
    """
    claims = []
    for line in raw.splitlines():
        cleaned = line.strip().lstrip("-*0123456789.) \t")
        if not cleaned or _looks_like_reasoning(cleaned):
            continue
        claim = _claim(cleaned)
        if claim is not None:
            claims.append(claim)
    return claims


def parse_claims(raw: str) -> list[dict]:
    """Parse the decomposition output into ``{"text", "kind"}`` dicts.

    Prefers the prompted JSON array; falls back to one claim per line when the
    model ignored the format. Returns claims in reading order, empty when
    nothing usable was found.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    claims = _parse_json_claims(raw)
    if claims is not None:
        return claims
    return _parse_line_claims(raw)


def extract_claims(image_path: str, caption: str) -> list[dict]:
    """Decompose a caption into atomic, checkable claims (one LLM call).

    ``image_path`` is passed because the loaded VLM expects an image alongside
    the prompt (the task itself is text). Returns ``{"text", "kind"}`` dicts in
    reading order; empty for a blank caption (no inference run).
    """
    caption = (caption or "").strip()
    if not caption:
        return []
    # A literal ``str.format`` would choke on the JSON braces the prompt
    # shows the model as its output shape.
    prompt = _prompt().replace("{caption}", caption)
    return parse_claims(_run_llm(image_path, prompt))
