"""Framing classification of a media from its booru tags.

The dataset auto-builder needs to know how a picture frames its subject
(a face close-up, an upper-body shot, a full-body shot, a body-part
focus...) to honor the per-bucket ratios of a target type. Rather than
running another vision model, the classification piggybacks on the tags
the WD auto-tagger already stores: danbooru's framing vocabulary
(``portrait``, ``upper_body``, ``full_body``, ``*_focus``...) is mapped
to buckets by ``config/default/autobuild.json`` (user-overridable, see
:func:`src.config.load_autobuild_config`).

Pure functions only — the callers (see :mod:`src.dataset_builder`) fetch
the tags and pass them in.
"""

# The bucket of a media whose tags match no configured framing tag. Kept
# selectable by the "any" ratio only: a quota-bearing bucket never counts
# an unclassified media.
UNKNOWN_BUCKET = "unknown"

# The pseudo-bucket of a ratio entry that accepts every media regardless
# of framing (the whole pool, unknown included) — used by target types
# where framing is irrelevant (style, concept).
ANY_BUCKET = "any"


def normalize_tag(name) -> str:
    """Return a tag name normalized for comparison.

    Lowercased, trimmed, spaces collapsed to underscores — so the config
    matches whether the tagger stored ``upper body`` or ``upper_body``.
    """
    return "_".join(str(name or "").strip().lower().split())


def classify(tag_names, buckets: dict) -> str:
    """Return the framing bucket of a media from its tag names.

    ``buckets`` is ``{bucket_id: [tag names]}`` (the ``framing_buckets``
    config); declaration order is precedence — the first bucket owning one of
    the media's tags wins (a ``portrait`` with ``eye_focus`` classifies as the
    body-part bucket when it's declared first). Returns the bucket id, or
    :data:`UNKNOWN_BUCKET` when no tag is a framing tag.
    """
    present = {normalize_tag(name) for name in tag_names or ()}
    for bucket_id, bucket_tags in (buckets or {}).items():
        if any(normalize_tag(tag) in present for tag in bucket_tags or ()):
            return bucket_id
    return UNKNOWN_BUCKET
