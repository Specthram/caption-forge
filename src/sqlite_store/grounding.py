"""SigLIP grounding repository: caption claims and tag scores.

Persists the output of :mod:`src.siglip_grounding` on both entry points.
A caption grounding hangs off a *revision* (like a review, so an edited
caption drops its scores) and owns its ordered claims; a tag grounding
hangs off a ``(media, tag, checkpoint)`` triple, because a picture keeps
its tag verdicts across every dataset that uses it.

Both sides store the raw 0-100 scores only. Nothing here knows about the
validation threshold — it lives in Settings and is applied on read, so
moving the slider never invalidates a run. Heat maps are never stored:
they are re-derived from the image on demand (one forward pass).

Bulk readers back the gallery paints — one query per page, never per card.
"""

from contextlib import closing

from src import db
from src.sqlite_store.base import _query_all, _query_one, _write

# The effective revision of a media's caption inside a dataset: the pinned
# revision when the dataset pins one, else the caption's head. Written once
# here and shared by the two paged readers below, exactly like the
# equivalent COALESCE in :mod:`src.sqlite_store.reviews`.
_EFFECTIVE_REVISION = (
    "COALESCE("
    "  (SELECT dc.revision_id FROM dataset_caption dc "
    "   WHERE dc.dataset_id = ? AND dc.caption_id = c.id "
    "     AND dc.mode = 'pinned' AND dc.revision_id IS NOT NULL), "
    "  c.head_revision_id)"
)


def _claim_rows(grounding_ids) -> dict:
    """Return ``{grounding_id: [claim dict, ...]}`` in reading order."""
    ids = [gid for gid in grounding_ids if gid]
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = _query_all(
        "SELECT id, grounding_id, position, text, kind, score, rejected "
        f"FROM caption_grounding_claim WHERE grounding_id IN ({placeholders}) "
        "ORDER BY grounding_id, position",
        ids,
    )
    claims: dict = {gid: [] for gid in ids}
    for row in rows:
        claim = dict(row)
        claim["rejected"] = bool(claim["rejected"])
        claims[claim["grounding_id"]].append(claim)
    return claims


def upsert_caption_grounding(revision_id: int, model_id: str, claims) -> int:
    """Replace a revision's grounding with a fresh run; return its id.

    A re-run is a new measurement, so the previous claims — and the user's
    ``rejected`` marks on them — are dropped rather than matched by text:
    silently carrying a rejection onto a claim the LLM re-worded would be
    a lie about what the user reviewed.

    Parameters
    ----------
    revision_id : int
        The grounded caption revision.
    model_id : str
        The SigLIP checkpoint that produced the scores.
    claims : sequence of dict
        ``{"text", "kind", "score"}`` dicts, in caption reading order.

    Returns
    -------
    int
        The ``caption_grounding`` row id.
    """
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "DELETE FROM caption_grounding WHERE revision_id = ?",
                (revision_id,),
            )
            cursor = conn.execute(
                "INSERT INTO caption_grounding "
                "(revision_id, model_id, created_at) "
                "VALUES (?, ?, datetime('now'))",
                (revision_id, model_id),
            )
            grounding_id = cursor.lastrowid
            conn.executemany(
                "INSERT INTO caption_grounding_claim "
                "(grounding_id, position, text, kind, score) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        grounding_id,
                        position,
                        claim["text"],
                        claim["kind"],
                        float(claim["score"]),
                    )
                    for position, claim in enumerate(claims)
                ],
            )
    return grounding_id


def get_caption_grounding(revision_id: int):
    """Return a revision's grounding with its claims, or None."""
    row = _query_one(
        "SELECT * FROM caption_grounding WHERE revision_id = ?",
        (revision_id,),
    )
    if row is None:
        return None
    grounding = dict(row)
    grounding["claims"] = _claim_rows([grounding["id"]])[grounding["id"]]
    return grounding


def caption_groundings_bulk(revision_ids) -> dict:
    """Return ``{revision_id: grounding dict}`` for a page of revisions.

    Two queries whatever the page size. Revisions with no grounding are
    absent from the result (the paint reads a missing key as "never
    grounded").
    """
    ids = [rid for rid in revision_ids if rid]
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = _query_all(
        "SELECT * FROM caption_grounding "
        f"WHERE revision_id IN ({placeholders})",
        ids,
    )
    groundings = [dict(row) for row in rows]
    claims = _claim_rows([row["id"] for row in groundings])
    for grounding in groundings:
        grounding["claims"] = claims.get(grounding["id"], [])
    return {row["revision_id"]: row for row in groundings}


def set_claim_rejected(claim_id: int, rejected: bool) -> None:
    """Mark (or restore) one grounded claim as non-validated by the user."""
    _write(
        "UPDATE caption_grounding_claim SET rejected = ? WHERE id = ?",
        (1 if rejected else 0, claim_id),
    )


def delete_caption_grounding(revision_id: int) -> None:
    """Drop a revision's grounding, claims included (no-op when absent)."""
    _write(
        "DELETE FROM caption_grounding WHERE revision_id = ?",
        (revision_id,),
    )


def grounded_media_ids(
    dataset_id: int, caption_type_id: int, model_id: str
) -> set:
    """Return the media already grounded under a checkpoint, for a type.

    Resolves each media's effective revision in SQL — the same COALESCE the
    caption reader uses — so a batch run can skip what it already scored
    without loading a single caption.
    """
    rows = _query_all(
        "SELECT c.media_id AS media_id FROM caption c "
        f"JOIN caption_grounding g ON g.revision_id = {_EFFECTIVE_REVISION} "
        "WHERE c.caption_type_id = ? AND g.model_id = ? "
        "  AND c.media_id IN "
        "    (SELECT media_id FROM dataset_media WHERE dataset_id = ?)",
        (dataset_id, caption_type_id, model_id, dataset_id),
    )
    return {row["media_id"] for row in rows}


def low_grounding_media_ids(
    dataset_id: int, caption_type_id: int, model_id: str, threshold: float
) -> set:
    """Return the media carrying at least one claim under the threshold.

    Backs the Caption tab's "weak grounding" filter. A rejected claim is
    already handled by the user, so it never re-flags its media. Paged
    DB-side like the "to review" filter, on the same effective revision.
    """
    rows = _query_all(
        "SELECT DISTINCT c.media_id AS media_id FROM caption c "
        f"JOIN caption_grounding g ON g.revision_id = {_EFFECTIVE_REVISION} "
        "JOIN caption_grounding_claim cl ON cl.grounding_id = g.id "
        "WHERE c.caption_type_id = ? AND g.model_id = ? "
        "  AND cl.rejected = 0 AND cl.score < ? "
        "  AND c.media_id IN "
        "    (SELECT media_id FROM dataset_media WHERE dataset_id = ?)",
        (dataset_id, caption_type_id, model_id, float(threshold), dataset_id),
    )
    return {row["media_id"] for row in rows}


def upsert_tag_grounding(
    media_id: int, tag_id: int, model_id: str, score: float
) -> None:
    """Insert or refresh one ``(media, tag, checkpoint)`` SigLIP score."""
    _write(
        "INSERT INTO media_tag_grounding "
        "(media_id, tag_id, model_id, score, created_at) "
        "VALUES (?, ?, ?, ?, datetime('now')) "
        "ON CONFLICT (media_id, tag_id, model_id) DO UPDATE SET "
        "score = excluded.score, created_at = excluded.created_at",
        (media_id, tag_id, model_id, float(score)),
    )


def tag_grounding_for_media(media_id: int, model_id: str) -> dict:
    """Return ``{tag_id: score}`` for a media's *currently attached* tags.

    The join on ``media_tag`` is what hides the rows a detached tag leaves
    behind (see the schema note in :mod:`src.db`).
    """
    rows = _query_all(
        "SELECT g.tag_id AS tag_id, g.score AS score "
        "FROM media_tag_grounding g "
        "JOIN media_tag mt "
        "  ON mt.media_id = g.media_id AND mt.tag_id = g.tag_id "
        "WHERE g.media_id = ? AND g.model_id = ?",
        (media_id, model_id),
    )
    return {row["tag_id"]: row["score"] for row in rows}


def tag_groundings_bulk(media_ids, model_id: str) -> dict:
    """Return ``{media_id: {tag_id: score}}`` for many media at once."""
    ids = [int(mid) for mid in media_ids if mid]
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = _query_all(
        "SELECT g.media_id AS media_id, g.tag_id AS tag_id, g.score AS score "
        "FROM media_tag_grounding g "
        "JOIN media_tag mt "
        "  ON mt.media_id = g.media_id AND mt.tag_id = g.tag_id "
        f"WHERE g.media_id IN ({placeholders}) AND g.model_id = ?",
        (*ids, model_id),
    )
    result: dict = {media_id: {} for media_id in ids}
    for row in rows:
        result[row["media_id"]][row["tag_id"]] = row["score"]
    return result


def delete_tag_grounding(media_id: int, tag_id: int) -> None:
    """Drop every checkpoint's score for one ``(media, tag)`` pair."""
    _write(
        "DELETE FROM media_tag_grounding WHERE media_id = ? AND tag_id = ?",
        (media_id, tag_id),
    )
