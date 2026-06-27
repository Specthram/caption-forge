"""Reference-free caption-score repository.

Persists the output of :mod:`src.caption_score`. A score hangs off a
*revision* (like a review or a grounding, so an edited caption drops its
numbers) and there is one row per ``(revision, encoder family)`` — the
``UNIQUE (revision_id, model_kind)`` makes :func:`upsert_caption_score`
idempotent, so re-scoring a family refreshes its line in place.

Only the raw 0-100 scores and the checkpoint that produced them are stored;
nothing here knows about the stale-checkpoint highlight, which the router
derives on read by comparing ``model_id`` to the configured one.
"""

from src.sqlite_store.base import _query_all, _write


def upsert_caption_score(
    revision_id: int, model_kind: str, model_id: str, score: float
) -> None:
    """Insert or refresh one revision's score for an encoder family."""
    _write(
        "INSERT INTO caption_score "
        "(revision_id, model_kind, model_id, score, created_at) "
        "VALUES (?, ?, ?, ?, datetime('now')) "
        "ON CONFLICT (revision_id, model_kind) DO UPDATE SET "
        "model_id = excluded.model_id, score = excluded.score, "
        "created_at = excluded.created_at",
        (revision_id, model_kind, model_id, float(score)),
    )


def get_caption_scores(revision_id: int) -> dict:
    """Return ``{model_kind: {model_id, score}}`` for a revision."""
    rows = _query_all(
        "SELECT model_kind, model_id, score FROM caption_score "
        "WHERE revision_id = ?",
        (revision_id,),
    )
    return {
        row["model_kind"]: {
            "model_id": row["model_id"],
            "score": row["score"],
        }
        for row in rows
    }


def caption_scores_bulk(revision_ids) -> dict:
    """Return ``{revision_id: {model_kind: {model_id, score}}}`` for a set.

    One query whatever the count — backs the dataset caption-score report,
    which reads every media's stored scores at once (never per card).
    """
    ids = [rid for rid in revision_ids if rid]
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = _query_all(
        "SELECT revision_id, model_kind, model_id, score "
        f"FROM caption_score WHERE revision_id IN ({placeholders})",
        ids,
    )
    result: dict = {}
    for row in rows:
        result.setdefault(row["revision_id"], {})[row["model_kind"]] = {
            "model_id": row["model_id"],
            "score": row["score"],
        }
    return result


def delete_caption_scores(revision_id: int) -> None:
    """Drop every encoder's score for a revision (no-op when absent)."""
    _write(
        "DELETE FROM caption_score WHERE revision_id = ?",
        (revision_id,),
    )


def upsert_media_tag_score(
    media_id: int,
    model_kind: str,
    model_id: str,
    score: float,
    scored_text: str,
) -> None:
    """Insert or refresh a media's tag score for one encoder family.

    ``scored_text`` is the comma-joined tag string the score was measured on,
    kept so the reader can tell whether the tags have changed since.
    """
    _write(
        "INSERT INTO media_tag_score "
        "(media_id, model_kind, model_id, score, scored_text, created_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now')) "
        "ON CONFLICT (media_id, model_kind) DO UPDATE SET "
        "model_id = excluded.model_id, score = excluded.score, "
        "scored_text = excluded.scored_text, created_at = excluded.created_at",
        (media_id, model_kind, model_id, float(score), scored_text),
    )


def get_media_tag_scores(media_id: int) -> dict:
    """Return ``{model_kind: {model_id, score, scored_text}}`` for a media."""
    rows = _query_all(
        "SELECT model_kind, model_id, score, scored_text "
        "FROM media_tag_score WHERE media_id = ?",
        (media_id,),
    )
    return {
        row["model_kind"]: {
            "model_id": row["model_id"],
            "score": row["score"],
            "scored_text": row["scored_text"],
        }
        for row in rows
    }


def delete_media_tag_scores(media_id: int) -> None:
    """Drop every encoder's tag score for a media (no-op when absent)."""
    _write(
        "DELETE FROM media_tag_score WHERE media_id = ?",
        (media_id,),
    )
