"""Caption-review repository: the integrity verdicts.

Persists the output of the caption-review heuristics (see
:mod:`src.caption_review`). A review is keyed on a *revision*, so a new
revision naturally supersedes the previous verdict; the ``UNIQUE`` on
``revision_id`` makes :func:`upsert_review` idempotent. Bulk readers back
the gallery paints (one query per page, never per card), and
:func:`flagged_media_ids` powers the Caption tab's "to review" filter by
resolving each media's effective revision exactly like the caption reader.

Judging a caption against its *image* is no longer this module's business:
that is SigLIP grounding, in :mod:`src.sqlite_store.grounding`.
"""

import json
from contextlib import closing

from src import db
from src.sqlite_store.base import _query_all


def _decode_review(row):
    """Return a review row as a dict with its ``issues`` JSON parsed."""
    if row is None:
        return None
    data = dict(row)
    data["issues"] = json.loads(data["issues"]) if data["issues"] else []
    return data


def upsert_review(
    revision_id: int, status: str, issues, judge_model: str = None
) -> None:
    """Insert or replace the review for a revision.

    Parameters
    ----------
    revision_id : int
        The reviewed revision. Its previous review, if any, is overwritten.
    status : str
        ``'ok'`` | ``'integrity'``.
    issues : list of dict
        The ``{code, detail}`` issue dicts, stored as JSON.
    judge_model : str, optional
        Legacy column of the retired veracity judge; always ``None``.
    """
    payload = json.dumps(issues or [])
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "INSERT INTO caption_review "
                "(revision_id, status, issues, judge_model, created_at) "
                "VALUES (?, ?, ?, ?, datetime('now')) "
                "ON CONFLICT (revision_id) DO UPDATE SET "
                "status = excluded.status, issues = excluded.issues, "
                "judge_model = excluded.judge_model, "
                "created_at = excluded.created_at",
                (revision_id, status, payload, judge_model),
            )


def get_review(revision_id: int):
    """Return the review dict for a revision, or None when unreviewed."""
    with closing(db.connect()) as conn:
        row = conn.execute(
            "SELECT * FROM caption_review WHERE revision_id = ?",
            (revision_id,),
        ).fetchone()
    return _decode_review(row)


def reviews_bulk(revision_ids) -> dict:
    """Return ``{revision_id: review dict}`` for the reviewed revisions.

    One query for a whole page; revisions with no review are simply absent
    from the result (the paint treats a missing key as "not reviewed yet").
    """
    ids = [rid for rid in revision_ids if rid]
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = _query_all(
        f"SELECT * FROM caption_review WHERE revision_id IN ({placeholders})",
        ids,
    )
    return {row["revision_id"]: _decode_review(row) for row in rows}


def delete_review(revision_id: int) -> None:
    """Delete the review attached to a revision (no-op when absent)."""
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "DELETE FROM caption_review WHERE revision_id = ?",
                (revision_id,),
            )


def flagged_media_ids(dataset_id: int, caption_type_id: int, statuses) -> set:
    """Return the media a dataset should re-review for a caption type.

    A media is flagged when its *effective* revision for the type (the
    dataset's pinned revision, else the caption's head) carries a review
    whose status is in ``statuses``. Resolving the effective revision in SQL
    keeps the "to review" filter DB-paged (one query, no per-card work).

    Parameters
    ----------
    dataset_id : int
        The dataset whose media are considered.
    caption_type_id : int
        The caption type the filter scopes to.
    statuses : iterable of str
        The review statuses that count as "to review" (e.g.
        ``("integrity",)``).

    Returns
    -------
    set of int
        The flagged media ids (empty when ``statuses`` is empty).
    """
    statuses = list(statuses)
    if not statuses:
        return set()
    status_marks = ",".join("?" * len(statuses))
    rows = _query_all(
        "SELECT c.media_id AS media_id "
        "FROM caption c "
        "JOIN caption_review r ON r.revision_id = COALESCE("
        "  (SELECT dc.revision_id FROM dataset_caption dc "
        "   WHERE dc.dataset_id = ? AND dc.caption_id = c.id "
        "     AND dc.mode = 'pinned' AND dc.revision_id IS NOT NULL), "
        "  c.head_revision_id) "
        "WHERE c.caption_type_id = ? "
        f"  AND r.status IN ({status_marks}) "
        "  AND c.media_id IN "
        "    (SELECT media_id FROM dataset_media WHERE dataset_id = ?)",
        (dataset_id, caption_type_id, *statuses, dataset_id),
    )
    return {row["media_id"] for row in rows}
