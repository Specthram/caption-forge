"""Maintenance sweeps over the index tables (the System view).

Report/purge pairs for the recomputable per-media index data: quality
scores, embedding vectors and the index columns (dimensions, perceptual
hashes, image stats). Everything here is rebuilt by the Libraries tab's
on-demand actions, so purging only costs the next recompute.
"""

from contextlib import closing

from src import db
from src.sqlite_store.base import _query_one


def quality_scores_report() -> dict:
    """Return ``{count, bytes}`` for the stored quality scores.

    Rows are fixed-width (a metric id, a REAL, a date); ``bytes`` is a
    rough per-row footprint estimate, not an exact figure.
    """
    row = _query_one("SELECT COUNT(*) AS n FROM media_quality")
    count = row["n"] if row else 0
    return {"count": count, "bytes": count * 50}


def purge_quality_scores() -> int:
    """Delete every stored quality score; return how many rows."""
    with closing(db.connect()) as conn:
        with conn:
            return conn.execute("DELETE FROM media_quality").rowcount


def embeddings_report() -> dict:
    """Return ``{count, bytes}`` for the stored embedding vectors."""
    row = _query_one(
        "SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(vector)), 0) AS b "
        "FROM media_embedding"
    )
    if row is None:
        return {"count": 0, "bytes": 0}
    return {"count": row["n"], "bytes": row["b"]}


def purge_embeddings() -> int:
    """Delete every stored embedding vector; return how many rows."""
    with closing(db.connect()) as conn:
        with conn:
            return conn.execute("DELETE FROM media_embedding").rowcount


# The per-media index columns the Libraries "Index" action fills; resetting
# them returns every media to the "never indexed" state.
_INDEX_COLUMNS = (
    "width",
    "height",
    "phash",
    "dhash",
    "sharpness",
    "clipping",
    "cleanliness",
    "indexed_at",
)


def media_index_report() -> dict:
    """Return ``{count, bytes}`` for the indexed media (dims/hashes/stats).

    ``bytes`` is a rough per-row estimate (two hex hashes + numbers).
    """
    row = _query_one(
        "SELECT COUNT(*) AS n FROM media WHERE indexed_at IS NOT NULL"
    )
    count = row["n"] if row else 0
    return {"count": count, "bytes": count * 100}


def purge_media_index() -> int:
    """Reset every media's index columns; return how many rows changed."""
    assignments = ", ".join(f"{col} = NULL" for col in _INDEX_COLUMNS)
    with closing(db.connect()) as conn:
        with conn:
            return conn.execute(
                f"UPDATE media SET {assignments} "
                "WHERE indexed_at IS NOT NULL"
            ).rowcount
