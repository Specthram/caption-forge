"""Shared low-level helpers for the sqlite_store package.

The per-call connection helpers every domain module builds on
(each opens its own short-lived connection — the safe pattern
under Gradio's worker threads) and the ``SORT_*`` keys accepted
by the library media listings.
"""

from contextlib import closing

from src import db
from src import quality

# Sort keys accepted by the library media listings (library_media_page and
# friends — the Medias tab and the Datasets "add media" picker). "quality"
# ranks by a media's score for the *selected* metric (or the normalized
# average across metrics — see :func:`_sort_order_by`); "dimension" reads the
# width/height the Libraries tab's "Index" action fills in. A media with no
# score for the ranked metric sorts to the bottom of both a best-first and a
# worst-first ranking (the ``score IS NULL`` guard below pushes it last).
SORT_DATE_DESC = "date_desc"
SORT_DATE_ASC = "date_asc"
SORT_QUALITY_DESC = "quality_desc"
SORT_QUALITY_ASC = "quality_asc"
SORT_DIMENSION_DESC = "dimension_desc"
SORT_DIMENSION_ASC = "dimension_asc"

# The non-quality sorts need no join and take no parameters.
_STATIC_ORDER_BY = {
    SORT_DATE_DESC: "m.created_at DESC, m.id DESC",
    SORT_DATE_ASC: "m.created_at ASC, m.id ASC",
    SORT_DIMENSION_DESC: "(m.width * m.height) DESC, m.id DESC",
    SORT_DIMENSION_ASC: "(m.width * m.height) ASC, m.id DESC",
}

# Directions of the two quality sorts.
_QUALITY_DIRECTION = {SORT_QUALITY_DESC: "DESC", SORT_QUALITY_ASC: "ASC"}


def _average_score_sql() -> str:
    """Return the correlated subquery for a media's normalized average score.

    Averages every ``media_quality`` row of ``m`` after normalizing each to
    0-100 with its metric's native bounds (``(score - min) / (max - min) *
    100``). The bounds come from :func:`src.quality.normalization_bounds`;
    the metric ids are :data:`src.quality.QUALITY_METRICS` keys (code
    constants, safe to inline), so the whole expression carries no bind
    parameters. A media with no scored metric yields ``NULL`` (AVG over no
    rows), matching how :func:`src.quality` normalizes "never scored".
    """
    cases = " ".join(
        f"WHEN '{metric_id}' THEN "
        f"(mq.score - {low}) / ({high - low}) * 100.0"
        for metric_id, (low, high) in quality.normalization_bounds().items()
    )
    return (
        "(SELECT AVG(CASE mq.metric_id "
        f"{cases} ELSE mq.score END) "
        "FROM media_quality mq WHERE mq.media_id = m.id)"
    )


def _sort_order_by(sort: str, metric: str = None):
    """Return ``(join_sql, order_by_sql, params)`` for a sort key.

    ``sort`` is one of the ``SORT_*`` constants — a fixed whitelist, never
    user input — so the fragments are safe to interpolate. Defaults to newest
    first for an unknown key.

    For the quality sorts, a specific ``metric`` (a
    :data:`src.quality.QUALITY_METRICS` key) is ranked by joining that
    metric's ``media_quality`` row (``join_sql`` carries one ``?`` bound to
    ``metric``, hence ``params``); ``metric`` being ``None`` or the
    :data:`src.quality.AVERAGE_METRIC_ID` pseudo-metric ranks by the
    normalized average instead (a correlated subquery, no join, no params).
    Either way a media with no score for the ranked metric sorts last (the
    ``IS NULL`` guard) in both directions.

    Parameters
    ----------
    sort : str
        One of the ``SORT_*`` constants.
    metric : str, optional
        The quality metric to rank by when ``sort`` is a quality sort.

    Returns
    -------
    tuple of (str, str, list)
        The FROM-clause join (possibly empty), the ORDER BY body and the
        join's bind parameters.
    """
    if sort in _QUALITY_DIRECTION:
        direction = _QUALITY_DIRECTION[sort]
        if metric and metric != quality.AVERAGE_METRIC_ID:
            join = (
                " LEFT JOIN media_quality mq "
                "ON mq.media_id = m.id AND mq.metric_id = ?"
            )
            order = f"(mq.score IS NULL), mq.score {direction}, m.id DESC"
            return join, order, [metric]
        expr = _average_score_sql()
        order = f"({expr} IS NULL), {expr} {direction}, m.id DESC"
        return "", order, []
    order = _STATIC_ORDER_BY.get(sort, _STATIC_ORDER_BY[SORT_DATE_DESC])
    return "", order, []


# SQLite caps the number of bound variables in one statement
# (SQLITE_MAX_VARIABLE_NUMBER — 999 on older builds, 32766 on recent ones). A
# whole-library read (a forced index, the lookalike set, a big deploy) can hand
# a bulk reader tens of thousands of ids, so every ``IN (?, ?, …)`` over an
# unbounded id list is split into chunks under the smallest limit.
_SQL_VARS_LIMIT = 900


def chunked(items, size: int = _SQL_VARS_LIMIT):
    """Yield successive ``size``-long slices of a list (for IN () lookups)."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _query_all(query: str, params=()) -> list:
    """Run a SELECT and return all rows."""
    with closing(db.connect()) as conn:
        return conn.execute(query, params).fetchall()


def _query_one(query: str, params=()):
    """Run a SELECT and return the first row, or None."""
    with closing(db.connect()) as conn:
        return conn.execute(query, params).fetchone()


def _write(query: str, params=()) -> int:
    """Run a single write in its own transaction; return ``lastrowid``."""
    with closing(db.connect()) as conn:
        with conn:
            return conn.execute(query, params).lastrowid
