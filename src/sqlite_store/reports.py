"""Dataset quality-report repository: the last report and its resolutions.

Two tables (see :mod:`src.db`):

* ``dataset_report`` — one JSON blob per dataset, the last evaluation, so
  the Quality report tab renders instantly instead of re-running the
  scorers on open.
* ``dataset_issue_resolution`` — how the user handled each finding
  (``removed`` | ``ignored`` | ``recaptioned``), keyed on the finding's
  stable ``issue_key`` and stamped with the ``fingerprint`` of the
  measurement behind it.

The fingerprint is what makes a re-run honest: a resolution survives while
the finding is *the same* finding, and is dropped the moment the
measurement changes (a recaptioned image, a similarity that moved).
"""

import json

from src.sqlite_store.base import _query_all, _query_one, _write

# The resolutions the report accepts, mirroring the tab's three actions.
RESOLUTIONS = ("removed", "ignored", "recaptioned")


def save_dataset_report(
    dataset_id: int,
    payload: dict,
    scorers,
    caption_type: str,
    duration_s: float,
) -> None:
    """Store (or replace) a dataset's last quality report.

    Parameters
    ----------
    dataset_id : int
        The evaluated dataset.
    payload : dict
        The serialised report (see :func:`src.dataset_quality.to_dict`).
    scorers : iterable of str
        The scorer ids the run enabled.
    caption_type : str
        The caption type the hygiene pillar and caption issues read.
    duration_s : float
        How long the run took, in seconds.
    """
    _write(
        "INSERT INTO dataset_report "
        "(dataset_id, payload, scorers, caption_type, duration_s) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT (dataset_id) DO UPDATE SET "
        "payload = excluded.payload, scorers = excluded.scorers, "
        "caption_type = excluded.caption_type, "
        "duration_s = excluded.duration_s, created_at = datetime('now')",
        (
            dataset_id,
            json.dumps(payload),
            ",".join(scorers),
            caption_type,
            float(duration_s),
        ),
    )


def get_dataset_report(dataset_id: int) -> dict | None:
    """Return a dataset's last report, or None when it was never run.

    Returns
    -------
    dict or None
        ``{"report", "scorers", "caption_type", "duration_s",
        "created_at"}``; ``report`` is the decoded blob.
    """
    row = _query_one(
        "SELECT payload, scorers, caption_type, duration_s, created_at "
        "FROM dataset_report WHERE dataset_id = ?",
        (dataset_id,),
    )
    if row is None:
        return None
    return {
        "report": json.loads(row["payload"]),
        "scorers": [s for s in row["scorers"].split(",") if s],
        "caption_type": row["caption_type"],
        "duration_s": row["duration_s"],
        "created_at": row["created_at"],
    }


def delete_dataset_report(dataset_id: int) -> None:
    """Drop a dataset's stored report (and nothing else)."""
    _write("DELETE FROM dataset_report WHERE dataset_id = ?", (dataset_id,))


def save_autobuild_recipe(
    dataset_id: int, recipe_json: str, live: bool
) -> None:
    """Store (or replace) the Studio recipe a dataset was built from.

    The recipe is what makes a dataset "living": while ``live`` is set, the
    upgrade runner replays it after each index and offers to swap a weaker
    pick for a stronger new candidate (see the ``autobuild`` runner).
    """
    _write(
        "INSERT INTO autobuild_recipe (dataset_id, payload, live) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT (dataset_id) DO UPDATE SET "
        "payload = excluded.payload, live = excluded.live, "
        "created_at = datetime('now')",
        (dataset_id, recipe_json, 1 if live else 0),
    )


def get_autobuild_recipe(dataset_id: int) -> dict | None:
    """Return a dataset's Studio recipe, or None when it has none.

    Returns
    -------
    dict or None
        ``{"recipe", "live", "created_at"}``; ``recipe`` is the decoded
        blob.
    """
    row = _query_one(
        "SELECT payload, live, created_at FROM autobuild_recipe "
        "WHERE dataset_id = ?",
        (dataset_id,),
    )
    if row is None:
        return None
    return {
        "recipe": json.loads(row["payload"]),
        "live": bool(row["live"]),
        "created_at": row["created_at"],
    }


def live_autobuild_recipes() -> list:
    """Return every living recipe as ``{dataset_id, recipe, created_at}``."""
    rows = _query_all(
        "SELECT dataset_id, payload, created_at FROM autobuild_recipe "
        "WHERE live = 1 ORDER BY dataset_id",
        (),
    )
    return [
        {
            "dataset_id": row["dataset_id"],
            "recipe": json.loads(row["payload"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def set_issue_resolution(
    dataset_id: int,
    issue_key: str,
    resolution: str,
    fingerprint: str = "",
    note: str = "",
) -> None:
    """Record how the user handled one finding.

    Raises
    ------
    ValueError
        When ``resolution`` is not one of :data:`RESOLUTIONS`.
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(f"unknown resolution: {resolution}")
    _write(
        "INSERT INTO dataset_issue_resolution "
        "(dataset_id, issue_key, resolution, fingerprint, note) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT (dataset_id, issue_key) DO UPDATE SET "
        "resolution = excluded.resolution, "
        "fingerprint = excluded.fingerprint, note = excluded.note, "
        "resolved_at = datetime('now')",
        (dataset_id, issue_key, resolution, fingerprint, note),
    )


def clear_issue_resolution(dataset_id: int, issue_key: str) -> None:
    """Forget how a finding was handled (the row becomes open again)."""
    _write(
        "DELETE FROM dataset_issue_resolution "
        "WHERE dataset_id = ? AND issue_key = ?",
        (dataset_id, issue_key),
    )


def dataset_resolutions(dataset_id: int) -> dict:
    """Return ``{issue_key: {resolution, fingerprint, note, resolved_at}}``."""
    rows = _query_all(
        "SELECT issue_key, resolution, fingerprint, note, resolved_at "
        "FROM dataset_issue_resolution WHERE dataset_id = ?",
        (dataset_id,),
    )
    return {
        row["issue_key"]: {
            "resolution": row["resolution"],
            "fingerprint": row["fingerprint"],
            "note": row["note"],
            "resolved_at": row["resolved_at"],
        }
        for row in rows
    }


def reconcile_resolutions(dataset_id: int, issues) -> int:
    """Drop the resolutions a fresh report has invalidated.

    A resolution is kept when the report still carries its finding *with
    the same fingerprint*. It is dropped when:

    * the finding is gone (the media was removed, the duplicate broken) —
      there is nothing left to mark as handled;
    * the finding is back but its measurement changed (a regenerated
      caption that is still degenerate, a similarity that drifted) — the
      user's earlier "ignore" was about a different finding.

    Parameters
    ----------
    dataset_id : int
        The evaluated dataset.
    issues : iterable
        The fresh report's issues (``key`` / ``fingerprint`` attributes).

    Returns
    -------
    int
        How many resolutions were dropped.
    """
    fresh = {issue.key: issue.fingerprint for issue in issues}
    stale = [
        key
        for key, stored in dataset_resolutions(dataset_id).items()
        if key not in fresh or fresh[key] != stored["fingerprint"]
    ]
    for key in stale:
        clear_issue_resolution(dataset_id, key)
    return len(stale)
