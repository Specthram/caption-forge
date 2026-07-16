"""Rule-based caption review repository: rules, runs and findings.

Backs the Caption tab's **Review** sub-tab. Three tables (see :mod:`src.db`):

* ``review_rule`` — the plain-language checks, scoped per dataset. ``kind``
  is ``'det'`` (deterministic), ``'text'`` (text-only LLM) or ``'vlm'``
  (vision judge); ``builtin`` marks a shipped preset.
* ``review_run`` — one pass of the judge over a dataset (or a single media).
* ``review_finding`` — one proposed correction awaiting the human decision.
  ``rule_id`` NULL (with ``rule_kind = 'integrity'``) marks an integrity
  finding from the heuristics of :mod:`src.caption_review`.

Nothing here applies a correction: a finding only records what the judge
proposed. Accepting one is a caption edit, done by :mod:`src.storage`. The
merge rule lives here: a whole-dataset run replaces the queue
(:func:`clear_dataset_findings`), a single-media run replaces only that
media's rows (:func:`clear_media_findings`).
"""

from contextlib import closing

from src import db
from src.sqlite_store.base import _query_all, _query_one, _write

# The rule kinds a judge run dispatches on. ``det`` needs no model; ``text``
# runs a text-only LLM pass; ``vlm`` loads the image into the vision judge.
KIND_DET = "det"
KIND_TEXT = "text"
KIND_VLM = "vlm"

# ``rule_kind`` stamped on an integrity finding (its ``rule_id`` is NULL).
KIND_INTEGRITY = "integrity"

# The finding kinds a human never has to judge: a deterministic rule or an
# integrity heuristic carries no model opinion, so "Accept all safe fixes"
# may apply them in bulk.
SAFE_KINDS = (KIND_DET, KIND_INTEGRITY)

# Human decisions recorded on a finding.
STATUS_PENDING = "pending"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"


# --- rules ---------------------------------------------------------------


def _decode_rule(row):
    """Return a rule row as a dict with its flags cast to bool."""
    if row is None:
        return None
    data = dict(row)
    for flag in ("needs_image", "enabled", "builtin"):
        data[flag] = bool(data[flag])
    return data


def list_rules(dataset_id: int) -> list:
    """Return every review rule of a dataset, oldest first (builtins lead)."""
    rows = _query_all(
        "SELECT * FROM review_rule WHERE dataset_id = ? "
        "ORDER BY builtin DESC, id ASC",
        (dataset_id,),
    )
    return [_decode_rule(row) for row in rows]


def get_rule(rule_id: int):
    """Return one rule dict, or None when it does not exist."""
    return _decode_rule(
        _query_one("SELECT * FROM review_rule WHERE id = ?", (rule_id,))
    )


def create_rule(
    dataset_id: int,
    text: str,
    kind: str,
    needs_image: bool = False,
    enabled: bool = True,
    builtin: bool = False,
) -> int:
    """Insert a review rule and return its id."""
    return _write(
        "INSERT INTO review_rule "
        "(dataset_id, text, kind, needs_image, enabled, builtin) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            dataset_id,
            text,
            kind,
            int(needs_image),
            int(enabled),
            int(builtin),
        ),
    )


def set_rule_enabled(rule_id: int, enabled: bool) -> None:
    """Toggle a rule on or off (the rail's checkbox)."""
    _write(
        "UPDATE review_rule SET enabled = ? WHERE id = ?",
        (int(enabled), rule_id),
    )


def update_rule_text(rule_id: int, text: str) -> None:
    """Rewrite a custom rule's text."""
    _write("UPDATE review_rule SET text = ? WHERE id = ?", (text, rule_id))


def delete_rule(rule_id: int) -> None:
    """Delete a rule (no-op when absent). The router guards builtins."""
    _write("DELETE FROM review_rule WHERE id = ?", (rule_id,))


def enabled_rules(dataset_id: int) -> list:
    """Return the enabled rules a run should apply, oldest first."""
    return [rule for rule in list_rules(dataset_id) if rule["enabled"]]


def has_rules(dataset_id: int) -> bool:
    """Return whether a dataset already carries any rule (builtin or not)."""
    return (
        _query_one(
            "SELECT 1 FROM review_rule WHERE dataset_id = ? LIMIT 1",
            (dataset_id,),
        )
        is not None
    )


# --- runs ----------------------------------------------------------------


def create_run(
    dataset_id: int, judge_model: str, scope: str, total: int
) -> int:
    """Open a review run and return its id (``finished_at`` still NULL)."""
    return _write(
        "INSERT INTO review_run "
        "(dataset_id, judge_model, scope, total) VALUES (?, ?, ?, ?)",
        (dataset_id, judge_model or "", scope, total),
    )


def finish_run(run_id: int, findings_count: int) -> None:
    """Stamp a run finished and record how many findings it produced."""
    _write(
        "UPDATE review_run SET findings_count = ?, "
        "finished_at = datetime('now') WHERE id = ?",
        (findings_count, run_id),
    )


def get_run(run_id: int):
    """Return one run dict, or None."""
    row = _query_one("SELECT * FROM review_run WHERE id = ?", (run_id,))
    return dict(row) if row is not None else None


# --- findings ------------------------------------------------------------

# Every finding read joins its run (for the dataset) and, when present, its
# rule (for the rule text the queue shows). ``rule_text`` is NULL for an
# integrity finding.
_FINDING_SELECT = (
    "SELECT f.*, run.dataset_id AS dataset_id, rule.text AS rule_text "
    "FROM review_finding f "
    "JOIN review_run run ON run.id = f.run_id "
    "LEFT JOIN review_rule rule ON rule.id = f.rule_id"
)


def add_finding(
    run_id: int,
    media_id: int,
    caption_type_id: int,
    note: str,
    caption_before: str,
    caption_after: str,
    rule_id: int = None,
    rule_kind: str = "",
) -> int:
    """Insert a pending finding and return its id."""
    return _write(
        "INSERT INTO review_finding "
        "(run_id, media_id, caption_type_id, rule_id, rule_kind, note, "
        " caption_before, caption_after) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            media_id,
            caption_type_id,
            rule_id,
            rule_kind,
            note,
            caption_before,
            caption_after,
        ),
    )


def get_finding(finding_id: int):
    """Return one enriched finding dict (with ``rule_text``), or None."""
    row = _query_one(f"{_FINDING_SELECT} WHERE f.id = ?", (finding_id,))
    return dict(row) if row is not None else None


def list_findings(dataset_id: int, status: str = None) -> list:
    """Return a dataset's findings, newest first, optionally by status."""
    params = [dataset_id]
    where = "WHERE run.dataset_id = ?"
    if status:
        where += " AND f.status = ?"
        params.append(status)
    rows = _query_all(f"{_FINDING_SELECT} {where} ORDER BY f.id DESC", params)
    return [dict(row) for row in rows]


def findings_counts(dataset_id: int) -> dict:
    """Return ``{pending, accepted, rejected}`` counts for a dataset."""
    rows = _query_all(
        "SELECT f.status AS status, COUNT(*) AS n FROM review_finding f "
        "JOIN review_run run ON run.id = f.run_id "
        "WHERE run.dataset_id = ? GROUP BY f.status",
        (dataset_id,),
    )
    counts = {
        STATUS_PENDING: 0,
        STATUS_ACCEPTED: 0,
        STATUS_REJECTED: 0,
    }
    for row in rows:
        counts[row["status"]] = row["n"]
    return counts


def pending_count(dataset_id: int) -> int:
    """Return the dataset's pending-finding count (the tab badge)."""
    row = _query_one(
        "SELECT COUNT(*) AS n FROM review_finding f "
        "JOIN review_run run ON run.id = f.run_id "
        "WHERE run.dataset_id = ? AND f.status = ?",
        (dataset_id, STATUS_PENDING),
    )
    return row["n"] if row is not None else 0


def safe_pending(dataset_id: int) -> list:
    """Return the pending *safe* findings (det + integrity) of a dataset.

    These carry no model judgement, so "Accept all safe fixes" may apply
    them without a per-finding human decision.
    """
    marks = ",".join("?" * len(SAFE_KINDS))
    rows = _query_all(
        f"{_FINDING_SELECT} WHERE run.dataset_id = ? AND f.status = ? "
        f"AND f.rule_kind IN ({marks}) ORDER BY f.id DESC",
        (dataset_id, STATUS_PENDING, *SAFE_KINDS),
    )
    return [dict(row) for row in rows]


def pending_for_rule(dataset_id: int, rule_id: int) -> list:
    """Return the pending findings of one rule (the wizard's bulk accept)."""
    rows = _query_all(
        f"{_FINDING_SELECT} WHERE run.dataset_id = ? AND f.rule_id = ? "
        "AND f.status = ? ORDER BY f.id DESC",
        (dataset_id, rule_id, STATUS_PENDING),
    )
    return [dict(row) for row in rows]


def decide_finding(
    finding_id: int, status: str, applied_caption: str = None
) -> None:
    """Record a human decision on a finding (accept / reject).

    ``applied_caption`` is the final text when the user edited the proposal
    before accepting; it is stored so an undo can tell what was applied.
    """
    _write(
        "UPDATE review_finding SET status = ?, applied_caption = ?, "
        "decided_at = datetime('now') WHERE id = ?",
        (status, applied_caption, finding_id),
    )


def reopen_finding(finding_id: int) -> None:
    """Return a decided finding to ``pending`` (the undo path)."""
    _write(
        "UPDATE review_finding SET status = ?, applied_caption = NULL, "
        "decided_at = NULL WHERE id = ?",
        (STATUS_PENDING, finding_id),
    )


def clear_dataset_findings(dataset_id: int) -> None:
    """Delete every finding of a dataset (a fresh whole-dataset run)."""
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "DELETE FROM review_finding WHERE run_id IN "
                "(SELECT id FROM review_run WHERE dataset_id = ?)",
                (dataset_id,),
            )


def clear_media_findings(dataset_id: int, media_id: int) -> None:
    """Delete one media's findings in a dataset (a single-media re-run)."""
    with closing(db.connect()) as conn:
        with conn:
            conn.execute(
                "DELETE FROM review_finding WHERE media_id = ? AND run_id IN "
                "(SELECT id FROM review_run WHERE dataset_id = ?)",
                (media_id, dataset_id),
            )
