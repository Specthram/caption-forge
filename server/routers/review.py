"""Caption-review routes: integrity heuristics + the rule-based judge queue.

Two mechanisms share this router:

* ``/integrity`` — the fast, model-free heuristics (see
  :mod:`src.caption_review`), attached to a caption revision.
* the **Review sub-tab** — plain-language *rules* checked by an independent
  *judge* model, producing *findings* the human accepts, rejects or edits.
  Nothing is ever applied silently; only "Accept all safe fixes" (the
  deterministic and integrity findings) skips a per-finding decision.

Judging a caption against its *image* for veracity still lives in SigLIP
grounding (see :mod:`server.routers.grounding`); the vision rules here are a
different, rule-scoped check.
"""

from fastapi import APIRouter, HTTPException

from server.jobs import manager
from server.runners.review_run import review_run_body
from server.schemas import (
    ReviewBulkDecideBody,
    ReviewDecideBody,
    ReviewRuleCreateBody,
    ReviewRuleUpdateBody,
    ReviewRunBody,
    ReviewTargetBody,
)
from src import sqlite_store as store
from src import storage

router = APIRouter(prefix="/api/review", tags=["review"])


@router.post("/integrity")
def integrity(body: ReviewTargetBody) -> dict:
    """Run the fast integrity heuristics and store the verdict."""
    status, issues = storage.run_integrity_review(
        body.dataset_id, body.key, body.caption_type
    )
    return {"status": status, "issues": issues}


# --- rules ---------------------------------------------------------------


@router.get("/rules")
def list_rules(dataset_id: int) -> dict:
    """Return a dataset's review rules (seeding the builtin presets once)."""
    return {"rules": storage.review_rules(dataset_id)}


@router.post("/rules")
def create_rule(body: ReviewRuleCreateBody) -> dict:
    """Add a custom rule; its kind is vision when it needs the image."""
    rule = storage.create_review_rule(
        body.dataset_id, body.text, body.needs_image
    )
    return {"rule": rule}


@router.patch("/rules/{rule_id}")
def update_rule(rule_id: int, body: ReviewRuleUpdateBody) -> dict:
    """Toggle a rule on/off or rewrite its text."""
    rule = store.get_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="rule not found")
    if body.enabled is not None:
        store.set_rule_enabled(rule_id, body.enabled)
    if body.text is not None:
        store.update_rule_text(rule_id, body.text)
    return {"rule": store.get_rule(rule_id)}


@router.delete("/rules/{rule_id}")
def delete_rule(rule_id: int) -> dict:
    """Delete a custom rule (builtins are toggled off, never removed)."""
    rule = store.get_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="rule not found")
    if rule["builtin"]:
        raise HTTPException(
            status_code=400, detail="builtin rules cannot be deleted"
        )
    store.delete_rule(rule_id)
    return {"deleted": rule_id}


# --- run + queue ---------------------------------------------------------


@router.post("/run")
def run_review(body: ReviewRunBody) -> dict:
    """Enqueue a review run over a dataset (or one media); return job id."""
    job = manager.submit(
        "review",
        f"Review dataset {body.dataset_id}",
        review_run_body(body),
        sub="preparing…",
    )
    return {"job_id": job.id}


@router.get("/findings")
def list_findings(dataset_id: int, status: str = None) -> dict:
    """Return the review queue for a dataset, optionally filtered by status."""
    return {
        "findings": storage.review_findings(dataset_id, status),
        "counts": storage.review_counts(dataset_id),
    }


@router.get("/counts")
def counts(dataset_id: int) -> dict:
    """Return the pending / accepted / rejected totals (the tab badge)."""
    return storage.review_counts(dataset_id)


@router.post("/findings/{finding_id}/decide")
def decide(finding_id: int, body: ReviewDecideBody) -> dict:
    """Accept (write a new revision) or reject one finding."""
    if body.action not in {"accept", "reject"}:
        raise HTTPException(status_code=400, detail="invalid action")
    finding = storage.decide_review_finding(
        finding_id, body.action, body.caption
    )
    if not finding:
        raise HTTPException(status_code=404, detail="finding not found")
    return {"finding": finding}


@router.post("/findings/{finding_id}/undo")
def undo(finding_id: int) -> dict:
    """Undo a decision: restore the caption and reopen the finding."""
    finding = storage.undo_review_finding(finding_id)
    if not finding:
        raise HTTPException(status_code=404, detail="finding not found")
    return {"finding": finding}


@router.post("/findings/decide_bulk")
def decide_bulk(body: ReviewBulkDecideBody) -> dict:
    """Accept every safe finding, or every pending finding of one rule."""
    if body.rule_id is None:
        applied = storage.accept_safe_fixes(body.dataset_id)
    else:
        applied = storage.accept_rule_fixes(body.dataset_id, body.rule_id)
    return {"accepted": applied}


@router.post("/findings/reject_all")
def reject_all(body: ReviewBulkDecideBody) -> dict:
    """Reject every pending finding of a dataset (captions untouched)."""
    return {"rejected": storage.reject_all_findings(body.dataset_id)}


@router.post("/findings/clear_history")
def clear_history(body: ReviewBulkDecideBody) -> dict:
    """Delete the decided findings (history); pending ones stay."""
    return {"cleared": storage.clear_review_history(body.dataset_id)}
