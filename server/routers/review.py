"""Caption-review routes: the integrity heuristics.

Judging a caption against its *image* moved to SigLIP grounding (see
:mod:`server.routers.grounding`), which replaced the VLM veracity judge and
its accept/reject proposal flow.
"""

from fastapi import APIRouter

from server.schemas import ReviewTargetBody
from src import storage

router = APIRouter(prefix="/api/review", tags=["review"])


@router.post("/integrity")
def integrity(body: ReviewTargetBody) -> dict:
    """Run the fast integrity heuristics and store the verdict."""
    status, issues = storage.run_integrity_review(
        body.dataset_id, body.key, body.caption_type
    )
    return {"status": status, "issues": issues}
