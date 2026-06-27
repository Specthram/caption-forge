"""Job-queue routes: list, read, stop and clear background jobs."""

from fastapi import APIRouter, HTTPException

from server.jobs import manager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def list_jobs() -> dict:
    """Return every job snapshot, newest first."""
    return {"jobs": manager.list_jobs()}


@router.get("/{job_id}/result")
def job_result(job_id: str) -> dict:
    """Return a finished job's payload.

    The escape hatch for a job whose *output* is the point rather than its
    side effects — the grounding heat maps, computed on the GPU worker and
    far too bulky to push through the progress WebSocket. A job still
    running answers with a null result, so the caller waits for the
    ``done`` event before reading.
    """
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.state == "error":
        raise HTTPException(status_code=409, detail=job.error or "job failed")
    return {"state": job.state, "result": job.result}


@router.post("/{job_id}/stop")
def stop_job(job_id: str) -> dict:
    """Request a running job to stop at its next checkpoint."""
    if not manager.request_stop(job_id):
        raise HTTPException(status_code=404, detail="job not stoppable")
    return {"ok": True}


@router.post("/clear")
def clear_finished() -> dict:
    """Drop finished jobs from the registry."""
    manager.clear_finished()
    return {"ok": True}
