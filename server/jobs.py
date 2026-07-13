"""Serialised background job queue with live progress broadcasting.

One worker task drains an :class:`asyncio.Queue`, so jobs run strictly one at a
time — the VRAM-safety invariant (only one heavy model loaded at once). Job
bodies are blocking callables run in a thread executor; progress updates are
pushed thread-safely onto every subscribed queue (WebSocket clients) and cached
in the registry.

A body is ``run(progress) -> result``. ``progress`` is a :class:`Progress`
callable ``progress(done=..., total=..., sub=...)`` that raises
:class:`JobStopped` when a stop was requested, so a cooperative loop stops at
its next update. ``progress.warn(message)`` records a non-fatal per-item
problem so the drawer can name it without aborting the run.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# A run over a broken library must not stream thousands of warnings to every
# WebSocket client: past this many the job keeps counting them, silently.
MAX_WARNINGS = 50

# How many recent progress samples the ETA rate is measured over. A short
# window makes the estimate follow the *current* step: a chain that moves
# from a fast pass (dimensions) to a slow one (quality) re-times itself
# instead of averaging both speeds for the whole run.
_ETA_WINDOW = 24


class JobStopped(Exception):
    """Raised inside a job body when a stop has been requested."""


@dataclass
class Job:  # pylint: disable=too-many-instance-attributes
    """One unit of background work and its live progress state."""

    id: str
    type: str
    name: str
    sub: str = ""
    state: str = "queued"  # queued | running | done | error | stopped
    done: int = 0
    total: int = 0
    error: str = ""
    # Non-fatal, per-item problems (an unreadable image, a file gone from
    # disk). Capped at MAX_WARNINGS; ``warning_count`` keeps the true total.
    warnings: list = field(default_factory=list)
    warning_count: int = 0
    result: Any = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # Wall-clock start of the *running* phase (None while still queued). ETA
    # measures from work beginning, not from the queue wait.
    started_at: float | None = None
    stop_requested: bool = False
    # Recent (timestamp, done) samples, newest last — the ETA rate window.
    # Not serialised; capped so a long run keeps only the tail.
    samples: deque = field(
        default_factory=lambda: deque(maxlen=_ETA_WINDOW), repr=False
    )

    @property
    def pct(self) -> int:
        """Return completion as an integer percentage (0 when unknown)."""
        if self.total <= 0:
            return 0
        return min(100, round(self.done * 100 / self.total))

    @property
    def eta_seconds(self) -> float | None:
        """Return estimated seconds remaining, or ``None`` when unknown.

        The rate is the throughput over the recent sample window (see
        :data:`_ETA_WINDOW`), so the estimate adapts to the step running now
        rather than the whole-run average. ``None`` until there is a
        countable total, real progress and two spaced samples to time.
        """
        if self.state != "running" or self.total <= 0:
            return None
        if self.done <= 0 or self.done >= self.total:
            return None
        if len(self.samples) < 2:
            return None
        old_time, old_done = self.samples[0]
        new_time, new_done = self.samples[-1]
        span = new_time - old_time
        advanced = new_done - old_done
        if span <= 0 or advanced <= 0:
            return None
        rate = advanced / span
        return (self.total - self.done) / rate

    def snapshot(self) -> dict:
        """Return a JSON-serialisable view of the job for the API/WS."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "sub": self.sub,
            "state": self.state,
            "done": self.done,
            "total": self.total,
            "pct": self.pct,
            "error": self.error,
            "warnings": list(self.warnings),
            "warning_count": self.warning_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "eta_seconds": self.eta_seconds,
        }


class Progress:  # pylint: disable=too-few-public-methods
    """Progress reporter handed to a running job body.

    Calling it updates the job counters/subtitle and broadcasts the new
    snapshot; it raises :class:`JobStopped` if a stop was requested, which
    a cooperative loop lets propagate to end the job cleanly.
    """

    def __init__(self, job_manager: "JobManager", job: Job) -> None:
        """Bind the reporter to its manager and job."""
        self._manager = job_manager
        self._job = job

    def __call__(
        self,
        done: int | None = None,
        total: int | None = None,
        sub: str | None = None,
    ) -> None:
        """Update counters/subtitle then broadcast; stop if requested."""
        job = self._job
        if done is not None:
            job.done = done
        if total is not None:
            job.total = total
        if sub is not None:
            job.sub = sub
        job.updated_at = time.time()
        if done is not None:
            job.samples.append((job.updated_at, job.done))
        self._manager.broadcast(job)
        if job.stop_requested:
            raise JobStopped()

    def warn(self, message: str) -> None:
        """Record a non-fatal problem and broadcast it.

        The run goes on: one bad image must not cost the 999 others.
        ``message`` is one line, media first — ``"foo.png: cannot identify"``.
        """
        job = self._job
        job.warning_count += 1
        if len(job.warnings) < MAX_WARNINGS:
            job.warnings.append(message)
        job.updated_at = time.time()
        self._manager.broadcast(job)


JobBody = Callable[[Progress], Any]


class JobManager:
    """Owns the queue, the worker task, the registry and subscribers."""

    def __init__(self) -> None:
        """Initialise empty registries, the queue and subscriber set."""
        self._registry: dict[str, Job] = {}
        self._bodies: dict[str, JobBody] = {}
        self._subscribers: set[asyncio.Queue] = set()
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker: asyncio.Task | None = None

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        """Capture the running loop and launch the single worker task.

        The manager is a singleton, so a second lifespan in the same process
        (a test's next ``TestClient``, an app restart) hands it a *different*
        loop. An :class:`asyncio.Queue` caches its first loop and refuses
        another, which would silently kill the new worker on its first
        ``get()``. Rebuilding the queue on a loop change (carrying pending job
        ids over) prevents that.
        """
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            pending = []
            while not self._queue.empty():
                pending.append(self._queue.get_nowait())
            self._queue = asyncio.Queue()
            for job_id in pending:
                self._queue.put_nowait(job_id)
            self._loop = loop
        if self._worker is None:
            self._worker = asyncio.create_task(self._run_worker())

    async def stop(self) -> None:
        """Cancel the worker task (on application shutdown)."""
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None

    # -- submission / control -------------------------------------------

    def submit(
        self, job_type: str, name: str, body: JobBody, sub: str = ""
    ) -> Job:
        """Register a job, enqueue it and return it immediately."""
        job = Job(id=uuid.uuid4().hex, type=job_type, name=name, sub=sub)
        self._registry[job.id] = job
        self._bodies[job.id] = body
        self._queue.put_nowait(job.id)
        self.broadcast(job)
        return job

    def request_stop(self, job_id: str) -> bool:
        """Flag a job to stop at its next progress checkpoint."""
        job = self._registry.get(job_id)
        if job is None or job.state in {"done", "error", "stopped"}:
            return False
        job.stop_requested = True
        return True

    def get(self, job_id: str) -> Job | None:
        """Return a job by id, or ``None``."""
        return self._registry.get(job_id)

    def list_jobs(self) -> list[dict]:
        """Return every job snapshot, newest first."""
        jobs = sorted(
            self._registry.values(),
            key=lambda j: j.created_at,
            reverse=True,
        )
        return [j.snapshot() for j in jobs]

    def clear_finished(self) -> None:
        """Drop finished jobs from the registry (jobs-drawer action)."""
        for job_id, job in list(self._registry.items()):
            if job.state in {"done", "error", "stopped"}:
                del self._registry[job_id]

    # -- subscription (WebSocket) ---------------------------------------

    def subscribe(self) -> asyncio.Queue:
        """Register a subscriber queue and return it."""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        self._subscribers.discard(queue)

    def broadcast(self, job: Job) -> None:
        """Push a job snapshot to every subscriber (thread safe).

        Safe to call from the worker thread: it hops back onto the event
        loop with :meth:`~asyncio.AbstractEventLoop.call_soon_threadsafe`.
        """
        snapshot = job.snapshot()
        loop = self._loop
        if loop is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._fan_out(snapshot)
        else:
            loop.call_soon_threadsafe(self._fan_out, snapshot)

    def _fan_out(self, snapshot: dict) -> None:
        """Deliver one snapshot to all subscriber queues (on the loop)."""
        for queue in list(self._subscribers):
            queue.put_nowait(snapshot)

    # -- worker ----------------------------------------------------------

    async def _run_worker(self) -> None:
        """Drain the queue, running one job at a time to completion."""
        while True:
            job_id = await self._queue.get()
            job = self._registry.get(job_id)
            body = self._bodies.pop(job_id, None)
            if job is None or body is None:
                self._queue.task_done()
                continue
            await self._execute(job, body)
            self._queue.task_done()

    async def _execute(self, job: Job, body: JobBody) -> None:
        """Run one job body in a thread, recording its terminal state."""
        job.state = "running"
        job.started_at = time.time()
        job.updated_at = job.started_at
        job.samples.clear()
        self.broadcast(job)
        progress = Progress(self, job)
        loop = asyncio.get_running_loop()
        try:
            job.result = await loop.run_in_executor(None, body, progress)
            job.state = "done"
        except JobStopped:
            job.state = "stopped"
        except Exception as exc:  # pylint: disable=broad-except
            job.state = "error"
            # ``str(exc)`` is empty for a bare raise and some torch/PIL errors,
            # leaving the drawer a red dot and nothing else. Always name the
            # class; full traceback goes to the log.
            job.error = f"{type(exc).__name__}: {exc}".rstrip(": ")
            logger.error(
                "job %s (%s) failed\n%s",
                job.id,
                job.type,
                "".join(traceback.format_exception(exc)),
            )
        job.updated_at = time.time()
        self.broadcast(job)


# Process-wide singleton (one worker, one GPU).
manager = JobManager()
