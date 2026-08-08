"""In-process job queue.

Prediction is job-based rather than synchronous because a single chain costs seconds to
tens of seconds (ESM-2 on CPU dominates), which is long enough that a synchronous endpoint
would be at the mercy of every reverse proxy's read timeout and would leave the user
staring at nothing. Jobs let the UI stream stage-by-stage progress instead.

Deliberately in-process: one worker holds one 2.6 GB ESM model, so the scaling knob is the
queue, not the process count. A Redis/Celery setup would add operational weight for no gain
at this size.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"          # queued | running | done | error
    stage: str = ""
    message: str = ""
    progress: float = 0.0           # 0..1, best effort
    result: Any = None
    error: Optional[dict] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    events: list[dict] = field(default_factory=list)

    def public(self, include_result: bool = True) -> dict:
        d = {
            "job_id": self.id, "kind": self.kind, "status": self.status,
            "stage": self.stage, "message": self.message, "progress": self.progress,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at, "error": self.error,
        }
        if include_result and self.status == "done":
            d["result"] = self.result
        return d


class JobQueue:
    def __init__(self, workers: int = 2, keep: int = 200):
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="jbbind-job")
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._keep = keep
        #: Serializes the memory-hungry section (ESM forward). Everything else can overlap.
        self.heavy = threading.Semaphore(1)

    def submit(self, kind: str, fn: Callable[[Callable[[str, str], None]], Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:16], kind=kind)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._trim()
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn) -> None:
        job.status = "running"
        job.started_at = time.time()

        def progress(stage: str, message: str) -> None:
            job.stage = stage
            job.message = message
            job.events.append({"t": time.time(), "stage": stage, "message": message})

        try:
            job.result = fn(progress)
            job.status = "done"
            job.progress = 1.0
        except Exception as exc:
            job.status = "error"
            code = getattr(exc, "code", exc.__class__.__name__)
            job.error = {
                "code": code,
                "message": getattr(exc, "message", str(exc)),
                "detail": getattr(exc, "stderr", "") or None,
            }
            job.events.append({"t": time.time(), "stage": "error", "message": str(exc)})
            if code in ("RuntimeError", "Exception"):  # unexpected: keep the traceback
                job.error["traceback"] = traceback.format_exc()[-2000:]
        finally:
            job.finished_at = time.time()

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def _trim(self) -> None:
        while len(self._order) > self._keep:
            old = self._order.pop(0)
            self._jobs.pop(old, None)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
