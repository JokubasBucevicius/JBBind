"""Job queue behaviour."""

from __future__ import annotations

import threading
import time

import pytest

from jbbind.core.jobs import JobQueue


def wait_for(job, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job.status in ("done", "error"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job stuck in {job.status}")


@pytest.fixture
def queue():
    q = JobQueue(workers=2)
    yield q
    q.shutdown()


def test_callback_receives_its_own_job(queue):
    """Regression: the callback used to close over the caller's `job = submit(...)`
    variable, which the worker could reach before the assignment completed."""
    seen = {}

    def work(progress, job):
        seen["id"] = job.id
        return "ok"

    job = wait_for(queue.submit("t", work))
    assert job.status == "done"
    assert seen["id"] == job.id
    assert job.result == "ok"


def test_progress_events_are_recorded_in_order(queue):
    def work(progress, job):
        for stage in ("fetch", "normalize", "model"):
            progress(stage, f"doing {stage}")
        return None

    job = wait_for(queue.submit("t", work))
    assert [e["stage"] for e in job.events] == ["fetch", "normalize", "model"]
    assert job.stage == "model"


def test_failure_is_captured_not_raised(queue):
    class Boom(Exception):
        code = "VoronotaFailed"
        message = "tessellation failed"
        stderr = "some stderr"

    def work(progress, job):
        raise Boom()

    job = wait_for(queue.submit("t", work))
    assert job.status == "error"
    assert job.error["code"] == "VoronotaFailed"
    assert job.error["detail"] == "some stderr"
    assert "traceback" not in job.error       # a typed error needs no traceback


def test_unexpected_error_keeps_a_traceback(queue):
    def work(progress, job):
        raise RuntimeError("unexpected")

    job = wait_for(queue.submit("t", work))
    assert job.status == "error"
    assert job.error["code"] == "RuntimeError"
    assert job.error.get("traceback")


def test_heavy_semaphore_serializes(queue):
    """The ESM section is the memory ceiling; only one job may hold it."""
    concurrent, peak = 0, 0
    lock = threading.Lock()

    def work(progress, job):
        nonlocal concurrent, peak
        with queue.heavy:
            with lock:
                concurrent += 1
                peak = max(peak, concurrent)
            time.sleep(0.15)
            with lock:
                concurrent -= 1
        return None

    jobs = [queue.submit("t", work) for _ in range(4)]
    for j in jobs:
        wait_for(j)
    assert peak == 1, f"{peak} jobs held the heavy semaphore at once"


def test_old_jobs_are_evicted(queue):
    queue._keep = 5
    jobs = [wait_for(queue.submit("t", lambda p, j: None)) for _ in range(8)]
    assert queue.get(jobs[-1].id) is not None
    assert queue.get(jobs[0].id) is None, "the queue must not grow without bound"


def test_public_hides_result_until_done(queue):
    job = wait_for(queue.submit("t", lambda p, j: {"big": "payload"}))
    assert job.public()["result"] == {"big": "payload"}
    assert "result" not in job.public(include_result=False)


def test_terminal_status_is_set_last(queue):
    """A poller that sees a terminal status must find the job fully populated.

    Setting job.status before job.error/job.result would let a client read
    status="error" with error=None -- which is exactly what the SSE stream and the
    /jobs/{id} endpoint do on every tick.
    """
    observations = []
    stop = threading.Event()

    def watch(job):
        while not stop.is_set():
            if job.status == "error":
                observations.append(("error", job.error, job.finished_at))
                return
            if job.status == "done":
                observations.append(("done", job.result, job.finished_at))
                return

    def work(progress, job):
        raise RuntimeError("boom")

    job = queue.submit("t", work)
    watcher = threading.Thread(target=watch, args=(job,), daemon=True)
    watcher.start()
    wait_for(job)
    stop.set()
    watcher.join(timeout=5)

    assert observations, "watcher never saw a terminal status"
    status, payload, finished = observations[0]
    assert status == "error"
    assert payload is not None, "status went terminal before the error was populated"
    assert finished is not None, "status went terminal before finished_at was set"
