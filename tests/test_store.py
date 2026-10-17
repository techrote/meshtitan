from pathlib import Path

from meshtitan.models import JobRequest, JobStatus
from meshtitan.store import JobStore


def test_store_transitions_and_recovery(tmp_path: Path):
    store = JobStore(tmp_path)
    job = store.create(JobRequest(backend="triposr", asset_id="a" * 64))
    assert job.status == JobStatus.queued

    running = store.mark_running(job.job_id)
    assert running.status == JobStatus.running

    recovered = store.recover_interrupted()
    assert recovered == [job.job_id]
    assert store.get(job.job_id).status == JobStatus.failed


def test_cancel_flag_is_durable(tmp_path: Path):
    store = JobStore(tmp_path)
    job = store.create(JobRequest(backend="triposr", asset_id="b" * 64))
    store.request_cancel(job.job_id)
    assert store.get(job.job_id).cancellation_requested is True
