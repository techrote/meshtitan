from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

from .models import JobRecord, JobRequest, JobStatus, TERMINAL_STATUSES, utcnow


class JobStoreError(ValueError):
    pass


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def job_dir(self, job_id: str) -> Path:
        if not job_id or any(c not in "0123456789abcdef" for c in job_id):
            raise JobStoreError("invalid job id")
        return self.root / job_id

    def create(self, request: JobRequest) -> JobRecord:
        with self._lock:
            job_id = uuid.uuid4().hex
            record = JobRecord(
                job_id=job_id,
                backend=request.backend,
                asset_id=request.asset_id,
                output_format=request.output_format,
                parameters=request.parameters,
            )
            self.job_dir(job_id).mkdir(parents=True, exist_ok=False)
            self._write(record)
            return record

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            path = self.job_dir(job_id) / "job.json"
            if not path.is_file():
                raise JobStoreError(f"job not found: {job_id}")
            return JobRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def update(self, record: JobRecord) -> JobRecord:
        with self._lock:
            self._write(record)
            return record

    def _write(self, record: JobRecord) -> None:
        directory = self.job_dir(record.job_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "job.json"
        tmp = directory / "job.json.tmp"
        tmp.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def mark_running(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        if record.status != JobStatus.queued:
            raise JobStoreError(f"cannot run job from state {record.status}")
        record.status = JobStatus.running
        record.started_at = utcnow()
        return self.update(record)

    def mark_terminal(
        self,
        job_id: str,
        status: JobStatus,
        *,
        result: dict | None = None,
        error: str | None = None,
    ) -> JobRecord:
        if status not in TERMINAL_STATUSES:
            raise JobStoreError(f"{status} is not terminal")
        record = self.get(job_id)
        record.status = status
        record.finished_at = utcnow()
        record.result = result
        record.error = error
        return self.update(record)

    def request_cancel(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        if record.status in TERMINAL_STATUSES:
            return record
        record.cancellation_requested = True
        return self.update(record)

    def recover_interrupted(self) -> list[str]:
        recovered: list[str] = []
        for path in self.root.glob("*/job.json"):
            record = JobRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
            if record.status in {JobStatus.queued, JobStatus.running}:
                record.status = JobStatus.failed
                record.finished_at = utcnow()
                record.error = "runtime restarted before job reached a terminal state"
                self.update(record)
                recovered.append(record.job_id)
        return recovered
