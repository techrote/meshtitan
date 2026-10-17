from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .assets import AssetStore
from .backends import BackendError, build_invocation
from .config import RuntimeConfig
from .gpu import GPUInfo, get_configured_gpu
from .models import JobRecord, JobRequest, JobStatus, TERMINAL_STATUSES
from .store import JobStore
from .telemetry import TelemetrySampler


class JobManager:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.assets = AssetStore(config.data_dir / "assets")
        self.jobs = JobStore(config.data_dir / "jobs")
        self.gpu: GPUInfo | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cancel: dict[str, threading.Event] = {}

    def start(self) -> None:
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.jobs.recover_interrupted()
        self.gpu = get_configured_gpu(self.config.gpu)
        self._thread = threading.Thread(target=self._worker, name="meshtitan-job-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        for event in self._cancel.values():
            event.set()
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=10)

    def submit(self, request: JobRequest) -> JobRecord:
        backend = self.config.backends.get(request.backend)
        if backend is None or not backend.enabled:
            raise BackendError(f"backend is unavailable: {request.backend}")
        self.assets.get(request.asset_id)
        record = self.jobs.create(request)
        self._cancel[record.job_id] = threading.Event()
        self._queue.put(record.job_id)
        return record

    def cancel(self, job_id: str) -> JobRecord:
        record = self.jobs.request_cancel(job_id)
        event = self._cancel.get(job_id)
        if event is not None:
            event.set()
        if record.status == JobStatus.queued:
            return self.jobs.mark_terminal(job_id, JobStatus.cancelled, error="cancelled before execution")
        return self.jobs.get(job_id)

    def _worker(self) -> None:
        while not self._stop.is_set():
            job_id = self._queue.get()
            if job_id is None:
                return
            try:
                record = self.jobs.get(job_id)
                if record.status in TERMINAL_STATUSES or record.cancellation_requested:
                    continue
                self._execute(job_id)
            except Exception as exc:  # fail closed: the queue must keep serving subsequent jobs
                try:
                    self.jobs.mark_terminal(job_id, JobStatus.failed, error=f"executor error: {exc}")
                except Exception:
                    pass
            finally:
                self._cancel.pop(job_id, None)

    def _execute(self, job_id: str) -> None:
        if self.gpu is None:
            raise RuntimeError("job manager was not started")
        record = self.jobs.mark_running(job_id)
        asset = self.assets.get(record.asset_id)
        backend = self.config.backends[record.backend]
        job_dir = self.jobs.job_dir(job_id)

        try:
            invocation = build_invocation(
                backend,
                Path(asset.path),
                job_dir,
                record.output_format,
                record.parameters,
            )
        except BackendError as exc:
            self.jobs.mark_terminal(job_id, JobStatus.failed, error=str(exc))
            return

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu.uuid
        env["PYTHONUNBUFFERED"] = "1"
        env["MESHTITAN_JOB_ID"] = job_id

        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        telemetry = TelemetrySampler(
            self.gpu.uuid,
            job_dir / "gpu-telemetry.jsonl",
            self.config.gpu.telemetry_interval_s,
        )
        cancel_event = self._cancel.setdefault(job_id, threading.Event())
        started = time.monotonic()

        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(
                invocation.command,
                cwd=invocation.cwd,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            telemetry.start()
            cancelled = False
            try:
                while process.poll() is None:
                    if self._stop.is_set() or cancel_event.wait(0.25):
                        cancelled = True
                        process.terminate()
                        try:
                            process.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                        break
            finally:
                telemetry_summary = telemetry.stop()

        wall_s = time.monotonic() - started
        if cancelled:
            self.jobs.mark_terminal(
                job_id,
                JobStatus.cancelled,
                error="cancelled during backend execution",
                result={"wall_seconds": wall_s, "telemetry": telemetry_summary},
            )
            return

        if process.returncode != 0:
            self.jobs.mark_terminal(
                job_id,
                JobStatus.failed,
                error=f"backend exited with code {process.returncode}; see stderr.log",
                result={"wall_seconds": wall_s, "telemetry": telemetry_summary},
            )
            return

        if not invocation.output_path.is_file() or invocation.output_path.stat().st_size == 0:
            self.jobs.mark_terminal(
                job_id,
                JobStatus.failed,
                error="backend reported success but mesh artifact is missing",
            )
            return

        runner_result: dict[str, Any] = {}
        if invocation.result_json.is_file():
            try:
                runner_result = json.loads(invocation.result_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                runner_result = {"warning": "runner-result.json could not be parsed"}

        result = {
            "mesh": str(invocation.output_path.resolve()),
            "mesh_bytes": invocation.output_path.stat().st_size,
            "wall_seconds": wall_s,
            "telemetry": telemetry_summary,
            "backend": runner_result,
            "provenance": {
                "gpu_uuid": self.gpu.uuid,
                "gpu_name": self.gpu.name,
                "gpu_compute_capability": self.gpu.compute_capability,
                "gpu_memory_total_mib": self.gpu.memory_total_mib,
                "driver_version": self.gpu.driver_version,
                "upstream_commit": backend.commit,
            },
        }
        self.jobs.mark_terminal(job_id, JobStatus.succeeded, result=result)
