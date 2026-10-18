from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .assets import AssetStore
from .backends import BackendError, backend_status, build_invocation, validate_parameters
from .config import RuntimeConfig
from .gpu import GPUInfo, get_configured_gpu
from .models import TERMINAL_STATUSES, BatchRequest, JobRecord, JobRequest, JobStatus
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
        self._submission_lock = threading.Lock()

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
        return self.submit_batch(BatchRequest(jobs=[request]), grouped=False)[0]

    def submit_batch(self, request: BatchRequest, *, grouped: bool = True) -> list[JobRecord]:
        # Validate the entire batch before creating or queuing any jobs.
        prepared = [self._validate(item) for item in request.jobs]
        batch_id = uuid.uuid4().hex if grouped else None
        with self._submission_lock:
            records = [
                self.jobs.create(
                    item, source_name=source_name, batch_id=batch_id,
                    batch_name=request.name if grouped else None, batch_index=index if grouped else None,
                )
                for index, (item, source_name) in enumerate(prepared)
            ]
            for record in records:
                self._cancel[record.job_id] = threading.Event()
                self._queue.put(record.job_id)
        return records

    def _validate(self, request: JobRequest) -> tuple[JobRequest, str]:
        backend = self.config.backends.get(request.backend)
        if backend is None or not backend.enabled:
            raise BackendError(f"backend is unavailable: {request.backend}")
        asset = self.assets.get(request.asset_id)
        if request.backend == "hunyuan_multiview":
            views = request.view_assets
            if (set(views) - {"front", "left", "back", "right"}
                    or "front" not in views or len(views) < 2):
                raise BackendError("multi-view requires front and at least one of left, back, right")
            if request.asset_id != views["front"]:
                raise BackendError("the primary asset must be the front view")
            for asset_id in views.values():
                self.assets.get(asset_id)
        elif request.view_assets:
            raise BackendError("view_assets is only supported by hunyuan_multiview")
        if request.backend == "ultrashape":
            parent = self._refinement_parent(request.parent_job_id)
            if request.asset_id != parent.asset_id:
                raise BackendError("refinement must use its source job's reference image")
        elif request.parent_job_id:
            raise BackendError("parent_job_id is only supported by ultrashape")
        parameters = validate_parameters(request.backend, request.parameters)
        source_name = request.source_name or asset.original_name
        return request.model_copy(update={"parameters": parameters}), source_name

    def _refinement_parent(self, job_id: str | None) -> JobRecord:
        if not job_id:
            raise BackendError("UltraShape requires parent_job_id")
        try:
            parent = self.jobs.get(job_id)
        except ValueError as exc:
            raise BackendError(str(exc)) from exc
        if parent.status != JobStatus.succeeded:
            raise BackendError("refinement source job must have succeeded")
        if not self._parent_mesh(parent).is_file():
            raise BackendError("refinement source mesh is unavailable")
        return parent

    def _parent_mesh(self, parent: JobRecord) -> Path:
        directory = self.jobs.job_dir(parent.job_id)
        preview = directory / "mesh.glb"
        return preview if preview.is_file() else directory / f"mesh.{parent.output_format}"

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
            status = backend_status(backend)
            if not status["ok"]:
                raise BackendError(status["error"])
            views = {
                name: self.assets.get(asset_id).model_dump() for name, asset_id in record.view_assets.items()
            }
            manifest = {"primary": asset.model_dump(), "views": views, "parent_job_id": record.parent_job_id}
            views_json = job_dir / "views.json"
            views_json.write_text(json.dumps({name: item["path"] for name, item in views.items()}),
                                  encoding="utf-8")
            parent_mesh = None
            input_up_axis = "y"
            if record.parent_job_id:
                parent = self._refinement_parent(record.parent_job_id)
                parent_mesh = self._parent_mesh(parent)
                digest = hashlib.sha256()
                with parent_mesh.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1048576), b""):
                        digest.update(chunk)
                manifest["parent_mesh_sha256"] = digest.hexdigest()
                input_up_axis = "z" if parent.backend == "triposr" else "y"
            (job_dir / "inputs.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            invocation = build_invocation(
                backend,
                Path(asset.path),
                job_dir,
                record.output_format,
                record.parameters,
                views_json=views_json if views else None,
                parent_mesh=parent_mesh,
                input_up_axis=input_up_axis,
            )
        except BackendError as exc:
            self.jobs.mark_terminal(job_id, JobStatus.failed, error=str(exc))
            return

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu.uuid
        env["PYTHONUNBUFFERED"] = "1"
        env["MESHTITAN_JOB_ID"] = job_id
        runner_sha256 = hashlib.sha256(Path(invocation.command[1]).read_bytes()).hexdigest()

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
                "runner_sha256": runner_sha256,
                "model_settings": backend.settings,
                "view_assets": record.view_assets,
                "parent_job_id": record.parent_job_id,
                "inputs": manifest,
            },
        }
        self.jobs.mark_terminal(job_id, JobStatus.succeeded, result=result)
