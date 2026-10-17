from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import __version__
from .assets import AssetError
from .backends import BackendError, backend_status
from .config import RuntimeConfig
from .executor import JobManager
from .models import AssetRecord, CapabilityResponse, JobRecord, JobRequest
from .store import JobStoreError


def create_app(config: RuntimeConfig) -> FastAPI:
    manager = JobManager(config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        manager.start()
        try:
            yield
        finally:
            manager.stop()

    app = FastAPI(title="meshtitan", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/capabilities", response_model=CapabilityResponse)
    def capabilities() -> CapabilityResponse:
        gpu = manager.gpu
        if gpu is None:
            raise HTTPException(status_code=503, detail="GPU manager is not ready")
        return CapabilityResponse(
            runtime_version=__version__,
            gpu={
                "index": gpu.index,
                "uuid": gpu.uuid,
                "name": gpu.name,
                "memory_total_mib": gpu.memory_total_mib,
                "compute_capability": gpu.compute_capability,
                "driver_version": gpu.driver_version,
            },
            backends={name: backend_status(cfg) for name, cfg in config.backends.items()},
        )

    @app.post("/v1/assets", response_model=AssetRecord)
    async def upload_asset(file: UploadFile = File(...)) -> AssetRecord:
        data = await file.read(config.max_upload_bytes + 1)
        if len(data) > config.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"asset exceeds {config.max_upload_mib} MiB limit")
        try:
            return manager.assets.add(data, file.filename or "input.png", file.content_type)
        except AssetError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/jobs", response_model=JobRecord, status_code=202)
    def submit_job(request: JobRequest) -> JobRecord:
        try:
            return manager.submit(request)
        except (AssetError, BackendError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        try:
            return manager.jobs.get(job_id)
        except JobStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobRecord)
    def cancel_job(job_id: str) -> JobRecord:
        try:
            return manager.cancel(job_id)
        except JobStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}/artifact")
    def job_artifact(job_id: str):
        try:
            record = manager.jobs.get(job_id)
        except JobStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        mesh = (record.result or {}).get("mesh")
        if not mesh or not Path(mesh).is_file():
            raise HTTPException(status_code=404, detail="mesh artifact is not available")
        return FileResponse(mesh, filename=Path(mesh).name, media_type="application/octet-stream")

    return app
