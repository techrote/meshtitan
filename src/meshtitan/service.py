from __future__ import annotations

import os
import struct
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .assets import AssetError
from .backends import BackendError, backend_status
from .config import RuntimeConfig
from .executor import JobManager
from .models import AssetRecord, BatchRequest, CapabilityResponse, JobRecord, JobRequest, JobStatus
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
    web_dir = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/", include_in_schema=False)
    def web_ui() -> FileResponse:
        return FileResponse(web_dir / "index.html", headers={"Cache-Control": "no-cache"})

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
            max_upload_mib=config.max_upload_mib,
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
    async def upload_asset(file: Annotated[UploadFile, File(...)]) -> AssetRecord:
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

    @app.get("/v1/assets/{asset_id}/image")
    def source_image(asset_id: str):
        try:
            asset = manager.assets.get(asset_id)
        except AssetError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
        return FileResponse(asset.path, media_type=mime[Path(asset.path).suffix.lower()])

    @app.post("/v1/batches", response_model=list[JobRecord], status_code=202)
    def submit_batch(request: BatchRequest):
        try:
            return manager.submit_batch(request)
        except (AssetError, BackendError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/batches/{batch_id}/cancel", response_model=list[JobRecord])
    def cancel_batch(batch_id: str):
        records = manager.jobs.batch(batch_id)
        if not records:
            raise HTTPException(status_code=404, detail="batch not found")
        return [manager.cancel(record.job_id) for record in records]

    @app.get("/v1/batches/{batch_id}/artifacts")
    def download_batch(batch_id: str):
        records = manager.jobs.batch(batch_id)
        if not records:
            raise HTTPException(status_code=404, detail="batch not found")
        completed = [record for record in records if record.status == JobStatus.succeeded]
        if not completed:
            raise HTTPException(status_code=404, detail="batch has no completed meshes")
        archive = tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024)
        try:
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                for record in completed:
                    mesh = manager.jobs.job_dir(record.job_id) / f"mesh.{record.output_format}"
                    if mesh.is_file():
                        prefix = f"{(record.batch_index or 0) + 1:03d}-{record.backend}-{record.job_id[:8]}"
                        bundle.write(mesh, f"{prefix}.{record.output_format}")
                        bundle.writestr(f"{prefix}.json", record.model_dump_json(indent=2))
            archive.seek(0)
        except Exception:
            archive.close()
            raise

        def chunks():
            try:
                while data := archive.read(65536):
                    yield data
            finally:
                archive.close()

        return StreamingResponse(chunks(), media_type="application/zip", headers={
            "Content-Disposition": f'attachment; filename="meshtitan-{records[0].batch_id}.zip"',
        })

    @app.get("/v1/gallery")
    def gallery(limit: Annotated[int, Query(ge=1, le=50)] = 20, offset: Annotated[int, Query(ge=0)] = 0):
        groups = {}
        for record in manager.jobs.recent(1_000_000):
            key = record.batch_id or "legacy"
            group = groups.setdefault(key, {
                "id": key, "name": record.batch_name or "Earlier generations",
                "created_at": record.created_at, "jobs": [],
            })
            item = record.model_dump(mode="json")
            if not item["source_name"]:
                try:
                    item["source_name"] = manager.assets.get(record.asset_id).original_name
                except AssetError:
                    item["source_name"] = record.job_id[:8]
            directory = manager.jobs.job_dir(record.job_id)
            item["thumbnail_available"] = (directory / "thumbnail.png").is_file()
            item["preview_available"] = (
                record.status == JobStatus.succeeded and (directory / "mesh.glb").is_file()
            )
            group["jobs"].append(item)
        ordered = list(groups.values())
        for group in ordered:
            group["jobs"].sort(key=lambda item: item["batch_index"] if item["batch_index"] is not None else 0)
        return {"groups": ordered[offset:offset + limit], "total": len(ordered)}

    @app.get("/v1/jobs/{job_id}/preview")
    def job_preview(job_id: str):
        try:
            record = manager.jobs.get(job_id)
        except JobStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        path = manager.jobs.job_dir(job_id) / "mesh.glb"
        if record.status != JobStatus.succeeded or not path.is_file():
            raise HTTPException(status_code=404, detail="preview is not available")
        return FileResponse(path, media_type="model/gltf-binary")

    @app.get("/v1/jobs/{job_id}/thumbnail")
    def job_thumbnail(job_id: str):
        try:
            manager.jobs.get(job_id)
        except JobStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        path = manager.jobs.job_dir(job_id) / "thumbnail.png"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="thumbnail is not available")
        return FileResponse(path, media_type="image/png")

    @app.put("/v1/jobs/{job_id}/thumbnail")
    async def save_thumbnail(job_id: str, request: Request):
        try:
            record = manager.jobs.get(job_id)
        except JobStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if record.status != JobStatus.succeeded:
            raise HTTPException(status_code=409, detail="job is not complete")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 524288:
                raise HTTPException(status_code=413, detail="thumbnail exceeds 512 KiB")
        if len(data) < 33 or data[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR":
            raise HTTPException(status_code=400, detail="expected a PNG thumbnail")
        width, height = struct.unpack(">II", data[16:24])
        if not (1 <= width <= 1024 and 1 <= height <= 1024):
            raise HTTPException(status_code=400, detail="thumbnail dimensions exceed 1024 pixels")
        directory = manager.jobs.job_dir(job_id)
        temporary = directory / f"thumbnail-{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_bytes(data)
            os.replace(temporary, directory / "thumbnail.png")
        finally:
            temporary.unlink(missing_ok=True)
        return {"ok": True}

    @app.get("/v1/jobs", response_model=list[JobRecord])
    def recent_jobs(limit: Annotated[int, Query(ge=1, le=100)] = 30) -> list[JobRecord]:
        return manager.jobs.recent(limit)

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
