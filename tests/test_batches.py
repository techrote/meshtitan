import io
import struct
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from meshtitan.backends import BackendError, validate_parameters
from meshtitan.config import BackendConfig, GPUConfig, RuntimeConfig
from meshtitan.executor import JobManager
from meshtitan.gpu import GPUInfo
from meshtitan.models import BatchRequest, JobRequest, JobStatus
from meshtitan.service import create_app


def config_for(tmp_path):
    return RuntimeConfig(
        tmp_path / "config.toml", tmp_path / "data", "127.0.0.1", 8088, 32, GPUConfig(),
        {name: BackendConfig(name, True, tmp_path / "python", tmp_path / "repo", "abc", {})
         for name in ("hunyuan_shape", "triposr")},
    )


def test_batch_validation_is_all_or_nothing_and_order_is_durable(tmp_path: Path):
    manager = JobManager(config_for(tmp_path))
    a = manager.assets.add(b"image-a", "a.png", "image/png")
    b = manager.assets.add(b"image-b", "b.png", "image/png")
    valid = JobRequest(backend="triposr", asset_id=a.asset_id)
    invalid = JobRequest(backend="hunyuan_shape", asset_id=b.asset_id, parameters={"box_v": 100})
    with pytest.raises(BackendError):
        manager.submit_batch(BatchRequest(name="invalid", jobs=[valid, invalid]))
    assert manager.jobs.recent() == []
    assert manager._queue.empty()

    batch = manager.submit_batch(BatchRequest(name="Product studies", jobs=[
        valid, JobRequest(backend="hunyuan_shape", asset_id=b.asset_id, parameters={"border_ratio": 0.2}),
    ]))
    assert len({record.batch_id for record in batch}) == 1
    assert [record.batch_index for record in batch] == [0, 1]
    assert [record.source_name for record in batch] == ["a.png", "b.png"]
    assert [manager._queue.get_nowait() for _ in batch] == [record.job_id for record in batch]
    restarted = JobManager(config_for(tmp_path))
    restarted.jobs.recover_interrupted()
    restored = restarted.jobs.batch(batch[0].batch_id)
    assert [record.job_id for record in restored] == [record.job_id for record in batch]
    assert all(
        record.batch_name == "Product studies" and record.status == JobStatus.failed for record in restored
    )
    duplicates = manager.submit_batch(BatchRequest(jobs=[
        JobRequest(backend="triposr", asset_id=a.asset_id, source_name="copy-1.png"),
        JobRequest(backend="triposr", asset_id=a.asset_id, source_name="copy-2.png"),
    ]))
    assert [record.source_name for record in duplicates] == ["copy-1.png", "copy-2.png"]


@pytest.mark.parametrize("backend,params", [
    ("hunyuan_shape", {"steps": 1.5}), ("hunyuan_shape", {"steps": True}),
    ("hunyuan_shape", {"seed": 10 ** 500}), ("hunyuan_shape", {"guidance_scale": float("nan")}),
    ("hunyuan_shape", {"mc_level": float("inf")}), ("hunyuan_shape", {"cpu_offload": "false"}),
    ("triposr", {"foreground_ratio": 0}), ("triposr", {"chunk_size": -1}),
    ("triposr", {"paint": True}),
])
def test_invalid_parameters_rejected_before_queueing(backend, params):
    with pytest.raises(BackendError):
        validate_parameters(backend, params)


@pytest.mark.parametrize("backend,key", [
    ("hunyuan_shape", "octree_resolution"), ("triposr", "mc_resolution"),
])
def test_experimental_resolution_requires_opt_in(backend, key):
    for resolution in (16, 32, 768, 1024):
        with pytest.raises(BackendError, match="experimental_resolution"):
            validate_parameters(backend, {key: resolution})
        assert validate_parameters(backend, {
            key: resolution, "experimental_resolution": True,
        })[key] == resolution
    for resolution in (15, 1025, 4096):
        with pytest.raises(BackendError):
            validate_parameters(backend, {key: resolution, "experimental_resolution": True})
    for resolution in (64, 128, 256, 384, 512):
        assert validate_parameters(backend, {key: resolution})[key] == resolution


def test_triposr_density_and_background_controls():
    values = validate_parameters("triposr", {"density_threshold": 15.5, "background_gray": 0.75})
    assert values["density_threshold"] == 15.5
    assert values["background_gray"] == 0.75
    assert validate_parameters("triposr", {})["density_threshold"] == 25
    for params in ({"density_threshold": -1}, {"density_threshold": float("inf")}, {"background_gray": 1.1}):
        with pytest.raises(BackendError):
            validate_parameters("triposr", params)


def test_gallery_thumbnail_preview_zip_and_batch_cancel(tmp_path: Path, monkeypatch):
    cfg = config_for(tmp_path)
    manager = JobManager(cfg)
    asset = manager.assets.add(b"source-image", "chair.png", "image/png")
    records = manager.submit_batch(BatchRequest(name="Saved batch", jobs=[
        JobRequest(backend="triposr", asset_id=asset.asset_id, output_format="obj"),
        JobRequest(backend="hunyuan_shape", asset_id=asset.asset_id),
    ]))
    first = records[0]
    directory = manager.jobs.job_dir(first.job_id)
    mesh = directory / "mesh.obj"
    mesh.write_bytes(b"obj-mesh")
    (directory / "mesh.glb").write_bytes(b"glTF-preview")
    manager.jobs.mark_terminal(first.job_id, JobStatus.succeeded, result={"mesh": str(mesh)})

    def start(instance):
        instance.gpu = GPUInfo(1, "GPU-test", "TITAN RTX", 24576, "7.5", "test")

    monkeypatch.setattr(JobManager, "start", start)  # Keep a real queued job without starting GPU work.
    with TestClient(create_app(cfg)) as client:
        groups = client.get("/v1/gallery").json()
        assert groups["total"] == 1
        group = groups["groups"][0]
        assert group["name"] == "Saved batch"
        assert [job["job_id"] for job in group["jobs"]] == [record.job_id for record in records]
        assert group["jobs"][0]["preview_available"] is True
        assert group["jobs"][0]["thumbnail_available"] is False
        assert client.get("/v1/gallery?offset=1").json()["groups"] == []
        assert client.get(f"/v1/assets/{asset.asset_id}/image").content == b"source-image"
        assert client.get(f"/v1/jobs/{first.job_id}/preview").content == b"glTF-preview"
        assert client.get(f"/v1/jobs/{records[1].job_id}/preview").status_code == 404
        endpoint = f"/v1/jobs/{first.job_id}/thumbnail"
        assert client.get(endpoint).status_code == 404
        assert client.put(endpoint, content=b"not png").status_code == 400
        assert client.put(endpoint, content=b"x" * 524289).status_code == 413
        header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        oversized_header = header + struct.pack(">II", 4096, 240) + b"\0" * 9
        assert client.put(endpoint, content=oversized_header).status_code == 400
        png = header + struct.pack(">II", 320, 240) + b"\0" * 9
        assert client.put(endpoint, content=png).status_code == 200
        assert client.get(endpoint).content == png
        assert client.get("/v1/gallery").json()["groups"][0]["jobs"][0]["thumbnail_available"] is True
        assert client.put(f"/v1/jobs/{records[1].job_id}/thumbnail", content=png).status_code == 409
        response = client.get(f"/v1/batches/{first.batch_id}/artifacts")
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert len(archive.namelist()) == 2  # One mesh plus its job metadata.
            obj_name = next(name for name in archive.namelist() if name.endswith(".obj"))
            assert archive.read(obj_name) == b"obj-mesh"
        cancelled = client.post(f"/v1/batches/{first.batch_id}/cancel").json()
        assert cancelled[0]["status"] == "succeeded"
        assert cancelled[1]["status"] == "cancelled"
        assert client.post("/v1/batches/missing/cancel").status_code == 404
        assert client.get("/v1/batches/missing/artifacts").status_code == 404


def test_batch_api_rejects_entire_invalid_request(tmp_path: Path, monkeypatch):
    cfg = config_for(tmp_path)
    manager = JobManager(cfg)
    asset = manager.assets.add(b"image", "image.png", "image/png")
    monkeypatch.setattr(JobManager, "start", lambda _: None)
    with TestClient(create_app(cfg)) as client:
        response = client.post("/v1/batches", json={"name": "Bad batch", "jobs": [
            {"backend": "triposr", "asset_id": asset.asset_id},
            {"backend": "triposr", "asset_id": asset.asset_id, "parameters": {"mc_resolution": 10000}},
        ]})
        assert response.status_code == 400
        assert client.get("/v1/jobs").json() == []
        assert client.post("/v1/batches", json={"jobs": []}).status_code == 422
