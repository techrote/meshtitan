from pathlib import Path

from fastapi.testclient import TestClient

from meshtitan.config import BackendConfig, GPUConfig, RuntimeConfig
from meshtitan.gpu import GPUInfo
from meshtitan.models import JobRequest, JobStatus
from meshtitan.service import create_app
from meshtitan.store import JobStore


def test_web_ui_and_saved_jobs(tmp_path: Path, monkeypatch):
    gpu = GPUInfo(1, "GPU-test", "NVIDIA TITAN RTX", 24576, "7.5", "test")
    monkeypatch.setattr("meshtitan.executor.get_configured_gpu", lambda _: gpu)
    config = RuntimeConfig(
        tmp_path / "config.toml", tmp_path / "data", "127.0.0.1", 8088, 12, GPUConfig(),
        {"triposr": BackendConfig("triposr", True, tmp_path / "python", tmp_path / "repo", "abc", {})},
    )
    store = JobStore(config.data_dir / "jobs")
    first = store.create(JobRequest(backend="triposr", asset_id="a" * 64))
    second = store.create(JobRequest(backend="triposr", asset_id="b" * 64))
    mesh = store.job_dir(first.job_id) / "mesh.glb"
    mesh.write_bytes(b"glTF-test-artifact")
    store.mark_terminal(first.job_id, JobStatus.succeeded, result={"mesh": str(mesh)})

    with TestClient(create_app(config)) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "Import images" in home.text
        for path in ["app.js", "batch.js", "viewer.js", "style.css", "vendor/three.module.min.js",
                     "vendor/three.core.min.js", "vendor/GLTFLoader.js", "vendor/OrbitControls.js",
                     "vendor/utils/BufferGeometryUtils.js"]:
            assert client.get(f"/static/{path}").status_code == 200
        assert client.get("/static/../service.py").status_code == 404
        assert client.get("/v1/capabilities").json()["max_upload_mib"] == 12
        listed = client.get("/v1/jobs").json()
        assert [item["job_id"] for item in listed] == [second.job_id, first.job_id]
        assert listed[0]["status"] == "failed"  # Startup recovers interrupted jobs.
        assert client.get("/v1/jobs?limit=1").json()[0]["job_id"] == second.job_id
        assert client.get("/v1/jobs?limit=0").status_code == 422
        assert client.get("/v1/jobs?limit=101").status_code == 422
        response = client.get(f"/v1/jobs/{first.job_id}/artifact")
        assert response.content == b"glTF-test-artifact"
        assert "attachment" in response.headers["content-disposition"]
        assert client.get(f"/v1/jobs/{second.job_id}/artifact").status_code == 404
        assert client.get("/v1/jobs/unknown").status_code == 404
