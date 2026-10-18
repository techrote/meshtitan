import hashlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from meshtitan.backends import BackendError, BackendInvocation, build_invocation, validate_parameters
from meshtitan.config import BackendConfig, GPUConfig, RuntimeConfig
from meshtitan.executor import JobManager
from meshtitan.gpu import GPUInfo
from meshtitan.models import JobRequest, JobStatus
from meshtitan.service import create_app


def manager_for(tmp_path):
    backends = {name: BackendConfig(name, True, tmp_path / "python", tmp_path / "repo", "abc", {})
                for name in ("hunyuan_shape", "triposr", "hunyuan_multiview", "ultrashape")}
    return JobManager(RuntimeConfig(tmp_path / "config", tmp_path / "data", "127.0.0.1", 8088,
                                    32, GPUConfig(), backends))


def test_multiview_inputs_validate_and_survive_reload(tmp_path):
    manager = manager_for(tmp_path)
    front = manager.assets.add(b"front", "front.png").asset_id
    back = manager.assets.add(b"back", "back.png").asset_id
    request = JobRequest(backend="hunyuan_multiview", asset_id=front,
                         view_assets={"front": front, "back": back})
    job = manager.submit(request)
    assert manager.jobs.get(job.job_id).view_assets == request.view_assets
    for invalid in ({}, {"front": front}, {"left": front, "back": back},
                    {"front": front, "top": back}, {"front": back, "left": front}):
        with pytest.raises(BackendError):
            manager.submit(request.model_copy(update={"view_assets": invalid}))
    with pytest.raises(BackendError, match="view_assets"):
        manager.submit(request.model_copy(update={"backend": "triposr"}))


def test_refinement_requires_successful_parent_and_matching_reference(tmp_path):
    manager = manager_for(tmp_path)
    asset = manager.assets.add(b"front", "front.png").asset_id
    parent = manager.submit(JobRequest(backend="hunyuan_shape", asset_id=asset))
    request = JobRequest(backend="ultrashape", asset_id=asset, parent_job_id=parent.job_id)
    with pytest.raises(BackendError, match="succeeded"):
        manager.submit(request)
    manager.jobs.mark_terminal(parent.job_id, JobStatus.succeeded)
    with pytest.raises(BackendError, match="unavailable"):
        manager.submit(request)
    mesh = manager.jobs.job_dir(parent.job_id) / "mesh.glb"
    mesh.write_bytes(b"source mesh")
    child = manager.submit(request)
    assert manager.jobs.get(child.job_id).parent_job_id == parent.job_id
    other = manager.assets.add(b"other", "other.png").asset_id
    with pytest.raises(BackendError, match="reference"):
        manager.submit(request.model_copy(update={"asset_id": other}))
    with pytest.raises(BackendError, match="parent_job_id"):
        manager.submit(request.model_copy(update={"parent_job_id": None}))
    with pytest.raises(BackendError):
        manager.submit(request.model_copy(update={"parent_job_id": "../outside"}))
    assert mesh.read_bytes() == b"source mesh"


def test_new_runner_invocations_and_parameter_guards(tmp_path: Path):
    python = tmp_path / "python.exe"
    python.touch()
    mv = BackendConfig("hunyuan_multiview", True, python, tmp_path, "abc", {})
    invocation = build_invocation(mv, tmp_path / "front.png", tmp_path, "obj", {},
                                  views_json=tmp_path / "views.json")
    command = " ".join(invocation.command)
    assert "tencent/Hunyuan3D-2mv" in command
    assert "--multiview" in command and "--views-json" in command
    with pytest.raises(BackendError, match="manifest"):
        build_invocation(mv, tmp_path / "front.png", tmp_path, "glb", {})
    ultra = BackendConfig("ultrashape", True, python, tmp_path, "abc", {})
    invocation = build_invocation(ultra, tmp_path / "front.png", tmp_path, "glb", {},
                                  parent_mesh=tmp_path / "source.glb", input_up_axis="z")
    command = " ".join(invocation.command)
    assert "--cpu-offload" in command
    assert "--num-latents 8192" in command
    assert "--input-up-axis z" in command
    for supplied in ({"num_latents": 32769}, {"num_latents": True}, {"normalize_scale": 1},
                     {"octree_resolution": 1024}, {"paint": True}, {"guidance_scale": float("nan")}):
        with pytest.raises(BackendError):
            validate_parameters("ultrashape", supplied)


def test_extension_api_batch_fails_atomically_and_returns_view_ids(tmp_path, monkeypatch):
    manager = manager_for(tmp_path)
    front = manager.assets.add(b"front", "front.png").asset_id
    back = manager.assets.add(b"back", "back.png").asset_id
    monkeypatch.setattr(JobManager, "start", lambda _: None)
    request = {"backend": "hunyuan_multiview", "asset_id": front,
               "view_assets": {"front": front, "back": back}}
    with TestClient(create_app(manager.config)) as client:
        response = client.post("/v1/batches", json={"jobs": [request, {
            "backend": "ultrashape", "asset_id": front, "parent_job_id": "a" * 32,
        }]})
        assert response.status_code == 400
        assert client.get("/v1/jobs").json() == []
        response = client.post("/v1/batches", json={"jobs": [request]})
        assert response.status_code == 202
        record = response.json()[0]
        assert record["view_assets"] == request["view_assets"]
        assert record["parameters"]["cpu_offload"] is True
        assert client.get(f'/v1/jobs/{record["job_id"]}').json()["view_assets"] == request["view_assets"]


def test_executor_keeps_input_and_runner_provenance_and_isolates_gpu(tmp_path, monkeypatch):
    manager = manager_for(tmp_path)
    manager.gpu = GPUInfo(1, "GPU-titan", "TITAN RTX", 24576, "7.5", "test")
    front = manager.assets.add(b"front", "front.png").asset_id
    parent = manager.submit(JobRequest(backend="hunyuan_shape", asset_id=front))
    mesh = manager.jobs.job_dir(parent.job_id) / "mesh.glb"
    mesh.write_bytes(b"original mesh")
    manager.jobs.mark_terminal(parent.job_id, JobStatus.succeeded)
    child = manager.submit(JobRequest(backend="ultrashape", asset_id=front, parent_job_id=parent.job_id))
    directory = manager.jobs.job_dir(child.job_id)
    runner = tmp_path / "fake_runner.py"
    runner.write_text(
        "import json, os, pathlib\n"
        f"pathlib.Path({str(directory / 'mesh.glb')!r}).write_bytes(b'refined mesh')\n"
        f"pathlib.Path({str(directory / 'runner-result.json')!r}).write_text("
        "json.dumps({'visible_gpu':os.environ['CUDA_VISIBLE_DEVICES']}))\n", encoding="utf-8",
    )

    def invocation(*args, **kwargs):
        assert kwargs["parent_mesh"] == mesh
        return BackendInvocation([sys.executable, str(runner)], tmp_path,
                                 directory / "mesh.glb", directory / "runner-result.json")

    class Sampler:
        def __init__(self, *args):
            pass

        def start(self):
            pass

        def stop(self):
            return {"test": True}

    monkeypatch.setattr("meshtitan.executor.build_invocation", invocation)
    monkeypatch.setattr("meshtitan.executor.backend_status", lambda _: {"ok": True})
    monkeypatch.setattr("meshtitan.executor.TelemetrySampler", Sampler)
    manager._execute(child.job_id)
    result = manager.jobs.get(child.job_id)
    assert result.status == JobStatus.succeeded
    assert result.result["backend"]["visible_gpu"] == "GPU-titan"
    provenance = result.result["provenance"]
    assert provenance["parent_job_id"] == parent.job_id
    assert provenance["runner_sha256"] == hashlib.sha256(runner.read_bytes()).hexdigest()
    assert provenance["inputs"]["parent_mesh_sha256"] == hashlib.sha256(b"original mesh").hexdigest()
    assert json.loads((directory / "inputs.json").read_text())["primary"]["sha256"] == front
    assert mesh.read_bytes() == b"original mesh"
