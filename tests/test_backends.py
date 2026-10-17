from pathlib import Path

import pytest

from meshtitan.backends import BackendError, build_invocation
from meshtitan.config import BackendConfig


def make_python(tmp_path: Path) -> Path:
    env = tmp_path / "venv"
    path = env / "bin" / "python"
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")
    return env


def test_hunyuan_invocation_contains_geometry_controls(tmp_path: Path):
    repo = tmp_path / "hunyuan"
    repo.mkdir()
    cfg = BackendConfig(
        name="hunyuan_shape",
        enabled=True,
        python=make_python(tmp_path),
        repo=repo,
        commit="abc",
        settings={"model_path": "m", "subfolder": "s"},
    )
    job = tmp_path / "job"
    job.mkdir()
    invocation = build_invocation(
        cfg,
        tmp_path / "image.png",
        job,
        "glb",
        {"octree_resolution": 384, "cpu_offload": True},
    )
    joined = " ".join(invocation.command)
    assert "--octree-resolution 384" in joined
    assert "--cpu-offload" in invocation.command
    assert invocation.output_path.name == "mesh.glb"


def test_unknown_parameter_fails_closed(tmp_path: Path):
    repo = tmp_path / "triposr"
    repo.mkdir()
    cfg = BackendConfig(
        name="triposr",
        enabled=True,
        python=make_python(tmp_path),
        repo=repo,
        commit="abc",
        settings={},
    )
    with pytest.raises(BackendError):
        build_invocation(cfg, tmp_path / "x.png", tmp_path, "glb", {"mystery": 1})
