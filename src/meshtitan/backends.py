from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BackendConfig, ConfigError, resolve_python


class BackendError(ValueError):
    pass


@dataclass(frozen=True)
class BackendInvocation:
    command: list[str]
    cwd: Path
    output_path: Path
    result_json: Path


HUNYUAN_DEFAULTS: dict[str, Any] = {
    "steps": 30,
    "guidance_scale": 5.0,
    "seed": 1234,
    "octree_resolution": 256,
    "num_chunks": 8000,
    "remove_background": True,
    "cpu_offload": False,
}

TRIPOSR_DEFAULTS: dict[str, Any] = {
    "chunk_size": 8192,
    "mc_resolution": 256,
    "remove_background": True,
    "foreground_ratio": 0.85,
}


def _merge(allowed: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(supplied) - set(allowed))
    if unknown:
        raise BackendError(f"unknown backend parameters: {', '.join(unknown)}")
    values = dict(allowed)
    values.update(supplied)
    return values


def _runner(name: str) -> Path:
    return Path(__file__).resolve().parent / "runners" / f"{name}.py"


def _python(config: BackendConfig) -> str:
    try:
        return str(resolve_python(config.python))
    except ConfigError as exc:
        raise BackendError(str(exc)) from exc


def build_invocation(
    config: BackendConfig,
    input_path: Path,
    job_dir: Path,
    output_format: str,
    parameters: dict[str, Any],
) -> BackendInvocation:
    output_path = job_dir / f"mesh.{output_format}"
    result_json = job_dir / "runner-result.json"

    if config.name == "hunyuan_shape":
        values = _merge(HUNYUAN_DEFAULTS, parameters)
        model_path = str(config.settings.get("model_path", "tencent/Hunyuan3D-2.1"))
        subfolder = str(config.settings.get("subfolder", "hunyuan3d-dit-v2-1"))
        cmd = [
            _python(config),
            str(_runner("hunyuan_shape")),
            "--repo", str(config.repo),
            "--input", str(input_path),
            "--output", str(output_path),
            "--result-json", str(result_json),
            "--model-path", model_path,
            "--subfolder", subfolder,
            "--steps", str(int(values["steps"])),
            "--guidance-scale", str(float(values["guidance_scale"])),
            "--seed", str(int(values["seed"])),
            "--octree-resolution", str(int(values["octree_resolution"])),
            "--num-chunks", str(int(values["num_chunks"])),
        ]
        if bool(values["remove_background"]):
            cmd.append("--remove-background")
        if bool(values["cpu_offload"]):
            cmd.append("--cpu-offload")
    elif config.name == "triposr":
        values = _merge(TRIPOSR_DEFAULTS, parameters)
        model = str(config.settings.get("model", "stabilityai/TripoSR"))
        cmd = [
            _python(config),
            str(_runner("triposr")),
            "--repo", str(config.repo),
            "--input", str(input_path),
            "--output", str(output_path),
            "--result-json", str(result_json),
            "--model", model,
            "--chunk-size", str(int(values["chunk_size"])),
            "--mc-resolution", str(int(values["mc_resolution"])),
            "--foreground-ratio", str(float(values["foreground_ratio"])),
        ]
        if bool(values["remove_background"]):
            cmd.append("--remove-background")
    else:
        raise BackendError(f"unsupported backend {config.name!r}")

    return BackendInvocation(cmd, config.repo, output_path, result_json)


def backend_status(config: BackendConfig) -> dict[str, Any]:
    status: dict[str, Any] = {
        "enabled": config.enabled,
        "repo": str(config.repo),
        "python": str(config.python),
        "expected_commit": config.commit,
    }
    try:
        status["resolved_python"] = str(resolve_python(config.python))
    except ConfigError as exc:
        status["ok"] = False
        status["error"] = str(exc)
        return status

    if not config.repo.is_dir():
        status["ok"] = False
        status["error"] = f"repository not found: {config.repo}"
        return status

    try:
        head = subprocess.run(
            ["git", "-C", str(config.repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        status["ok"] = False
        status["error"] = f"cannot read repository HEAD: {exc}"
        return status

    status["actual_commit"] = head
    status["ok"] = not config.commit or head == config.commit
    if not status["ok"]:
        status["error"] = "repository HEAD does not match configured upstream pin"
    return status
