from __future__ import annotations

import math
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
    "box_v": 1.01,
    "mc_level": 0.0,
    "border_ratio": 0.15,
    "experimental_resolution": False,
}

ULTRASHAPE_DEFAULTS: dict[str, Any] = {
    "steps": 30, "guidance_scale": 5.0, "seed": 1234,
    "octree_resolution": 256, "num_chunks": 2048, "num_latents": 8192,
    "normalize_scale": 0.99, "remove_background": True, "cpu_offload": True,
    "experimental_resolution": False,
}
MULTIVIEW_DEFAULTS = {**HUNYUAN_DEFAULTS, "cpu_offload": True}


def validate_parameters(backend: str, supplied: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "hunyuan_shape": HUNYUAN_DEFAULTS, "hunyuan_multiview": MULTIVIEW_DEFAULTS,
        "ultrashape": ULTRASHAPE_DEFAULTS, "triposr": TRIPOSR_DEFAULTS,
    }.get(backend)
    if defaults is None:
        raise BackendError(f"unsupported backend {backend!r}")
    values = _merge(defaults, supplied)
    ranges = {
        "steps": (1, 100), "guidance_scale": (0, 30), "seed": (0, 4294967295),
        "octree_resolution": (16, 1024), "num_chunks": (256, 65536),
        "box_v": (0.5, 2), "mc_level": (-1, 1), "border_ratio": (0, 0.5),
        "chunk_size": (256, 65536), "mc_resolution": (16, 1024), "foreground_ratio": (0.1, 1),
        "density_threshold": (0, 1000), "background_gray": (0, 1),
        "num_latents": (1024, 32768), "normalize_scale": (0.5, 0.9999),
    }
    for key, value in values.items():
        default = defaults[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise BackendError(f"{key} must be true or false")
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or (isinstance(value, float) and not math.isfinite(value))):
            raise BackendError(f"{key} must be a finite number")
        if isinstance(default, int) and not isinstance(value, int):
            raise BackendError(f"{key} must be an integer")
        low, high = ranges[key]
        if not low <= value <= high:
            raise BackendError(f"{key} must be between {low} and {high}")
    resolution = values["mc_resolution" if backend == "triposr" else "octree_resolution"]
    if not 64 <= resolution <= 512 and not values["experimental_resolution"]:
        raise BackendError("resolutions outside 64–512 require experimental_resolution=true")
    return values

TRIPOSR_DEFAULTS: dict[str, Any] = {
    "chunk_size": 8192,
    "mc_resolution": 256,
    "remove_background": True,
    "foreground_ratio": 0.85,
    "density_threshold": 25.0,
    "background_gray": 0.5,
    "experimental_resolution": False,
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
    *,
    views_json: Path | None = None,
    parent_mesh: Path | None = None,
    input_up_axis: str = "y",
) -> BackendInvocation:
    output_path = job_dir / f"mesh.{output_format}"
    result_json = job_dir / "runner-result.json"

    if config.name in {"hunyuan_shape", "hunyuan_multiview"}:
        values = validate_parameters(config.name, parameters)
        multiview = config.name == "hunyuan_multiview"
        model_path = str(config.settings.get("model_path", "tencent/Hunyuan3D-2mv" if multiview
                                             else "tencent/Hunyuan3D-2.1"))
        subfolder = str(config.settings.get("subfolder", "hunyuan3d-dit-v2-mv" if multiview
                                            else "hunyuan3d-dit-v2-1"))
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
            "--box-v", str(float(values["box_v"])),
            "--mc-level", str(float(values["mc_level"])),
            "--border-ratio", str(float(values["border_ratio"])),
        ]
        if bool(values["remove_background"]):
            cmd.append("--remove-background")
        if bool(values["cpu_offload"]):
            cmd.append("--cpu-offload")
        if config.name == "hunyuan_multiview":
            if views_json is None:
                raise BackendError("multi-view requires a view manifest")
            cmd.extend(["--multiview", "--views-json", str(views_json)])
    elif config.name == "ultrashape":
        if parent_mesh is None:
            raise BackendError("UltraShape requires a completed source mesh")
        values = validate_parameters(config.name, parameters)
        checkpoint = Path(str(config.settings.get("checkpoint", "checkpoints/ultrashape_v1.pt")))
        if not checkpoint.is_absolute():
            checkpoint = config.repo / checkpoint
        cmd = [
            _python(config), str(_runner("ultrashape")), "--repo", str(config.repo),
            "--input", str(input_path), "--mesh", str(parent_mesh),
            "--output", str(output_path), "--result-json", str(result_json),
            "--checkpoint", str(checkpoint), "--input-up-axis", input_up_axis,
        ]
        for key, value in values.items():
            if isinstance(value, bool):
                if value and key != "experimental_resolution":
                    cmd.append("--" + key.replace("_", "-"))
            else:
                cmd.extend(["--" + key.replace("_", "-"), str(value)])
    elif config.name == "triposr":
        values = validate_parameters(config.name, parameters)
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
            "--density-threshold", str(float(values["density_threshold"])),
            "--background-gray", str(float(values["background_gray"])),
        ]
        if bool(values["remove_background"]):
            cmd.append("--remove-background")
    else:
        raise BackendError(f"unsupported backend {config.name!r}")

    if values["experimental_resolution"]:
        cmd.append("--experimental-resolution")
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
