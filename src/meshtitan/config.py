from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class GPUConfig:
    selector: str = "0"
    expected_name_contains: str = "TITAN RTX"
    min_vram_mib: int = 22000
    min_compute_capability: str = "7.5"
    telemetry_interval_s: float = 1.0


@dataclass(frozen=True)
class BackendConfig:
    name: str
    enabled: bool
    python: Path
    repo: Path
    commit: str
    settings: dict[str, Any]


@dataclass(frozen=True)
class RuntimeConfig:
    config_path: Path
    data_dir: Path
    host: str
    port: int
    max_upload_mib: int
    gpu: GPUConfig
    backends: dict[str, BackendConfig]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mib * 1024 * 1024


def _resolve(base: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def load_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"configuration file not found: {config_path}")

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    base = config_path.parent
    runtime = raw.get("runtime", {})
    gpu_raw = raw.get("gpu", {})
    backend_raw = raw.get("backends", {})

    gpu = GPUConfig(
        selector=str(gpu_raw.get("selector", "0")),
        expected_name_contains=str(gpu_raw.get("expected_name_contains", "TITAN RTX")),
        min_vram_mib=int(gpu_raw.get("min_vram_mib", 22000)),
        min_compute_capability=str(gpu_raw.get("min_compute_capability", "7.5")),
        telemetry_interval_s=float(gpu_raw.get("telemetry_interval_s", 1.0)),
    )

    backends: dict[str, BackendConfig] = {}
    for name, item in backend_raw.items():
        if not isinstance(item, dict):
            raise ConfigError(f"backends.{name} must be a table")
        python_value = str(item.get("python", ""))
        repo_value = str(item.get("repo", ""))
        if not python_value or not repo_value:
            raise ConfigError(f"backends.{name} requires python and repo")
        known = {"enabled", "python", "repo", "commit"}
        backends[name] = BackendConfig(
            name=name,
            enabled=bool(item.get("enabled", True)),
            python=_resolve(base, python_value),
            repo=_resolve(base, repo_value),
            commit=str(item.get("commit", "")),
            settings={k: v for k, v in item.items() if k not in known},
        )

    if not backends:
        raise ConfigError("at least one backend must be configured")

    return RuntimeConfig(
        config_path=config_path,
        data_dir=_resolve(base, str(runtime.get("data_dir", "./var"))),
        host=str(runtime.get("host", "127.0.0.1")),
        port=int(runtime.get("port", 8088)),
        max_upload_mib=int(runtime.get("max_upload_mib", 32)),
        gpu=gpu,
        backends=backends,
    )


def resolve_python(path: Path) -> Path:
    """Resolve either a Python executable or a venv directory to its interpreter."""
    if path.is_file():
        return path
    if path.is_dir():
        candidates = (
            path / "Scripts" / "python.exe",
            path / "bin" / "python",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise ConfigError(f"backend Python interpreter not found at {path}")
