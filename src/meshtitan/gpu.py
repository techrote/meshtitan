from __future__ import annotations

import csv
import io
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass

from .config import GPUConfig


class GPUError(RuntimeError):
    pass


@dataclass(frozen=True)
class GPUInfo:
    index: int
    uuid: str
    name: str
    memory_total_mib: int
    compute_capability: str | None
    driver_version: str | None


def _run_nvidia_smi(fields: list[str]) -> str:
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise GPUError(f"nvidia-smi query failed: {exc}") from exc
    return result.stdout


def parse_gpu_csv(text: str, fields: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    reader = csv.reader(io.StringIO(text))
    for values in reader:
        if not values:
            continue
        cleaned = [value.strip() for value in values]
        if len(cleaned) != len(fields):
            raise GPUError(f"unexpected nvidia-smi row: {values!r}")
        rows.append(dict(zip(fields, cleaned, strict=True)))
    return rows


def query_gpus() -> list[GPUInfo]:
    preferred = ["index", "uuid", "name", "memory.total", "compute_cap", "driver_version"]
    try:
        rows = parse_gpu_csv(_run_nvidia_smi(preferred), preferred)
        has_cc = True
    except GPUError:
        fallback = ["index", "uuid", "name", "memory.total", "driver_version"]
        rows = parse_gpu_csv(_run_nvidia_smi(fallback), fallback)
        has_cc = False

    gpus = [
        GPUInfo(
            index=int(row["index"]),
            uuid=row["uuid"],
            name=row["name"],
            memory_total_mib=int(float(row["memory.total"])),
            compute_capability=row.get("compute_cap") if has_cc else None,
            driver_version=row.get("driver_version"),
        )
        for row in rows
    ]
    if not gpus:
        raise GPUError("nvidia-smi returned no GPUs")
    return gpus


def select_gpu(gpus: Iterable[GPUInfo], selector: str) -> GPUInfo:
    candidates = list(gpus)
    for gpu in candidates:
        if selector == str(gpu.index) or selector.lower() == gpu.uuid.lower():
            return gpu
    raise GPUError(f"configured GPU selector {selector!r} did not match any NVIDIA GPU")


def _cc_tuple(value: str) -> tuple[int, int]:
    try:
        major, minor = value.split(".", 1)
        return int(major), int(minor)
    except (ValueError, AttributeError) as exc:
        raise GPUError(f"invalid compute capability {value!r}") from exc


def assert_gpu_contract(gpu: GPUInfo, config: GPUConfig) -> None:
    if config.expected_name_contains.lower() not in gpu.name.lower():
        raise GPUError(
            f"GPU {gpu.index} is {gpu.name!r}; expected name containing {config.expected_name_contains!r}"
        )
    if gpu.memory_total_mib < config.min_vram_mib:
        raise GPUError(
            f"GPU {gpu.index} reports {gpu.memory_total_mib} MiB VRAM; "
            f"configured minimum is {config.min_vram_mib} MiB"
        )
    if gpu.compute_capability is not None:
        if _cc_tuple(gpu.compute_capability) < _cc_tuple(config.min_compute_capability):
            raise GPUError(
                f"GPU {gpu.index} compute capability {gpu.compute_capability} is below "
                f"{config.min_compute_capability}"
            )


def get_configured_gpu(config: GPUConfig) -> GPUInfo:
    gpu = select_gpu(query_gpus(), config.selector)
    assert_gpu_contract(gpu, config)
    return gpu
