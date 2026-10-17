from __future__ import annotations

import csv
import io
import json
import subprocess
import threading
from pathlib import Path
from typing import Any


_FIELDS = [
    "timestamp",
    "index",
    "uuid",
    "name",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
]


def _number(value: str) -> float | None:
    value = value.strip()
    if not value or value in {"N/A", "[Not Supported]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def sample_gpu(uuid: str) -> dict[str, Any] | None:
    cmd = [
        "nvidia-smi",
        "-i", uuid,
        f"--query-gpu={','.join(_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    rows = list(csv.reader(io.StringIO(out)))
    if not rows:
        return None
    values = [x.strip() for x in rows[0]]
    if len(values) != len(_FIELDS):
        return None
    raw = dict(zip(_FIELDS, values, strict=True))
    return {
        "timestamp": raw["timestamp"],
        "index": int(raw["index"]),
        "uuid": raw["uuid"],
        "name": raw["name"],
        "memory_used_mib": _number(raw["memory.used"]),
        "memory_total_mib": _number(raw["memory.total"]),
        "utilization_gpu_pct": _number(raw["utilization.gpu"]),
        "power_draw_w": _number(raw["power.draw"]),
        "temperature_c": _number(raw["temperature.gpu"]),
        "sm_clock_mhz": _number(raw["clocks.sm"]),
    }


class TelemetrySampler:
    def __init__(self, uuid: str, path: Path, interval_s: float):
        self.uuid = uuid
        self.path = path
        self.interval_s = max(0.2, interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="meshtitan-telemetry", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            while not self._stop.is_set():
                sample = sample_gpu(self.uuid)
                if sample is not None:
                    self._samples.append(sample)
                    f.write(json.dumps(sample, sort_keys=True) + "\n")
                    f.flush()
                self._stop.wait(self.interval_s)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s + 2)
        return summarize(self._samples)


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def peak(key: str) -> float | None:
        vals = [float(s[key]) for s in samples if s.get(key) is not None]
        return max(vals) if vals else None

    def mean(key: str) -> float | None:
        vals = [float(s[key]) for s in samples if s.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "samples": len(samples),
        "peak_memory_used_mib": peak("memory_used_mib"),
        "peak_temperature_c": peak("temperature_c"),
        "peak_power_draw_w": peak("power_draw_w"),
        "mean_power_draw_w": mean("power_draw_w"),
        "mean_gpu_utilization_pct": mean("utilization_gpu_pct"),
        "peak_sm_clock_mhz": peak("sm_clock_mhz"),
    }
