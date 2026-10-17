from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import uvicorn

from . import __version__
from .assets import AssetError
from .backends import backend_status
from .config import ConfigError, load_config
from .executor import JobManager
from .gpu import GPUError, get_configured_gpu
from .models import TERMINAL_STATUSES, JobRequest
from .service import create_app


def _doctor(config_path: str) -> int:
    try:
        cfg = load_config(config_path)
        gpu = get_configured_gpu(cfg.gpu)
    except (ConfigError, GPUError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2

    backends = {name: backend_status(item) for name, item in cfg.backends.items()}
    ok = all((not item.enabled) or backends[name].get("ok", False) for name, item in cfg.backends.items())
    payload = {
        "ok": ok,
        "version": __version__,
        "gpu": gpu.__dict__,
        "backends": backends,
        "data_dir": str(cfg.data_dir),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if ok else 3


def _run(config_path: str, backend: str, image: str, output_format: str, params: list[str]) -> int:
    cfg = load_config(config_path)
    image_path = Path(image).resolve()
    if not image_path.is_file():
        print(f"input image not found: {image_path}", file=sys.stderr)
        return 2

    supplied: dict[str, object] = {}
    for item in params:
        if "=" not in item:
            print(f"parameter must be KEY=VALUE: {item}", file=sys.stderr)
            return 2
        key, raw = item.split("=", 1)
        value: object
        if raw.lower() in {"true", "false"}:
            value = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        supplied[key] = value

    manager = JobManager(cfg)
    manager.start()
    try:
        data = image_path.read_bytes()
        try:
            asset = manager.assets.add(data, image_path.name, None)
        except AssetError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        record = manager.submit(
            JobRequest(
                backend=backend,
                asset_id=asset.asset_id,
                output_format=output_format,
                parameters=supplied,
            )
        )
        print(f"job {record.job_id} submitted")
        while True:
            record = manager.jobs.get(record.job_id)
            if record.status in TERMINAL_STATUSES:
                print(record.model_dump_json(indent=2))
                return 0 if record.status.value == "succeeded" else 4
            time.sleep(0.5)
    finally:
        manager.stop()


def main() -> None:
    parser = argparse.ArgumentParser(prog="meshtitan")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", default="meshtitan.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="validate GPU and backend installations")

    serve = sub.add_parser("serve", help="start the HTTP service")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)

    run = sub.add_parser("run", help="run one local image-to-mesh job")
    run.add_argument("--backend", choices=["hunyuan_shape", "triposr"], required=True)
    run.add_argument("--input", required=True)
    run.add_argument("--format", default="glb", choices=["glb", "obj", "ply", "stl"])
    run.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")

    args = parser.parse_args()
    try:
        if args.command == "doctor":
            raise SystemExit(_doctor(args.config))
        cfg = load_config(args.config)
        if args.command == "serve":
            uvicorn.run(create_app(cfg), host=args.host or cfg.host, port=args.port or cfg.port)
            return
        if args.command == "run":
            raise SystemExit(_run(args.config, args.backend, args.input, args.format, args.param))
    except (ConfigError, GPUError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
