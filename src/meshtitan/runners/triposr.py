from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--model", default="stabilityai/TripoSR")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--mc-resolution", type=int, default=256)
    parser.add_argument("--foreground-ratio", type=float, default=0.85)
    parser.add_argument("--remove-background", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    import numpy as np
    import rembg
    import torch
    from PIL import Image
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in TripoSR backend environment")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"backend expected exactly one CUDA-visible GPU, found {torch.cuda.device_count()}; "
            "the broker must pin CUDA_VISIBLE_DEVICES"
        )

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.perf_counter()
    model = TSR.from_pretrained(args.model, config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(args.chunk_size)
    model.to("cuda:0")
    load_s = time.perf_counter() - t0

    image = Image.open(args.input).convert("RGB")
    preprocess_s = 0.0
    if args.remove_background:
        t_pre = time.perf_counter()
        session = rembg.new_session()
        rgba = remove_background(image, session)
        rgba = resize_foreground(rgba, args.foreground_ratio)
        array = np.array(rgba).astype(np.float32) / 255.0
        rgb = array[:, :, :3] * array[:, :, 3:4] + (1 - array[:, :, 3:4]) * 0.5
        image = Image.fromarray((rgb * 255.0).astype(np.uint8))
        preprocess_s = time.perf_counter() - t_pre

    t1 = time.perf_counter()
    with torch.no_grad():
        scene_codes = model([image], device="cuda:0")
    inference_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    meshes = model.extract_mesh(
        scene_codes,
        has_vertex_color=False,
        resolution=args.mc_resolution,
    )
    extraction_s = time.perf_counter() - t2
    if not meshes:
        raise RuntimeError("TripoSR returned no mesh")
    mesh = meshes[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    t3 = time.perf_counter()
    mesh.export(output)
    export_s = time.perf_counter() - t3

    result = {
        "backend": "triposr",
        "model": args.model,
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "load_seconds": load_s,
        "preprocess_seconds": preprocess_s,
        "inference_seconds": inference_s,
        "extraction_seconds": extraction_s,
        "export_seconds": export_s,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1048576,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 1048576,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "parameters": {
            "chunk_size": args.chunk_size,
            "mc_resolution": args.mc_resolution,
            "remove_background": args.remove_background,
            "foreground_ratio": args.foreground_ratio,
            "vertex_colors": False,
        },
    }
    Path(args.result_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
