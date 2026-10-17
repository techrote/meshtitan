from __future__ import annotations

import argparse
import gc
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
    parser.add_argument("--model-path", default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-1")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--octree-resolution", type=int, default=256)
    parser.add_argument("--num-chunks", type=int, default=8000)
    parser.add_argument("--remove-background", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "hy3dshape"))

    import torch
    from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.rembg import BackgroundRemover
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in Hunyuan backend environment")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"backend expected exactly one CUDA-visible GPU, found {torch.cuda.device_count()}; "
            "the broker must pin CUDA_VISIBLE_DEVICES"
        )

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.perf_counter()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        subfolder=args.subfolder,
        use_safetensors=False,
        device="cuda",
    )
    load_s = time.perf_counter() - t0
    if args.cpu_offload:
        pipeline.enable_model_cpu_offload(device="cuda")

    image = Image.open(args.input).convert("RGBA")
    if args.remove_background:
        image = BackgroundRemover()(image.convert("RGB"))

    generator = torch.Generator().manual_seed(args.seed)
    t1 = time.perf_counter()
    outputs = pipeline(
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        output_type="trimesh",
    )
    inference_s = time.perf_counter() - t1

    if not outputs:
        raise RuntimeError("Hunyuan returned no mesh")
    mesh = outputs[0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    t2 = time.perf_counter()
    mesh.export(output)
    export_s = time.perf_counter() - t2

    result = {
        "backend": "hunyuan_shape",
        "model_path": args.model_path,
        "subfolder": args.subfolder,
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "load_seconds": load_s,
        "inference_seconds": inference_s,
        "export_seconds": export_s,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1048576,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 1048576,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "parameters": {
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "octree_resolution": args.octree_resolution,
            "num_chunks": args.num_chunks,
            "remove_background": args.remove_background,
            "cpu_offload": args.cpu_offload,
        },
    }
    Path(args.result_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
