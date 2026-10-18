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
    parser.add_argument("--box-v", type=float, default=1.01)
    parser.add_argument("--mc-level", type=float, default=0.0)
    parser.add_argument("--border-ratio", type=float, default=0.15)
    parser.add_argument("--remove-background", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--experimental-resolution", action="store_true")
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--views-json")
    args = parser.parse_args()
    if not 16 <= args.octree_resolution <= 1024:
        parser.error("octree resolution must be between 16 and 1024")
    if not 64 <= args.octree_resolution <= 512 and not args.experimental_resolution:
        parser.error("resolutions outside 64–512 require --experimental-resolution")

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo if args.multiview else repo / "hy3dshape"))

    import torch
    if args.multiview:
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    else:
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

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.perf_counter()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        subfolder=args.subfolder,
        use_safetensors=args.multiview,
        device="cpu" if args.cpu_offload else "cuda",
        dtype=torch.float16,
    )
    load_s = time.perf_counter() - t0
    pipeline.image_processor.border_ratio = args.border_ratio
    if args.cpu_offload:
        # The pinned 2mv pipeline's offload method expects a registry its constructor omits.
        if not hasattr(pipeline, "components"):
            pipeline.components = {name: getattr(pipeline, name) for name in ("conditioner", "model", "vae")}
        pipeline.enable_model_cpu_offload(device="cuda")
        pipeline.device = torch.device("cuda")

    remover = BackgroundRemover() if args.remove_background else None

    def load_image(path):
        with Image.open(path) as source:
            image = source.convert("RGBA")
        return remover(image.convert("RGB")) if remover else image

    if args.multiview:
        if not args.views_json:
            parser.error("multi-view requires --views-json")
        paths = json.loads(Path(args.views_json).read_text(encoding="utf-8"))
        if "front" not in paths or len(paths) < 2 or set(paths) - {"front", "left", "back", "right"}:
            parser.error("expected front and at least one additional labelled view")
        image = {name: load_image(path) for name, path in paths.items()}
    else:
        image = load_image(args.input)

    generator = torch.Generator().manual_seed(args.seed)
    t1 = time.perf_counter()
    outputs = pipeline(
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        box_v=args.box_v,
        mc_level=args.mc_level,
        output_type="trimesh",
    )
    inference_s = time.perf_counter() - t1

    if not outputs or outputs[0] is None:
        raise RuntimeError("Hunyuan returned no mesh")
    mesh = outputs[0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    t2 = time.perf_counter()
    mesh.export(output)
    if output.suffix.lower() != ".glb":
        mesh.export(output.with_suffix(".glb"))
    export_s = time.perf_counter() - t2

    result = {
        "backend": "hunyuan_multiview" if args.multiview else "hunyuan_shape",
        "precision": "float16",
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
            "box_v": args.box_v,
            "mc_level": args.mc_level,
            "border_ratio": args.border_ratio,
            "experimental_resolution": args.experimental_resolution,
        },
    }
    Path(args.result_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
