from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from titan_compat import cpu_marching_cubes, prepare_ultrashape


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("repo", "input", "mesh", "output", "result-json", "checkpoint"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--input-up-axis", choices=("y", "z"), default="y")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--octree-resolution", type=int, default=256)
    parser.add_argument("--num-chunks", type=int, default=2048)
    parser.add_argument("--num-latents", type=int, default=8192)
    parser.add_argument("--normalize-scale", type=float, default=0.99)
    parser.add_argument("--remove-background", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--experimental-resolution", action="store_true")
    args = parser.parse_args()
    if not 16 <= args.octree_resolution <= 1024:
        parser.error("resolution must be between 16 and 1024")
    if not 64 <= args.octree_resolution <= 512 and not args.experimental_resolution:
        parser.error("resolutions outside 64–512 require --experimental-resolution")
    repo = Path(args.repo).resolve()
    output = Path(args.output).resolve()
    compatibility = prepare_ultrashape(repo, output.parent / "compat")
    sys.path.insert(0, str(output.parent / "compat"))
    for key in ("USE_SAGEATTN", "CA_USE_SAGEATTN"):
        os.environ[key] = "0"

    import numpy as np
    import torch
    import trimesh
    from omegaconf import OmegaConf
    from PIL import Image
    from transformers import Dinov2Config
    from ultrashape.models.autoencoders.surface_extractors import MCSurfaceExtractor
    from ultrashape.pipelines import UltraShapePipeline
    from ultrashape.rembg import BackgroundRemover
    from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
    from ultrashape.utils import instantiate_from_config, voxelize_from_point

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("UltraShape requires exactly one CUDA-visible GPU")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    MCSurfaceExtractor.run = cpu_marching_cubes
    start = time.perf_counter()
    print("Loading UltraShape components on CPU (TITAN FP16 profile)", flush=True)
    cfg = OmegaConf.load(repo / "configs/infer_dit_refine.yaml").model.params
    dino_revision = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
    # Its trained weights are already in the UltraShape checkpoint; only fetch the architecture config.
    cfg.conditioner_config.params.main_image_encoder.kwargs.config = Dinov2Config.from_pretrained(
        "facebook/dinov2-large", revision=dino_revision,
    ).to_dict()
    components = {
        name: instantiate_from_config(OmegaConf.to_container(cfg[key], resolve=True)) for name, key in (
            ("vae", "vae_config"), ("model", "dit_cfg"), ("conditioner", "conditioner_config"),
            ("scheduler", "scheduler_cfg"), ("image_processor", "image_processor_cfg"),
        )
    }
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    print("Loading checkpoint weights", flush=True)
    for name, weight_key in (("vae", "vae"), ("model", "dit"), ("conditioner", "conditioner")):
        components[name].load_state_dict(weights[weight_key], strict=True)
        components[name].eval().to(dtype=torch.float16)
    del weights
    # Construct on CPU first so all three models are never resident on the GPU together.
    pipeline = UltraShapePipeline(**components, device="cpu", dtype=torch.float16)
    if args.cpu_offload:
        pipeline.enable_model_cpu_offload(device="cuda")
        pipeline.device = torch.device("cuda")
    else:
        pipeline.to("cuda", torch.float16)
    load_s = time.perf_counter() - start
    print(f"Models ready in {load_s:.1f}s; preparing reference and coarse mesh", flush=True)
    with Image.open(args.input) as source:
        image = source.convert("RGBA")
    if args.remove_background:
        image = BackgroundRemover()(image.convert("RGB"))
    mesh = trimesh.load(args.mesh, force="mesh")
    if args.input_up_axis == "z":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    loader = SharpEdgeSurfaceLoader(num_sharp_points=204800, num_uniform_points=204800)
    # CPU voxelization avoids the large CUDA unique/sort workspace.
    surface = loader(mesh, normalize_scale=args.normalize_scale)
    _, voxel_idx = voxelize_from_point(surface[:, :, :3], args.num_latents,
                                        resolution=cfg.vae_config.params.voxel_query_res)
    start = time.perf_counter()
    print("Running refinement", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        meshes, _ = pipeline(
            image=image, voxel_cond=voxel_idx.to("cuda"),
            generator=torch.Generator("cuda").manual_seed(args.seed),
            num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
            octree_resolution=args.octree_resolution, num_chunks=args.num_chunks,
            box_v=1.0, mc_level=0.0, mc_algo="mc",
        )
    if not meshes or meshes[0] is None:
        raise RuntimeError("UltraShape returned no mesh; see extraction log")
    mesh = meshes[0]
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise RuntimeError("UltraShape produced empty or nonfinite geometry")
    mesh.export(output)
    if output.suffix.lower() != ".glb":
        mesh.export(output.with_suffix(".glb"))
    checkpoint_hash = hashlib.sha256()
    with Path(args.checkpoint).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            checkpoint_hash.update(chunk)
    result = {
        "backend": "ultrashape", "precision": "float16", "compatibility": compatibility,
        "checkpoint_sha256": checkpoint_hash.hexdigest(), "input_up_axis": args.input_up_axis,
        "conditioning_config_revision": dino_revision,
        "device_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda, "load_seconds": load_s,
        "inference_seconds": time.perf_counter() - start,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1048576,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 1048576,
        "vertices": len(mesh.vertices), "faces": len(mesh.faces),
        "parameters": {key: value for key, value in vars(args).items() if key not in {
            "repo", "input", "mesh", "output", "result_json", "checkpoint", "input_up_axis",
        }},
    }
    Path(args.result_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
