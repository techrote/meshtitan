"""Reviewed inference adaptations for the pinned UltraShape source, never the broker."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


def prepare_ultrashape(repo: Path, destination: Path) -> dict:
    """Keep vendor checkout intact; retain the exact adapted source with each result."""
    shutil.copytree(repo / "ultrashape", destination / "ultrashape",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    changes = {
        "surface_loaders.py": (
            "f8f75d089b8c406d0b47b0c60eca79a18d3aa65667b5bb4b796955e9fe326122",
            [("np.random.default_rng()",
              "np.random.default_rng(seed=int(np.random.randint(0, 2**32, dtype=np.int64)))")],
        ),
        "models/autoencoders/__init__.py": (
            "a84a34f6d14a75341805656725cc1ae6adbd8ae5c65da837fe04f5c2d917402f",
            [("from .vae_trainer import VAETrainer", "# Training dependencies excluded from inference")],
        ),
        "models/denoisers/dit_mask.py": (
            "4cb94699a54b4410410972bdbfdb34e5103bd83f57b017a13e59eeac9dab68a1",
            [("from flash_attn import flash_attn_varlen_func",
              "from titan_compat import varlen_attention as flash_attn_varlen_func"),
             ("torch.bfloat16", "torch.float16"),
             ("enable_flash=True", "enable_flash=False"),
             ("enable_math=False", "enable_math=True")],
        ),
        "models/autoencoders/surface_extractors.py": (
            "811fca0e27d75245c8a257b2c11654751d6366c0849651ca23491401f84bc69c",
            [("import cubvh", "# CPU marching cubes installed by the TITAN runner")],
        ),
    }
    manifest = {
        "profile": "titan-sm75-fp16-sdpa-cpu-mc-v1",
        "source_hash_encoding": "UTF-8 with LF line endings", "files": {},
    }
    for relative, (expected, replacements) in changes.items():
        path = destination / "ultrashape" / relative
        # Normalize checkout line endings before checking the reviewed source.
        source = path.read_text(encoding="utf-8")
        original_hash = hashlib.sha256(source.encode()).hexdigest()
        if original_hash != expected:
            raise RuntimeError(f"unreviewed UltraShape source: {relative} ({original_hash})")
        for old, new in replacements:
            if old not in source:
                raise RuntimeError(f"compatibility edit no longer applies: {relative}: {old}")
            source = source.replace(old, new)
        path.write_text(source, encoding="utf-8", newline="\n")
        manifest["files"][relative] = {
            "original_sha256": original_hash, "adapted_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
    manifest["adapter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (destination / "compatibility.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                     dropout_p=0.0, softmax_scale=None, causal=False):
    """Equivalent independent sequences; chunk queries to bound math-SDPA memory."""
    import torch
    import torch.nn.functional as functional

    if causal or dropout_p:
        raise ValueError("TITAN inference adapter only supports noncausal attention without dropout")
    outputs = []
    qs, ks = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for start, end, kstart, kend in zip(qs[:-1], qs[1:], ks[:-1], ks[1:], strict=True):
        key = k[kstart:kend].transpose(0, 1).unsqueeze(0)
        value = v[kstart:kend].transpose(0, 1).unsqueeze(0)
        for offset in range(start, end, 512):
            query = q[offset:min(offset + 512, end)].transpose(0, 1).unsqueeze(0)
            output = functional.scaled_dot_product_attention(query, key, value, scale=softmax_scale)
            outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def cpu_marching_cubes(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
    import numpy as np
    from skimage.measure import marching_cubes

    grid = grid_logit.detach().float().cpu().numpy()
    valid = np.isfinite(grid)
    # The hierarchical decoder leaves unqueried voxels as NaN; never treat them as geometry.
    vertices, faces, _, _ = marching_cubes(
        np.nan_to_num(grid, nan=-10000.0), level=mc_level, mask=valid,
    )
    grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
    vertices = vertices / grid_size * bbox_size + bbox_min
    return vertices, faces
