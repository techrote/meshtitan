# TITAN RTX physical acceptance

Cloud CI can qualify control flow, serialization, parameter validation and repository pinning. It cannot qualify
the physical TITAN RTX. MT-004 remains open until the following is run on the target machine.

## Preflight

1. Stop other CUDA workloads.
2. Run `nvidia-smi` and record the complete device table.
3. Set `gpu.selector` in `meshtitan.toml` to the TITAN RTX UUID.
4. Run:

```powershell
.\.venv-runtime\Scripts\meshtitan.exe --config meshtitan.toml doctor
```

Acceptance requires the intended TITAN RTX, >=22,000 MiB VRAM and the exact pinned backend commits.

## Frozen corpus

Initial smoke/benchmark inputs:

- TripoSR `examples/chair.png`
- TripoSR `examples/teapot.png`
- TripoSR `examples/robot.png`

Copy/hash the exact files before measurement. Additional user images may be evaluated later but must not replace
the frozen baseline.

## TripoSR baseline

- chunk size: 8192
- marching cubes: 256
- background removal: fixed consistently
- output: GLB

Warm the system once, then retain at least three measured jobs per image.

## Hunyuan baseline and resolution ladder

Start with:

- steps: 30
- guidance: 5.0
- seed: 1234
- octree: 256
- chunks: 8000
- CPU offload: false

Only after a clean 256 run, test 384. Only after 384 is clean and VRAM margin is understood, test 512.

Do not reduce quality settings automatically on OOM. Record the failure as evidence, then begin a separately named
fallback experiment.

## Evidence to retain

Every accepted run must retain the complete job directory:

- `job.json`
- `runner-result.json`
- `gpu-telemetry.jsonl`
- stdout/stderr logs
- generated mesh

Also record host CPU/RAM, OS, NVIDIA driver, display use, PCIe link and any power-limit change.

## Geometry review

Validate each mesh independently of the generating application:

- file parses successfully;
- nonzero vertices/faces;
- finite coordinates;
- bounding box is nondegenerate;
- connected-component count;
- obvious floating fragments/non-manifold regions;
- fixed-view renders for silhouette and thin-feature comparison.

Performance acceptance and geometry-quality judgment are separate conclusions.

## Completion language

Allowed:

- `software-qualified / physical gate held`
- `TITAN RTX baseline passed`
- `384 passed; 512 OOM at recorded settings`

Not allowed without evidence:

- `supports the TITAN RTX`
- `512 works`
- `Hunyuan is faster/slower than TripoSR`
