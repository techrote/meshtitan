# Backend contracts

## Hunyuan3D 2.1 shape

Pinned source:

- repository: `Tencent-Hunyuan/Hunyuan3D-2.1`
- commit: `82920d643c0dc2f7bfd7255f45f62d386edfe60c`
- model: `tencent/Hunyuan3D-2.1`
- subfolder: `hunyuan3d-dit-v2-1`

Only `hy3dshape` is used. The Hunyuan Paint model, custom paint rasterizer and PBR texture pipeline are outside
the current scope.

Default job parameters:

| Parameter | Default | Purpose |
|---|---:|---|
| `steps` | 30 | diffusion steps |
| `guidance_scale` | 5.0 | image guidance |
| `seed` | 1234 | deterministic generator seed |
| `octree_resolution` | 256 | mesh decode resolution |
| `num_chunks` | 8000 | volume decode chunk count |
| `remove_background` | true | upstream rembg path |
| `cpu_offload` | false | diagnostic fallback only |

The TITAN RTX should first qualify at 256. Then test 384 and 512 as separate hardware experiments. Do not make
512 the default merely because it fits.

`cpu_offload=true` calls the upstream pipeline's real Accelerate-backed `enable_model_cpu_offload()`. It is
available for diagnosis but is not expected to be necessary on a 24 GB TITAN RTX for shape-only inference.

## TripoSR

Pinned source:

- repository: `VAST-AI-Research/TripoSR`
- commit: `107cefdc244c39106fa830359024f6a2f1c78871`
- model: `stabilityai/TripoSR`

Default job parameters:

| Parameter | Default | Purpose |
|---|---:|---|
| `chunk_size` | 8192 | surface-query chunk size |
| `mc_resolution` | 256 | marching-cubes grid |
| `remove_background` | true | upstream rembg path |
| `foreground_ratio` | 0.85 | object framing after background removal |

Texture baking and rendered video output are intentionally omitted. The mesh uses TripoSR's normal geometry
extraction path.

## Output formats

Both backends support mesh export through Trimesh to:

- GLB
- OBJ
- PLY
- STL

GLB is the qualification default.

## Result evidence

Successful jobs include:

- phase timing from the model runner;
- PyTorch peak CUDA allocation/reservation;
- vertex and face counts;
- runtime wall clock;
- NVML/nvidia-smi telemetry summary;
- physical GPU identity and driver;
- exact upstream Git commit configured for the backend.

A successful process exit is insufficient if the mesh artifact is missing or empty.
