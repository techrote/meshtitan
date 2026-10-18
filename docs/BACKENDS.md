# Backend contracts

## Multi-view and refinement extensions

Windows setup also configures `hunyuan_multiview` and `ultrashape`; see [Windows setup](WINDOWS.md)
for exact pins, compatibility adaptations, controls and operating limits. All four backends share the same
single-TITAN queue and isolated-process contract.

Multi-view requests use `backend="hunyuan_multiview"`, `asset_id` equal to the front image ID and
`view_assets={"front": "<asset-id>", "left": "<asset-id>", ...}`. Front plus at least one other canonical
view is required; supported keys are front/left/back/right. Parameters are the same as Hunyuan shape,
with CPU offload enabled by default. View IDs persist in job records, and all source hashes are written
to the execution input manifest.

Refinement requests use `backend="ultrashape"`, `parent_job_id` of a succeeded job with an available mesh,
and the same primary `asset_id` as that job. Defaults: steps 30, guidance_scale 5, seed 1234,
octree_resolution 256, num_chunks 2048, num_latents 8192, normalize_scale 0.99,
remove_background true, cpu_offload true, experimental_resolution false. There are no arbitrary mesh paths
in the public API. The source mesh remains intact; its SHA-256 is recorded with the child job.

UltraShape uses a per-job compatibility copy of reviewed source, FP16 and PyTorch attention. It records
the adapter and source hashes and checkpoint hash. Its geometry is normalised to Y-up; TripoSR source
meshes are rotated before refinement. Multi-view source jobs supply their front image for refinement.

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
| `box_v` | 1.01 | half-width of sampled 3D bounds |
| `mc_level` | 0.0 | marching-cubes surface threshold |
| `border_ratio` | 0.15 | empty image border around the recentered foreground |

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
| `density_threshold` | 25.0 | activated-density isovalue passed to upstream mesh extraction |
| `background_gray` | 0.5 | background fill after removal: 0 black, 1 white |
| `experimental_resolution` | false | opt in to resolutions outside 64–512 (16–1024 allowed) |

Higher density thresholds generally contract the occupied region and can remove thin parts; lower thresholds can
expand it and add floating fragments. Threshold is independent of grid resolution. Extreme thresholds can yield no
surface. Foreground ratio and background gray affect only the background-removal preprocessing path.

TripoSR is feed-forward; diffusion steps, classifier-free guidance and sampling seeds do not apply. Renderer ray
sample counts and camera controls affect rendered images, not this geometry extraction path.

Texture baking, rendered video and vertex-colour extraction are intentionally omitted. MeshTitan calls upstream
`extract_mesh(..., has_vertex_color=False)` so the geometry-only path does not perform the additional per-vertex
colour query.

## Output formats

Both backends support mesh export through Trimesh to:

- GLB
- OBJ
- PLY
- STL

GLB is the qualification default.
Hunyuan also accepts `experimental_resolution=false` by default, with the same 16–1024 custom range and explicit
opt-in outside 64–512. Dense-grid costs grow approximately cubically with resolution; higher-resolution hardware
qualification is still pending. Neither backend's default resolution or GPU contract is changed by this option.
Non-GLB exports also save `mesh.glb` for browser previews and thumbnail rendering.
All exposed backend parameters are type/range checked before any job in a batch is queued.

## Result evidence

Successful jobs include:

- phase timing from the model runner;
- PyTorch peak CUDA allocation/reservation;
- vertex and face counts;
- runtime wall clock;
- NVML/nvidia-smi telemetry summary;
- physical GPU identity and driver;
- exact upstream Git commit configured for the backend.

TripoSR results explicitly record `vertex_colors: false` in their parameter provenance.

A successful process exit is insufficient if the mesh artifact is missing or empty.
