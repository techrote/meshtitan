# meshtitan

A small, auditable runtime for **geometry generation on one NVIDIA TITAN RTX 24 GB**.

See [the current project checkpoint](docs/PROJECT-STATE.md) for implemented features, verified local runs
and the remaining hardware-qualification limits.

Current backends:

- **Hunyuan3D 2.1 shape** — image to high-detail mesh, no Hunyuan Paint/PBR pipeline.
- **TripoSR** — image to mesh, no texture baking by default.
- **Hunyuan3D-2mv** — labelled front/side/back views combined into one shape.
- **UltraShape** — image-guided refinement of a completed mesh, with a TITAN FP16 compatibility profile.

On Windows, use `windows.bat setup`, then `windows.bat run`. See [Windows setup and controls](docs/WINDOWS.md)
for prerequisites, multi-view inputs, refinement parameters and hardware verification limits.

The runtime intentionally keeps model environments outside the broker process. Hunyuan and TripoSR can use their
own pinned Python/PyTorch dependencies while sharing one job API, one Titan GPU identity contract and one evidence
format.

## Status

Software preparation is implemented under MT-001/002/003. **Physical TITAN RTX acceptance remains MT-004 and cannot
be completed by GitHub-hosted CI.**

Upstream pins:

| Backend | Repository | Commit |
|---|---|---|
| Hunyuan shape | `Tencent-Hunyuan/Hunyuan3D-2.1` | `82920d643c0dc2f7bfd7255f45f62d386edfe60c` |
| TripoSR | `VAST-AI-Research/TripoSR` | `107cefdc244c39106fa830359024f6a2f1c78871` |
| Hunyuan multi-view | `Tencent-Hunyuan/Hunyuan3D-2` | `f8db63096c8282cb27354314d896feba5ba6ff8a` |
| UltraShape | `PKU-YuanGroup/UltraShape-1.0` | `5e8dcef05df101ab00ab6cd5fdd0ed0c74fbca66` |

## Bootstrap

### Windows

Requirements: Git, Python 3.10 and a CUDA 12.4-compatible NVIDIA driver. Fresh TripoSR installs also need
CUDA Toolkit 12.4 and MSVC 14.38; see [the complete Windows instructions](docs/WINDOWS.md).

```powershell
windows.bat setup
windows.bat doctor
windows.bat run
```

### Linux

```bash
./scripts/bootstrap_titan.sh
.venv-runtime/bin/meshtitan --config meshtitan.toml doctor
```

The core bootstrap scripts:

1. clone both upstream repositories into ignored `.vendor/` directories;
2. detach them at the reviewed commits above;
3. create independent runtime/Hunyuan/TripoSR virtual environments;
4. install PyTorch 2.5.1 + CUDA 12.4 wheels in the model environments;
5. install each pinned upstream dependency set;
6. create `meshtitan.toml` from the example if one does not already exist.

No model weights are committed here. Windows setup additionally installs the multi-view and refinement
environments and downloads their pinned primary weights. Core weights download on first use.
The Linux bootstrap currently prepares the original two backends; extension setup is documented for Windows.

## Run

HTTP service:

```powershell
.\.venv-runtime\Scripts\meshtitan.exe --config meshtitan.toml serve
```

Default address: `http://127.0.0.1:8088`.

Open that address in your browser for the local studio. Import multiple PNG, JPEG or WebP images, name the batch,
select Hunyuan3D 2.1 shape and/or TripoSR, and adjust each model's parameters. Jobs run sequentially in image import
order, with selected backends and Hunyuan seed variations processed for each image. A batch supports up to 200 jobs;
TripoSR runs once per image. A failed job does not stop subsequent jobs. Cancelling a batch stops its running job and
cancels its pending jobs while preserving completed outputs. Restarted jobs fail closed and retain batch membership.

The gallery groups jobs into collapsible batches with source thumbnails and cached renders of completed meshes.
Search, status filters, a queue view and adjustable tile size help browse results. Open a tile to inspect its mesh,
reuse its parameters, or add its source image to the input tray. Download individual meshes or all completed batch
meshes plus job metadata as a ZIP. GLB, OBJ, PLY and STL exports all retain a GLB copy for browser previews.
Draft, Balanced and High detail presets are available; named custom presets and current settings are saved in the
browser. High detail uses 384 resolution and remains an experiment, not a hardware qualification claim.

Hunyuan batches can sweep guidance from the main guidance value using a count and positive or negative increment
(up to 16 values, each within 0–30). Each seed variation is held fixed across its guidance sweep. The batch planner
counts the full image × seed × guidance product toward the 200-job limit; TripoSR still runs once per image.
For example, two images × two seeds × three guidance values yield 12 Hunyuan jobs, plus two TripoSR jobs if selected.

Both resolution fields accept custom integers from 16–1024. Values outside 64–512 require the per-backend
`experimental_resolution` toggle. These bounds are application limits, not upstream guarantees. Resolution 256
remains the default. Large dense grids can exhaust host RAM or VRAM; small grids can lose thin features or fail to
produce a useful surface. No resolution is reduced automatically on failure.

The 3D viewer is bundled locally (Three.js 0.180.0, MIT); no CDN or frontend build step is required. WebGL is
required for previews and mesh thumbnail rendering, while source thumbnails, generation and downloads remain
available if WebGL is unavailable. Mesh thumbnail PNGs are generated on demand in the browser and cached beside
the durable job records, independently of model execution.

Useful endpoints:

- `GET /health`
- `GET /v1/capabilities`
- `POST /v1/assets`
- `POST /v1/jobs`
- `POST /v1/batches` — `{ "name": "Batch name", "jobs": [<job request>, ...] }`, 1–200 jobs
- `POST /v1/batches/{id}/cancel`
- `GET /v1/batches/{id}/artifacts` — ZIP of completed meshes and job metadata
- `GET /v1/gallery?limit=20&offset=0` — complete groups with thumbnail/preview availability
- `GET /v1/assets/{id}/image`
- `GET /v1/jobs?limit=30` — recent jobs, newest first (maximum 100)
- `GET /v1/jobs/{id}`
- `POST /v1/jobs/{id}/cancel`
- `GET /v1/jobs/{id}/artifact`
- `GET /v1/jobs/{id}/preview` — GLB preview, including for other export formats
- `GET /v1/jobs/{id}/thumbnail` / `PUT /v1/jobs/{id}/thumbnail` — cached PNG render, up to 512 KiB / 1024 px

Batch planning checks can also be run with `node --test tests/test_batch_plan.mjs` (Node.js with ES module support).

One-shot local execution is simpler for first hardware qualification:

```powershell
.\.venv-runtime\Scripts\meshtitan.exe --config meshtitan.toml run `
  --backend hunyuan_shape `
  --input .\input.png `
  --param octree_resolution=256 `
  --param seed=1234
```

TripoSR:

```powershell
.\.venv-runtime\Scripts\meshtitan.exe --config meshtitan.toml run `
  --backend triposr `
  --input .\.vendor\TripoSR\examples\chair.png `
  --param chunk_size=8192 `
  --param mc_resolution=256
```

## Why no Hunyuan `--low_vram_mode`?

For the current geometry-only objective, it is unnecessary as the default strategy. Hunyuan's shape stage is far
smaller than the combined shape+paint pipeline. The adapter therefore runs shape normally on the 24 GB Titan and
exposes the upstream **real model CPU offload** only as `cpu_offload=true` for diagnostics/fallback.

This is intentionally different from relying on the upstream Gradio `--low_vram_mode`, which does not by itself
provide a complete live-model eviction strategy for the paint pipeline.

## Evidence

Each job retains:

```text
var/jobs/<job-id>/
  job.json
  runner-result.json
  gpu-telemetry.jsonl
  stdout.log
  stderr.log
  mesh.glb
```

The broker captures physical GPU/driver/upstream-commit provenance. Model runners capture phase timings, PyTorch peak
CUDA memory and mesh vertex/face counts.

See:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/BACKENDS.md`](docs/BACKENDS.md)
- [`docs/HARDWARE-ACCEPTANCE.md`](docs/HARDWARE-ACCEPTANCE.md)
- [`docs/RAG.md`](docs/RAG.md)
