# meshtitan

A small, auditable runtime for **geometry generation on one NVIDIA TITAN RTX 24 GB**.

Current backends:

- **Hunyuan3D 2.1 shape** — image to high-detail mesh, no Hunyuan Paint/PBR pipeline.
- **TripoSR** — image to mesh, no texture baking by default.

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

## Bootstrap

### Windows

Requirements: Git, Python 3.10, NVIDIA driver suitable for the TITAN RTX, and a CUDA 12.x toolkit/MSVC toolchain
where TripoSR's `torchmcubes` build requires it.

```powershell
.\scripts\bootstrap_titan.ps1
.\.venv-runtime\Scripts\meshtitan.exe --config meshtitan.toml doctor
```

### Linux

```bash
./scripts/bootstrap_titan.sh
.venv-runtime/bin/meshtitan --config meshtitan.toml doctor
```

The scripts:

1. clone both upstream repositories into ignored `.vendor/` directories;
2. detach them at the reviewed commits above;
3. create independent runtime/Hunyuan/TripoSR virtual environments;
4. install PyTorch 2.5.1 + CUDA 12.4 wheels in the model environments;
5. install each pinned upstream dependency set;
6. create `meshtitan.toml` from the example if one does not already exist.

No model weights are committed here. Hugging Face downloads occur on first backend use.

## Run

HTTP service:

```powershell
.\.venv-runtime\Scripts\meshtitan.exe --config meshtitan.toml serve
```

Default address: `http://127.0.0.1:8088`.

Useful endpoints:

- `GET /health`
- `GET /v1/capabilities`
- `POST /v1/assets`
- `POST /v1/jobs`
- `GET /v1/jobs/{id}`
- `POST /v1/jobs/{id}/cancel`
- `GET /v1/jobs/{id}/artifact`

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
