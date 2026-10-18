# Windows setup and use

Run from the project directory:

```bat
windows.bat setup
windows.bat doctor
windows.bat run
```

Double-clicking `windows.bat` or `run_windows.bat` starts the service in its console.
`setup_windows.bat` runs setup. Open http://127.0.0.1:8088/ once the service reports ready.
Stop it with Ctrl+C before rerunning setup; Windows locks a running console entry point.
The launcher uses the project's Python environment, regardless of PATH or current directory.

## Prerequisites

- NVIDIA TITAN RTX with the existing SM75 / 22 GiB minimum contract and a CUDA 12.4-compatible driver.
- Python 3.10 with the Windows Python launcher (`py -3.10`), or an existing MeshTitan environment.
  Existing uv-managed environments are supported without requiring uv on PATH.
- Git and internet access to PyPI, PyTorch and Hugging Face for initial setup.
- For a **fresh TripoSR install**, CUDA Toolkit 12.4 and Visual Studio 2022 Build Tools with
  MSVC v143 **14.38** and the Windows SDK. Setup builds pinned torchmcubes for SM75 using Ninja.
  These system tools must already be installed; setup reports missing prerequisites.
- Allow disk space for four separate environments and model caches. The new primary weights alone
  are approximately 4.93 GB (multi-view) and 7.37 GB (UltraShape), plus package downloads and temporary files.
  UltraShape model loading also uses substantial system RAM; 32 GB or more is advisable.

Setup preserves an existing `meshtitan.toml`, its GPU UUID, model environments and job history.
On first setup it selects the unique TITAN RTX by UUID, including machines with another GPU at index 0.
It adds missing backend tables but does not overwrite custom backend settings. If those tables already
exist but point elsewhere, review them manually. To install just the extensions on an existing setup:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -ExtensionsOnly
```

The setup is repeatable, stops on command failures, and never replaces a different upstream commit
when that checkout has local edits. Core model pins remain unchanged. Core environments that already
exist are preserved; remove/repair an incomplete environment deliberately rather than expecting setup
to reset it. New environments use `requirements-inference.txt`; training, Paint/PBR, FlashAttention,
SageAttention, DeepSpeed and CUDA Python dependencies are excluded from the broker.

## Multi-view generation

Choose **Multi-view · one object** in Input mode. Import 2–4 images of the same object and assign a
unique direction to each: **front** is required, with any of **left**, **back**, **right**. Assignment
initially follows import order; check the labels before generating. These are canonical quarter-turn
views, not arbitrary camera angles. Use consistent pose, framing and object proportions.

Hunyuan3D-2mv is a separate checkpoint from the single-image Hunyuan3D 2.1 model. The Hunyuan sampling,
geometry, seed variation and guidance sweep controls apply to either input mode. All views are fused
into each generated mesh. CPU offload is enabled when switching to multi-view mode. This is generative
reconstruction and does not establish real-world scale or scanning accuracy.

## UltraShape refinement

Open **UltraShape refinement settings**, then click **Refine with UltraShape** on a completed mesh.
The result is a new queued job linked to the original job, using its reference image and coarse mesh.
The original mesh is preserved. Multi-view refinement uses the front reference image. TripoSR inputs
are rotated from Z-up to Y-up before refinement; UltraShape results use Y-up. Output coordinates are
normalised, so do not assume dimensions or transforms match the original asset.

| Setting | Effect |
| --- | --- |
| Sampling steps | More denoising iterations; longer runtime, with diminishing quality gains. |
| Guidance | Greater adherence to the image; extremes can exaggerate or damage geometry. |
| Seed | Controls diffusion and mesh sampling randomness; useful for comparisons. |
| Mesh resolution | Extraction grid density; more potential surface detail and much greater memory/time cost. |
| Decoder chunk size | Queries processed together; smaller values reduce peak memory and can slow extraction. |
| Geometry tokens | Spatial conditioning/latent capacity; more tokens may preserve finer structure but increase attention memory and computation substantially. |
| Normalisation scale | How tightly the coarse mesh fits inside the normalised cube; too small shrinks the conditioning, too close to the boundary can clip it. |
| CPU offload | Moves the image encoder, denoiser and VAE to the GPU when needed. Recommended for 24 GB; needs host RAM and adds transfers. |
| Remove background | Segments the reference; disable to preserve an existing good alpha mask. |
| Experimental resolution | Required outside 64–512; permits 16–1024 with no automatic quality fallback. |

Defaults are 30 steps, guidance 5, 8192 tokens, resolution 256 and chunk size 2048 with CPU offload.
Higher token counts and resolutions require separate operator testing. Refinement can invent detail;
compare the original and refined previews rather than assuming more polygons mean greater accuracy.

## TITAN compatibility and provenance

The new source pins and immutable primary model revisions are in `meshtitan.toml.example` and setup scripts.
UltraShape's upstream example uses BF16 and FlashAttention; our runner instead uses FP16, PyTorch SDPA,
CPU voxel preparation and scikit-image marching cubes. Its variable-length attention adapter preserves
independent sequence boundaries and chunks queries. No FlashAttention or SageAttention installation is needed.
FlashVDM is the upstream decoder algorithm's name; its presence does not mean FlashAttention is installed.

The UltraShape vendor checkout stays intact. Each job receives a compatibility copy of its Python source,
with reviewed source hashes, adapted hashes and the adapter hash in `compat/compatibility.json`.
An unexpected source version fails closed. The unused VAE trainer import is excluded, and surface
sampling is seeded. DINO's configuration is pinned; its trained weights come from the UltraShape checkpoint.
The runner records checkpoint SHA-256, effective parameters, GPU identity, timing, memory and compatibility
profile. Job records retain view asset IDs and parent job ID; `inputs.json` records all input hashes
and the source mesh hash. Stdout, stderr and GPU telemetry remain alongside every job.

These are inference compatibility adaptations. Successful local jobs are evidence for those particular
settings and inputs; they do not complete MT-004 hardware qualification or certify every exposed setting.
