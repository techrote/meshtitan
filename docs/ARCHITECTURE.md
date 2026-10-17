# Architecture

## Purpose

`meshtitan` is a deliberately small single-node runtime for geometry generation on one NVIDIA TITAN RTX.
It is not a general CUDA serving framework and it does not vendor model weights.

The broker process has no PyTorch dependency. Hunyuan3D and TripoSR execute in isolated backend
environments so dependency changes in one model do not destabilize the service itself.

## Data flow

```text
image upload/local file
        |
        v
content-addressed AssetStore
        |
        v
serialized JobManager queue
        |
        +---- Hunyuan3D 2.1 shape runner ----+
        |                                     |
        +---- TripoSR runner -----------------+--> mesh + runner-result.json
        |
        +--> nvidia-smi telemetry --> gpu-telemetry.jsonl
        |
        v
durable job.json + mesh artifact + logs
```

One job runs at a time. This is intentional: both target models can occupy significant VRAM, and the first
hardware qualification needs deterministic resource ownership more than concurrency.

## GPU identity contract

The broker probes NVIDIA hardware before accepting work. By default it requires:

- name containing `TITAN RTX`;
- at least 22,000 MiB reported VRAM;
- compute capability >= 7.5 when `nvidia-smi` exposes the field.

After first physical qualification, replace the numeric selector in `meshtitan.toml` with the Titan's stable
GPU UUID. Every backend subprocess receives that UUID as `CUDA_VISIBLE_DEVICES`, so the backend sees one
device and always uses `cuda:0`.

## Process isolation

The runtime venv contains FastAPI/Pydantic only.

Backend venvs contain their own PyTorch/model packages:

- `.venv-hunyuan`
- `.venv-triposr`

The job executor invokes runner scripts with the backend interpreter. This keeps Hunyuan and TripoSR dependency
resolution independent while maintaining one API and one provenance format.

## Persistence and restart semantics

Inputs are keyed by SHA-256 under `var/assets/`. Each job has its own directory under `var/jobs/`.

Queued or running jobs found during startup are marked **failed**, not silently resumed. Model inference is not
checkpointed, so retrying invisibly could create ambiguous evidence. A caller can explicitly submit a new job.

## Telemetry

While a backend process runs, the broker samples `nvidia-smi` into JSONL and summarizes:

- peak VRAM;
- mean GPU utilization;
- mean/peak board power;
- peak temperature;
- peak SM clock.

Backend runners separately record PyTorch peak allocated/reserved memory and phase timing.

These are complementary measurements: NVML-level used memory includes CUDA context/library allocations that
PyTorch's allocator does not own.

## Deliberate non-goals for the initial implementation

- Hunyuan Paint/PBR texture generation;
- multi-GPU model parallelism;
- K80/legacy CUDA support;
- GPU concurrency;
- distributed scheduling;
- claims of TITAN RTX performance before MT-004 physical acceptance.
