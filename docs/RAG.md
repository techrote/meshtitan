# meshtitan RAG context

Use this file as the compact authority map for future implementation turns.

## Project intent

`techrote/meshtitan` is a **single-TITAN-RTX geometry-generation runtime and qualification harness**.

Current target backends:

1. Hunyuan3D 2.1 **shape only**
2. TripoSR **geometry only**
3. Hunyuan3D-2mv **multi-view shape only**, isolated `.venv-hunyuan-mv`
4. UltraShape **geometry refinement only**, isolated `.venv-ultrashape`

Texture painting/PBR generation is deliberately excluded. Future height/normal/displacement baking should consume
the generated high-detail mesh as a separate deterministic mesh-processing stage.

## Authority

- repository code and open issue state are live authority;
- do not rely on old prompt SHAs without reconciling current GitHub state;
- do not claim physical GPU qualification from GitHub-hosted CI.

Upstream pins:

- Hunyuan3D 2.1: `82920d643c0dc2f7bfd7255f45f62d386edfe60c`
- TripoSR: `107cefdc244c39106fa830359024f6a2f1c78871`
- Hunyuan3D-2 (2mv): `f8db63096c8282cb27354314d896feba5ba6ff8a`
- UltraShape-1.0: `5e8dcef05df101ab00ab6cd5fdd0ed0c74fbca66`

Primary model revisions and TITAN adaptations are documented in `docs/WINDOWS.md`.
New inference dependencies use `requirements-inference.txt`. `windows.bat setup|run|doctor` is the
Windows entry point. Multi-view inputs and refinement parent links are durable job fields;
UltraShape compatibility source is retained with each job, leaving pinned vendor sources intact.

## Design invariants

- broker must not import PyTorch;
- model environments remain isolated;
- exactly one configured GPU is exposed to a backend process;
- default GPU contract expects TITAN RTX / SM75 / >=22 GiB;
- one inference job at a time;
- job records are durable;
- interrupted jobs fail closed;
- image inputs are content-addressed;
- backend parameter sets reject unknown keys;
- outputs must exist and be nonempty before success;
- every result carries upstream/GPU provenance;
- hardware acceptance is explicit and separate from software CI.

## Issues

- MT-001: runtime foundation
- MT-002: Hunyuan3D 2.1 shape adapter
- MT-003: TripoSR adapter/comparison harness
- MT-004: physical TITAN RTX qualification

## Operator entry points

Windows:

```powershell
.\scripts\bootstrap_titan.ps1
.\.venv-runtime\Scripts\meshtitan.exe --config meshtitan.toml doctor
.\.venv-runtime\Scripts\meshtitan.exe --config meshtitan.toml serve
```

Linux:

```bash
./scripts/bootstrap_titan.sh
.venv-runtime/bin/meshtitan --config meshtitan.toml doctor
.venv-runtime/bin/meshtitan --config meshtitan.toml serve
```

Direct one-shot jobs are available through `meshtitan run`.

## Current held work

Do not add K80, V100, GTX 1070, Quadro RTX 4000, multi-GPU scheduling or Hunyuan Paint to the initial acceptance
branch. They are intentionally deferred while the Titan path is being established.
