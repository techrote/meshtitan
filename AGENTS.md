# Agent instructions

Read `docs/RAG.md` before implementation work.

This repository is currently Titan RTX-specific. Preserve these boundaries unless an issue explicitly changes them:

- Hunyuan3D 2.1 means shape generation only.
- TripoSR means geometry generation only.
- Do not add PBR/paint dependencies to the core runtime.
- The broker must stay free of PyTorch/CUDA Python dependencies.
- Backends run in isolated subprocesses with one GPU exposed through `CUDA_VISIBLE_DEVICES`.
- Do not broaden the default GPU contract away from TITAN RTX while MT-004 is pending.
- GitHub-hosted CI is software evidence only. Never claim physical performance or VRAM acceptance without operator
  evidence from the target machine.
- Keep upstream model repositories pinned. Any pin update needs explicit compatibility review and its own issue/PR.
- Preserve durable provenance and logs when changing job execution.

Run before requesting review:

```bash
python -m compileall -q src
ruff check src tests
pytest -q
```
