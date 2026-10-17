#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HUNYUAN_COMMIT="82920d643c0dc2f7bfd7255f45f62d386edfe60c"
TRIPOSR_COMMIT="107cefdc244c39106fa830359024f6a2f1c78871"

ensure_repo() {
  local url="$1" path="$2" commit="$3"
  if [[ ! -d "$path/.git" ]]; then
    git clone "$url" "$path"
  fi
  git -C "$path" fetch origin "$commit"
  git -C "$path" checkout --detach "$commit"
  test "$(git -C "$path" rev-parse HEAD)" = "$commit"
}

ensure_venv() {
  local path="$1"
  if [[ ! -x "$path/bin/python" ]]; then
    python3.10 -m venv "$path"
  fi
  "$path/bin/python" -m pip install --upgrade pip wheel setuptools
}

mkdir -p .vendor
ensure_repo https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git .vendor/Hunyuan3D-2.1 "$HUNYUAN_COMMIT"
ensure_repo https://github.com/VAST-AI-Research/TripoSR.git .vendor/TripoSR "$TRIPOSR_COMMIT"

ensure_venv .venv-runtime
.venv-runtime/bin/python -m pip install -e '.[dev]'

ensure_venv .venv-hunyuan
.venv-hunyuan/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
.venv-hunyuan/bin/python -m pip install -r .vendor/Hunyuan3D-2.1/hy3dshape/requirements.txt

ensure_venv .venv-triposr
.venv-triposr/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
TORCH_CUDA_ARCH_LIST=7.5 .venv-triposr/bin/python -m pip install -r .vendor/TripoSR/requirements.txt

[[ -f meshtitan.toml ]] || cp meshtitan.toml.example meshtitan.toml

echo "Bootstrap complete."
echo "Next: .venv-runtime/bin/meshtitan --config meshtitan.toml doctor"
echo "Then: .venv-runtime/bin/meshtitan --config meshtitan.toml serve"
