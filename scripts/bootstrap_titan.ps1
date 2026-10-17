$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$HunyuanCommit = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
$TripoCommit = "107cefdc244c39106fa830359024f6a2f1c78871"

function Ensure-Repo($Url, $Path, $Commit) {
    if (-not (Test-Path $Path)) {
        git clone $Url $Path
    }
    git -C $Path fetch origin $Commit
    git -C $Path checkout --detach $Commit
    $Head = (git -C $Path rev-parse HEAD).Trim()
    if ($Head -ne $Commit) { throw "Failed to pin $Path to $Commit (got $Head)" }
}

function Ensure-Venv($Path) {
    if (-not (Test-Path $Path)) {
        py -3.10 -m venv $Path
    }
    & "$Path\Scripts\python.exe" -m pip install --upgrade pip wheel setuptools
}

New-Item -ItemType Directory -Force ".vendor" | Out-Null

Ensure-Repo "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git" ".vendor\Hunyuan3D-2.1" $HunyuanCommit
Ensure-Repo "https://github.com/VAST-AI-Research/TripoSR.git" ".vendor\TripoSR" $TripoCommit

Ensure-Venv ".venv-runtime"
& ".venv-runtime\Scripts\python.exe" -m pip install -e ".[dev]"

Ensure-Venv ".venv-hunyuan"
& ".venv-hunyuan\Scripts\python.exe" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
& ".venv-hunyuan\Scripts\python.exe" -m pip install -r ".vendor\Hunyuan3D-2.1\hy3dshape\requirements.txt"

Ensure-Venv ".venv-triposr"
& ".venv-triposr\Scripts\python.exe" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
$env:TORCH_CUDA_ARCH_LIST = "7.5"
& ".venv-triposr\Scripts\python.exe" -m pip install -r ".vendor\TripoSR\requirements.txt"

if (-not (Test-Path "meshtitan.toml")) {
    Copy-Item "meshtitan.toml.example" "meshtitan.toml"
}

Write-Host ""
Write-Host "Bootstrap complete."
Write-Host "Next: .\.venv-runtime\Scripts\meshtitan.exe --config meshtitan.toml doctor"
Write-Host "Then: .\.venv-runtime\Scripts\meshtitan.exe --config meshtitan.toml serve"
