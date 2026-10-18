param([switch]$ExtensionsOnly)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $Root
# Updating a Windows console entry point while it is running fails halfway through pip uninstall.
$activeServer = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -like '*meshtitan*serve*' -and
    ($_.CommandLine -like "*$Root*" -or $_.CommandLine -like '*-m meshtitan*')
}
if ($activeServer) { throw 'Stop the MeshTitan server (Ctrl+C in its terminal), then rerun windows.bat setup. Existing jobs and models are preserved.' }

function Invoke-Checked([string]$Command, [string[]]$Arguments) {
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command failed with exit code $LASTEXITCODE" }
}
function Ensure-Repo([string]$Url, [string]$Path, [string]$Commit) {
    if (-not (Test-Path -LiteralPath $Path)) { Invoke-Checked 'git' @('clone', $Url, $Path) }
    $head = (& git -C $Path rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Not a Git checkout: $Path" }
    if ($head -ne $Commit) {
        $dirty = & git -C $Path status --porcelain
        if ($dirty) { throw "Preserve local edits in $Path before setup can select its pinned commit." }
        Invoke-Checked 'git' @('-C', $Path, 'fetch', 'origin', $Commit)
        Invoke-Checked 'git' @('-C', $Path, 'checkout', '--detach', $Commit)
    }
    if ((& git -C $Path rev-parse HEAD).Trim() -ne $Commit) { throw "Pin verification failed: $Path" }
}

# Reuse the base interpreter of an existing environment, including uv-managed Python.
$Python = $null
foreach ($candidate in @('.venv-runtime\Scripts\python.exe', '.venv-hunyuan\Scripts\python.exe')) {
    if (Test-Path -LiteralPath $candidate) {
        $Python = (& $candidate -c 'import sys; print(sys._base_executable)').Trim()
        break
    }
}
if (-not $Python) {
    $Python = (& py -3.10 -c 'import sys; print(sys.executable)').Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.10 and the Python launcher, then rerun setup.' }
}
Invoke-Checked $Python @('-c', 'import sys; assert sys.version_info[:2] == (3,10), "Python 3.10 is required"')

function Ensure-Environment([string]$Path) {
    if (-not (Test-Path -LiteralPath "$Path\Scripts\python.exe")) {
        Invoke-Checked $Python @('-m', 'venv', $Path)
    }
    Invoke-Checked "$Path\Scripts\python.exe" @('-m', 'pip', 'install', '--upgrade', 'pip', 'wheel', 'setuptools')
}
function Install-ShapeEnvironment([string]$Path) {
    Ensure-Environment $Path
    Invoke-Checked "$Path\Scripts\python.exe" @('-m', 'pip', 'install', 'torch==2.5.1+cu124', 'torchvision==0.20.1+cu124', '--extra-index-url', 'https://download.pytorch.org/whl/cu124')
    Invoke-Checked "$Path\Scripts\python.exe" @('-m', 'pip', 'install', '-r', 'requirements-inference.txt')
    Invoke-Checked "$Path\Scripts\python.exe" @('-m', 'pip', 'check')
}

New-Item -ItemType Directory -Force '.vendor', 'var\models' | Out-Null
Ensure-Environment '.venv-runtime'
Invoke-Checked '.venv-runtime\Scripts\python.exe' @('-m', 'pip', 'install', '-e', '.[dev]')

if (-not $ExtensionsOnly) {
    Ensure-Repo 'https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git' '.vendor\Hunyuan3D-2.1' '82920d643c0dc2f7bfd7255f45f62d386edfe60c'
    Ensure-Repo 'https://github.com/VAST-AI-Research/TripoSR.git' '.vendor\TripoSR' '107cefdc244c39106fa830359024f6a2f1c78871'
    # Leave working, previously configured core environments intact.
    if (-not (Test-Path '.venv-hunyuan\Scripts\python.exe')) { Install-ShapeEnvironment '.venv-hunyuan' }
    if (-not (Test-Path '.venv-triposr\Scripts\python.exe')) {
        Install-ShapeEnvironment '.venv-triposr'
        Invoke-Checked '.venv-triposr\Scripts\python.exe' @('-m', 'pip', 'install', 'imageio[ffmpeg]', 'moderngl==5.10.0', 'setuptools==79.0.1', 'ninja==1.13.2', 'cmake==3.31.10', 'pybind11==3.1.0', 'scikit-build-core==1.1.1')
        Ensure-Repo 'https://github.com/tatsy/torchmcubes.git' '.vendor\torchmcubes' '879926d0ef58e6ce0ac2630fdecb5e53af7ed3ff'
        # Native CUDA extension: use the installed CUDA 12.4 toolkit and MSVC 14.38.
        $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Install Visual Studio Build Tools with MSVC 14.38 and CUDA Toolkit 12.4. See docs/WINDOWS.md.' }
        $installation = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
        $vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
        & cmd.exe /d /s /c "`"`"$vcvars`" -vcvars_ver=14.38 >nul && set`"" | ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
        }
        if ($LASTEXITCODE -ne 0) { throw 'MSVC 14.38 environment setup failed.' }
        $env:CUDA_PATH = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA\v12.4'
        if (-not (Test-Path "$env:CUDA_PATH\bin\nvcc.exe")) { throw 'CUDA Toolkit 12.4 was not found.' }
        $env:CUDA_HOME = $env:CUDA_PATH
        $env:PATH = "$Root\.venv-triposr\Scripts;$env:CUDA_PATH\bin;$env:PATH"
        $env:TORCH_CUDA_ARCH_LIST = '7.5'
        $env:CMAKE_ARGS = '-DCMAKE_CUDA_ARCHITECTURES=75'
        $env:CMAKE_GENERATOR = 'Ninja'
        $env:CMAKE_BUILD_PARALLEL_LEVEL = '4'
        Invoke-Checked '.venv-triposr\Scripts\python.exe' @('-m', 'pip', 'install', '--no-build-isolation', '--no-deps', '.vendor\torchmcubes')
    }
}

Ensure-Repo 'https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git' '.vendor\Hunyuan3D-2' 'f8db63096c8282cb27354314d896feba5ba6ff8a'
Ensure-Repo 'https://github.com/PKU-YuanGroup/UltraShape-1.0.git' '.vendor\UltraShape-1.0' '5e8dcef05df101ab00ab6cd5fdd0ed0c74fbca66'
Install-ShapeEnvironment '.venv-hunyuan-mv'
Install-ShapeEnvironment '.venv-ultrashape'
Invoke-Checked '.venv-hunyuan-mv\Scripts\python.exe' @('scripts\setup_models.py')
Invoke-Checked '.venv-runtime\Scripts\python.exe' @('scripts\configure_windows.py')
Invoke-Checked '.venv-runtime\Scripts\meshtitan.exe' @('--config', 'meshtitan.toml', 'doctor')
Write-Host 'Setup complete. Run windows.bat run to open the local service.'
