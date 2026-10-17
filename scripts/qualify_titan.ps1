param(
    [string]$Input = ".vendor\TripoSR\examples\chair.png"
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$MT = ".\.venv-runtime\Scripts\meshtitan.exe"

& $MT --config meshtitan.toml doctor

Write-Host "=== TripoSR baseline ==="
& $MT --config meshtitan.toml run --backend triposr --input $Input --format glb `
  --param chunk_size=8192 --param mc_resolution=256

Write-Host "=== Hunyuan shape 256 baseline ==="
& $MT --config meshtitan.toml run --backend hunyuan_shape --input $Input --format glb `
  --param steps=30 --param guidance_scale=5.0 --param octree_resolution=256 `
  --param num_chunks=8000 --param seed=1234

Write-Host "Baseline qualification completed. Review var\jobs\*/job.json and GPU telemetry before attempting 384/512."
