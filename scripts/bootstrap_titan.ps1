# Backwards-compatible entry point for the complete Windows setup.
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'setup_windows.ps1')
exit $LASTEXITCODE
