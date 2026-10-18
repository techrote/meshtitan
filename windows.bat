@echo off
setlocal
cd /d "%~dp0"
if /i "%~1"=="setup" goto setup
if /i "%~1"=="doctor" goto doctor
if /i "%~1"=="help" goto help
if "%~1"=="" goto run
if /i "%~1"=="run" goto run
echo Unknown command: %~1
goto help
:setup
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_windows.ps1"
exit /b %errorlevel%
:doctor
if not exist ".venv-runtime\Scripts\meshtitan.exe" goto missing
".venv-runtime\Scripts\python.exe" -m meshtitan --config meshtitan.toml doctor
exit /b %errorlevel%
:run
if not exist ".venv-runtime\Scripts\meshtitan.exe" goto missing
if not exist "meshtitan.toml" goto missing
echo MeshTitan Studio: http://127.0.0.1:8088/
".venv-runtime\Scripts\python.exe" -m meshtitan --config meshtitan.toml serve
exit /b %errorlevel%
:missing
echo Run windows.bat setup first. See docs\WINDOWS.md for prerequisites.
exit /b 1
:help
echo Usage: windows.bat [setup^|run^|doctor^|help]
echo Setup installs isolated geometry backends and downloads pinned model weights.
exit /b 0
