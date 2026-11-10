# Setup for Windows PowerShell.
#
#   .\scripts\setup.ps1
#
# Creates a virtual environment in .venv, installs the pinned dependencies into
# it, fetches the hand landmark model, and runs the tests. Nothing is installed
# outside this folder and nothing on your system is changed.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "`n[1/4] Python" -ForegroundColor Cyan
$py = "python"
try { & $py --version } catch { throw "Python was not found on PATH. Install it from python.org first." }

if (-not (Test-Path ".venv")) {
    Write-Host "`n[2/4] Creating .venv" -ForegroundColor Cyan
    & $py -m venv .venv
} else {
    Write-Host "`n[2/4] .venv already exists, reusing it" -ForegroundColor Cyan
}

$venvPy = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "Could not find $venvPy - delete the .venv folder and run this again." }

Write-Host "`n[3/4] Installing dependencies (this pulls ~400 MB, give it a minute)" -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nInstall failed. If the message mentions mediapipe and you are on Python 3.13," -ForegroundColor Yellow
    Write-Host "see docs/INSTALL.md - the short version is: pip install `"mediapipe>=0.10.30`" opencv-python numpy" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n[4/4] Model and tests" -ForegroundColor Cyan
& $venvPy scripts\get_model.py
& $venvPy tests\test_geometry.py

Write-Host "`nDone. Start it with:" -ForegroundColor Green
Write-Host "    .venv\Scripts\activate"
Write-Host "    python -m handportal --input 0 --preview`n"
