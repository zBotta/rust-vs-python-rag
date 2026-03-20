# run_benchmark.ps1 — Orchestrate the full benchmark on Windows.
#
# Usage: .\run_benchmark.ps1
# Assumes setup.ps1 has already been run and .venv exists.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Activate the Python virtual environment
$activateScript = ".\.venv\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Error "Virtual environment not found. Run .\setup.ps1 first."
    exit 1
}
. $activateScript

# ---------------------------------------------------------------------------
# 1. Run Python pipeline
# ---------------------------------------------------------------------------
Write-Host "==> Running Python pipeline..." -ForegroundColor Cyan
python -m python_pipeline.pipeline

# ---------------------------------------------------------------------------
# 2. Cool-down between pipeline runs (Requirement 7.4)
# ---------------------------------------------------------------------------
Write-Host "Waiting 10 seconds for LLM server to drain..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# ---------------------------------------------------------------------------
# 3. Run Rust pipeline
# ---------------------------------------------------------------------------
Write-Host "==> Running Rust pipeline..." -ForegroundColor Cyan

$rustBinary = ".\rust_pipeline\target\release\rust_pipeline.exe"
if (-not (Test-Path $rustBinary)) {
    Write-Error "Rust binary not found at '$rustBinary'. Run .\setup.ps1 first."
    exit 1
}
& $rustBinary

# ---------------------------------------------------------------------------
# 4. Generate report
# ---------------------------------------------------------------------------
Write-Host "==> Generating report..." -ForegroundColor Cyan
python report\generate_report.py

Write-Host ""
Write-Host "Benchmark complete. See benchmark_report.md for results." -ForegroundColor Green
