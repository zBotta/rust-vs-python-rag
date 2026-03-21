# run_benchmark.ps1 — Orchestrate the full benchmark on Windows.
#
# Usage: .\run_benchmark.ps1
# Assumes setup.ps1 has already been run.
#
# Set DISABLE_SSL_VERIFY=1 if behind a corporate proxy with SSL inspection:
#   $env:DISABLE_SSL_VERIFY=1; .\run_benchmark.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# ---------------------------------------------------------------------------
# 1. Run Python pipeline
# ---------------------------------------------------------------------------
Write-Host "==> Running Python pipeline..." -ForegroundColor Cyan
uv run python -m python_pipeline.pipeline

# ---------------------------------------------------------------------------
# 2. Cool-down between pipeline runs (Requirement 7.4)
# ---------------------------------------------------------------------------
Write-Host "Waiting 10 seconds for LLM server to drain..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# ---------------------------------------------------------------------------
# 3. Run Rust pipeline
# ---------------------------------------------------------------------------
Write-Host "==> Running Rust pipeline..." -ForegroundColor Cyan

$rustBinary = ".\target\release\rust_pipeline.exe"
if (-not (Test-Path $rustBinary)) {
    Write-Host "Rust binary not found. Building now..." -ForegroundColor Yellow
    cargo build --release --manifest-path rust_pipeline\Cargo.toml
}
& $rustBinary

# ---------------------------------------------------------------------------
# 4. Generate report
# ---------------------------------------------------------------------------
Write-Host "==> Generating report..." -ForegroundColor Cyan
uv run python report\generate_report.py

Write-Host ""
Write-Host "Benchmark complete. See benchmark_report.md for results." -ForegroundColor Green
