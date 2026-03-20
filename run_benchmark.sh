#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Run Python pipeline
echo "==> Running Python pipeline..."
python -m python_pipeline.pipeline

# 2. Wait 10 seconds cool-down
echo "Waiting 10 seconds for LLM server to drain..."
sleep 10

# 3. Run Rust pipeline
echo "==> Running Rust pipeline..."
./rust_pipeline/target/release/rust_pipeline

# 4. Run report generator
echo "==> Generating report..."
python report/generate_report.py

echo ""
echo "Benchmark complete. See benchmark_report.md for results."
