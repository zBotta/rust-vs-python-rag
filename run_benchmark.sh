#!/usr/bin/env bash
set -euo pipefail
# Set DISABLE_SSL_VERIFY=1 if behind a corporate proxy with SSL inspection:
#   DISABLE_SSL_VERIFY=1 ./run_benchmark.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Run Python pipeline
echo "==> Running Python pipeline..."
uv run python -m python_pipeline.pipeline

# 2. Wait 10 seconds cool-down
echo "Waiting 10 seconds for LLM server to drain..."
sleep 10

# 3. Run Rust pipeline
echo "==> Running Rust pipeline..."
RUST_BIN="./target/release/rust_pipeline"
if [ ! -f "$RUST_BIN" ]; then
    echo "Rust binary not found. Building now..."
    cargo build --release --manifest-path rust_pipeline/Cargo.toml
fi
"$RUST_BIN"

# 4. Run report generator
echo "==> Generating report..."
uv run python report/generate_report.py

echo ""
echo "Benchmark complete. See benchmark_report.md for results."
