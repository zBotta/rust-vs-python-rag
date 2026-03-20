# Rust vs Python RAG Benchmark

A proof-of-concept benchmark comparing end-to-end latency and resource usage of a
Retrieval-Augmented Generation (RAG) pipeline implemented in Python and Rust.

---

## Prerequisites

| Requirement | Version |
|---|---|
| [uv](https://docs.astral.sh/uv/) | latest |
| Python (managed by uv) | ≥ 3.11 |
| Rust toolchain (stable) | ≥ 1.78 |
| Ollama | latest |
| System libraries | `libssl`, `libstdc++` (standard on most Linux/macOS) |

### Installing uv

uv is the Python package manager used by this project. It manages the Python version and virtual environment automatically.

**Windows:**
```powershell
winget install --id=astral-sh.uv -e
```

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other options.

### Installing Rust

Install the Rust stable toolchain (≥ 1.78) via [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
```

### Installing Ollama

Follow the instructions at <https://ollama.com/download> for your platform, then pull the model:

```bash
ollama pull llama3.2:3b
```

---

## Installation / Setup

**Linux / macOS:**
```bash
# 1. Clone the repo and enter the project directory
git clone <repo-url>
cd rust-vs-python-rag-benchmark

# 2. Run the setup script
bash setup.sh
```

**Windows (PowerShell):**
```powershell
git clone <repo-url>
cd rust-vs-python-rag-benchmark
.\setup.ps1
```

`setup.sh` / `setup.ps1` performs the following steps automatically:
- Initialises a uv project (`uv init`)
- Adds all direct dependencies from `requirements.in` constrained by `requirements.txt` (`uv add -r requirements.in -c requirements.txt`)
- Syncs the environment (`uv sync`)
- Builds the Rust binary in release mode via `cargo build --release`

No manual Python installation or virtual environment activation is needed — uv handles it all.

---

## Running the Benchmark

```bash
# Start Ollama (if not already running as a service)
ollama serve &

# Run the full benchmark (Python pipeline → 10s cool-down → Rust pipeline → report)
bash run_benchmark.sh
```

Results are written to `output/` and the final report is generated as `benchmark_report.md`.

---

## Configuration

All tunable parameters live in `benchmark_config.toml`:

```toml
dataset_name    = "wikipedia"
dataset_subset  = "20220301.simple"
num_documents   = 1000
chunk_size      = 512
chunk_overlap   = 64
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
top_k           = 5
llm_model       = "llama3.2:3b"
llm_host        = "http://localhost:11434"
query_set_path  = "query_set.json"
output_dir      = "output/"
```

Override the LLM model at runtime via the `BENCHMARK_MODEL` environment variable:

```bash
BENCHMARK_MODEL=mistral:7b bash run_benchmark.sh
```

---

## Running Tests

```bash
# Python property and unit tests
uv run pytest tests/

# Rust tests (including proptest property tests)
cargo test --manifest-path rust_pipeline/Cargo.toml
```

---

## Project Structure

```
rust-vs-python-rag-benchmark/
├── benchmark_config.toml   # Shared configuration
├── query_set.json           # 50 benchmark questions
├── requirements.in          # Unpinned direct Python dependencies
├── requirements.txt         # Pinned constraints (used with uv)
├── Cargo.toml               # Rust workspace
├── setup.sh                 # Environment bootstrap (Linux/macOS)
├── setup.ps1                # Environment bootstrap (Windows)
├── run_benchmark.sh         # Benchmark orchestrator (Linux/macOS)
├── run_benchmark.ps1        # Benchmark orchestrator (Windows)
├── python_pipeline/         # Python RAG implementation
├── rust_pipeline/           # Rust RAG implementation
├── report/                  # Report generator
└── tests/                   # Python property and unit tests
```
