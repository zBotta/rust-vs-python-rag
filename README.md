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

If not working due to proxy issues, go to the local LLM from HuggingFace part [here](#using-a-local-huggingface-model)

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

### Using a Local HuggingFace Model

(Proxy-Free Setup)

If you have issues with a corporate proxy blocking model downloads, you can serve a locally downloaded HuggingFace GGUF model through Ollama. The steps below use `wareef/LFM2-8B-A1B-Q4_K_M-GGUF` as an example, which fits comfortably in 8 GB of GPU memory.

**Step 1 — Download the model from HuggingFace:**

Download model .gguf file (we recommend the Q_4_K_M for a 8 GB GPU computer) from its HuggingFace repo [wareef/LFM2-8B-A1B-Q4_K_M-GGUF](https://huggingface.co/LiquidAI/LFM2-8B-A1B-GGUF/tree/main)

After downloading, check the exact filename inside `./models/` — it may differ slightly from the repo name.

**Step 2 — Create a `Modelfile` in the project root:**

```
FROM ./models/LFM2-8B-A1B-Q4_K_M.gguf

PARAMETER num_ctx 4096
PARAMETER num_gpu 99
```

`num_gpu 99` offloads as many layers as possible to the GPU.

**Step 3 — Register the model with Ollama:**

```bash
ollama create lfm2-8b -f Modelfile
ollama list   # verify it appears
```

**Step 4 — Update `benchmark_config.toml`:**

```toml
llm_model = "lfm2-8b"
llm_host  = "http://localhost:11434"
```

**Step 5 — Start Ollama and verify:**

```bash
ollama serve

curl http://localhost:11434/api/generate -d '{
  "model": "lfm2-8b",
  "prompt": "Hello",
  "stream": false
}'
```

No code changes are required — both pipelines read `llm_model` from `benchmark_config.toml` and route requests to the local Ollama server.

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
