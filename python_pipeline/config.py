"""Configuration loader for the RAG benchmark.

Reads benchmark_config.toml, validates required keys, and applies defaults
for optional keys. Raises BenchmarkError with the missing key name if any
required key is absent.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


class BenchmarkError(Exception):
    """Base exception for all benchmark errors."""


REQUIRED_KEYS: list[str] = [
    "dataset_name",
    "dataset_subset",
    "num_documents",
    "chunk_size",
    "chunk_overlap",
    "embedding_model",
    "top_k",
    "llm_model",
    "llm_host",
    "query_set_path",
    "output_dir",
]

# Optional keys with their documented defaults.
# Currently all keys are required; this dict is kept for forward-compatibility.
OPTIONAL_DEFAULTS: dict[str, object] = {}


@dataclass
class BenchmarkConfig:
    dataset_name: str
    dataset_subset: str
    num_documents: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    top_k: int
    llm_model: str
    llm_host: str
    query_set_path: str
    output_dir: str


def load_config(config_path: str | Path = "benchmark_config.toml") -> BenchmarkConfig:
    """Load and validate benchmark configuration from a TOML file.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        A validated BenchmarkConfig dataclass.

    Raises:
        BenchmarkError: If a required key is missing, naming the missing key.
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    with path.open("rb") as fh:
        raw: dict = tomllib.load(fh)

    # Apply optional defaults first so required-key validation sees the full picture.
    for key, default in OPTIONAL_DEFAULTS.items():
        raw.setdefault(key, default)

    # Validate required keys.
    for key in REQUIRED_KEYS:
        if key not in raw:
            raise BenchmarkError(
                f"Missing required configuration key: '{key}' in {config_path}"
            )

    return BenchmarkConfig(
        dataset_name=raw["dataset_name"],
        dataset_subset=raw["dataset_subset"],
        num_documents=int(raw["num_documents"]),
        chunk_size=int(raw["chunk_size"]),
        chunk_overlap=int(raw["chunk_overlap"]),
        embedding_model=raw["embedding_model"],
        top_k=int(raw["top_k"]),
        llm_model=raw["llm_model"],
        llm_host=raw["llm_host"],
        query_set_path=raw["query_set_path"],
        output_dir=raw["output_dir"],
    )
