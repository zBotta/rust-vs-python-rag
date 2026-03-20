"""Report generator — Task 11 implementation.

Reads metrics_python.jsonl and metrics_rust.jsonl and produces benchmark_report.md
with a summary table, per-query latency plots, and optional high-failure-rate warnings.
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> tuple[list[dict], dict]:
    """Load query records and summary record from a JSONL file.

    Returns (query_records, summary_record).
    Raises FileNotFoundError with a descriptive message if the file is missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Metrics file not found: '{path}'. "
            "Ensure the pipeline has been run before generating the report."
        )

    queries: list[dict] = []
    summary: dict = {}

    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "query":
                queries.append(obj)
            elif obj.get("type") == "summary":
                summary = obj

    return queries, summary


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_histogram(
    python_latencies: list[float],
    rust_latencies: list[float],
    output_dir: Path,
    filename: str = "latency_histogram.png",
) -> str:
    """Save a per-query latency histogram PNG and return the filename."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = 30
    if python_latencies:
        ax.hist(python_latencies, bins=bins, alpha=0.6, label="Python", color="steelblue")
    if rust_latencies:
        ax.hist(rust_latencies, bins=bins, alpha=0.6, label="Rust", color="darkorange")
    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("Query count")
    ax.set_title("Per-query latency distribution")
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / filename
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)
    return filename


def _save_cdf(
    python_latencies: list[float],
    rust_latencies: list[float],
    output_dir: Path,
    filename: str = "latency_cdf.png",
) -> str:
    """Save a per-query latency CDF PNG and return the filename."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for latencies, label, color in [
        (python_latencies, "Python", "steelblue"),
        (rust_latencies, "Rust", "darkorange"),
    ]:
        if latencies:
            sorted_lats = sorted(latencies)
            cdf = np.arange(1, len(sorted_lats) + 1) / len(sorted_lats)
            ax.plot(sorted_lats, cdf, label=label, color=color)

    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Per-query latency CDF")
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / filename
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)
    return filename


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(
    python_jsonl: str,
    rust_jsonl: str,
    output_path: str = "benchmark_report.md",
    query_set_size: int = 50,
) -> None:
    """Read both JSONL files and produce benchmark_report.md.

    Parameters
    ----------
    python_jsonl:
        Path to the Python pipeline metrics JSONL file.
    rust_jsonl:
        Path to the Rust pipeline metrics JSONL file.
    output_path:
        Destination path for the generated Markdown report.
    query_set_size:
        Total number of queries in the query set (used for failure-rate check).

    Raises
    ------
    FileNotFoundError
        If either JSONL file does not exist.
    """
    py_queries, py_summary = _load_jsonl(python_jsonl)
    rs_queries, rs_summary = _load_jsonl(rust_jsonl)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Extract per-query latencies (successful only for percentiles) ---
    py_lats = [q["end_to_end_ms"] for q in py_queries if not q.get("failed", False)]
    rs_lats = [q["end_to_end_ms"] for q in rs_queries if not q.get("failed", False)]

    py_retrieval = [q["retrieval_ms"] for q in py_queries if not q.get("failed", False)]
    rs_retrieval = [q["retrieval_ms"] for q in rs_queries if not q.get("failed", False)]

    py_ttft = [q["ttft_ms"] for q in py_queries if not q.get("failed", False)]
    rs_ttft = [q["ttft_ms"] for q in rs_queries if not q.get("failed", False)]

    py_tokens = [q["total_tokens"] for q in py_queries if not q.get("failed", False)]
    rs_tokens = [q["total_tokens"] for q in rs_queries if not q.get("failed", False)]

    # --- Summary values ---
    py_p50 = py_summary.get("p50_latency_ms", 0.0)
    py_p95 = py_summary.get("p95_latency_ms", 0.0)
    py_embed = py_summary.get("embedding_phase_ms", 0.0)
    py_index = py_summary.get("index_build_ms", 0.0)
    py_failures = int(py_summary.get("failure_count", 0))

    rs_p50 = rs_summary.get("p50_latency_ms", 0.0)
    rs_p95 = rs_summary.get("p95_latency_ms", 0.0)
    rs_embed = rs_summary.get("embedding_phase_ms", 0.0)
    rs_index = rs_summary.get("index_build_ms", 0.0)
    rs_failures = int(rs_summary.get("failure_count", 0))

    py_mean_ttft = _mean(py_ttft)
    rs_mean_ttft = _mean(rs_ttft)
    py_mean_retrieval = _mean(py_retrieval)
    rs_mean_retrieval = _mean(rs_retrieval)
    py_mean_tokens = _mean([float(t) for t in py_tokens])
    rs_mean_tokens = _mean([float(t) for t in rs_tokens])

    # --- Generate plots ---
    hist_file = _save_histogram(py_lats, rs_lats, output_dir)
    cdf_file = _save_cdf(py_lats, rs_lats, output_dir)

    # --- Build table rows ---
    def _row(label: str, py_val: float, rs_val: float, fmt: str = ".2f") -> str:
        delta = rs_val - py_val
        delta_pct = (delta / py_val * 100.0) if py_val != 0.0 else float("nan")
        delta_pct_str = f"{delta_pct:.2f}%" if not (delta_pct != delta_pct) else "N/A"
        return (
            f"| {label} | {py_val:{fmt}} | {rs_val:{fmt}} "
            f"| {delta:+{fmt}} | {delta_pct_str} |"
        )

    def _row_int(label: str, py_val: float, rs_val: float) -> str:
        delta = rs_val - py_val
        delta_pct = (delta / py_val * 100.0) if py_val != 0.0 else float("nan")
        delta_pct_str = f"{delta_pct:.2f}%" if not (delta_pct != delta_pct) else "N/A"
        return (
            f"| {label} | {int(py_val)} | {int(rs_val)} "
            f"| {int(delta):+d} | {delta_pct_str} |"
        )

    rows = [
        _row("p50 end-to-end latency", py_p50, rs_p50),
        _row("p95 end-to-end latency", py_p95, rs_p95),
        _row("mean TTFT", py_mean_ttft, rs_mean_ttft),
        _row("mean retrieval latency", py_mean_retrieval, rs_mean_retrieval),
        _row("mean Total_Tokens", py_mean_tokens, rs_mean_tokens),
        _row("embedding phase time", py_embed, rs_embed),
        _row("index build time", py_index, rs_index),
        _row_int("failure count", float(py_failures), float(rs_failures)),
    ]

    # --- Warnings ---
    warnings: list[str] = []
    if query_set_size > 0:
        if py_failures / query_set_size > 0.10:
            warnings.append(
                f"> ⚠️ **Warning**: Python pipeline failure count ({py_failures}) exceeds 10% of "
                f"query set size ({query_set_size}). Results may not be statistically reliable."
            )
        if rs_failures / query_set_size > 0.10:
            warnings.append(
                f"> ⚠️ **Warning**: Rust pipeline failure count ({rs_failures}) exceeds 10% of "
                f"query set size ({query_set_size}). Results may not be statistically reliable."
            )

    # --- Assemble report ---
    lines: list[str] = [
        "# Rust vs Python RAG Benchmark Report",
        "",
        "## Summary Table",
        "",
        "| Metric | Python value | Rust value | Delta | Delta % |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(rows)

    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        lines.extend(warnings)

    lines += [
        "",
        "## Per-Query Latency Distribution",
        "",
        f"![Latency Histogram]({hist_file})",
        "",
        f"![Latency CDF]({cdf_file})",
        "",
    ]

    report_text = "\n".join(lines)
    Path(output_path).write_text(report_text, encoding="utf-8")
