"""Metrics collector — Task 6 implementation.

Provides QueryMetrics and PipelineMetrics dataclasses, percentile computation,
and JSONL serialization/deserialization.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class QueryMetrics:
    query_id: int
    end_to_end_ms: float
    retrieval_ms: float
    ttft_ms: float
    generation_ms: float
    total_tokens: int
    failed: bool
    failure_reason: Optional[str]


@dataclass
class PipelineMetrics:
    embedding_phase_ms: float
    index_build_ms: float
    queries: list[QueryMetrics] = field(default_factory=list)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


def compute_percentiles(latencies: list[float]) -> tuple[float, float]:
    """Return (p50, p95) for the given list of latency values.

    Uses statistics.median for p50 and a linear-interpolation percentile
    for p95 (equivalent to numpy.percentile with interpolation='linear').

    Returns (0.0, 0.0) for an empty list.
    """
    if not latencies:
        return 0.0, 0.0

    sorted_vals = sorted(latencies)
    n = len(sorted_vals)

    # p50 — standard median
    p50 = statistics.median(sorted_vals)

    # p95 — linear interpolation (matches numpy default)
    p95 = _percentile(sorted_vals, 95.0)

    return p50, p95


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Compute a percentile using linear interpolation (numpy-compatible)."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]

    # numpy uses: index = pct/100 * (n-1)
    idx = pct / 100.0 * (n - 1)
    lo = int(idx)
    hi = lo + 1
    frac = idx - lo

    if hi >= n:
        return sorted_vals[-1]

    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def serialize_to_jsonl(metrics: PipelineMetrics, output_path: str) -> None:
    """Write PipelineMetrics to a JSONL file.

    Each query is written as one JSON object per line with ``"type": "query"``.
    The final line is a summary object with ``"type": "summary"``.
    """
    successful_latencies = [
        q.end_to_end_ms for q in metrics.queries if not q.failed
    ]
    p50, p95 = compute_percentiles(successful_latencies)
    failure_count = sum(1 for q in metrics.queries if q.failed)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        for q in metrics.queries:
            record = {
                "type": "query",
                "query_id": q.query_id,
                "end_to_end_ms": q.end_to_end_ms,
                "retrieval_ms": q.retrieval_ms,
                "ttft_ms": q.ttft_ms,
                "generation_ms": q.generation_ms,
                "total_tokens": q.total_tokens,
                "failed": q.failed,
                "failure_reason": q.failure_reason,
            }
            fh.write(json.dumps(record) + "\n")

        summary = {
            "type": "summary",
            "embedding_phase_ms": metrics.embedding_phase_ms,
            "index_build_ms": metrics.index_build_ms,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "failure_count": failure_count,
        }
        fh.write(json.dumps(summary) + "\n")


def deserialize_from_jsonl(input_path: str) -> PipelineMetrics:
    """Read a JSONL file produced by serialize_to_jsonl and return PipelineMetrics.

    Expects query records followed by a single summary record as the last line.
    """
    path = Path(input_path)
    queries: list[QueryMetrics] = []
    summary: dict = {}

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "query":
                queries.append(
                    QueryMetrics(
                        query_id=obj["query_id"],
                        end_to_end_ms=obj["end_to_end_ms"],
                        retrieval_ms=obj["retrieval_ms"],
                        ttft_ms=obj["ttft_ms"],
                        generation_ms=obj["generation_ms"],
                        total_tokens=obj["total_tokens"],
                        failed=obj["failed"],
                        failure_reason=obj.get("failure_reason"),
                    )
                )
            elif obj.get("type") == "summary":
                summary = obj

    return PipelineMetrics(
        embedding_phase_ms=summary.get("embedding_phase_ms", 0.0),
        index_build_ms=summary.get("index_build_ms", 0.0),
        queries=queries,
        p50_latency_ms=summary.get("p50_latency_ms", 0.0),
        p95_latency_ms=summary.get("p95_latency_ms", 0.0),
    )
