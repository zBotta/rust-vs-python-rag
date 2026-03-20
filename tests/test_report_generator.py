"""Tests for the report generator — Task 11.

Sub-task 11.1 — Property 13: Report contains all required column headers and row labels
Sub-task 11.2 — Property 14: Report contains warning when failure_count / query_set_size > 0.10
Sub-task 11.3 — Unit tests for report generator

Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from report.generate_report import generate_report

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_COLUMN_HEADERS = ["Metric", "Python value", "Rust value", "Delta", "Delta %"]
REQUIRED_ROW_LABELS = [
    "p50 end-to-end latency",
    "p95 end-to-end latency",
    "mean TTFT",
    "mean retrieval latency",
    "mean Total_Tokens",
    "embedding phase time",
    "index build time",
    "failure count",
]


def _write_jsonl(path: str, queries: list[dict], summary: dict) -> None:
    """Write a minimal JSONL file with query records and a summary record."""
    with open(path, "w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q) + "\n")
        fh.write(json.dumps(summary) + "\n")


def _make_query(query_id: int, end_to_end_ms: float = 100.0, failed: bool = False) -> dict:
    return {
        "type": "query",
        "query_id": query_id,
        "end_to_end_ms": end_to_end_ms,
        "retrieval_ms": 10.0,
        "ttft_ms": 50.0,
        "generation_ms": 40.0,
        "total_tokens": 200,
        "failed": failed,
        "failure_reason": "err" if failed else None,
    }


def _make_summary(
    embedding_phase_ms: float = 1000.0,
    index_build_ms: float = 200.0,
    p50: float = 100.0,
    p95: float = 150.0,
    failure_count: int = 0,
) -> dict:
    return {
        "type": "summary",
        "embedding_phase_ms": embedding_phase_ms,
        "index_build_ms": index_build_ms,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "failure_count": failure_count,
    }


# ---------------------------------------------------------------------------
# Strategies for property tests
# ---------------------------------------------------------------------------

non_neg_float = st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)
pos_float = st.floats(min_value=1.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)
non_neg_int = st.integers(min_value=0, max_value=1000)


def _jsonl_strategy(failure_count: int = 0):
    """Strategy that builds a valid JSONL content (list of query dicts + summary dict)."""
    return st.fixed_dictionaries({
        "queries": st.lists(
            st.fixed_dictionaries({
                "type": st.just("query"),
                "query_id": st.integers(min_value=0, max_value=999),
                "end_to_end_ms": non_neg_float,
                "retrieval_ms": non_neg_float,
                "ttft_ms": non_neg_float,
                "generation_ms": non_neg_float,
                "total_tokens": non_neg_int,
                "failed": st.just(False),
                "failure_reason": st.just(None),
            }),
            min_size=1,
            max_size=10,
        ),
        "summary": st.fixed_dictionaries({
            "type": st.just("summary"),
            "embedding_phase_ms": non_neg_float,
            "index_build_ms": non_neg_float,
            "p50_latency_ms": non_neg_float,
            "p95_latency_ms": non_neg_float,
            "failure_count": st.just(failure_count),
        }),
    })


# ---------------------------------------------------------------------------
# Sub-task 11.1 — Property 13: Report table completeness
# Feature: rust-vs-python-rag-benchmark, Property 13: Report contains all required column headers and row labels
# ---------------------------------------------------------------------------

# Feature: rust-vs-python-rag-benchmark, Property 13: Report contains all required column headers and row labels


@given(
    py_data=_jsonl_strategy(),
    rs_data=_jsonl_strategy(),
)
@settings(max_examples=100, deadline=None)
def test_report_contains_all_required_headers_and_rows(
    py_data: dict,
    rs_data: dict,
) -> None:
    """Property 13: Report contains all required column headers and row labels.

    # Feature: rust-vs-python-rag-benchmark, Property 13: Report contains all required column headers and row labels
    Validates: Requirements 8.2, 8.3
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        py_path = str(Path(tmpdir) / "metrics_python.jsonl")
        rs_path = str(Path(tmpdir) / "metrics_rust.jsonl")
        report_path = str(Path(tmpdir) / "benchmark_report.md")

        _write_jsonl(py_path, py_data["queries"], py_data["summary"])
        _write_jsonl(rs_path, rs_data["queries"], rs_data["summary"])

        generate_report(py_path, rs_path, output_path=report_path, query_set_size=50)

        report_text = Path(report_path).read_text(encoding="utf-8")

        for header in REQUIRED_COLUMN_HEADERS:
            assert header in report_text, (
                f"Required column header '{header}' not found in report"
            )

        for row_label in REQUIRED_ROW_LABELS:
            assert row_label in report_text, (
                f"Required row label '{row_label}' not found in report"
            )


# ---------------------------------------------------------------------------
# Sub-task 11.2 — Property 14: High failure rate warning
# Feature: rust-vs-python-rag-benchmark, Property 14: Report contains warning when failure_count / query_set_size > 0.10
# ---------------------------------------------------------------------------

# Feature: rust-vs-python-rag-benchmark, Property 14: Report contains warning when failure_count / query_set_size > 0.10


@given(
    query_set_size=st.integers(min_value=10, max_value=200),
    py_failure_frac=st.floats(min_value=0.11, max_value=1.0, allow_nan=False, allow_infinity=False),
    rs_failure_frac=st.floats(min_value=0.0, max_value=0.05, allow_nan=False, allow_infinity=False),
    py_queries=st.lists(
        st.fixed_dictionaries({
            "type": st.just("query"),
            "query_id": st.integers(min_value=0, max_value=999),
            "end_to_end_ms": non_neg_float,
            "retrieval_ms": non_neg_float,
            "ttft_ms": non_neg_float,
            "generation_ms": non_neg_float,
            "total_tokens": non_neg_int,
            "failed": st.just(False),
            "failure_reason": st.just(None),
        }),
        min_size=1,
        max_size=5,
    ),
    rs_queries=st.lists(
        st.fixed_dictionaries({
            "type": st.just("query"),
            "query_id": st.integers(min_value=0, max_value=999),
            "end_to_end_ms": non_neg_float,
            "retrieval_ms": non_neg_float,
            "ttft_ms": non_neg_float,
            "generation_ms": non_neg_float,
            "total_tokens": non_neg_int,
            "failed": st.just(False),
            "failure_reason": st.just(None),
        }),
        min_size=1,
        max_size=5,
    ),
)
@settings(max_examples=100, deadline=None)
def test_report_contains_warning_when_failure_rate_exceeds_10_percent(
    query_set_size: int,
    py_failure_frac: float,
    rs_failure_frac: float,
    py_queries: list[dict],
    rs_queries: list[dict],
) -> None:
    """Property 14: Report contains warning when failure_count / query_set_size > 0.10.

    # Feature: rust-vs-python-rag-benchmark, Property 14: Report contains warning when failure_count / query_set_size > 0.10
    Validates: Requirements 8.5
    """
    # Compute failure counts that exceed 10% for Python and stay below for Rust
    py_failure_count = max(int(py_failure_frac * query_set_size), int(query_set_size * 0.10) + 1)
    rs_failure_count = int(rs_failure_frac * query_set_size)

    py_summary = _make_summary(failure_count=py_failure_count)
    rs_summary = _make_summary(failure_count=rs_failure_count)

    with tempfile.TemporaryDirectory() as tmpdir:
        py_path = str(Path(tmpdir) / "metrics_python.jsonl")
        rs_path = str(Path(tmpdir) / "metrics_rust.jsonl")
        report_path = str(Path(tmpdir) / "benchmark_report.md")

        _write_jsonl(py_path, py_queries, py_summary)
        _write_jsonl(rs_path, rs_queries, rs_summary)

        generate_report(py_path, rs_path, output_path=report_path, query_set_size=query_set_size)

        report_text = Path(report_path).read_text(encoding="utf-8")

        # The report must contain a warning about statistical reliability
        warning_keywords = ["warning", "Warning", "WARNING"]
        reliability_keywords = ["statistically reliable", "reliable", "failure"]

        has_warning = any(kw in report_text for kw in warning_keywords)
        has_reliability = any(kw in report_text for kw in reliability_keywords)

        assert has_warning and has_reliability, (
            f"Report must contain a warning about statistical reliability when "
            f"failure_count ({py_failure_count}) / query_set_size ({query_set_size}) > 0.10. "
            f"Report text:\n{report_text[:500]}"
        )


# ---------------------------------------------------------------------------
# Sub-task 11.3 — Unit tests for report generator
# Validates: Requirements 8.1, 8.4
# ---------------------------------------------------------------------------

class TestReportGeneratorUnit:
    """Unit tests for the report generator."""

    def test_markdown_file_produced_from_minimal_fixtures(self, tmp_path: Path) -> None:
        """Test that a markdown file is produced from two minimal JSONL fixtures.

        Validates: Requirements 8.1
        """
        py_path = str(tmp_path / "metrics_python.jsonl")
        rs_path = str(tmp_path / "metrics_rust.jsonl")
        report_path = str(tmp_path / "benchmark_report.md")

        py_queries = [_make_query(0, 100.0), _make_query(1, 120.0), _make_query(2, 90.0)]
        rs_queries = [_make_query(0, 80.0), _make_query(1, 95.0), _make_query(2, 75.0)]
        py_summary = _make_summary(p50=100.0, p95=120.0, failure_count=0)
        rs_summary = _make_summary(p50=80.0, p95=95.0, failure_count=0)

        _write_jsonl(py_path, py_queries, py_summary)
        _write_jsonl(rs_path, rs_queries, rs_summary)

        generate_report(py_path, rs_path, output_path=report_path, query_set_size=50)

        assert Path(report_path).exists(), "benchmark_report.md was not created"
        report_text = Path(report_path).read_text(encoding="utf-8")

        # Must be non-empty markdown
        assert len(report_text) > 0
        assert "# Rust vs Python RAG Benchmark Report" in report_text

        # All required column headers present
        for header in REQUIRED_COLUMN_HEADERS:
            assert header in report_text, f"Missing column header: {header}"

        # All required row labels present
        for row_label in REQUIRED_ROW_LABELS:
            assert row_label in report_text, f"Missing row label: {row_label}"

    def test_image_references_embedded_in_output(self, tmp_path: Path) -> None:
        """Test that PNG image references are embedded in the report output.

        Validates: Requirements 8.4
        """
        py_path = str(tmp_path / "metrics_python.jsonl")
        rs_path = str(tmp_path / "metrics_rust.jsonl")
        report_path = str(tmp_path / "benchmark_report.md")

        py_queries = [_make_query(0, 100.0)]
        rs_queries = [_make_query(0, 80.0)]
        py_summary = _make_summary()
        rs_summary = _make_summary()

        _write_jsonl(py_path, py_queries, py_summary)
        _write_jsonl(rs_path, rs_queries, rs_summary)

        generate_report(py_path, rs_path, output_path=report_path, query_set_size=50)

        report_text = Path(report_path).read_text(encoding="utf-8")

        # Report must contain markdown image references
        assert "![" in report_text, "No image references found in report"
        assert ".png" in report_text, "No PNG image references found in report"

        # PNG files must actually exist in the output directory
        png_files = list(tmp_path.glob("*.png"))
        assert len(png_files) >= 1, f"No PNG files found in {tmp_path}"

    def test_error_returned_when_python_jsonl_missing(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised when the Python JSONL file is missing.

        Validates: Requirements 8.1
        """
        py_path = str(tmp_path / "metrics_python.jsonl")  # does not exist
        rs_path = str(tmp_path / "metrics_rust.jsonl")
        report_path = str(tmp_path / "benchmark_report.md")

        # Create only the Rust JSONL
        rs_queries = [_make_query(0, 80.0)]
        rs_summary = _make_summary()
        _write_jsonl(rs_path, rs_queries, rs_summary)

        with pytest.raises(FileNotFoundError) as exc_info:
            generate_report(py_path, rs_path, output_path=report_path)

        assert "metrics_python.jsonl" in str(exc_info.value), (
            f"Error message should identify the missing file. Got: {exc_info.value}"
        )

    def test_error_returned_when_rust_jsonl_missing(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised when the Rust JSONL file is missing.

        Validates: Requirements 8.1
        """
        py_path = str(tmp_path / "metrics_python.jsonl")
        rs_path = str(tmp_path / "metrics_rust.jsonl")  # does not exist
        report_path = str(tmp_path / "benchmark_report.md")

        # Create only the Python JSONL
        py_queries = [_make_query(0, 100.0)]
        py_summary = _make_summary()
        _write_jsonl(py_path, py_queries, py_summary)

        with pytest.raises(FileNotFoundError) as exc_info:
            generate_report(py_path, rs_path, output_path=report_path)

        assert "metrics_rust.jsonl" in str(exc_info.value), (
            f"Error message should identify the missing file. Got: {exc_info.value}"
        )

    def test_warning_emitted_when_failure_rate_exceeds_threshold(self, tmp_path: Path) -> None:
        """Test that a warning is emitted when failure count > 10% of query set size.

        Validates: Requirements 8.5
        """
        py_path = str(tmp_path / "metrics_python.jsonl")
        rs_path = str(tmp_path / "metrics_rust.jsonl")
        report_path = str(tmp_path / "benchmark_report.md")

        # 6 failures out of 50 = 12% > 10%
        py_queries = [_make_query(i) for i in range(44)] + [_make_query(i + 44, failed=True) for i in range(6)]
        rs_queries = [_make_query(i) for i in range(50)]
        py_summary = _make_summary(failure_count=6)
        rs_summary = _make_summary(failure_count=0)

        _write_jsonl(py_path, py_queries, py_summary)
        _write_jsonl(rs_path, rs_queries, rs_summary)

        generate_report(py_path, rs_path, output_path=report_path, query_set_size=50)

        report_text = Path(report_path).read_text(encoding="utf-8")
        assert "Warning" in report_text or "warning" in report_text, (
            "Report must contain a warning when failure rate > 10%"
        )
        assert "statistically reliable" in report_text, (
            "Warning must mention statistical reliability"
        )

    def test_no_warning_when_failure_rate_below_threshold(self, tmp_path: Path) -> None:
        """Test that no warning is emitted when failure count <= 10% of query set size."""
        py_path = str(tmp_path / "metrics_python.jsonl")
        rs_path = str(tmp_path / "metrics_rust.jsonl")
        report_path = str(tmp_path / "benchmark_report.md")

        # 5 failures out of 50 = 10% — not strictly greater than 10%
        py_queries = [_make_query(i) for i in range(50)]
        rs_queries = [_make_query(i) for i in range(50)]
        py_summary = _make_summary(failure_count=5)
        rs_summary = _make_summary(failure_count=5)

        _write_jsonl(py_path, py_queries, py_summary)
        _write_jsonl(rs_path, rs_queries, rs_summary)

        generate_report(py_path, rs_path, output_path=report_path, query_set_size=50)

        report_text = Path(report_path).read_text(encoding="utf-8")
        assert "statistically reliable" not in report_text, (
            "Report must NOT contain a reliability warning when failure rate <= 10%"
        )
