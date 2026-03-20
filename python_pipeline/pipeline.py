"""Top-level Python RAG pipeline — Task 9.

Wires: dataset_loader → chunker → embedder → vector_store → retriever → llm_client → metrics_collector
Reads config from benchmark_config.toml, runs all queries sequentially, and writes
metrics_python.jsonl to output_dir.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from python_pipeline import config as config_module
from python_pipeline import dataset_loader, chunker, embedder
from python_pipeline.vector_store import VectorStore
from python_pipeline.retriever import Retriever
from python_pipeline import llm_client
from python_pipeline.metrics_collector import (
    QueryMetrics,
    PipelineMetrics,
    compute_percentiles,
    serialize_to_jsonl,
)


def run_pipeline(config_path: str = "benchmark_config.toml") -> None:
    """Run the full Python RAG pipeline and write metrics_python.jsonl."""

    # 1. Load config
    cfg = config_module.load_config(config_path)

    # Ensure output directory exists
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load documents
    print("Loading documents...")
    docs = dataset_loader.load_documents(cfg.dataset_name, cfg.dataset_subset, cfg.num_documents)
    print(f"Loaded {len(docs)} documents.")

    # 3. Chunk documents
    print("Chunking documents...")
    chunks = chunker.chunk_documents(docs, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
    print(f"Produced {len(chunks)} chunks.")

    # 4. Embed all chunks — record embedding_phase_ms
    print("Embedding chunks...")
    embed_start = time.perf_counter()
    embeddings = embedder.embed_chunks(chunks)
    embedding_phase_ms = (time.perf_counter() - embed_start) * 1000.0
    print(f"Embedding done in {embedding_phase_ms:.1f} ms.")

    # 5. Build vector index — record index_build_ms
    print("Building vector index...")
    vs = VectorStore(dim=384, space="cosine")
    index_start = time.perf_counter()
    vs.build_index(embeddings)
    index_build_ms = (time.perf_counter() - index_start) * 1000.0
    print(f"Index built in {index_build_ms:.1f} ms.")

    # embedder_fn for Retriever: embed a single query string → list[float]
    def embedder_fn(query: str) -> list[float]:
        return embedder.embed_chunks([query])[0]

    retriever = Retriever(chunks=chunks, vector_store=vs, embedder_fn=embedder_fn)

    # 6. Load query set
    query_set_path = Path(cfg.query_set_path)
    with query_set_path.open("r", encoding="utf-8") as fh:
        queries = json.load(fh)
    total_queries = len(queries)
    print(f"Loaded {total_queries} queries.")

    # 7. Run each query sequentially
    query_metrics_list: list[QueryMetrics] = []

    for i, entry in enumerate(queries):
        query_id: int = entry["id"]
        question: str = entry["question"]
        print(f"Running query {i + 1}/{total_queries}...")

        e2e_start = time.perf_counter()

        try:
            # a. Retrieve top-k chunks — record retrieval_ms
            retrieval_start = time.perf_counter()
            retrieved_chunks = retriever.retrieve(question, top_k=cfg.top_k)
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

            # b. Generate answer — record ttft_ms, generation_ms, total_tokens
            response = llm_client.generate(
                query=question,
                chunks=retrieved_chunks,
                llm_host=cfg.llm_host,
            )

            # c. Record end-to-end latency
            end_to_end_ms = (time.perf_counter() - e2e_start) * 1000.0

            if response.failed:
                qm = QueryMetrics(
                    query_id=query_id,
                    end_to_end_ms=end_to_end_ms,
                    retrieval_ms=retrieval_ms,
                    ttft_ms=0.0,
                    generation_ms=0.0,
                    total_tokens=0,
                    failed=True,
                    failure_reason=response.failure_reason,
                )
            else:
                qm = QueryMetrics(
                    query_id=query_id,
                    end_to_end_ms=end_to_end_ms,
                    retrieval_ms=retrieval_ms,
                    ttft_ms=response.ttft_ms,
                    generation_ms=response.generation_ms,
                    total_tokens=response.total_tokens,
                    failed=False,
                    failure_reason=None,
                )

        except Exception as exc:
            end_to_end_ms = (time.perf_counter() - e2e_start) * 1000.0
            qm = QueryMetrics(
                query_id=query_id,
                end_to_end_ms=end_to_end_ms,
                retrieval_ms=0.0,
                ttft_ms=0.0,
                generation_ms=0.0,
                total_tokens=0,
                failed=True,
                failure_reason=str(exc),
            )

        query_metrics_list.append(qm)

    # 8. Compute p50/p95 from successful queries
    successful_latencies = [q.end_to_end_ms for q in query_metrics_list if not q.failed]
    p50, p95 = compute_percentiles(successful_latencies)

    # 9. Create PipelineMetrics and serialize to {output_dir}/metrics_python.jsonl
    pipeline_metrics = PipelineMetrics(
        embedding_phase_ms=embedding_phase_ms,
        index_build_ms=index_build_ms,
        queries=query_metrics_list,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
    )

    output_path = output_dir / "metrics_python.jsonl"
    serialize_to_jsonl(pipeline_metrics, str(output_path))
    print(f"Metrics written to {output_path}")
    print(f"p50={p50:.1f} ms  p95={p95:.1f} ms  failures={sum(1 for q in query_metrics_list if q.failed)}/{total_queries}")


if __name__ == "__main__":
    run_pipeline()
