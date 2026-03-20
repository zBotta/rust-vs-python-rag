"""Retriever: embeds a query, queries the vector store, returns top-k chunk texts."""
from __future__ import annotations

from typing import Callable

from python_pipeline.vector_store import VectorStore


class Retriever:
    """Retrieves the most relevant chunks for a query."""

    def __init__(
        self,
        chunks: list[str],
        vector_store: VectorStore,
        embedder_fn: Callable[[str], list[float]],
    ) -> None:
        self._chunks = chunks
        self._vector_store = vector_store
        self._embedder_fn = embedder_fn

    def retrieve(self, query: str, top_k: int) -> list[str]:
        """Embed query, query vector store, return top-k chunk texts."""
        query_embedding = self._embedder_fn(query)
        results = self._vector_store.query(query_embedding, top_k)
        return [self._chunks[chunk_id] for chunk_id, _ in results]
