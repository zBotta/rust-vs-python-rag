"""Vector store using hnswlib HNSW index."""
from __future__ import annotations

import hnswlib
import numpy as np


class VectorStore:
    """In-memory HNSW vector index backed by hnswlib."""

    def __init__(self, dim: int = 384, space: str = "cosine") -> None:
        self._dim = dim
        self._space = space
        self._index: hnswlib.Index | None = None

    def build_index(self, embeddings: list[list[float]]) -> None:
        """Build HNSW index from a list of embedding vectors."""
        n = len(embeddings)
        if n == 0:
            raise ValueError("Cannot build index from empty embeddings list")
        index = hnswlib.Index(space=self._space, dim=self._dim)
        index.init_index(max_elements=n, ef_construction=200, M=16)
        arr = np.array(embeddings, dtype=np.float32)
        index.add_items(arr, list(range(n)))
        index.set_ef(50)
        self._index = index

    def query(self, embedding: list[float], top_k: int) -> list[tuple[int, float]]:
        """Return list of (chunk_id, distance) tuples for the top_k nearest neighbours."""
        if self._index is None:
            raise RuntimeError("Index has not been built. Call build_index first.")
        arr = np.array(embedding, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self._index.get_current_count())
        labels, distances = self._index.knn_query(arr, k=k)
        return list(zip(labels[0].tolist(), distances[0].tolist()))
