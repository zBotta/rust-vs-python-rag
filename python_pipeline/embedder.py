"""Embedder — Task 4 implementation.

Uses sentence-transformers/all-MiniLM-L6-v2 to produce 384-dimensional
float vectors for each input text chunk.
"""
from __future__ import annotations

from python_pipeline.config import BenchmarkError


class EmbedError(BenchmarkError):
    """Raised when the embedding model cannot be loaded or encoding fails."""


_model = None  # module-level cache


def _get_model():
    global _model
    if _model is None:
        try:
            import logging
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            from sentence_transformers import SentenceTransformer  # type: ignore
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:
            raise EmbedError(f"Failed to load embedding model: {exc}") from exc
    return _model


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed text chunks using sentence-transformers/all-MiniLM-L6-v2.

    Args:
        chunks: List of text strings to embed.

    Returns:
        List of 384-dimensional float vectors (one per chunk).

    Raises:
        EmbedError: If the model cannot be loaded.
    """
    model = _get_model()
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return [vec.tolist() for vec in embeddings]
