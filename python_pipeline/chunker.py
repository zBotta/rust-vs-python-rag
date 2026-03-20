"""Chunker — Task 3 implementation.

Uses langchain_text_splitters.RecursiveCharacterTextSplitter with character-based
chunk_size=512 and chunk_overlap=64.
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(docs: list[str], chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split documents into fixed-size overlapping chunks.

    Short documents (length < chunk_size characters) are returned as a single
    chunk equal to the original text — no padding is added.

    Args:
        docs: List of raw document strings.
        chunk_size: Maximum chunk size in characters (default 512).
        overlap: Overlap between consecutive chunks in characters (default 64).

    Returns:
        Flat list of chunk strings from all documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    chunks: list[str] = []
    for doc in docs:
        if len(doc) < chunk_size:
            # Short document: store as single chunk without padding
            chunks.append(doc)
        else:
            chunks.extend(splitter.split_text(doc))
    return chunks
