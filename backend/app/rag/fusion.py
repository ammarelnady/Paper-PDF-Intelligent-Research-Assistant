"""Rank-fusion helpers for optional hybrid retrieval."""

from __future__ import annotations

from collections.abc import Sequence

from .vector_store import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], top_k: int, rrf_k: int = 60
) -> list[RetrievedChunk]:
    """Fuse ranked lists using ``sum(1 / (rrf_k + rank))`` with one-based ranks."""

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer.")
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            chunks.setdefault(chunk.chunk_id, chunk)
    return [
        RetrievedChunk(item.document_id, item.chunk_id, item.page_number, item.section, item.text, scores[item.chunk_id])
        for item in sorted(chunks.values(), key=lambda item: (-scores[item.chunk_id], item.chunk_id))[:top_k]
    ]
