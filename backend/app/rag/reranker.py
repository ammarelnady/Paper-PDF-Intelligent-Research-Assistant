"""Replaceable rerankers for vector-search candidates."""

from __future__ import annotations

import re
from typing import Protocol, Sequence, runtime_checkable

from .vector_store import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    """A component that returns a reordered, scored subset of candidates."""

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Rank candidates for ``query`` and return no more than ``top_k`` results."""


class ScoreReranker:
    """Dependency-free reranker combining semantic and lexical relevance."""

    _TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)

    def __init__(self, semantic_weight: float = 0.75) -> None:
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be between 0.0 and 1.0.")
        self.semantic_weight = semantic_weight

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Blend normalized FAISS score with query-token coverage.

        The returned score is the combined relevance score. A cross-encoder can
        implement ``Reranker`` and replace this class without changing callers.
        """

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        candidate_list = list(candidates)
        if not candidate_list:
            return []
        query_terms = set(self._tokens(query))
        scores = [candidate.score for candidate in candidate_list]
        low, high = min(scores), max(scores)
        ranked: list[RetrievedChunk] = []
        for candidate in candidate_list:
            semantic_score = (candidate.score - low) / (high - low) if high != low else 1.0
            document_terms = set(self._tokens(candidate.text))
            lexical_score = len(query_terms & document_terms) / len(query_terms) if query_terms else 0.0
            score = self.semantic_weight * semantic_score + (1.0 - self.semantic_weight) * lexical_score
            ranked.append(
                RetrievedChunk(candidate.document_id, candidate.chunk_id, candidate.page_number,
                               candidate.section, candidate.text, float(score))
            )
        return sorted(ranked, key=lambda item: (-item.score, item.chunk_id))[:top_k]

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        return cls._TOKEN_PATTERN.findall(text.lower())
