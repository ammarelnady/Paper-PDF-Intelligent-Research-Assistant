"""High-level semantic retrieval orchestration."""

from __future__ import annotations

from typing import Literal

from .bm25_store import Bm25Index
from .embeddings import EmbeddingError, EmbeddingProvider
from .fusion import reciprocal_rank_fusion
from .reranker import Reranker
from .vector_store import FaissVectorStore, RetrievedChunk


class RetrievalError(ValueError):
    """Raised when retrieval arguments are invalid or query embedding fails."""


class SemanticRetriever:
    """Embed a query, retrieve FAISS candidates, then optionally rerank them."""

    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: FaissVectorStore,
                 reranker: Reranker | None = None, bm25_index: Bm25Index | None = None,
                 rrf_k: int = 60) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.reranker = reranker
        self.bm25_index = bm25_index
        if not isinstance(rrf_k, int) or rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer.")
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 20,
                 strategy: Literal["faiss", "hybrid_rrf"] = "faiss") -> list[RetrievedChunk]:
        """Return citation-ready chunks relevant to a non-empty query."""

        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("query must be a non-empty string.")
        if not isinstance(top_k, int) or top_k <= 0:
            raise RetrievalError("top_k must be a positive integer.")
        if not isinstance(candidate_k, int) or candidate_k <= 0:
            raise RetrievalError("candidate_k must be a positive integer.")
        if candidate_k < top_k:
            raise RetrievalError("candidate_k must be greater than or equal to top_k.")
        if strategy not in ("faiss", "hybrid_rrf"):
            raise RetrievalError("strategy must be 'faiss' or 'hybrid_rrf'.")
        try:
            query_embedding = self.embedding_provider.embed_query(query)
        except EmbeddingError as exc:
            raise RetrievalError("Unable to embed the query.") from exc
        candidates = self.vector_store.search(query_embedding, candidate_k)
        if strategy == "hybrid_rrf":
            if self.bm25_index is None:
                raise RetrievalError("A Bm25Index is required for the 'hybrid_rrf' strategy.")
            return reciprocal_rank_fusion(
                (candidates, self.bm25_index.search(query, candidate_k)), top_k=top_k, rrf_k=self.rrf_k
            )
        if self.reranker is None:
            return candidates[:top_k]
        return self.reranker.rerank(query, candidates, top_k)
