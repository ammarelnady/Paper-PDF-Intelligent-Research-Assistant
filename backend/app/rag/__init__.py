<<<<<<< HEAD
"""Reserved for Reem's RAG implementation."""
=======
"""Public interfaces for semantic retrieval and RAG."""

from .bm25_store import Bm25Index
from .embeddings import EmbeddingError, EmbeddingProvider, SentenceTransformerEmbeddingProvider
from .fusion import reciprocal_rank_fusion
from .reranker import Reranker, ScoreReranker
from .retriever import RetrievalError, SemanticRetriever
from .vector_store import (
    DocumentChunkLike, DuplicateChunkIDError, EmbeddingDimensionError,
    EmptyVectorStoreError, FaissVectorStore, RetrievedChunk, VectorStoreError,
)
from .bm25_store import Bm25Index

__all__ = [
    "Bm25Index", "DocumentChunkLike", "DuplicateChunkIDError", "EmbeddingDimensionError", "EmbeddingError",
    "EmbeddingProvider", "EmptyVectorStoreError", "FaissVectorStore", "RetrievedChunk",
    "RetrievalError", "Reranker", "ScoreReranker", "SemanticRetriever",
    "SentenceTransformerEmbeddingProvider", "VectorStoreError", "reciprocal_rank_fusion",
]
>>>>>>> origin/main
