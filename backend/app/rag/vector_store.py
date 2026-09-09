"""FAISS-backed, in-memory vector storage with citation metadata preservation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .embeddings import EmbeddingError, normalize_embeddings


class VectorStoreError(RuntimeError):
    """Base exception for vector-store failures."""


class EmptyVectorStoreError(VectorStoreError):
    """Raised when a search is attempted before chunks have been indexed."""


class DuplicateChunkIDError(VectorStoreError):
    """Raised when attempting to add a chunk ID already present in the store."""


class EmbeddingDimensionError(VectorStoreError):
    """Raised when an embedding does not match the configured index dimension."""


@runtime_checkable
class DocumentChunkLike(Protocol):
    """Temporary structural contract ``DocumentChunk`` model."""

    document_id: str
    chunk_id: str
    page_number: int
    section: str
    text: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A retrieval result whose fields are sufficient for answer citations."""

    document_id: str
    chunk_id: str
    page_number: int
    section: str
    text: str
    score: float


class FaissVectorStore:
    """An in-memory normalized-vector FAISS ``IndexFlatIP`` store."""

    def __init__(self, embedding_dimension: int) -> None:
        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be a positive integer.")
        self.embedding_dimension = embedding_dimension
        self._index = self._create_index(embedding_dimension)
        self._chunks: list[DocumentChunkLike] = []
        self._chunk_ids: set[str] = set()

    @property
    def size(self) -> int:
        """Number of chunks currently indexed."""

        return len(self._chunks)

    def add(self, chunks: Sequence[DocumentChunkLike], embeddings: np.ndarray) -> None:
        """Add chunks and their corresponding embeddings atomically."""

        chunk_list = list(chunks)
        if not chunk_list:
            raise VectorStoreError("At least one chunk is required when adding embeddings.")
        self._validate_chunks(chunk_list)
        vectors = self._validate_vectors(embeddings, len(chunk_list))
        try:
            self._index.add(vectors)
        except Exception as exc:  # pragma: no cover - FAISS backend dependent
            raise VectorStoreError("FAISS failed to add the supplied embeddings.") from exc
        self._chunks.extend(chunk_list)
        self._chunk_ids.update(chunk.chunk_id for chunk in chunk_list)

    def search(self, query_embedding: np.ndarray, k: int) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks ranked by normalized inner-product score."""

        if self.size == 0:
            raise EmptyVectorStoreError("Cannot search an empty vector store; add chunks first.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        vector = self._validate_query(query_embedding)
        limit = min(k, self.size)
        try:
            scores, indices = self._index.search(vector.reshape(1, -1), limit)
        except Exception as exc:  # pragma: no cover - FAISS backend dependent
            raise VectorStoreError("FAISS failed to search the query embedding.") from exc
        return [
            self._to_retrieved_chunk(self._chunks[int(index)], float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

    @staticmethod
    def _create_index(dimension: int):  # type: ignore[no-untyped-def]
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise VectorStoreError(
                "faiss-cpu is required for FaissVectorStore. Install it before creating a vector store."
            ) from exc
        return faiss.IndexFlatIP(dimension)

    def _validate_vectors(self, embeddings: np.ndarray, expected_count: int) -> np.ndarray:
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != expected_count:
            raise VectorStoreError("embeddings must be a two-dimensional array with one row per chunk.")
        if array.shape[1] != self.embedding_dimension:
            raise EmbeddingDimensionError(
                f"Embedding dimension {array.shape[1]} does not match index dimension {self.embedding_dimension}."
            )
        try:
            return normalize_embeddings(array)
        except EmbeddingError as exc:
            raise VectorStoreError(str(exc)) from exc

    def _validate_query(self, query_embedding: np.ndarray) -> np.ndarray:
        array = np.asarray(query_embedding, dtype=np.float32)
        if array.ndim != 1:
            raise VectorStoreError("query_embedding must be a one-dimensional vector.")
        if array.shape[0] != self.embedding_dimension:
            raise EmbeddingDimensionError(
                f"Query dimension {array.shape[0]} does not match index dimension {self.embedding_dimension}."
            )
        try:
            return normalize_embeddings(array)[0]
        except EmbeddingError as exc:
            raise VectorStoreError(str(exc)) from exc

    def _validate_chunks(self, chunks: Sequence[DocumentChunkLike]) -> None:
        incoming_ids: set[str] = set()
        for chunk in chunks:
            required = ("document_id", "chunk_id", "page_number", "section", "text")
            missing = [name for name in required if not hasattr(chunk, name)]
            if missing:
                raise VectorStoreError(f"Chunk is missing required fields: {', '.join(missing)}.")
            if not isinstance(chunk.chunk_id, str) or not chunk.chunk_id.strip():
                raise VectorStoreError("Every chunk_id must be a non-empty string.")
            if chunk.chunk_id in self._chunk_ids or chunk.chunk_id in incoming_ids:
                raise DuplicateChunkIDError(f"Chunk ID '{chunk.chunk_id}' is already indexed.")
            incoming_ids.add(chunk.chunk_id)

    @staticmethod
    def _to_retrieved_chunk(chunk: DocumentChunkLike, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            section=chunk.section,
            text=chunk.text,
            score=score,
        )
