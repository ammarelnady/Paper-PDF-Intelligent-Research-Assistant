"""Optional BM25 lexical retrieval over metadata-preserving document chunks."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from .vector_store import DocumentChunkLike, DuplicateChunkIDError, EmptyVectorStoreError, RetrievedChunk, VectorStoreError


class Bm25Index:
    """An in-memory BM25 index kept separate from the FAISS vector store."""

    _TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)

    def __init__(self, tokenizer: Callable[[str], list[str]] | None = None) -> None:
        self._tokenizer = tokenizer or self._default_tokenize
        self._chunks: list[DocumentChunkLike] = []
        self._chunk_ids: set[str] = set()
        self._index = None

    @property
    def size(self) -> int:
        """Number of chunks indexed for lexical retrieval."""

        return len(self._chunks)

    def add(self, chunks: Sequence[DocumentChunkLike]) -> None:
        """Index chunks, preserving their source metadata for result construction."""

        chunk_list = list(chunks)
        if not chunk_list:
            raise VectorStoreError("At least one chunk is required when building a BM25 index.")
        for chunk in chunk_list:
            required = ("document_id", "chunk_id", "page_number", "section", "text")
            missing = [field for field in required if not hasattr(chunk, field)]
            if missing:
                raise VectorStoreError(f"Chunk is missing required fields: {', '.join(missing)}.")
            if not isinstance(chunk.chunk_id, str) or not chunk.chunk_id.strip():
                raise VectorStoreError("Every chunk_id must be a non-empty string.")
            if chunk.chunk_id in self._chunk_ids or sum(item.chunk_id == chunk.chunk_id for item in chunk_list) > 1:
                raise DuplicateChunkIDError(f"Chunk ID '{chunk.chunk_id}' is already indexed.")
        self._chunks.extend(chunk_list)
        self._chunk_ids.update(chunk.chunk_id for chunk in chunk_list)
        self._index = self._build_index([self._tokenizer(chunk.text) for chunk in self._chunks])

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the highest-scoring BM25 matches for a non-empty query."""

        if self.size == 0 or self._index is None:
            raise EmptyVectorStoreError("Cannot search an empty BM25 index; add chunks first.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        query_tokens = self._tokenizer(query)
        if not query_tokens:
            raise ValueError("query must contain at least one searchable token.")
        scores = self._index.get_scores(query_tokens)
        ranked_indices = sorted(range(self.size), key=lambda index: (-float(scores[index]), self._chunks[index].chunk_id))[:k]
        return [
            RetrievedChunk(
                document_id=self._chunks[index].document_id,
                chunk_id=self._chunks[index].chunk_id,
                page_number=self._chunks[index].page_number,
                section=self._chunks[index].section,
                text=self._chunks[index].text,
                score=float(scores[index]),
            )
            for index in ranked_indices
        ]

    @staticmethod
    def _default_tokenize(text: str) -> list[str]:
        return Bm25Index._TOKEN_PATTERN.findall(text.lower())

    @staticmethod
    def _build_index(tokenized_corpus: list[list[str]]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise VectorStoreError("rank-bm25 is required for Bm25Index. Install it to use lexical retrieval.") from exc
        return BM25Okapi(tokenized_corpus)
