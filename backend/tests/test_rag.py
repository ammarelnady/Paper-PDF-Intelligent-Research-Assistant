"""Unit tests for the FAISS semantic-retrieval module.

The deterministic provider avoids Sentence Transformers model downloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pytest

from backend.app.rag import (
    DuplicateChunkIDError,
    EmbeddingDimensionError,
    EmptyVectorStoreError,
    FaissVectorStore,
    RetrievalError,
    RetrievedChunk,
    ScoreReranker,
    SemanticRetriever,
    reciprocal_rank_fusion,
)


@dataclass(frozen=True)
class MockDocumentChunk:
    document_id: str
    chunk_id: str
    page_number: int
    section: str
    text: str


class DeterministicEmbeddingProvider:
    """Maps known text values to test vectors without network/model access."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = {key: np.asarray(value, dtype=np.float32) for key, value in vectors.items()}

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self.vectors[text] for text in texts]).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.vectors[query].astype(np.float32)


@pytest.fixture
def chunks() -> list[MockDocumentChunk]:
    return [
        MockDocumentChunk("paper-1", "c-method", 2, "Methods", "neural network training method"),
        MockDocumentChunk("paper-1", "c-data", 3, "Dataset", "image dataset collection"),
        MockDocumentChunk("paper-1", "c-result", 7, "Results", "accuracy evaluation result"),
    ]


@pytest.fixture
def provider() -> DeterministicEmbeddingProvider:
    return DeterministicEmbeddingProvider(
        {
            "neural network training method": [1.0, 0.0, 0.0],
            "image dataset collection": [0.0, 1.0, 0.0],
            "accuracy evaluation result": [0.0, 0.0, 1.0],
            "how was the neural model trained?": [0.95, 0.05, 0.0],
            "dataset": [0.0, 1.0, 0.0],
        }
    )


@pytest.fixture
def populated_store(chunks: list[MockDocumentChunk], provider: DeterministicEmbeddingProvider) -> FaissVectorStore:
    store = FaissVectorStore(embedding_dimension=3)
    store.add(chunks, provider.embed_texts([chunk.text for chunk in chunks]))
    return store


def test_semantic_top_k_ordering(populated_store: FaissVectorStore, provider: DeterministicEmbeddingProvider) -> None:
    results = populated_store.search(provider.embed_query("how was the neural model trained?"), k=2)
    assert [result.chunk_id for result in results] == ["c-method", "c-data"]
    assert results[0].score > results[1].score


def test_retrieval_preserves_citation_metadata(
    populated_store: FaissVectorStore, provider: DeterministicEmbeddingProvider
) -> None:
    result = SemanticRetriever(provider, populated_store).retrieve("dataset", top_k=1, candidate_k=1)[0]
    assert (result.document_id, result.chunk_id, result.page_number, result.section, result.text) == (
        "paper-1", "c-data", 3, "Dataset", "image dataset collection"
    )


def test_reranker_can_promote_lexically_relevant_candidate() -> None:
    candidates = [
        MockDocumentChunk("doc", "semantic", 1, "A", "unrelated passage"),
        MockDocumentChunk("doc", "lexical", 2, "B", "dataset collection details"),
    ]
    provider = DeterministicEmbeddingProvider(
        {
            "unrelated passage": [1.0, 0.0],
            "dataset collection details": [0.8, 0.2],
            "dataset": [1.0, 0.0],
        }
    )
    store = FaissVectorStore(2)
    store.add(candidates, provider.embed_texts([chunk.text for chunk in candidates]))
    result = SemanticRetriever(provider, store, ScoreReranker(semantic_weight=0.2)).retrieve(
        "dataset", top_k=1, candidate_k=2
    )
    assert result[0].chunk_id == "lexical"


def test_empty_query_is_rejected(populated_store: FaissVectorStore, provider: DeterministicEmbeddingProvider) -> None:
    with pytest.raises(RetrievalError, match="non-empty"):
        SemanticRetriever(provider, populated_store).retrieve("   ")


@pytest.mark.parametrize(("top_k", "candidate_k"), [(0, 1), (1, 0), (2, 1)])
def test_invalid_k_values_are_rejected(
    populated_store: FaissVectorStore, provider: DeterministicEmbeddingProvider, top_k: int, candidate_k: int
) -> None:
    with pytest.raises(RetrievalError):
        SemanticRetriever(provider, populated_store).retrieve("dataset", top_k=top_k, candidate_k=candidate_k)


def test_empty_vector_store_is_rejected(provider: DeterministicEmbeddingProvider) -> None:
    with pytest.raises(EmptyVectorStoreError, match="empty"):
        FaissVectorStore(3).search(provider.embed_query("dataset"), k=1)


def test_duplicate_chunk_ids_are_rejected(
    chunks: list[MockDocumentChunk], provider: DeterministicEmbeddingProvider
) -> None:
    store = FaissVectorStore(3)
    store.add([chunks[0]], provider.embed_texts([chunks[0].text]))
    with pytest.raises(DuplicateChunkIDError, match="c-method"):
        store.add([chunks[0]], provider.embed_texts([chunks[0].text]))


def test_embedding_dimension_mismatch_is_rejected(chunks: list[MockDocumentChunk]) -> None:
    store = FaissVectorStore(3)
    with pytest.raises(EmbeddingDimensionError, match="does not match"):
        store.add([chunks[0]], np.asarray([[1.0, 0.0]], dtype=np.float32))


def test_basic_end_to_end_retrieval(
    populated_store: FaissVectorStore, provider: DeterministicEmbeddingProvider
) -> None:
    results = SemanticRetriever(provider, populated_store).retrieve(
        "how was the neural model trained?", top_k=2, candidate_k=3
    )
    assert [item.chunk_id for item in results] == ["c-method", "c-data"]
    assert all(isinstance(item.score, float) for item in results)


def test_reciprocal_rank_fusion_combines_rankings() -> None:
    first = RetrievedChunk("doc", "first", 1, "A", "first text", 0.9)
    shared = RetrievedChunk("doc", "shared", 2, "B", "shared text", 0.8)
    second = RetrievedChunk("doc", "second", 3, "C", "second text", 0.7)
    results = reciprocal_rank_fusion([[first, shared], [shared, second]], top_k=3, rrf_k=60)
    assert [item.chunk_id for item in results] == ["shared", "first", "second"]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)


class FakeBm25Index:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        return self.results[:k]


def test_hybrid_rrf_retrieval_is_explicit_and_preserves_metadata(
    populated_store: FaissVectorStore, provider: DeterministicEmbeddingProvider
) -> None:
    bm25_result = RetrievedChunk("paper-1", "c-result", 7, "Results", "accuracy evaluation result", 4.2)
    retriever = SemanticRetriever(provider, populated_store, bm25_index=FakeBm25Index([bm25_result]))  # type: ignore[arg-type]
    results = retriever.retrieve("dataset", top_k=2, candidate_k=2, strategy="hybrid_rrf")
    assert results[0].chunk_id == "c-data"
    assert {item.chunk_id for item in results} == {"c-data", "c-result"}
    result = next(item for item in results if item.chunk_id == "c-result")
    assert (result.document_id, result.page_number, result.section, result.text) == (
        "paper-1", 7, "Results", "accuracy evaluation result"
    )


def test_hybrid_rrf_requires_bm25_index(
    populated_store: FaissVectorStore, provider: DeterministicEmbeddingProvider
) -> None:
    with pytest.raises(RetrievalError, match="Bm25Index"):
        SemanticRetriever(provider, populated_store).retrieve("dataset", strategy="hybrid_rrf")
