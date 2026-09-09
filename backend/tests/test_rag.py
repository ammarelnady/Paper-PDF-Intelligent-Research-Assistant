"""Unit tests for the FAISS semantic-retrieval module.

The deterministic provider avoids Sentence Transformers model downloads.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from app.rag import (
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
except ImportError:
    from backend.app.rag import (  # type: ignore
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


class FakeBm25Index:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        return self.results[:k]


class TestRagRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import faiss
        except ImportError:
            self.skipTest("faiss-cpu is not installed in current environment; will run in Colab/CI with requirements.txt")

        self.chunks = [
            MockDocumentChunk("paper-1", "c-method", 2, "Methods", "neural network training method"),
            MockDocumentChunk("paper-1", "c-data", 3, "Dataset", "image dataset collection"),
            MockDocumentChunk("paper-1", "c-result", 7, "Results", "accuracy evaluation result"),
        ]
        self.provider = DeterministicEmbeddingProvider(
            {
                "neural network training method": [1.0, 0.0, 0.0],
                "image dataset collection": [0.0, 1.0, 0.0],
                "accuracy evaluation result": [0.0, 0.0, 1.0],
                "how was the neural model trained?": [0.95, 0.05, 0.0],
                "dataset": [0.0, 1.0, 0.0],
            }
        )
        self.store = FaissVectorStore(embedding_dimension=3)
        self.store.add(self.chunks, self.provider.embed_texts([c.text for c in self.chunks]))

    def test_semantic_top_k_ordering(self) -> None:
        results = self.store.search(self.provider.embed_query("how was the neural model trained?"), k=2)
        self.assertEqual([result.chunk_id for result in results], ["c-method", "c-data"])
        self.assertGreater(results[0].score, results[1].score)

    def test_retrieval_preserves_citation_metadata(self) -> None:
        result = SemanticRetriever(self.provider, self.store).retrieve("dataset", top_k=1, candidate_k=1)[0]
        self.assertEqual(
            (result.document_id, result.chunk_id, result.page_number, result.section, result.text),
            ("paper-1", "c-data", 3, "Dataset", "image dataset collection"),
        )

    def test_reranker_can_promote_lexically_relevant_candidate(self) -> None:
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
        self.assertEqual(result[0].chunk_id, "lexical")

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaises(RetrievalError):
            SemanticRetriever(self.provider, self.store).retrieve("   ")

    def test_invalid_k_values_are_rejected(self) -> None:
        retriever = SemanticRetriever(self.provider, self.store)
        with self.assertRaises(RetrievalError):
            retriever.retrieve("dataset", top_k=0, candidate_k=1)
        with self.assertRaises(RetrievalError):
            retriever.retrieve("dataset", top_k=1, candidate_k=0)
        with self.assertRaises(RetrievalError):
            retriever.retrieve("dataset", top_k=2, candidate_k=1)

    def test_empty_vector_store_is_rejected(self) -> None:
        with self.assertRaises(EmptyVectorStoreError):
            FaissVectorStore(3).search(self.provider.embed_query("dataset"), k=1)

    def test_duplicate_chunk_ids_are_rejected(self) -> None:
        store = FaissVectorStore(3)
        store.add([self.chunks[0]], self.provider.embed_texts([self.chunks[0].text]))
        with self.assertRaises(DuplicateChunkIDError):
            store.add([self.chunks[0]], self.provider.embed_texts([self.chunks[0].text]))

    def test_embedding_dimension_mismatch_is_rejected(self) -> None:
        store = FaissVectorStore(3)
        with self.assertRaises(EmbeddingDimensionError):
            store.add([self.chunks[0]], np.asarray([[1.0, 0.0]], dtype=np.float32))

    def test_basic_end_to_end_retrieval(self) -> None:
        results = SemanticRetriever(self.provider, self.store).retrieve(
            "how was the neural model trained?", top_k=2, candidate_k=3
        )
        self.assertEqual([item.chunk_id for item in results], ["c-method", "c-data"])
        self.assertTrue(all(isinstance(item.score, float) for item in results))

    def test_reciprocal_rank_fusion_combines_rankings(self) -> None:
        first = RetrievedChunk("doc", "first", 1, "A", "first text", 0.9)
        shared = RetrievedChunk("doc", "shared", 2, "B", "shared text", 0.8)
        second = RetrievedChunk("doc", "second", 3, "C", "second text", 0.7)
        results = reciprocal_rank_fusion([[first, shared], [shared, second]], top_k=3, rrf_k=60)
        self.assertEqual([item.chunk_id for item in results], ["shared", "first", "second"])
        self.assertAlmostEqual(results[0].score, 1 / 62 + 1 / 61, places=5)

    def test_hybrid_rrf_retrieval_is_explicit_and_preserves_metadata(self) -> None:
        bm25_result = RetrievedChunk("paper-1", "c-result", 7, "Results", "accuracy evaluation result", 4.2)
        retriever = SemanticRetriever(self.provider, self.store, bm25_index=FakeBm25Index([bm25_result]))  # type: ignore[arg-type]
        results = retriever.retrieve("dataset", top_k=2, candidate_k=2, strategy="hybrid_rrf")
        self.assertEqual(results[0].chunk_id, "c-data")
        self.assertEqual({item.chunk_id for item in results}, {"c-data", "c-result"})
        result = next(item for item in results if item.chunk_id == "c-result")
        self.assertEqual(
            (result.document_id, result.page_number, result.section, result.text),
            ("paper-1", 7, "Results", "accuracy evaluation result"),
        )

    def test_hybrid_rrf_requires_bm25_index(self) -> None:
        with self.assertRaises(RetrievalError):
            SemanticRetriever(self.provider, self.store).retrieve("dataset", strategy="hybrid_rrf")


if __name__ == "__main__":
    unittest.main()
