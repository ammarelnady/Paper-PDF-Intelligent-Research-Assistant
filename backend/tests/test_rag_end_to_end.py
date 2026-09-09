"""Real-PDF integration test for document processing and hybrid RAG retrieval.

Set ``PAPER_PDF_PATH`` to an absolute PDF path, or place the test paper at
``backend/data/uploads/attention_is_all_you_need.pdf``. The test is skipped
when no test PDF is available, so cloned checkouts do not require paper data.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF_PATH = REPOSITORY_ROOT / "backend" / "data" / "uploads" / "attention_is_all_you_need.pdf"
PDF_PATH = Path(os.environ.get("PAPER_PDF_PATH", DEFAULT_PDF_PATH))


class TestRagEndToEnd(unittest.TestCase):
    def test_real_pdf_document_processing_to_hybrid_rag(self) -> None:
        """Exercise Salma's pipeline followed by FAISS, BM25, and RRF retrieval."""
        if not PDF_PATH.is_file():
            self.skipTest(
                f"Real integration PDF is unavailable at {DEFAULT_PDF_PATH}. Set PAPER_PDF_PATH to run."
            )

        try:
            import faiss
            import pymupdf
            import rank_bm25
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            self.skipTest(f"Skipping E2E RAG test due to missing optional dependency: {e}")

        from app.document_processing import chunk_document, detect_sections, extract_text_from_pdf
        from app.rag import Bm25Index, FaissVectorStore, SemanticRetriever, SentenceTransformerEmbeddingProvider

        document = extract_text_from_pdf(PDF_PATH)
        sections = detect_sections(document)
        chunks = chunk_document(document)

        self.assertGreater(document.num_pages, 0)
        self.assertTrue(document.pages)
        self.assertTrue(sections)
        self.assertTrue(chunks)

        source_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.assertEqual(len(source_by_id), len(chunks))
        for chunk in chunks:
            self.assertEqual(chunk.document_id, document.document_id)
            self.assertGreaterEqual(chunk.page_number, 1)
            self.assertTrue(chunk.section)
            self.assertTrue(chunk.text.strip())

        provider = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        embeddings = provider.embed_texts([chunk.text for chunk in chunks])
        vector_store = FaissVectorStore(embedding_dimension=embeddings.shape[1])
        vector_store.add(chunks, embeddings)

        bm25_index = Bm25Index()
        bm25_index.add(chunks)
        faiss_retriever = SemanticRetriever(provider, vector_store)
        hybrid_retriever = SemanticRetriever(provider, vector_store, bm25_index=bm25_index, rrf_k=60)

        queries = (
            "What is the main contribution of the paper?",
            "Which optimizer and learning-rate schedule did the authors use?",
            "What machine translation results did the Transformer achieve?",
        )
        for query in queries:
            faiss_results = faiss_retriever.retrieve(query, top_k=3, candidate_k=10)
            bm25_results = bm25_index.search(query, k=3)
            hybrid_results = hybrid_retriever.retrieve(query, top_k=3, candidate_k=10, strategy="hybrid_rrf")

            for results in (faiss_results, bm25_results, hybrid_results):
                self.assertTrue(results)
                self.assertLessEqual(len(results), 3)
                for result in results:
                    source = source_by_id[result.chunk_id]
                    self.assertEqual(result.document_id, source.document_id)
                    self.assertEqual(result.page_number, source.page_number)
                    self.assertEqual(result.section, source.section)
                    self.assertEqual(result.text, source.text)
                    self.assertIsInstance(result.score, float)

        optimizer_results = hybrid_retriever.retrieve(
            "Which optimizer and learning-rate schedule did the authors use?",
            top_k=10,
            candidate_k=20,
            strategy="hybrid_rrf",
        )
        self.assertTrue(any("adam" in result.text.lower() for result in optimizer_results))


if __name__ == "__main__":
    unittest.main()
