"""
Unit tests for Paper Summarization module (Salma - Member 1).
Tests overall structured summaries, section-level summaries, chunk traceability,
groundedness evaluation, and LLM / extractive configurable strategies.
"""

import json
import unittest
from app.document_processing.chunker import chunk_document
from app.document_processing.section_detector import detect_sections
from app.models.document import Document, PageContent
from app.models.summary import PaperSummary
from app.summarization.summarizer import (
    PaperSummarizer,
    summarize_paper,
    summarize_section,
)


class TestPaperSummarization(unittest.TestCase):
    """Test suite for Paper Summarization module."""

    def setUp(self):
        # Comprehensive synthetic paper content
        self.doc_id = "doc_sum_test_456"
        self.pages = [
            PageContent(
                page_number=1,
                text=(
                    "Attention Is All You Need for Scientific Papers\n"
                    "Abstract\n"
                    "We propose a novel neural architecture called the Transformer. "
                    "The problem addressed is that recurrent models are fundamentally sequential and slow.\n"
                    "1. Introduction\n"
                    "Sequential computation remains a bottleneck in natural language processing. "
                    "Our main contributions are: (1) We introduce the Transformer model based solely on attention. "
                    "(2) We eliminate recurrence and convolutions. (3) We achieve state-of-the-art translation."
                ),
            ),
            PageContent(
                page_number=2,
                text=(
                    "2. Methodology\n"
                    "The Transformer follows an encoder-decoder architecture using stacked self-attention and point-wise feed-forward layers. "
                    "Multi-head attention allows the model to jointly attend to information from different representation subspaces."
                ),
            ),
            PageContent(
                page_number=3,
                text=(
                    "3. Results\n"
                    "On the WMT 2014 English-to-German task, the model achieves 28.4 BLEU, outperforming existing models by 2.0 BLEU.\n"
                    "4. Limitations\n"
                    "A limitation of this work is the quadratic memory complexity with respect to sequence length."
                ),
            ),
        ]
        self.document = Document(
            document_id=self.doc_id,
            filename="transformer_paper.pdf",
            title="Attention Is All You Need for Scientific Papers",
            num_pages=3,
            pages=self.pages,
            full_text="\n\n".join(p.text for p in self.pages),
        )
        detect_sections(self.document)
        self.chunks = chunk_document(self.document)

    # -------------------------------------------------------------
    # 1. Overall Paper Summary Tests (Extractive Strategy)
    # -------------------------------------------------------------
    def test_overall_summary_contract(self):
        summary_result = summarize_paper(self.document, chunks=self.chunks, strategy="extractive")

        self.assertIsInstance(summary_result, PaperSummary)
        self.assertEqual(summary_result.document_id, self.doc_id)
        self.assertIsNone(summary_result.section)
        self.assertTrue(len(summary_result.summary) > 0)
        self.assertTrue(len(summary_result.source_chunks) > 0)

        # Verify structured attributes
        self.assertIsNotNone(summary_result.problem_statement)
        self.assertIsNotNone(summary_result.methodology)
        self.assertIsNotNone(summary_result.findings)
        self.assertIsNotNone(summary_result.limitations)
        self.assertIn("quadratic memory complexity", summary_result.limitations.lower())

    def test_overall_summary_empty_document(self):
        empty_doc = Document(document_id="doc_empty", filename="empty.pdf", pages=[], num_pages=0)
        result = summarize_paper(empty_doc)
        self.assertEqual(result.document_id, "doc_empty")
        self.assertIn("empty", result.summary.lower())

    # -------------------------------------------------------------
    # 2. Section-Level Summary Tests
    # -------------------------------------------------------------
    def test_section_summary_contract(self):
        method_summary = summarize_section(
            self.document, section_name="Methodology", chunks=self.chunks
        )

        self.assertIsInstance(method_summary, PaperSummary)
        self.assertEqual(method_summary.document_id, self.doc_id)
        self.assertEqual(method_summary.section, "Methodology")
        self.assertTrue(len(method_summary.summary) > 0)
        self.assertIn("encoder-decoder", method_summary.summary.lower())
        self.assertTrue(len(method_summary.source_chunks) > 0)

    def test_section_summary_nonexistent_section(self):
        result = summarize_section(self.document, section_name="NonExistentSection", chunks=self.chunks)
        self.assertEqual(result.section, "NonExistentSection")
        self.assertIn("No content found", result.summary)

    # -------------------------------------------------------------
    # 3. LLM Strategy & Mock Integration Tests
    # -------------------------------------------------------------
    def test_llm_strategy_with_mock_json_response(self):
        def mock_llm(prompt: str, system_prompt: str) -> str:
            return json.dumps({
                "summary": "This paper presents the Transformer, a novel attention-based architecture.",
                "problem_statement": "Sequential computation bottlenecks in NLP.",
                "methodology": "Stacked multi-head self-attention encoder-decoder layers.",
                "key_contributions": ["Introduced Transformer", "Eliminated recurrence"],
                "findings": "Achieved 28.4 BLEU on English-to-German translation.",
                "limitations": "Quadratic memory complexity on long sequences.",
            })

        summarizer = PaperSummarizer(strategy="llm", llm_caller=mock_llm, model_name="mock-gpt4")
        summary_result = summarizer.summarize_paper(self.document, chunks=self.chunks)

        self.assertEqual(summary_result.document_id, self.doc_id)
        self.assertEqual(summary_result.model_used, "mock-gpt4")
        self.assertEqual(summary_result.key_contributions, ["Introduced Transformer", "Eliminated recurrence"])
        self.assertEqual(summary_result.problem_statement, "Sequential computation bottlenecks in NLP.")
        self.assertEqual(summary_result.findings, "Achieved 28.4 BLEU on English-to-German translation.")

    # -------------------------------------------------------------
    # 4. Groundedness Evaluation Tests
    # -------------------------------------------------------------
    def test_groundedness_evaluation(self):
        summarizer = PaperSummarizer()
        grounded_summary = "The Transformer uses multi-head self-attention and achieves 28.4 BLEU."
        score = summarizer.evaluate_groundedness(grounded_summary, self.document.full_text)
        self.assertGreater(score, 0.6)

        hallucinated_summary = "The quantum teleportation spaceship traveled beyond Jupiter in 2099."
        score_hallucinated = summarizer.evaluate_groundedness(hallucinated_summary, self.document.full_text)
        self.assertLess(score_hallucinated, 0.2)


if __name__ == "__main__":
    unittest.main()
