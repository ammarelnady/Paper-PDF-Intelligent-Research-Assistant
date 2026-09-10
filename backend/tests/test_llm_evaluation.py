"""
Unit tests for LLM generation, Prompt Construction, Citation Formatting, and Evaluation metrics (Mohamed - Member 5).
"""

import unittest
from app.contracts import WebSource
from app.evaluation import (
    compute_faithfulness,
    compute_section_coverage,
    evaluate_answer_groundedness,
    evaluate_hallucination_rate,
    evaluate_routing_accuracy,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from app.llm import CitationFormatter, LLMClient, PromptBuilder
from app.rag.vector_store import RetrievedChunk


class TestLLMAndPromptConstruction(unittest.TestCase):
    def test_prompt_builder_with_rag_and_web(self):
        builder = PromptBuilder()
        chunk = RetrievedChunk("doc-1", "chunk-101", 3, "Methodology", "Transformer uses self-attention.", 0.88)
        web = WebSource("LLaMA 3 Overview", "https://ai.meta.com/llama", "ai.meta.com", "Llama 3 is a state-of-the-art model.")

        prompt = builder.build_prompt(
            query="Compare Transformer with Llama 3",
            retrieved_chunks=[chunk],
            web_sources=[web],
            route="HYBRID",
        )

        self.assertIn("=== RETRIEVED PAPER CONTEXT ===", prompt)
        self.assertIn("=== EXTERNAL WEB SEARCH CONTEXT ===", prompt)
        self.assertIn("Transformer uses self-attention.", prompt)
        self.assertIn("ai.meta.com", prompt)
        self.assertIn("Compare Transformer with Llama 3", prompt)

    def test_citation_formatter(self):
        chunk = RetrievedChunk("doc-1", "chunk-101", 3, "Methodology", "Self-attention mechanism details.", 0.92)
        web = WebSource("Attention Explainer", "https://example.com/attn", "example.com", "Explaining self-attention.")

        result = CitationFormatter.format_citations(
            answer="The Transformer uses self-attention [Page 3, Methodology].",
            chunks=[chunk],
            web_sources=[web],
        )

        self.assertIn("The Transformer uses self-attention", result["answer"])
        self.assertTrue(result["has_citations"])
        self.assertEqual(len(result["paper_citations"]), 1)
        self.assertEqual(result["paper_citations"][0]["chunk_id"], "chunk-101")
        self.assertEqual(result["paper_citations"][0]["page_number"], 3)
        self.assertEqual(len(result["web_citations"]), 1)
        self.assertEqual(result["web_citations"][0]["domain"], "example.com")

    def test_llm_client_local_fallback(self):
        client = LLMClient(provider="none", api_key=None)
        resp = client.generate("Test prompt context")
        self.assertTrue(len(resp) > 0)


class TestEvaluationFramework(unittest.TestCase):
    def test_retrieval_metrics(self):
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        gold = ["c2", "c5"]

        r_at_3 = recall_at_k(retrieved, gold, k=3)
        self.assertEqual(r_at_3, 0.5)

        r_at_5 = recall_at_k(retrieved, gold, k=5)
        self.assertEqual(r_at_5, 1.0)

        p_at_2 = precision_at_k(retrieved, gold, k=2)
        self.assertEqual(p_at_2, 0.5)

        mrr = mean_reciprocal_rank([["c1", "c2"], ["c3", "c4"]], [["c2"], ["c3"]])
        self.assertEqual(mrr, 0.75)

    def test_summarization_metrics(self):
        source = "The Transformer model uses multi-head attention and feed-forward networks for translation."
        summary = "Transformer uses multi-head attention and feed-forward networks."
        faithfulness = compute_faithfulness(summary, source)
        self.assertGreater(faithfulness, 0.8)

        sections = {
            "Method": "multi-head attention encoder decoder architecture",
            "Results": "achieves state-of-the-art BLEU score",
        }
        cov = compute_section_coverage("multi-head attention BLEU", sections)
        self.assertGreater(cov["Method"], 0.0)

    def test_routing_metrics(self):
        preds = ["RAG", "WEB", "HYBRID", "RAG"]
        gold = ["RAG", "WEB", "RAG", "RAG"]
        stats = evaluate_routing_accuracy(preds, gold)
        self.assertEqual(stats["accuracy"], 0.75)
        self.assertEqual(stats["precision_RAG"], 1.0)

    def test_hallucination_metrics(self):
        context = "The model was evaluated on the WMT 2014 English-to-German benchmark dataset."
        grounded_ans = "The model was evaluated on the WMT English-to-German benchmark dataset."
        hr_grounded = evaluate_hallucination_rate(grounded_ans, context)
        self.assertLess(hr_grounded, 0.2)

        hallucinated_ans = "The quantum spaceship explored Alpha Centauri in parallel universes."
        hr_hallucinated = evaluate_hallucination_rate(hallucinated_ans, context)
        self.assertGreater(hr_hallucinated, 0.7)


if __name__ == "__main__":
    unittest.main()
