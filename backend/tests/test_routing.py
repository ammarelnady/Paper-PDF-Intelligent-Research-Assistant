"""Offline tests for the isolated Research Agent routing module."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts import RetrievedChunk, RouteDecision, WebSource
from app.routing.query_classifier import DeterministicQueryClassifier
from app.routing.router import ResearchRouter
from app.web_search.web_search_client import WebSearchClient


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.queries: list[str] = []

    def search(self, query: str) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.chunks


class FakeSearchProvider:
    def __init__(self, sources: list[WebSource]) -> None:
        self.sources = sources
        self.queries: list[str] = []

    def search(self, query: str) -> list[WebSource]:
        self.queries.append(query)
        return self.sources


class FakeFallback:
    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision
        self.calls: list[str] = []

    def classify(self, query: str, _intent: object) -> RouteDecision:
        self.calls.append(query)
        return self.decision


class RoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunk = RetrievedChunk(
            document_id="paper-17",
            chunk_id="chunk-4",
            text="The paper uses the ExampleSet dataset.",
            page_number=3,
            section="Methods",
            score=0.88,
        )
        self.source = WebSource(
            title="Transformer research update",
            url="https://research.example.org/transformers",
            domain="research.example.org",
            snippet="A provider-supplied update.",
        )

    def test_paper_specific_query_routes_to_rag_with_confidence(self) -> None:
        decision = DeterministicQueryClassifier().classify(
            "What dataset did the paper use?"
        )
        self.assertEqual(decision.route, "RAG")
        self.assertGreater(decision.confidence, 0.0)

    def test_main_results_query_routes_to_rag(self) -> None:
        decision = DeterministicQueryClassifier().classify("What are the main results?")

        self.assertEqual(decision.route, "RAG")
        self.assertGreaterEqual(decision.confidence, 0.90)

    def test_current_query_routes_to_web(self) -> None:
        decision = DeterministicQueryClassifier().classify(
            "What are the latest developments related to Transformer architectures?"
        )
        self.assertEqual(decision.route, "WEB")

    def test_external_research_query_routes_to_web(self) -> None:
        decision = DeterministicQueryClassifier().classify(
            "What does the research literature say about Transformer architectures?"
        )

        self.assertEqual(decision.route, "WEB")
        self.assertGreaterEqual(decision.confidence, 0.80)

    def test_paper_and_current_comparison_routes_to_hybrid(self) -> None:
        decision = DeterministicQueryClassifier().classify(
            "How does the Transformer architecture in this paper compare with recent approaches?"
        )
        self.assertEqual(decision.route, "HYBRID")

    def test_ambiguous_query_uses_injected_fallback(self) -> None:
        fallback = FakeFallback(RouteDecision(route="WEB", confidence=0.77))
        classifier = DeterministicQueryClassifier(fallback=fallback)

        decision = classifier.classify("Explain Transformer architectures")

        self.assertEqual(decision.route, "WEB")
        self.assertEqual(fallback.calls, ["Explain Transformer architectures"])

    def test_high_confidence_query_does_not_use_fallback(self) -> None:
        fallback = FakeFallback(RouteDecision(route="WEB", confidence=0.77))
        classifier = DeterministicQueryClassifier(fallback=fallback)

        decision = classifier.classify("What methodology does this paper propose?")

        self.assertEqual(decision.route, "RAG")
        self.assertEqual(fallback.calls, [])

    def test_comparison_only_query_is_low_confidence_hybrid(self) -> None:
        decision = DeterministicQueryClassifier().classify(
            "Compare Transformer architectures"
        )

        self.assertEqual(decision.route, "HYBRID")
        self.assertLess(decision.confidence, 0.60)

    def test_completely_ambiguous_query_is_low_confidence_hybrid(self) -> None:
        decision = DeterministicQueryClassifier().classify("Explain photosynthesis")

        self.assertEqual(decision.route, "HYBRID")
        self.assertLess(decision.confidence, 0.60)

    def test_short_query_is_safe_low_confidence_hybrid(self) -> None:
        decision = DeterministicQueryClassifier().classify("AI")

        self.assertEqual(decision.route, "HYBRID")
        self.assertLess(decision.confidence, 0.60)

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "query must not be empty"):
            DeterministicQueryClassifier().classify("")

    def test_whitespace_only_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "query must not be empty"):
            DeterministicQueryClassifier().classify("   \t\n")

    def test_punctuation_only_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "letters or digits"):
            DeterministicQueryClassifier().classify("?!...")

    def test_fake_retriever_is_used_and_chunk_provenance_is_preserved(self) -> None:
        retriever = FakeRetriever([self.chunk])
        result = ResearchRouter(retriever=retriever).route(
            "What methodology does this paper propose?"
        )

        self.assertEqual(result.decision.route, "RAG")
        self.assertEqual(retriever.queries, ["What methodology does this paper propose?"])
        self.assertEqual(result.retrieved_chunks, (self.chunk,))
        self.assertEqual(result.retrieved_chunks[0].document_id, "paper-17")
        self.assertEqual(result.retrieved_chunks[0].page_number, 3)

    def test_rag_route_without_retriever_returns_empty_chunks(self) -> None:
        result = ResearchRouter().route("What methodology does this paper propose?")

        self.assertEqual(result.decision.route, "RAG")
        self.assertEqual(result.retrieved_chunks, ())
        self.assertEqual(result.web_sources, ())

    def test_fake_web_provider_is_used_and_source_provenance_is_preserved(self) -> None:
        provider = FakeSearchProvider([self.source])
        result = ResearchRouter(web_search=WebSearchClient(provider)).route(
            "What happened recently in Transformer research?"
        )

        self.assertEqual(result.decision.route, "WEB")
        self.assertEqual(provider.queries, ["What happened recently in Transformer research?"])
        self.assertEqual(result.web_sources, (self.source,))
        self.assertEqual(result.web_sources[0].url, "https://research.example.org/transformers")
        self.assertEqual(result.web_sources[0].domain, "research.example.org")

    def test_web_route_without_client_returns_empty_sources(self) -> None:
        result = ResearchRouter().route("What are the latest Transformer developments?")

        self.assertEqual(result.decision.route, "WEB")
        self.assertEqual(result.retrieved_chunks, ())
        self.assertEqual(result.web_sources, ())

    def test_web_search_client_delegates_to_provider(self) -> None:
        provider = FakeSearchProvider([self.source])
        client = WebSearchClient(provider)

        sources = client.search("Transformer update")

        self.assertEqual(provider.queries, ["Transformer update"])
        self.assertEqual(sources, (self.source,))

    def test_web_search_client_rejects_empty_query(self) -> None:
        client = WebSearchClient(FakeSearchProvider([self.source]))

        with self.assertRaisesRegex(ValueError, "query must not be empty"):
            client.search("   ")

    def test_hybrid_route_uses_both_boundaries(self) -> None:
        retriever = FakeRetriever([self.chunk])
        provider = FakeSearchProvider([self.source])
        router = ResearchRouter(
            retriever=retriever,
            web_search=WebSearchClient(provider),
        )

        result = router.route(
            "How does this paper compare with recent Transformer approaches?"
        )

        self.assertEqual(result.decision.route, "HYBRID")
        self.assertEqual(result.retrieved_chunks, (self.chunk,))
        self.assertEqual(result.web_sources, (self.source,))
        self.assertEqual(len(retriever.queries), 1)
        self.assertEqual(len(provider.queries), 1)

    def test_default_routing_requires_no_ollama_or_network(self) -> None:
        result = ResearchRouter().route("What are the main results of the paper?")

        self.assertEqual(result.decision.route, "RAG")
        self.assertEqual(result.retrieved_chunks, ())
        self.assertEqual(result.web_sources, ())


if __name__ == "__main__":
    unittest.main()
