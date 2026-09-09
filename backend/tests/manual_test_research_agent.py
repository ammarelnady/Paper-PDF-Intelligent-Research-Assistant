"""Manual live check for Research Agent routing and DuckDuckGo web search.

Run from ``backend`` with:
    set PYTHONPATH=.
    python tests/manual_test_research_agent.py
"""

from __future__ import annotations

from app.routing.query_classifier import DeterministicQueryClassifier
from app.routing.router import ResearchRouter
from app.web_search.duckduckgo_provider import (
    DuckDuckGoHtmlSearchProvider,
    WebSearchProviderError,
)
from app.web_search.web_search_client import WebSearchClient


SCENARIOS = (
    ("What dataset did the paper use?", "RAG"),
    ("What are the latest developments in Transformer architectures?", "WEB"),
    (
        "How does the Transformer architecture in this paper compare with recent approaches?",
        "HYBRID",
    ),
)


def main() -> None:
    classifier = DeterministicQueryClassifier()
    web_search = WebSearchClient(DuckDuckGoHtmlSearchProvider())
    router = ResearchRouter(classifier=classifier, web_search=web_search)

    for query, expected_route in SCENARIOS:
        decision = classifier.classify(query)
        if decision.route != expected_route:
            raise AssertionError(
                f"Expected {expected_route} for {query!r}, got {decision.route}"
            )

        print("=" * 70)
        print("QUERY:", query)
        print("EXPECTED ROUTE:", expected_route)

        try:
            result = router.route(query)
        except WebSearchProviderError as error:
            # Classification has still occurred through the concrete classifier.
            print("ROUTE:", decision.route)
            print("CONFIDENCE:", decision.confidence)
            print("WEB SEARCH: real DuckDuckGo provider request failed:", error)
            continue

        if result.decision.route != expected_route:
            raise AssertionError(
                f"Router returned {result.decision.route}, expected {expected_route}"
            )

        print("ROUTE:", result.decision.route)
        print("CONFIDENCE:", result.decision.confidence)

        if result.decision.route == "RAG":
            print("RAG: route selected, but no real retriever is connected yet.")

        if result.decision.route in {"WEB", "HYBRID"}:
            print("WEB SOURCES:")
            if result.web_sources:
                for source in result.web_sources:
                    print(source)
            else:
                print("No sources were returned by the real provider.")


if __name__ == "__main__":
    main()
