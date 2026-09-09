"""Route queries and invoke only the configured source boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.contracts import RetrievedChunk, RouteDecision, WebSource
from app.routing.query_classifier import DeterministicQueryClassifier, QueryClassifier
from app.routing.retrieval import Retriever
from app.web_search.web_search_client import WebSearchClient


@dataclass(frozen=True)
class RoutingResult:
    """A route decision and unmodified, provenance-bearing source results."""

    decision: RouteDecision
    retrieved_chunks: tuple[RetrievedChunk, ...] = ()
    web_sources: tuple[WebSource, ...] = ()


class ResearchRouter:
    """Coordinates source selection without knowing source implementation details."""

    def __init__(
        self,
        classifier: QueryClassifier | None = None,
        retriever: Retriever | None = None,
        web_search: WebSearchClient | None = None,
    ) -> None:
        self._classifier = classifier or DeterministicQueryClassifier()
        self._retriever = retriever
        self._web_search = web_search

    def route(self, query: str) -> RoutingResult:
        """Classify then query the source boundaries selected by that decision."""
        decision = self._classifier.classify(query)
        chunks: Sequence[RetrievedChunk] = ()
        sources: Sequence[WebSource] = ()

        if decision.route in {"RAG", "HYBRID"} and self._retriever is not None:
            chunks = self._retriever.search(query)
        if decision.route in {"WEB", "HYBRID"} and self._web_search is not None:
            sources = self._web_search.search(query)

        return RoutingResult(
            decision=decision,
            retrieved_chunks=tuple(chunks),
            web_sources=tuple(sources),
        )
