"""A provider-agnostic web search abstraction with no network implementation."""

from __future__ import annotations

from typing import Protocol, Sequence

from app.contracts import WebSource


class SearchProvider(Protocol):
    """Implemented by an adapter for a chosen web-search service."""

    def search(self, query: str) -> Sequence[WebSource]:
        """Return source metadata and snippets for a query."""


class WebSearchClient:
    """Stable search boundary; provider adapters are injected by composition."""

    def __init__(self, provider: SearchProvider) -> None:
        self._provider = provider

    def search(self, query: str) -> tuple[WebSource, ...]:
        if not query.strip():
            raise ValueError("query must not be empty")
        return tuple(self._provider.search(query))
