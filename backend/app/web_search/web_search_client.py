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


class FallbackSearchProvider:
    """Try providers in order and continue when one is blocked or empty."""

    def __init__(self, providers: Sequence[SearchProvider]) -> None:
        if not providers:
            raise ValueError("at least one search provider is required")
        self._providers = tuple(providers)

    def search(self, query: str) -> tuple[WebSource, ...]:
        last_error: Exception | None = None
        for provider in self._providers:
            try:
                sources = tuple(provider.search(query))
                if sources:
                    return sources
            except Exception as error:
                last_error = error
        if last_error:
            raise last_error
        return ()
