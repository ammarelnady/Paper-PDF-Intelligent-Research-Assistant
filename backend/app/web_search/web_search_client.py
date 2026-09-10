"""A provider-agnostic web search abstraction with no network implementation."""

from __future__ import annotations

from typing import Protocol, Sequence

from app.contracts import WebSource
from app.web_search.query_preparation import SearchQueryPreparer


class WebSearchProviderError(RuntimeError):
    """Expected network/provider failure at the web-search boundary."""


class SearchProvider(Protocol):
    """Implemented by an adapter for a chosen web-search service."""

    def search(self, query: str) -> Sequence[WebSource]:
        """Return source metadata and snippets for a query."""


class WebSearchClient:
    """Stable search boundary; provider adapters are injected by composition."""

    def __init__(
        self, provider: SearchProvider, query_preparer: SearchQueryPreparer | None = None
    ) -> None:
        self._provider = provider
        self._query_preparer = query_preparer or SearchQueryPreparer()

    def search(self, query: str) -> tuple[WebSource, ...]:
        if not query.strip():
            raise ValueError("query must not be empty")
        # Providers receive concise keywords.  The API and answer generation
        # retain the original query supplied by the user.
        return tuple(self._provider.search(self._query_preparer.prepare(query)))


class FallbackSearchProvider:
    """Try providers in order and continue when one is blocked or empty."""

    def __init__(self, providers: Sequence[SearchProvider]) -> None:
        if not providers:
            raise ValueError("at least one search provider is required")
        self._providers = tuple(providers)

    def search(self, query: str) -> tuple[WebSource, ...]:
        errors: list[WebSearchProviderError] = []
        for provider in self._providers:
            try:
                sources = tuple(provider.search(query))
                if sources:
                    return sources
            except WebSearchProviderError as error:
                errors.append(error)
        if errors:
            raise WebSearchProviderError("All configured web-search providers failed") from errors[-1]
        return ()
