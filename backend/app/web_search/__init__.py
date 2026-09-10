"""Provider-neutral web search boundary."""

from .duckduckgo_provider import DuckDuckGoHtmlSearchProvider, WebSearchProviderError
from .openalex_provider import OpenAlexSearchProvider
from .web_search_client import FallbackSearchProvider, SearchProvider, WebSearchClient

__all__ = [
    "DuckDuckGoHtmlSearchProvider",
    "OpenAlexSearchProvider",
    "FallbackSearchProvider",
    "SearchProvider",
    "WebSearchClient",
    "WebSearchProviderError",
]
