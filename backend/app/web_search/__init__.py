"""Provider-neutral web search boundary."""

from .duckduckgo_provider import DuckDuckGoHtmlSearchProvider, WebSearchProviderError
from .web_search_client import SearchProvider, WebSearchClient

__all__ = [
    "DuckDuckGoHtmlSearchProvider",
    "SearchProvider",
    "WebSearchClient",
    "WebSearchProviderError",
]
