"""Provider-neutral web search boundary."""

from .duckduckgo_provider import DuckDuckGoHtmlSearchProvider
from .openalex_provider import OpenAlexSearchProvider
from .query_preparation import LLMSearchQueryRewriter, SearchQueryPreparer
from .web_search_client import FallbackSearchProvider, SearchProvider, WebSearchClient, WebSearchProviderError


def build_search_provider(provider_name: str) -> SearchProvider:
    """Build the configured key-free provider chain without changing .env names."""
    name = provider_name.strip().lower()
    duckduckgo = DuckDuckGoHtmlSearchProvider()
    openalex = OpenAlexSearchProvider()
    if name == "openalex":
        return FallbackSearchProvider([openalex, duckduckgo])
    # ``duckduckgo`` remains the compatible default.  ``auto`` also keeps the
    # general-web source first; OpenAlex supplies academic fallback coverage.
    return FallbackSearchProvider([duckduckgo, openalex])

__all__ = [
    "DuckDuckGoHtmlSearchProvider",
    "OpenAlexSearchProvider",
    "FallbackSearchProvider",
    "SearchProvider",
    "WebSearchClient",
    "WebSearchProviderError",
    "SearchQueryPreparer",
    "LLMSearchQueryRewriter",
    "build_search_provider",
]
