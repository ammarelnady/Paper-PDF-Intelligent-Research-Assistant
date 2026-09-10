from app.routing import ResearchRouter
from app.web_search import (
    WebSearchClient,
    DuckDuckGoHtmlSearchProvider,
    OpenAlexSearchProvider,
    FallbackSearchProvider,
)

fallback = FallbackSearchProvider(
    [
        DuckDuckGoHtmlSearchProvider(max_results=5),
        OpenAlexSearchProvider(max_results=5),
    ]
)

router = ResearchRouter(
    web_search=WebSearchClient(provider=fallback)
)

queries = [
    "What is the latest research on RAG?",
    "What are the recent developments in large language models?",
    "What methodology does this paper use?",
    "Compare this paper with recent research on transformers.",
    "What is the latest version of GPT?",
]

for query in queries:
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")

    try:
        result = router.route(query)

        print(f"ROUTE: {result.decision.route}")
        print(f"CONFIDENCE: {result.decision.confidence}")
        print(f"WEB RESULTS: {len(result.web_sources)}")

        for i, source in enumerate(result.web_sources, 1):
            print(f"  {i}. {source.title}")
            print(f"     {source.url}")

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")