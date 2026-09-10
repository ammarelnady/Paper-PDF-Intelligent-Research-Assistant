"""Offline integration coverage for LLM classification through web search."""

from __future__ import annotations

import unittest

from app.contracts import RetrievedChunk, WebSource
from app.routing import LLMQueryClassifier, ResearchRouter
from app.web_search import SearchQueryPreparer, WebSearchClient


class _QuestionAwareLLM:
    def generate(self, *, prompt: str, **_kwargs: object) -> str:
        question = prompt.split("User question:", 1)[1].split("Decision examples:", 1)[0].lower()
        if "methodology" in question:
            return '{"route":"RAG","confidence":0.95}'
        if "compare this paper" in question:
            return '{"route":"HYBRID","confidence":0.95}'
        return '{"route":"WEB","confidence":0.95}'


class _Retriever:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str):
        self.calls.append(query)
        return (RetrievedChunk("paper", "chunk", "method", 1, "Methods", 0.9),)


class _DuckDuckGoMock:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str):
        self.calls.append(query)
        return (WebSource("Current source", "https://example.org/current", "example.org", "snippet"),)


class RoutingIntegrationTests(unittest.TestCase):
    def test_realistic_questions_route_and_search_as_expected(self) -> None:
        retriever = _Retriever()
        duckduckgo = _DuckDuckGoMock()
        router = ResearchRouter(
            classifier=LLMQueryClassifier(_QuestionAwareLLM()),
            retriever=retriever,
            web_search=WebSearchClient(duckduckgo, SearchQueryPreparer()),
        )
        cases = {
            "What is the latest research on RAG?": "WEB",
            "What are the recent developments in large language models?": "WEB",
            "What methodology does this paper use?": "RAG",
            "Compare this paper with recent research on transformers.": "HYBRID",
            "What is the latest version of GPT?": "WEB",
        }
        results = {query: router.route(query) for query in cases}
        self.assertEqual({query: result.decision.route for query, result in results.items()}, cases)
        self.assertEqual(len(retriever.calls), 2)  # RAG and HYBRID only
        self.assertEqual(len(duckduckgo.calls), 4)  # WEB and HYBRID only
        self.assertTrue(any("retrieval augmented generation" in query.lower() for query in duckduckgo.calls))
        self.assertTrue(any("openai" in query.lower() for query in duckduckgo.calls))
