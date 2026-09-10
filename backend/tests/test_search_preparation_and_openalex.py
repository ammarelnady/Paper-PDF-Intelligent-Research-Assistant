"""Offline tests for search preparation and OpenAlex boundary failures."""

from __future__ import annotations

import json
import unittest
from urllib.error import HTTPError

from app.web_search import OpenAlexSearchProvider, SearchQueryPreparer, WebSearchProviderError


class SearchPreparationTests(unittest.TestCase):
    def test_rag_question_keeps_concepts_and_expands_abbreviation(self) -> None:
        prepared = SearchQueryPreparer().prepare("What is the latest research on RAG?").lower()
        for term in ("latest", "research", "retrieval", "augmented", "generation", "rag"):
            self.assertIn(term, prepared)

    def test_gpt_and_llm_questions_are_search_friendly(self) -> None:
        self.assertIn("openai", SearchQueryPreparer().prepare("What is the latest version of GPT?").lower())
        self.assertIn("large language models", SearchQueryPreparer().prepare(
            "What are the recent developments in large language models?"
        ).lower())


class OpenAlexProviderTests(unittest.TestCase):
    def test_valid_academic_response_maps_result(self) -> None:
        payload = {"results": [{
            "display_name": "Retrieval Augmented Generation",
            "primary_location": {"landing_page_url": "https://example.edu/paper"},
            "publication_year": 2025,
        }]}
        provider = OpenAlexSearchProvider(transport=lambda _url, _timeout: json.dumps(payload))
        sources = provider.search("latest research on RAG")
        self.assertEqual(sources[0].title, "Retrieval Augmented Generation")
        self.assertEqual(sources[0].domain, "example.edu")

    def test_empty_and_malformed_responses_are_safe(self) -> None:
        self.assertEqual(OpenAlexSearchProvider(transport=lambda _u, _t: '{"results": []}').search("RAG"), ())
        with self.assertRaisesRegex(WebSearchProviderError, "malformed JSON"):
            OpenAlexSearchProvider(transport=lambda _u, _t: "not json").search("RAG")

    def test_http_error_is_wrapped(self) -> None:
        def failing(_url: str, _timeout: float) -> str:
            raise HTTPError("https://api.openalex.org/works", 400, "Bad Request", None, None)

        with self.assertRaisesRegex(WebSearchProviderError, "OpenAlex search request failed"):
            OpenAlexSearchProvider(transport=failing).search("natural language query")
