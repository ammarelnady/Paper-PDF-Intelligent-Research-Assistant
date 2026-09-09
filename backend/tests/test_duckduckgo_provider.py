"""Offline unit tests for the DuckDuckGo HTML provider."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.web_search.duckduckgo_provider import (
    DuckDuckGoHtmlSearchProvider,
    WebSearchProviderError,
)


ONE_RESULT_HTML = """
<html><body>
  <a class="result__a" href="https://example.org/paper">Example &amp; Paper</a>
  <div class="result__snippet"> A useful  research result. </div>
</body></html>
"""

MULTIPLE_RESULT_HTML = """
<html><body>
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fone.example%2Farticle">One</a>
  <div class="result__snippet">First source</div>
  <a class="result__a" href="https://two.example/path">Two</a>
  <div class="result__snippet">Second source</div>
</body></html>
"""


class DuckDuckGoProviderTests(unittest.TestCase):
    def test_successful_search_maps_to_web_source(self) -> None:
        requests: list[tuple[str, float]] = []

        def transport(url: str, timeout: float) -> str:
            requests.append((url, timeout))
            return ONE_RESULT_HTML

        sources = DuckDuckGoHtmlSearchProvider(transport=transport).search("paper search")

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].title, "Example & Paper")
        self.assertEqual(sources[0].url, "https://example.org/paper")
        self.assertEqual(sources[0].domain, "example.org")
        self.assertEqual(sources[0].snippet, "A useful research result.")
        self.assertIn("q=paper+search", requests[0][0])

    def test_multiple_results_are_returned_and_redirect_urls_are_unwrapped(self) -> None:
        provider = DuckDuckGoHtmlSearchProvider(
            max_results=2,
            transport=lambda _url, _timeout: MULTIPLE_RESULT_HTML,
        )

        sources = provider.search("multiple")

        self.assertEqual([source.title for source in sources], ["One", "Two"])
        self.assertEqual(sources[0].url, "https://one.example/article")
        self.assertEqual(sources[0].domain, "one.example")

    def test_malformed_results_are_skipped(self) -> None:
        malformed_html = """
        <a class="result__a" href="javascript:alert(1)">Unsafe</a>
        <div class="result__snippet">Ignored</div>
        <a class="result__a" href="https://valid.example">Valid</a>
        <div class="result__snippet">Kept</div>
        """
        provider = DuckDuckGoHtmlSearchProvider(
            transport=lambda _url, _timeout: malformed_html,
        )

        sources = provider.search("malformed")

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].title, "Valid")

    def test_empty_results_return_an_empty_tuple(self) -> None:
        provider = DuckDuckGoHtmlSearchProvider(
            transport=lambda _url, _timeout: "<html><body>No results</body></html>",
        )

        self.assertEqual(provider.search("nothing"), ())

    def test_network_failure_raises_provider_error(self) -> None:
        def failing_transport(_url: str, _timeout: float) -> str:
            raise OSError("offline")

        provider = DuckDuckGoHtmlSearchProvider(transport=failing_transport)

        with self.assertRaisesRegex(WebSearchProviderError, "search request failed"):
            provider.search("network failure")

    def test_provider_rejects_empty_query(self) -> None:
        provider = DuckDuckGoHtmlSearchProvider(
            transport=lambda _url, _timeout: ONE_RESULT_HTML,
        )

        with self.assertRaisesRegex(ValueError, "query must not be empty"):
            provider.search("   ")


if __name__ == "__main__":
    unittest.main()
