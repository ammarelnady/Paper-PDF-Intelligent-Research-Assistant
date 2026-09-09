"""A lightweight DuckDuckGo HTML search provider with no API-key requirement."""

from __future__ import annotations

from collections.abc import Callable
from html.parser import HTMLParser
import re
from typing import Final
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from app.contracts import WebSource


SearchTransport = Callable[[str, float], str]
_SEARCH_ENDPOINT: Final = "https://html.duckduckgo.com/html/"
_USER_AGENT: Final = "PaperResearchAssistant/1.0 (web-search boundary)"


class WebSearchProviderError(RuntimeError):
    """Raised when a provider cannot complete a search request."""


class DuckDuckGoHtmlSearchProvider:
    """Search DuckDuckGo's HTML endpoint and map valid entries to WebSource."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        max_results: int = 5,
        transport: SearchTransport | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_results <= 0:
            raise ValueError("max_results must be positive")
        self._timeout_seconds = timeout_seconds
        self._max_results = max_results
        self._transport = transport or _fetch_html

    def search(self, query: str) -> tuple[WebSource, ...]:
        """Search for a query, returning valid source metadata or a provider error."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        request_url = f"{_SEARCH_ENDPOINT}?{urlencode({'q': normalized_query})}"
        try:
            response_html = self._transport(request_url, self._timeout_seconds)
        except Exception as error:
            raise WebSearchProviderError("DuckDuckGo search request failed") from error

        if not isinstance(response_html, str):
            raise WebSearchProviderError("DuckDuckGo returned a non-text response")

        parser = _DuckDuckGoResultParser()
        try:
            parser.feed(response_html)
            parser.close()
            raw_results = parser.results()
        except Exception as error:
            raise WebSearchProviderError("DuckDuckGo response could not be parsed") from error

        sources: list[WebSource] = []
        for title, href, snippet in raw_results:
            source = _to_web_source(title, href, snippet)
            if source is not None:
                sources.append(source)
            if len(sources) == self._max_results:
                break
        return tuple(sources)


def _fetch_html(url: str, timeout_seconds: float) -> str:
    """Fetch a DuckDuckGo HTML response using only the standard library."""
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=timeout_seconds) as response:
        response_bytes = response.read()
        content_type = response.headers.get_content_charset() or "utf-8"
    return response_bytes.decode(content_type, errors="replace")


class _DuckDuckGoResultParser(HTMLParser):
    """Extract result title, link, and snippet fields from DuckDuckGo HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._results: list[tuple[str, str, str]] = []
        self._current: dict[str, list[str] | str] | None = None
        self._capture: str | None = None
        self._capture_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self._finish_current()
            self._current = {
                "title": [],
                "href": attributes.get("href") or "",
                "snippet": [],
            }
            self._start_capture("title", tag)
        elif self._current is not None and "result__snippet" in classes:
            self._start_capture("snippet", tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == self._capture_tag:
            self._capture = None
            self._capture_tag = None

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._capture is not None:
            captured = self._current[self._capture]
            assert isinstance(captured, list)
            captured.append(data)

    def close(self) -> None:
        super().close()
        self._finish_current()

    def results(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(self._results)

    def _start_capture(self, field: str, tag: str) -> None:
        self._capture = field
        self._capture_tag = tag

    def _finish_current(self) -> None:
        if self._current is None:
            return
        title = _normalize_text(self._current["title"])
        href = self._current["href"]
        snippet = _normalize_text(self._current["snippet"])
        assert isinstance(href, str)
        self._results.append((title, href, snippet))
        self._current = None
        self._capture = None
        self._capture_tag = None


def _to_web_source(title: str, href: str, snippet: str) -> WebSource | None:
    normalized_title = _normalize_text(title)
    normalized_snippet = _normalize_text(snippet)
    normalized_url = _normalize_url(href)
    if not normalized_title or not normalized_url:
        return None

    domain = urlparse(normalized_url).hostname
    if not domain:
        return None
    return WebSource(
        title=normalized_title,
        url=normalized_url,
        domain=domain.lower(),
        snippet=normalized_snippet,
    )


def _normalize_url(href: str) -> str | None:
    candidate = href.strip()
    if candidate.startswith("//"):
        candidate = f"https:{candidate}"
    parsed = urlparse(candidate)

    if parsed.hostname and parsed.hostname.lower().endswith("duckduckgo.com"):
        redirect_url = parse_qs(parsed.query).get("uddg", [None])[0]
        if redirect_url:
            candidate = redirect_url
            parsed = urlparse(candidate)

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return urlunparse(parsed)


def _normalize_text(value: object) -> str:
    if isinstance(value, list):
        value = " ".join(value)
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()
