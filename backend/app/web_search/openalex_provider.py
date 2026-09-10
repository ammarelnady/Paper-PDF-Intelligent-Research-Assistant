"""Key-free academic search fallback using the OpenAlex works API."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.contracts import WebSource


OpenAlexTransport = Callable[[str, float], str]
_ENDPOINT = "https://api.openalex.org/works"


class OpenAlexSearchProvider:
    """Return real academic sources when HTML search is blocked or empty."""

    def __init__(self, *, timeout_seconds: float = 12.0, max_results: int = 5, transport: OpenAlexTransport | None = None):
        if timeout_seconds <= 0 or max_results <= 0:
            raise ValueError("timeout_seconds and max_results must be positive")
        self._timeout_seconds = timeout_seconds
        self._max_results = max_results
        self._transport = transport or _fetch_json

    def search(self, query: str) -> tuple[WebSource, ...]:
        normalized = query.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        url = f"{_ENDPOINT}?{urlencode({'search': normalized, 'per-page': self._max_results})}"
        payload = json.loads(self._transport(url, self._timeout_seconds))
        results = payload.get("results", []) if isinstance(payload, dict) else []
        sources: list[WebSource] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("display_name") or item.get("title") or "").strip()
            location = item.get("primary_location") or {}
            url_value = location.get("landing_page_url") or item.get("doi") or item.get("id")
            if not title or not url_value:
                continue
            snippet = _abstract(item.get("abstract_inverted_index"))
            if not snippet:
                year = item.get("publication_year") or ""
                authors = ", ".join(
                    str((author.get("author") or {}).get("display_name", ""))
                    for author in (item.get("authorships") or [])[:3]
                    if isinstance(author, dict)
                )
                snippet = f"Academic work published in {year}. Authors: {authors}.".strip()
            domain = url_value.split("/")[2] if "://" in url_value else "openalex.org"
            sources.append(WebSource(title=title, url=url_value, domain=domain, snippet=snippet[:500]))
            if len(sources) >= self._max_results:
                break
        return tuple(sources)


def _fetch_json(url: str, timeout_seconds: float) -> str:
    request = Request(url, headers={"User-Agent": "PaperResearchAssistant/1.0 (academic-search)"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read().decode(response.headers.get_content_charset() or "utf-8", errors="replace")


def _abstract(inverted: Any) -> str:
    if not isinstance(inverted, dict):
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        if isinstance(positions, list):
            words.extend((int(position), str(word)) for position in positions if isinstance(position, int))
    return " ".join(word for _, word in sorted(words))
