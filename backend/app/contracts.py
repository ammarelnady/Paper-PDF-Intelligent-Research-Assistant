"""Small shared data contracts used by the Research Agent."""

from __future__ import annotations

from dataclasses import dataclass


ALLOWED_ROUTES = frozenset({"RAG", "WEB", "HYBRID"})


@dataclass(frozen=True)
class RouteDecision:
    """The selected knowledge route and the classifier's confidence."""

    route: str
    confidence: float

    def __post_init__(self) -> None:
        if self.route not in ALLOWED_ROUTES:
            raise ValueError(f"Unsupported route: {self.route}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class RetrievedChunk:
    """A provenance-preserving result supplied by an external retriever."""

    document_id: str
    chunk_id: str
    text: str
    page_number: int
    section: str | None
    score: float


@dataclass(frozen=True)
class WebSource:
    """A provenance-preserving result supplied by a web-search provider."""

    title: str
    url: str
    domain: str
    snippet: str
