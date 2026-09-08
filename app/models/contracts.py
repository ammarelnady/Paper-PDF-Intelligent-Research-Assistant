"""Framework-independent shared contracts for future project modules.

These classes define data shapes only. They intentionally contain no provider,
retrieval, routing, or generation logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Topic:
    topic_id: str
    name: str
    score: float
    source_chunk_ids: list[str]


@dataclass(frozen=True, slots=True)
class Concept:
    name: str
    score: float
    source_chunk_ids: list[str]


@dataclass(frozen=True, slots=True)
class PaperAnalysis:
    document_id: str
    topics: list[Topic]
    concepts: list[Concept]


@dataclass(frozen=True, slots=True)
class SuggestedQuestion:
    question_id: str
    document_id: str
    question: str
    source_chunk_ids: list[str]
    category: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    document_id: str
    chunk_id: str
    text: str
    page_number: int
    section: str | None
    score: float


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: str
    confidence: float


@dataclass(frozen=True, slots=True)
class WebSource:
    title: str
    url: str
    domain: str
    snippet: str


@dataclass(frozen=True, slots=True)
class Citation:
    citation_id: str
    source_type: str
    source_id: str
    page_number: int | None
    section: str | None
    title: str | None
    url: str | None


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: list[Citation]
