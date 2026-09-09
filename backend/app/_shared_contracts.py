"""Temporary compatibility layer for the frozen v1 shared contracts.

Salma owns ``app.models`` and is still finalising it. Shahd's feature imports
the shared models when they exist and otherwise uses equivalent local fallback
classes. Keeping this module at package level avoids coupling topics and
questions to each other.
"""

from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - exercised after the shared models are integrated
    from app.models.contracts import (  # type: ignore[import-not-found]
        Concept,
        PaperAnalysis,
        SuggestedQuestion,
        Topic,
    )
except (ImportError, ModuleNotFoundError):

    @dataclass(frozen=True)
    class Topic:
        topic_id: str
        name: str
        score: float
        source_chunk_ids: list[str]

    @dataclass(frozen=True)
    class Concept:
        name: str
        score: float
        source_chunk_ids: list[str]

    @dataclass(frozen=True)
    class PaperAnalysis:
        document_id: str
        topics: list[Topic]
        concepts: list[Concept]

    @dataclass(frozen=True)
    class SuggestedQuestion:
        question_id: str
        document_id: str
        question: str
        source_chunk_ids: list[str]
        category: str


__all__ = ["Concept", "PaperAnalysis", "SuggestedQuestion", "Topic"]
