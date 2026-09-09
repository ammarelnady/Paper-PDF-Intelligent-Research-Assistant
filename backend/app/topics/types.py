"""Internal typed values used by the paper-understanding pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from app._shared_contracts import PaperAnalysis, SuggestedQuestion


@dataclass(frozen=True)
class Keyword:
    text: str
    score: float
    source_chunk_ids: list[str]


@dataclass(frozen=True)
class ImportantSection:
    section: str
    score: float
    source_chunk_ids: list[str]
    reasons: list[str]


@dataclass(frozen=True)
class UnderstandingResult:
    """Complete Shahd feature output, including frozen shared contracts."""

    document_id: str
    analysis: PaperAnalysis
    keywords: list[Keyword]
    important_sections: list[ImportantSection]
    questions: list[SuggestedQuestion]


__all__ = ["ImportantSection", "Keyword", "UnderstandingResult"]
