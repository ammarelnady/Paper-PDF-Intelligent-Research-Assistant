from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SectionSpan:
    """A character range of page text associated with one active section."""

    section: str | None
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class PageContent:
    document_id: str
    page_number: int
    text: str
    section: str | None = None
    section_spans: tuple[SectionSpan, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    document_id: str
    page_number: int
    section: str | None
    chunk_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Document:
    """Shared document contract with Phase 1 processing metadata retained."""

    document_id: str
    filename: str
    page_count: int
    # These fields preserve the existing Phase 1 in-memory processing result.
    # Future modules should use the frozen fields above as their shared contract.
    source_path: Path | None = None
    pages: tuple[PageContent, ...] = field(default_factory=tuple)
    chunks: tuple[DocumentChunk, ...] = field(default_factory=tuple)
