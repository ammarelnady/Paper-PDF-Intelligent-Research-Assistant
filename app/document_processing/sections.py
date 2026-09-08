from __future__ import annotations

import re

from app.models.document import PageContent, SectionSpan

_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+.{2,80}$")
_KNOWN_HEADINGS = {"abstract", "introduction", "background", "related work", "method", "methods", "methodology", "materials and methods", "results", "discussion", "conclusion", "conclusions", "references", "acknowledgments", "appendix"}


def assign_sections(pages: tuple[PageContent, ...]) -> tuple[PageContent, ...]:
    """Detect section transitions and retain section spans within each page."""
    current_section: str | None = None
    assigned: list[PageContent] = []
    for page in pages:
        spans, current_section = _section_spans(page.text, current_section)
        page_section = spans[0].section if spans else current_section
        assigned.append(PageContent(
            document_id=page.document_id,
            page_number=page.page_number,
            text=page.text,
            section=page_section,
            section_spans=spans,
        ))
    return tuple(assigned)


def _section_spans(text: str, initial_section: str | None) -> tuple[tuple[SectionSpan, ...], str | None]:
    """Return contiguous character spans, changing section at each heading line."""
    if not text:
        return (), initial_section

    current_section = initial_section
    span_start = 0
    spans: list[SectionSpan] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if _is_heading(line):
            if offset > span_start:
                spans.append(SectionSpan(current_section, span_start, offset))
            current_section = _clean_heading(line)
            span_start = offset
        offset += len(line)
    if span_start < len(text):
        spans.append(SectionSpan(current_section, span_start, len(text)))
    return tuple(spans), current_section


def _is_heading(line: str) -> bool:
    text = line.strip()
    if not text or len(text) > 100 or text.endswith((".", ",", ";", ":", "?", "!")):
        return False
    if _clean_heading(text).lower() in _KNOWN_HEADINGS or _NUMBERED_HEADING.match(text):
        return True
    return 1 <= len(text.split()) <= 10 and (text.isupper() or text.istitle())


def _clean_heading(text: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*[.)]?\s*", "", text).strip()
