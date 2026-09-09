"""
Document + PageContent models.

Document is the top-level processed-paper object: page-by-page raw
content plus detected sections. Other model modules (chunk.py,
summary.py) import `new_id` from here to keep id generation consistent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    """Generate a short, human-readable unique id like 'doc_3f9a1c'."""
    return f"{prefix}_{uuid4().hex[:8]}"


class PageContent(BaseModel):
    """Raw extracted content for a single PDF page."""

    page_number: int = Field(..., description="1-indexed page number.")
    text: str = Field(..., description="Extracted plain text for this page.")
    char_count: int = Field(default=0, description="len(text), cached for convenience.")

    def model_post_init(self, __context) -> None:
        if not self.char_count:
            self.char_count = len(self.text)


class SectionInfo(BaseModel):
    """A detected logical section of the paper (e.g. 'Methodology')."""

    section_id: str = Field(default_factory=lambda: new_id("sec"))
    title: str = Field(..., description="Normalized section title, e.g. 'Introduction'.")
    raw_heading: str = Field(..., description="Heading text as it appeared in the PDF.")
    start_page: int
    end_page: int
    order: int = Field(..., description="Position of this section in reading order.")


class Document(BaseModel):
    """A processed paper: metadata + full page/section breakdown."""

    document_id: str = Field(default_factory=lambda: new_id("doc"))
    filename: str
    title: Optional[str] = None
    num_pages: int = 0
    pages: List[PageContent] = Field(default_factory=list)
    sections: List[SectionInfo] = Field(default_factory=list)
    full_text: str = Field(default="", description="Concatenated, cleaned page text.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def get_section_for_page(self, page_number: int) -> Optional[str]:
        """Return the section title covering a given page, if any."""
        for section in self.sections:
            if section.start_page <= page_number <= section.end_page:
                return section.title
        return None
