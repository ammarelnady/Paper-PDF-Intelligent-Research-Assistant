"""
PaperSummary model — output contract of the summarization module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class PaperSummary(BaseModel):
    """
    Output of the summarization module.

    Fixed contract per spec: document_id, summary, section, source_chunks.
    `section=None` means this is the overall paper summary; otherwise
    it is a section-level summary for that section title.
    """

    document_id: str
    summary: str
    section: Optional[str] = None
    source_chunks: List[str] = Field(default_factory=list)

    # Structured, paper-specific fields (populated for the overall summary).
    key_contributions: Optional[List[str]] = None
    problem_statement: Optional[str] = None
    methodology: Optional[str] = None
    findings: Optional[str] = None
    limitations: Optional[str] = None

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_used: Optional[str] = None
