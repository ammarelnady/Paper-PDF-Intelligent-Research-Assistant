"""
DocumentChunk model — the metadata-aware chunk contract shared with
the rest of the team (RAG indexing, citations).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.document import new_id


class DocumentChunk(BaseModel):
    """
    A metadata-aware chunk of a document, ready for embedding/retrieval.

    Fixed contract per the project spec:
    document_id, chunk_id, page_number, section, text
    """

    document_id: str
    chunk_id: str = Field(default_factory=lambda: new_id("chunk"))
    page_number: int
    section: str = Field(default="Unknown", description="Section title this chunk falls under.")
    text: str

    # Extra metadata (additive — does not break the fixed contract above).
    chunk_index: int = Field(default=0, description="Order of this chunk within the document.")
    token_estimate: int = Field(default=0, description="Rough token count, for budget-aware retrieval.")

    def model_post_init(self, __context) -> None:
        if not self.token_estimate:
            # Cheap heuristic: ~4 chars per token for English text.
            self.token_estimate = max(1, len(self.text) // 4)
