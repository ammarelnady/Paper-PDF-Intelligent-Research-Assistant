"""
API Pydantic Schemas for Request and Response models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    title: Optional[str] = None
    num_pages: int
    num_chunks: int
    num_sections: int
    message: str = "Document uploaded, processed, and indexed successfully."


class SectionResponse(BaseModel):
    section_id: str
    title: str
    raw_heading: str
    start_page: int
    end_page: int
    order: int


class PaperSummaryResponse(BaseModel):
    document_id: str
    summary: str
    section: Optional[str] = None
    key_contributions: Optional[List[str]] = None
    problem_statement: Optional[str] = None
    methodology: Optional[str] = None
    findings: Optional[str] = None
    limitations: Optional[str] = None
    source_chunks: List[str] = Field(default_factory=list)
    model_used: Optional[str] = None


class TopicItem(BaseModel):
    topic_id: str
    name: str
    score: float
    source_chunk_ids: List[str] = Field(default_factory=list)


class KeywordItem(BaseModel):
    text: str
    score: float
    source_chunk_ids: List[str] = Field(default_factory=list)


class ConceptItem(BaseModel):
    concept_id: str
    name: str
    definition: Optional[str] = None
    source_chunk_ids: List[str] = Field(default_factory=list)


class PaperUnderstandingResponse(BaseModel):
    document_id: str
    topics: List[TopicItem] = Field(default_factory=list)
    keywords: List[KeywordItem] = Field(default_factory=list)
    concepts: List[ConceptItem] = Field(default_factory=list)


class SuggestedQuestionResponse(BaseModel):
    question_id: str
    question: str
    category: str
    source_chunk_ids: List[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str
    document_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    strategy: Optional[Literal["faiss", "hybrid_rrf"]] = None


class PaperCitation(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    section: str
    score: float
    snippet: str


class WebCitation(BaseModel):
    title: str
    url: str
    domain: str
    snippet: str


class QueryResponse(BaseModel):
    query: str
    route: str  # 'RAG', 'WEB', 'HYBRID'
    confidence: float
    answer: str
    paper_citations: List[PaperCitation] = Field(default_factory=list)
    web_citations: List[WebCitation] = Field(default_factory=list)
    has_citations: bool = False
