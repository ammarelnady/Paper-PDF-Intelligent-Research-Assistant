"""
Suggested Questions API Endpoint.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from app.api.documents import ANALYSIS_REGISTRY, CHUNKS_REGISTRY, DOCUMENTS_REGISTRY
from app.api.schemas import SuggestedQuestionResponse
from app.questions import QuestionGenerator
from app.topics import PaperUnderstandingAnalyzer

router = APIRouter(prefix="/questions", tags=["Questions"])


@router.get("/{document_id}", response_model=List[SuggestedQuestionResponse])
def get_suggested_questions(document_id: str):
    """
    Get grounded suggested research questions for a specific document.
    """
    analysis = ANALYSIS_REGISTRY.get(document_id)
    if not analysis:
        doc = DOCUMENTS_REGISTRY.get(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        chunks = CHUNKS_REGISTRY.get(document_id, [])
        analyzer = PaperUnderstandingAnalyzer()
        analysis = analyzer.analyze(chunks)
        ANALYSIS_REGISTRY[document_id] = analysis

    return [
        SuggestedQuestionResponse(
            question_id=q.question_id,
            question=q.question,
            category=q.category,
            source_chunk_ids=list(q.source_chunk_ids),
        )
        for q in analysis.questions
    ]
