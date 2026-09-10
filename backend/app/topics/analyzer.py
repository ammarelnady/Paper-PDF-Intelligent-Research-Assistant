"""High-level orchestration for the Paper Understanding feature."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Callable, Optional

from app.questions.question_generator import QuestionGenerator

from app._shared_contracts import PaperAnalysis
from .concept_extractor import ConceptExtractor
from .keyword_extractor import KeywordExtractor
from .section_ranker import ImportantSectionIdentifier
from app.chunk_adapter import normalize_chunks
from .topic_extractor import TopicExtractor
from .types import UnderstandingResult


class PaperUnderstandingAnalyzer:
    """Run all understanding stages once over a paper's chunks."""

    def __init__(
        self,
        *,
        max_keywords: int = 15,
        max_topics: int = 8,
        max_concepts: int = 12,
        max_sections: int = 6,
        max_questions: int = 8,
        llm_caller: Optional[Callable[[str, str], str]] = None,
    ) -> None:
        self.keyword_extractor = KeywordExtractor(max_keywords=max_keywords)
        self.topic_extractor = TopicExtractor(max_topics=max_topics)
        self.concept_extractor = ConceptExtractor(max_concepts=max_concepts)
        self.section_identifier = ImportantSectionIdentifier(max_sections=max_sections)
        self.question_generator = QuestionGenerator(
            max_questions=max_questions,
            llm_caller=llm_caller,
        )

    def analyze(self, chunks: Iterable[Any]) -> UnderstandingResult:
        normalized_chunks = normalize_chunks(chunks)

        # Each line represents one stage in the pipeline.
        keywords = self.keyword_extractor.extract(normalized_chunks)
        topics = self.topic_extractor.extract(normalized_chunks)
        concepts = self.concept_extractor.extract(normalized_chunks)
        important_sections = self.section_identifier.identify(normalized_chunks)
        questions = self.question_generator.generate(
            normalized_chunks,
            topics=topics,
            concepts=concepts,
        )

        analysis = PaperAnalysis(
            document_id=normalized_chunks[0].document_id,
            topics=topics,
            concepts=concepts,
        )
        return UnderstandingResult(
            document_id=normalized_chunks[0].document_id,
            analysis=analysis,
            keywords=keywords,
            important_sections=important_sections,
            questions=questions,
        )


__all__ = ["PaperUnderstandingAnalyzer"]
