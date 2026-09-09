"""Generate diverse, source-linked questions from paper analysis."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from app._shared_contracts import SuggestedQuestion
from app.chunk_adapter import normalize_chunks, remove_duplicates


QUESTION_RULES = [
    {
        "category": "methodology",
        "signals": ["method", "methodology", "approach", "architecture"],
        "template": "How does the methodology described in the {section} section work?",
    },
    {
        "category": "dataset",
        "signals": ["dataset", "data", "corpus"],
        "template": (
            "What data or dataset is used in the {section} section, "
            "and how is it prepared?"
        ),
    },
    {
        "category": "results",
        "signals": ["result", "evaluation", "experiment", "finding"],
        "template": "What are the main findings reported in the {section} section?",
    },
    {
        "category": "limitations",
        "signals": ["limitation", "threat", "future work"],
        "template": (
            "What limitations or future-work directions are identified in "
            "the {section} section?"
        ),
    },
    {
        "category": "summary",
        "signals": ["abstract", "introduction", "conclusion"],
        "template": (
            "What central problem and contribution are presented in "
            "the {section} section?"
        ),
    },
]


def find_question_rule(text: str) -> dict[str, Any] | None:
    """Return the first question rule whose signal appears in the text."""

    lowercase_text = text.lower()
    for rule in QUESTION_RULES:
        if any(signal in lowercase_text for signal in rule["signals"]):
            return rule
    return None


class QuestionGenerator:
    def __init__(self, *, max_questions: int = 8) -> None:
        if max_questions < 1:
            raise ValueError("max_questions must be at least 1")
        self.max_questions = max_questions

    def generate(
        self,
        chunks: Iterable[Any],
        *,
        topics: Sequence[Any] = (),
        concepts: Sequence[Any] = (),
    ) -> list[SuggestedQuestion]:
        normalized_chunks = normalize_chunks(chunks)
        document_id = normalized_chunks[0].document_id
        candidates: list[tuple[str, str, list[str]]] = []
        seen_sections: set[tuple[str, str]] = set()

        # First, create questions from meaningful sections in the paper.
        for chunk in normalized_chunks:
            section = chunk.section or "relevant paper content"
            rule = find_question_rule(f"{section} {chunk.text[:300]}")
            if rule is None:
                continue

            category = rule["category"]
            section_key = (category, section.lower())
            if section_key in seen_sections:
                continue

            question = rule["template"].format(section=section)
            candidates.append((category, question, [chunk.chunk_id]))
            seen_sections.add(section_key)

        # Then, add questions about the extracted topics and concepts.
        for topic in topics:
            candidates.append(
                (
                    "comparison",
                    f"How does the paper position {topic.name} relative to the other approaches it discusses?",
                    list(topic.source_chunk_ids),
                )
            )
        for concept in concepts:
            candidates.append(
                (
                    "concept",
                    f"What is {concept.name}, and what role does it play in this paper?",
                    list(concept.source_chunk_ids),
                )
            )

        if not candidates:
            candidates.append(
                (
                    "summary",
                    "What problem does this paper address, and what is its main contribution?",
                    [normalized_chunks[0].chunk_id],
                )
            )

        # Remove repeated questions and assign stable IDs.
        questions: list[SuggestedQuestion] = []
        seen_text: set[str] = set()
        for category, question, source_ids in candidates:
            if question.lower() in seen_text:
                continue
            seen_text.add(question.lower())
            questions.append(
                SuggestedQuestion(
                    question_id=f"question-{len(questions) + 1:03d}",
                    document_id=document_id,
                    question=question,
                    source_chunk_ids=remove_duplicates(source_ids),
                    category=category,
                )
            )
            if len(questions) == self.max_questions:
                break
        return questions


__all__ = ["QuestionGenerator"]
