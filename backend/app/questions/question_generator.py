"""Generate diverse, source-linked questions from paper analysis."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
import re
from typing import Any, Callable, Optional

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
    def __init__(
        self,
        *,
        max_questions: int = 8,
        llm_caller: Optional[Callable[[str, str], str]] = None,
    ) -> None:
        if max_questions < 1:
            raise ValueError("max_questions must be at least 1")
        self.max_questions = max_questions
        self.llm_caller = llm_caller

    def generate(
        self,
        chunks: Iterable[Any],
        *,
        topics: Sequence[Any] = (),
        concepts: Sequence[Any] = (),
    ) -> list[SuggestedQuestion]:
        normalized_chunks = normalize_chunks(chunks)
        document_id = normalized_chunks[0].document_id

        if self.llm_caller:
            llm_questions = self._generate_with_llm(
                normalized_chunks, topics=topics, concepts=concepts
            )
            if llm_questions:
                return llm_questions

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

    def _generate_with_llm(
        self,
        chunks: Sequence[Any],
        *,
        topics: Sequence[Any],
        concepts: Sequence[Any],
    ) -> list[SuggestedQuestion]:
        """Generate source-linked questions from the paper using structured JSON."""
        allowed_ids = {chunk.chunk_id for chunk in chunks}
        context = "\n\n".join(
            f"[{chunk.chunk_id}] Section: {chunk.section}\n{chunk.text[:700]}"
            for chunk in chunks[:12]
        )
        topic_names = ", ".join(str(topic.name) for topic in topics[:8]) or "none"
        concept_names = ", ".join(str(concept.name) for concept in concepts[:8]) or "none"
        prompt = f"""Generate up to {self.max_questions} insightful research questions grounded in this paper.

Return JSON only as an array of objects with exactly these keys:
category, question, source_chunk_ids.
Categories should be meaningful values such as summary, methodology, dataset,
results, limitations, or concept. Every source_chunk_ids value must come from
the chunk IDs shown below. Do not invent facts or ask questions unrelated to
the supplied paper.

Topics: {topic_names}
Concepts: {concept_names}

Paper context:
{context}
"""
        try:
            raw = self.llm_caller(prompt, "You create grounded research questions. Return valid JSON only.")
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
            try:
                items = json.loads(cleaned)
            except json.JSONDecodeError:
                match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
                if not match:
                    return []
                items = json.loads(match.group(0))
            if not isinstance(items, list):
                return []

            questions: list[SuggestedQuestion] = []
            seen: set[str] = set()
            for item in items:
                if not isinstance(item, dict):
                    continue
                question = str(item.get("question", "")).strip()
                category = str(item.get("category", "summary")).strip().lower() or "summary"
                source_ids = [str(value) for value in item.get("source_chunk_ids", []) if str(value) in allowed_ids]
                if not question or not source_ids or question.lower() in seen:
                    continue
                seen.add(question.lower())
                questions.append(
                    SuggestedQuestion(
                        question_id=f"question-{len(questions) + 1:03d}",
                        document_id=chunks[0].document_id,
                        question=question,
                        source_chunk_ids=remove_duplicates(source_ids),
                        category=category,
                    )
                )
                if len(questions) >= self.max_questions:
                    break
            return questions
        except Exception:
            return []


__all__ = ["QuestionGenerator"]
