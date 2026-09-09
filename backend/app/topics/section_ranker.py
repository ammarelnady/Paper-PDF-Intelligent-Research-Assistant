"""Identify sections that are likely to contain the paper's core evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from app.chunk_adapter import normalize_chunks, remove_duplicates
from .types import ImportantSection


SECTION_SIGNALS = {
    "abstract": 0.75,
    "method": 1.0,
    "methodology": 1.0,
    "experiment": 0.95,
    "dataset": 0.9,
    "result": 1.0,
    "evaluation": 0.95,
    "discussion": 0.85,
    "limitation": 0.9,
    "conclusion": 0.8,
}


class ImportantSectionIdentifier:
    def __init__(self, *, max_sections: int = 6) -> None:
        if max_sections < 1:
            raise ValueError("max_sections must be at least 1")
        self.max_sections = max_sections

    def identify(self, chunks: Iterable[Any]) -> list[ImportantSection]:
        normalized_chunks = normalize_chunks(chunks)
        grouped: dict[str, list[Any]] = defaultdict(list)
        for chunk in normalized_chunks:
            grouped[chunk.section or "Unsectioned"].append(chunk)
        maximum_length = max(sum(len(chunk.text) for chunk in group) for group in grouped.values())

        ranked: list[ImportantSection] = []
        for section, group in grouped.items():
            lower = section.lower()
            matched = [signal for signal in SECTION_SIGNALS if signal in lower]
            signal_score = max((SECTION_SIGNALS[signal] for signal in matched), default=0.25)
            length_score = sum(len(chunk.text) for chunk in group) / maximum_length
            # Heading meaning is more important than section length.
            score = (0.75 * signal_score) + (0.25 * length_score)
            reasons = [f"heading matches '{signal}'" for signal in matched]
            if length_score >= 0.75:
                reasons.append("contains substantial paper content")
            if not reasons:
                reasons.append("ranked by content coverage")
            ranked.append(
                ImportantSection(
                    section=section,
                    score=round(min(score, 1.0), 4),
                    source_chunk_ids=remove_duplicates(
                        chunk.chunk_id for chunk in group
                    ),
                    reasons=reasons,
                )
            )
        return sorted(ranked, key=lambda item: (-item.score, item.section.lower()))[
            : self.max_sections
        ]


__all__ = ["ImportantSectionIdentifier"]
