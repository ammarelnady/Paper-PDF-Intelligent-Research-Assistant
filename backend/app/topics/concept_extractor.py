"""Scientific concept and acronym extraction."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from app._shared_contracts import Concept
from app.chunk_adapter import content_tokens, normalize_chunks, remove_duplicates


DEFINITION_RE = re.compile(
    r"""
    \b(
        [A-Z][A-Za-z-]+
        (?:\s+(?:of\s+|and\s+)?[A-Z][A-Za-z-]+){1,6}
    )
    \s*\(([A-Z][A-Z0-9-]{1,9})\)
    """,
    re.VERBOSE,
)
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,9}\b")


class ConceptExtractor:
    def __init__(self, *, max_concepts: int = 12, min_frequency: int = 2) -> None:
        if max_concepts < 1 or min_frequency < 1:
            raise ValueError("max_concepts and min_frequency must be positive")
        self.max_concepts = max_concepts
        self.min_frequency = min_frequency

    def extract(self, chunks: Iterable[Any]) -> list[Concept]:
        normalized_chunks = normalize_chunks(chunks)
        counts: Counter[str] = Counter()
        sources: dict[str, list[str]] = defaultdict(list)

        for chunk in normalized_chunks:
            found_in_chunk: set[str] = set()

            # Example: "Graph Neural Network (GNN)" is a strong concept.
            for full_name, acronym in DEFINITION_RE.findall(chunk.text):
                concept = f"{full_name} ({acronym})"
                # A definition is stronger evidence than an incidental repeated
                # n-gram, so it should win during overlap de-duplication.
                counts[concept] += 5
                found_in_chunk.add(concept)
            # Standalone uppercase abbreviations are weaker concept candidates.
            for acronym in ACRONYM_RE.findall(chunk.text):
                if acronym not in {"PDF", "FIG", "TABLE", "AND", "THE"}:
                    counts[acronym] += 1
                    found_in_chunk.add(acronym)

            # Repeated two- and three-word phrases can also represent concepts.
            words = content_tokens(chunk.text)
            for size in (2, 3):
                for index in range(len(words) - size + 1):
                    phrase = " ".join(words[index : index + size])
                    counts[phrase] += 1
                    found_in_chunk.add(phrase)
            for concept in found_in_chunk:
                sources[concept].append(chunk.chunk_id)

        eligible = [item for item in counts.items() if item[1] >= self.min_frequency]
        eligible.sort(key=lambda item: (-item[1], -len(item[0].split()), item[0].lower()))
        if not eligible:
            return []
        highest = eligible[0][1]
        selected: list[tuple[str, int]] = []
        for name, count in eligible:
            lower = name.lower()
            # Avoid returning a shorter n-gram when a stronger selected phrase
            # already contains it.
            if any(
                lower != chosen.lower()
                and (lower in chosen.lower() or chosen.lower() in lower)
                for chosen, _ in selected
            ):
                continue
            selected.append((name, count))
            if len(selected) == self.max_concepts:
                break
        return [
            Concept(
                name=name,
                score=round(count / highest, 4),
                source_chunk_ids=remove_duplicates(sources[name]),
            )
            for name, count in selected
        ]


__all__ = ["ConceptExtractor"]
