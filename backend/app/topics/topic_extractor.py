"""Topic extraction grounded in detected sections and document keywords."""
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from app._shared_contracts import Topic
from .keyword_extractor import KeywordExtractor
from app.chunk_adapter import normalize_chunks, remove_duplicates


GENERIC_SECTIONS = {"abstract", "introduction", "background", "conclusion", "references"}


class TopicExtractor:
    def __init__(self, *, max_topics: int = 8) -> None:
        if max_topics < 1:
            raise ValueError("max_topics must be at least 1")
        self.max_topics = max_topics

    def extract(self, chunks: Iterable[Any]) -> list[Topic]:
        normalized_chunks = normalize_chunks(chunks)

        # Section headings are useful human-readable topic names.
        by_section: dict[str, list[str]] = defaultdict(list)
        section_size: dict[str, int] = defaultdict(int)
        for chunk in normalized_chunks:
            if chunk.section:
                section = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", chunk.section).strip()
                is_page_label = bool(re.fullmatch(r"page\s+\d+", section, flags=re.IGNORECASE))
                if section and section.lower() not in GENERIC_SECTIONS and not is_page_label:
                    by_section[section].append(chunk.chunk_id)
                    section_size[section] += len(chunk.text)

        candidates: list[tuple[str, float, list[str]]] = []
        total_size = sum(section_size.values()) or 1
        for section, source_ids in by_section.items():
            score = min(1.0, 0.55 + 0.45 * section_size[section] / total_size)
            candidates.append((section, score, remove_duplicates(source_ids)))

        # Keywords fill any remaining topic slots when headings are not enough.
        keywords = KeywordExtractor(max_keywords=self.max_topics * 3).extract(
            normalized_chunks
        )
        used_words = {name.lower() for name, _, _ in candidates}
        for keyword in keywords:
            if keyword.text in used_words:
                continue
            # Multiword phrases make stronger topic labels.  High-scoring single
            # terms are used only when sections do not provide enough coverage.
            if " " not in keyword.text and len(candidates) >= self.max_topics // 2:
                continue
            candidates.append((keyword.text.title(), keyword.score * 0.85, keyword.source_chunk_ids))
            used_words.add(keyword.text)
            if len(candidates) >= self.max_topics * 2:
                break

        candidates.sort(key=lambda item: (-item[1], item[0].lower()))
        return [
            Topic(
                topic_id=f"topic-{index:03d}",
                name=name,
                score=round(score, 4),
                source_chunk_ids=source_ids,
            )
            for index, (name, score, source_ids) in enumerate(
                candidates[: self.max_topics], start=1
            )
        ]


__all__ = ["TopicExtractor"]
