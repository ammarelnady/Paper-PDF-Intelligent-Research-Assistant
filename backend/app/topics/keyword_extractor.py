"""Keyword extraction using chunk-aware TF-IDF scoring."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from app.chunk_adapter import content_tokens, normalize_chunks
from .types import Keyword


class KeywordExtractor:
    """Extract reproducible unigram and bigram keywords from paper chunks."""

    def __init__(self, *, max_keywords: int = 15, include_bigrams: bool = True) -> None:
        if max_keywords < 1:
            raise ValueError("max_keywords must be at least 1")
        self.max_keywords = max_keywords
        self.include_bigrams = include_bigrams

    def extract(self, chunks: Iterable[Any]) -> list[Keyword]:
        normalized_chunks = normalize_chunks(chunks)
        term_counts_per_chunk: list[Counter[str]] = []
        source_chunks: dict[str, list[str]] = defaultdict(list)

        # Step 1: collect important words and two-word phrases from every chunk.
        for chunk in normalized_chunks:
            words = content_tokens(chunk.text)
            terms = list(words)
            if self.include_bigrams:
                bigrams = [f"{left} {right}" for left, right in zip(words, words[1:])]
                terms.extend(bigrams)

            counts = Counter(terms)
            term_counts_per_chunk.append(counts)
            for term in counts:
                source_chunks[term].append(chunk.chunk_id)

        # Step 2: calculate TF-IDF. Frequent terms get a high TF score, while
        # terms that appear in every chunk are reduced by IDF.
        document_frequency = Counter(
            term for chunk_counts in term_counts_per_chunk for term in chunk_counts
        )
        raw_scores: Counter[str] = Counter()
        number_of_chunks = len(normalized_chunks)

        for counts in term_counts_per_chunk:
            total = sum(counts.values()) or 1
            for term, count in counts.items():
                term_frequency = count / total
                inverse_document_frequency = (
                    math.log((1 + number_of_chunks) / (1 + document_frequency[term])) + 1
                )
                phrase_bonus = 1.25 if " " in term else 1.0
                raw_scores[term] += (
                    term_frequency * inverse_document_frequency * phrase_bonus
                )

        if not raw_scores:
            return []

        # Step 3: convert scores to the simple range 0..1 and return the top N.
        highest_score = max(raw_scores.values())
        ranked = sorted(raw_scores, key=lambda term: (-raw_scores[term], term))
        return [
            Keyword(
                text=term,
                score=round(raw_scores[term] / highest_score, 4),
                source_chunk_ids=source_chunks[term],
            )
            for term in ranked[: self.max_keywords]
        ]


__all__ = ["KeywordExtractor"]
