"""
Summarization Evaluation Metrics: Faithfulness, Content Coverage, and Compression Ratio.
"""

from __future__ import annotations

import re
from typing import Dict, Set


def compute_faithfulness(summary: str, source_text: str) -> float:
    """Calculate token-level factual groundedness score between 0.0 and 1.0."""
    if not summary.strip() or not source_text.strip():
        return 0.0

    def get_keywords(text: str) -> Set[str]:
        words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
        stop_words = {
            "the", "and", "for", "that", "this", "with", "from", "are", "was",
            "were", "been", "have", "has", "had", "will", "would", "can", "could",
            "about", "into", "through", "over", "after", "other", "some", "such"
        }
        return {w for w in words if w not in stop_words}

    sum_tokens = get_keywords(summary)
    if not sum_tokens:
        return 1.0

    src_tokens = get_keywords(source_text)
    overlap = sum_tokens.intersection(src_tokens)
    return round(len(overlap) / len(sum_tokens), 4)


def compute_section_coverage(summary: str, key_sections: Dict[str, str]) -> Dict[str, float]:
    """Calculate coverage percentage across key sections."""
    coverage: Dict[str, float] = {}
    sum_lower = summary.lower()
    for sec_name, sec_text in key_sections.items():
        if not sec_text:
            coverage[sec_name] = 0.0
            continue
        words = set(re.findall(r"\b[a-zA-Z]{4,}\b", sec_text.lower()))
        if not words:
            coverage[sec_name] = 0.0
            continue
        overlap = [w for w in words if w in sum_lower]
        coverage[sec_name] = round(len(overlap) / len(words), 4)
    return coverage
