"""
LLM Answer Evaluation Metrics: Groundedness and Hallucination Rate.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Set


def _stem(word: str) -> str:
    """Basic rule-based suffix normalization for English tokens."""
    w = word.lower()
    for suffix in ("ation", "tions", "tion", "ing", "ed", "es", "s", "ly", "al", "ment"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[:-len(suffix)]
    return w


def evaluate_hallucination_rate(answer: str, context_text: str) -> float:
    """
    Estimate hallucination rate as the ratio of key factual tokens in the answer
    that have zero presence in the supporting context (using suffix normalization).
    Returns a score between 0.0 (perfectly grounded) and 1.0 (completely hallucinated).
    """
    if not answer.strip():
        return 0.0
    if not context_text.strip():
        return 1.0

    stop_words = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was",
        "were", "been", "have", "has", "had", "will", "would", "can", "could",
        "about", "into", "through", "over", "after", "other", "some", "such",
        "also", "than", "then", "which", "when", "where", "what", "their", "perform"
    }

    def extract_entity_stems(text: str) -> Set[str]:
        words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
        return {_stem(w) for w in words if w not in stop_words and not w.isdigit()}

    ans_stems = extract_entity_stems(answer)
    if not ans_stems:
        return 0.0

    ctx_stems = extract_entity_stems(context_text)
    unsupported = [w for w in ans_stems if w not in ctx_stems]

    return round(len(unsupported) / len(ans_stems), 4)


def evaluate_answer_groundedness(answer: str, context_text: str) -> float:
    """Return groundedness score: 1.0 - hallucination_rate."""
    hr = evaluate_hallucination_rate(answer, context_text)
    return round(1.0 - hr, 4)
