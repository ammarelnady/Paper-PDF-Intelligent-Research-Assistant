"""
Retrieval Evaluation Metrics: Recall@K, Precision@K, Mean Reciprocal Rank (MRR).
"""

from __future__ import annotations

from typing import List, Sequence, Set


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Calculate Recall@K."""
    if not gold_ids:
        return 1.0
    top_k = retrieved_ids[:k]
    matched = set(top_k).intersection(set(gold_ids))
    return len(matched) / len(set(gold_ids))


def precision_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Calculate Precision@K."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    matched = set(top_k).intersection(set(gold_ids))
    return len(matched) / min(k, len(top_k))


def mean_reciprocal_rank(queries_retrieved: Sequence[Sequence[str]], queries_gold: Sequence[Sequence[str]]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) across a list of test queries."""
    if not queries_retrieved or not queries_gold:
        return 0.0

    reciprocal_ranks: List[float] = []
    for retrieved, gold in zip(queries_retrieved, queries_gold):
        gold_set = set(gold)
        rr = 0.0
        for rank, item in enumerate(retrieved, start=1):
            if item in gold_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
