"""
Routing Classification Evaluation Metrics.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def evaluate_routing_accuracy(
    predictions: Sequence[str], ground_truth: Sequence[str]
) -> Dict[str, float]:
    """Compute overall accuracy and per-route precision."""
    if not predictions or not ground_truth or len(predictions) != len(ground_truth):
        return {"accuracy": 0.0}

    correct = sum(1 for p, g in zip(predictions, ground_truth) if p.upper() == g.upper())
    total = len(predictions)

    stats: Dict[str, float] = {"accuracy": round(correct / total, 4)}

    routes = set(g.upper() for g in ground_truth)
    for r in routes:
        r_pred = [p.upper() == r for p in predictions]
        r_gold = [g.upper() == r for g in ground_truth]
        tp = sum(1 for p, g in zip(r_pred, r_gold) if p and g)
        pred_pos = sum(1 for p in r_pred if p)
        stats[f"precision_{r}"] = round(tp / pred_pos, 4) if pred_pos else 0.0

    return stats
