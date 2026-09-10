"""Evaluation framework for Retrieval, Summarization, Routing, and LLM Groundedness."""

from app.evaluation.llm_eval import evaluate_answer_groundedness, evaluate_hallucination_rate
from app.evaluation.retrieval_eval import mean_reciprocal_rank, precision_at_k, recall_at_k
from app.evaluation.routing_eval import evaluate_routing_accuracy
from app.evaluation.summarization_eval import compute_faithfulness, compute_section_coverage

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mean_reciprocal_rank",
    "compute_faithfulness",
    "compute_section_coverage",
    "evaluate_routing_accuracy",
    "evaluate_hallucination_rate",
    "evaluate_answer_groundedness",
]
