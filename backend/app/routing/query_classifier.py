"""Deterministic query classifier with an optional ambiguity fallback boundary."""

from __future__ import annotations

from typing import Protocol

from app.contracts import RouteDecision
from app.routing.intent_detector import IntentDetector, QueryIntent


class ClassifierFallback(Protocol):
    """Optional model-backed classifier; no provider is required by default."""

    def classify(self, query: str, intent: QueryIntent) -> RouteDecision | None:
        """Return a decision for an ambiguous query, or None to retain rules."""


class QueryClassifier(Protocol):
    def classify(self, query: str) -> RouteDecision:
        """Classify a query into RAG, WEB, or HYBRID."""


class DeterministicQueryClassifier:
    """Rule-first classifier; an injected fallback runs only below the threshold."""

    def __init__(
        self,
        intent_detector: IntentDetector | None = None,
        fallback: ClassifierFallback | None = None,
        fallback_threshold: float = 0.60,
    ) -> None:
        if not 0.0 <= fallback_threshold <= 1.0:
            raise ValueError("fallback_threshold must be between 0.0 and 1.0")
        self._intent_detector = intent_detector or IntentDetector()
        self._fallback = fallback
        self._fallback_threshold = fallback_threshold

    def classify(self, query: str) -> RouteDecision:
        intent = self._intent_detector.detect(query)
        rule_decision = self._classify_intent(intent)
        if self._fallback and rule_decision.confidence < self._fallback_threshold:
            fallback_decision = self._fallback.classify(query, intent)
            if fallback_decision is not None:
                return fallback_decision
        return rule_decision

    @staticmethod
    def _classify_intent(intent: QueryIntent) -> RouteDecision:
        if intent.mixed_knowledge:
            return RouteDecision(route="HYBRID", confidence=0.94)
        if intent.current_information:
            return RouteDecision(route="WEB", confidence=0.92)
        if intent.external_research:
            return RouteDecision(route="WEB", confidence=0.82)
        if intent.paper_specific:
            return RouteDecision(route="RAG", confidence=0.91)
        if intent.comparison:
            return RouteDecision(route="HYBRID", confidence=0.55)
        return RouteDecision(route="HYBRID", confidence=0.40)
