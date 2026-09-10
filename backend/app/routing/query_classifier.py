"""Rule-based and LLM-orchestrated query classification."""

from __future__ import annotations

import json
import re
from typing import Protocol

from app.contracts import RouteDecision
from app.routing.intent_detector import IntentDetector, QueryIntent
from app.llm.client import LLMClient


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


class LLMQueryClassifier:
    """Use the configured LLM as an orchestrator with deterministic fallback."""

    SYSTEM_PROMPT = (
        "You are a query-routing orchestrator for a research assistant. "
        "Choose the minimum useful evidence sources. Return JSON only, with "
        "exactly the keys route and confidence. route must be RAG, WEB, or HYBRID; "
        "confidence must be a number from 0 to 1."
    )

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        fallback: DeterministicQueryClassifier | None = None,
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._fallback = fallback or DeterministicQueryClassifier()
        self._intent_detector = IntentDetector()

    def classify(self, query: str) -> RouteDecision:
        """Ask the LLM to select a route, falling back safely on any failure."""
        intent = self._intent_detector.detect(query)
        prompt = f"""Select a route for this user query.

Routes:
- RAG: answer from the uploaded paper only.
- WEB: answer from current or external research only.
- HYBRID: combine the uploaded paper and external research.

Signals detected by the fast path: {intent}
User query: {query}

Return JSON only, for example: {{"route": "RAG", "confidence": 0.91}}"""
        try:
            raw = self._llm.generate(
                prompt=prompt,
                system_instruction=self.SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=80,
            )
            data = self._parse_json(raw)
            route = str(data.get("route", "")).upper()
            confidence = float(data.get("confidence", 0.0))
            if route not in {"RAG", "WEB", "HYBRID"}:
                raise ValueError("LLM returned an unsupported route")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("LLM returned an invalid confidence")
            return RouteDecision(route=route, confidence=confidence)
        except Exception:
            # The application remains usable if the model is unavailable or returns prose.
            return self._fallback.classify(query)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError("LLM route response must be a JSON object")
        return value
