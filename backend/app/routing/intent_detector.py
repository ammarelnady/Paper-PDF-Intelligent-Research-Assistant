"""Intent signal detection, kept separate from route selection."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class QueryIntent:
    """Signals that describe a query without choosing a downstream route."""

    paper_specific: bool
    current_information: bool
    external_research: bool
    comparison: bool

    @property
    def mixed_knowledge(self) -> bool:
        """Whether the query explicitly combines the paper and outside context."""
        return self.paper_specific and (
            self.current_information or self.external_research or self.comparison
        )


class IntentDetector:
    """Lightweight, explainable intent detector for the routing fast path."""

    _PAPER_PATTERNS = (
        r"\bthis paper\b",
        r"\bthe paper\b",
        r"\bthis study\b",
        r"\bthe study\b",
        r"\bthe authors?\b",
        r"\bthe document\b",
        r"\bthe dataset (?:did|used|use)\b",
        r"\b(?:main )?results?\b",
        r"\bmethodolog(?:y|ies)\b",
        r"\b(?:the )?experimental (?:setup|results?)\b",
    )
    _CURRENT_PATTERNS = (
        r"\blatest\b",
        r"\brecent(?:ly)?\b",
        r"\bcurrent\b",
        r"\btoday\b",
        r"\bnew developments?\b",
        r"\bstate[- ]of[- ]the[- ]art\b",
        r"\bwhat happened\b",
    )
    _EXTERNAL_PATTERNS = (
        r"\bexternal research\b",
        r"\bbeyond (?:the )?paper\b",
        r"\bin the literature\b",
        r"\b(?:the )?research literature\b",
        r"\boutside (?:this |the )?(?:paper|study)\b",
        r"\bother (?:research|studies|approaches)\b",
    )
    _COMPARISON_PATTERNS = (
        r"\bcompar(?:e|ed|ing|ison)\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bdiffer(?:ent|ence|s)?\b",
    )

    def detect(self, query: str) -> QueryIntent:
        """Extract coarse intent signals from a non-empty user query."""
        normalized = query.strip().lower()
        if not normalized:
            raise ValueError("query must not be empty")
        if not re.search(r"[^\W_]", normalized):
            raise ValueError("query must contain letters or digits")

        return QueryIntent(
            paper_specific=self._matches(normalized, self._PAPER_PATTERNS),
            current_information=self._matches(normalized, self._CURRENT_PATTERNS),
            external_research=self._matches(normalized, self._EXTERNAL_PATTERNS),
            comparison=self._matches(normalized, self._COMPARISON_PATTERNS),
        )

    @staticmethod
    def _matches(text: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)
