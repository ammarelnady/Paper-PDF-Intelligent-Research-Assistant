"""Prepare provider-neutral search queries without changing the user question."""

from __future__ import annotations

import json
import re
from typing import Protocol


class SearchQueryRewriter(Protocol):
    """Optional LLM boundary used to turn a question into search keywords."""

    def rewrite(self, query: str) -> str | None:
        """Return a concise search query, or ``None`` to use the safe fallback."""


class SearchQueryPreparer:
    """Keep original and provider-facing queries separate.

    An injected rewriter is normally an LLM orchestration component.  The small
    deterministic fallback intentionally only expands well-known abbreviations
    and removes conversational framing; it is not a second route classifier.
    """

    def __init__(self, rewriter: SearchQueryRewriter | None = None) -> None:
        self._rewriter = rewriter

    def prepare(self, original_query: str) -> str:
        query = original_query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if self._rewriter is not None:
            try:
                rewritten = self._rewriter.rewrite(query)
                if isinstance(rewritten, str) and self._is_usable(rewritten):
                    return self._clean(rewritten)
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        return self._deterministic_prepare(query)

    @staticmethod
    def _is_usable(value: str) -> bool:
        return bool(value.strip()) and len(value.strip()) <= 300

    @staticmethod
    def _clean(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    def _deterministic_prepare(self, query: str) -> str:
        lowered = query.lower()
        terms: list[str] = []
        if "latest" in lowered:
            terms.append("latest")
        elif "recent" in lowered or "current" in lowered:
            terms.append("recent")

        # These expansions improve provider recall while retaining the original
        # abbreviation, which is often the user's most precise term.
        if re.search(r"\brag\b", lowered):
            terms.extend(["research", "retrieval augmented generation", "RAG"])
        elif re.search(r"\bgpt\b", lowered):
            terms.extend(["GPT", "version", "OpenAI"])
        elif "large language model" in lowered or re.search(r"\bllms?\b", lowered):
            terms.extend(["developments", "large language models", "LLMs"])
        else:
            stripped = re.sub(
                r"^(what is|what are|what does|how does|tell me about)\s+",
                "",
                query.strip(),
                flags=re.IGNORECASE,
            ).rstrip("?.!")
            terms.append(stripped)
        return self._clean(" ".join(terms))


class LLMSearchQueryRewriter:
    """Small LLM adapter for search rewriting, with no routing responsibility."""

    _SYSTEM_PROMPT = (
        "Rewrite research questions into concise web-search keywords. Preserve "
        "the user's subject, expand useful abbreviations, and never answer the "
        "question. Return JSON only: {\"search_query\": \"...\"}."
    )

    def __init__(self, llm_client: object) -> None:
        self._llm_client = llm_client

    def rewrite(self, query: str) -> str | None:
        try:
            generate = getattr(self._llm_client, "generate")
            raw = generate(
                prompt=(
                    f"Question: {query}\n"
                    "Examples: latest research on RAG -> latest research retrieval "
                    "augmented generation RAG; latest version of GPT -> latest GPT "
                    "version OpenAI."
                ),
                system_instruction=self._SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=80,
            )
            data = json.loads(str(raw).strip().removeprefix("```json").removesuffix("```").strip())
            value = data.get("search_query") if isinstance(data, dict) else None
            return value if isinstance(value, str) else None
        except Exception:
            # The preparer will use its deterministic, provider-neutral fallback.
            return None
