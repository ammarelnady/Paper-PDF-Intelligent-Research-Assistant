"""
Citation Formatter module.

Normalizes inline citations, extracts source references, and builds
citation cards for API and UI presentation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from app.contracts import WebSource
from app.rag.vector_store import RetrievedChunk


class CitationFormatter:
    """Formats answer text with clean citations and generates structured source metadata."""

    @staticmethod
    def format_citations(
        answer: str,
        chunks: Sequence[RetrievedChunk] = (),
        web_sources: Sequence[WebSource] = (),
    ) -> Dict[str, Any]:
        """
        Produce formatted answer text along with structured citation metadata.
        """
        paper_citations: List[Dict[str, Any]] = []
        for c in chunks:
            paper_citations.append({
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "page_number": c.page_number,
                "section": c.section,
                "score": round(float(c.score), 4),
                "snippet": c.text[:250] + ("..." if len(c.text) > 250 else ""),
            })

        web_citations: List[Dict[str, Any]] = []
        for w in web_sources:
            web_citations.append({
                "title": w.title,
                "url": w.url,
                "domain": w.domain,
                "snippet": w.snippet[:250] + ("..." if len(w.snippet) > 250 else ""),
            })

        # Build clean markdown bibliography/sources footer if not present
        footer_parts: List[str] = []
        if paper_citations:
            footer_parts.append("### 📚 Paper Citations")
            for c in paper_citations:
                footer_parts.append(
                    f"- **[Page {c['page_number']} | {c['section']}]** `{c['chunk_id']}` (Score: {c['score']}):\n"
                    f"  > *\"{c['snippet'][:150]}...\"*"
                )

        if web_citations:
            footer_parts.append("### 🌐 External Web Sources")
            for w in web_citations:
                footer_parts.append(f"- **[{w['title']}]({w['url']})** ({w['domain']})")

        formatted_answer = answer.strip()

        citation_pattern = re.compile(r"\[(?:Page\s+\d+[^\]]*|Chunk:\s*[^\]]+|Web:\s*[^\]]+)\]", re.IGNORECASE)
        has_inline_citations = bool(citation_pattern.search(formatted_answer))
        return {
            "answer": formatted_answer,
            "paper_citations": paper_citations,
            "web_citations": web_citations,
            "has_citations": has_inline_citations and (len(paper_citations) > 0 or len(web_citations) > 0),
        }
