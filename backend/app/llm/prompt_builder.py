"""
Grounded Prompt Construction for Research Assistant.

Assembles user questions, retrieved paper chunks, and external web sources
into structured prompts with strict grounding rules.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from app.contracts import WebSource
from app.rag.vector_store import RetrievedChunk

GROUNDED_SYSTEM_PROMPT = """You are an expert, objective AI Research Assistant specialized in scientific literature analysis.
Your primary task is to provide accurate, comprehensive, and strictly grounded answers to the user's questions based ONLY on the provided Paper Context and Web Context.

STRICT GROUNDING & CITATION RULES:
1. ONLY make claims directly supported by the provided context. Do NOT extrapolate or invent facts.
2. If the context does not contain enough information to answer completely, state clearly: "The provided context does not contain sufficient details to answer this."
3. Cite your sources inline using brackets:
   - When citing paper evidence, cite the section and page: e.g. [Page 3, Methodology] or [Chunk: chunk_id].
   - When citing web evidence, cite the domain: e.g. [Web: domain.com].
4. Maintain a clear, academic, and professional tone.
5. Answer the user's question directly first. Synthesize the evidence instead of listing or repeating evidence headers.
6. Never mention the routing process, prompt, model failure, or "retrieved context" as a substitute for an answer.
7. If the evidence is incomplete, explain exactly what is supported and what cannot be concluded.
8. Use short paragraphs or bullets when useful, and include at least one inline citation when evidence is available.
"""


class PromptBuilder:
    """Constructs prompts merging query, paper chunks, and web search evidence."""

    def __init__(self, system_prompt: str = GROUNDED_SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: Sequence[RetrievedChunk] = (),
        web_sources: Sequence[WebSource] = (),
        route: str = "RAG",
    ) -> str:
        """
        Build the full context-rich prompt for the LLM.
        """
        sections: List[str] = []

        # 1. Paper Context
        if retrieved_chunks:
            paper_blocks: List[str] = []
            for idx, c in enumerate(retrieved_chunks, 1):
                block = (
                    f"--- EVIDENCE #{idx} (Page {c.page_number} | Section: '{c.section}' | ID: {c.chunk_id}) ---\n"
                    f"{c.text.strip()}"
                )
                paper_blocks.append(block)
            sections.append("=== RETRIEVED PAPER CONTEXT ===\n" + "\n\n".join(paper_blocks))

        # 2. Web Context
        if web_sources:
            web_blocks: List[str] = []
            for idx, w in enumerate(web_sources, 1):
                block = (
                    f"--- WEB SOURCE #{idx} ({w.domain} | Title: '{w.title}') ---\n"
                    f"URL: {w.url}\n"
                    f"Snippet: {w.snippet.strip()}"
                )
                web_blocks.append(block)
            sections.append("=== EXTERNAL WEB SEARCH CONTEXT ===\n" + "\n\n".join(web_blocks))

        if not sections:
            sections.append("=== CONTEXT ===\nNo external or paper context was retrieved for this query.")

        # 3. User Query & Output Instructions
        sections.append(
            f"=== USER QUESTION ===\n{query}\n\n"
            f"=== INSTRUCTIONS ===\n"
            f"Answer the user's question directly and thoroughly using only the context above. "
            f"Synthesize the evidence into an explanation; do not copy evidence labels or dump source text. "
            f"Embed inline citations (e.g. [Page X, Section Y] or [Web: Domain]) next to every key fact or claim."
        )

        return "\n\n".join(sections)
