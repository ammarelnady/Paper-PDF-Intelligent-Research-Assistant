"""
Paper Summarization Module.

Generates overall structured summaries and section-level summaries grounded in the
extracted paper content. Supports both configurable LLM generation and robust local
extractive summarization with full source chunk traceability.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set, Union

from app.config import settings
from app.document_processing.chunker import chunk_document
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.summary import PaperSummary
from app.summarization.prompts import (
    GROUNDEDNESS_SYSTEM_INSTRUCTION,
    OVERALL_PAPER_SUMMARY_PROMPT,
    SECTION_SUMMARY_PROMPT,
)

logger = logging.getLogger(__name__)


class PaperSummarizer:
    """
    Summarizer for research papers providing structured summaries,
    section-level summaries, and groundedness evaluation.
    """

    def __init__(
        self,
        strategy: str = settings.DEFAULT_SUMMARY_STRATEGY,
        llm_caller: Optional[Callable[[str, str], str]] = None,
        model_name: Optional[str] = None,
    ):
        """
        Args:
            strategy: 'extractive', 'llm', or 'auto'.
            llm_caller: Optional callable taking (prompt, system_prompt) -> response string.
            model_name: Name of model for metadata tracking.
        """
        self.strategy = strategy
        self.llm_caller = llm_caller
        self.model_name = model_name or ("extractive-nlp" if not llm_caller else "llm-summarizer")

    def summarize_paper(
        self,
        document: Document,
        chunks: Optional[List[DocumentChunk]] = None,
        strategy: Optional[str] = None,
    ) -> PaperSummary:
        """
        Generate a comprehensive, structured summary for an entire paper.

        Args:
            document: Processed Document object.
            chunks: Optional pre-computed DocumentChunk list.
            strategy: Optional override for summarization strategy ('extractive' or 'llm').

        Returns:
            PaperSummary: Grounded summary with key contributions, problem, methodology, etc.
        """
        if not document.pages and not document.full_text:
            return PaperSummary(
                document_id=document.document_id,
                summary="The document is empty and contains no extractable text.",
                section=None,
                source_chunks=[],
                model_used=self.model_name,
            )

        doc_chunks = chunks if chunks is not None else chunk_document(document)
        chosen_strategy = strategy or self.strategy

        if chosen_strategy == "llm" and self.llm_caller:
            try:
                return self._summarize_with_llm(document, doc_chunks)
            except Exception as e:
                logger.warning(f"LLM summarization failed ({e}); falling back to extractive summary.")

        return self._summarize_extractive(document, doc_chunks)

    def summarize_section(
        self,
        document: Document,
        section_name: str,
        chunks: Optional[List[DocumentChunk]] = None,
        strategy: Optional[str] = None,
    ) -> PaperSummary:
        """
        Generate a summary for a specific section of the paper.

        Args:
            document: Processed Document.
            section_name: Target section name (e.g. 'Methodology', 'Introduction').
            chunks: Optional pre-computed DocumentChunk list.
            strategy: Optional override for summarization strategy.

        Returns:
            PaperSummary with section set to section_name and source_chunks populated.
        """
        doc_chunks = chunks if chunks is not None else chunk_document(document)

        # Match chunks belonging to this section
        target_norm = section_name.strip().lower()
        matched_chunks = [
            c for c in doc_chunks if target_norm in c.section.lower() or c.section.lower() in target_norm
        ]

        if not matched_chunks:
            # Fallback to document section lookup
            section_text = document.get_text_for_section(section_name)
            if not section_text:
                return PaperSummary(
                    document_id=document.document_id,
                    summary=f"No content found for section '{section_name}'.",
                    section=section_name,
                    source_chunks=[],
                    model_used=self.model_name,
                )
            context_text = section_text
            source_chunk_ids = []
        else:
            context_text = "\n\n".join(c.text for c in matched_chunks)
            source_chunk_ids = [c.chunk_id for c in matched_chunks]

        chosen_strategy = strategy or self.strategy

        if chosen_strategy == "llm" and self.llm_caller:
            try:
                prompt = SECTION_SUMMARY_PROMPT.format(section_name=section_name, context=context_text[:4000])
                response = self.llm_caller(prompt, GROUNDEDNESS_SYSTEM_INSTRUCTION)
                return PaperSummary(
                    document_id=document.document_id,
                    summary=response.strip(),
                    section=section_name,
                    source_chunks=source_chunk_ids,
                    model_used=self.model_name,
                )
            except Exception as e:
                logger.warning(f"Section LLM summarization failed: {e}")

        # Extractive section summary
        sentences = self._extract_key_sentences(context_text, num_sentences=4)
        sec_summary = " ".join(sentences) if sentences else context_text[:300]

        return PaperSummary(
            document_id=document.document_id,
            summary=sec_summary,
            section=section_name,
            source_chunks=source_chunk_ids,
            model_used=f"{self.model_name} (extractive)",
        )

    def evaluate_groundedness(
        self, summary: str, source_text_or_chunks: Union[str, List[DocumentChunk]]
    ) -> float:
        """
        Compute a factual groundedness / token overlap score between summary and source.
        Returns a float score between 0.0 and 1.0.
        """
        if not summary or not summary.strip():
            return 0.0

        if isinstance(source_text_or_chunks, list):
            source_text = " ".join(c.text for c in source_text_or_chunks)
        else:
            source_text = str(source_text_or_chunks)

        if not source_text.strip():
            return 0.0

        def tokenize(text: str) -> Set[str]:
            words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())
            stop_words = {
                "the", "and", "for", "that", "this", "with", "from", "are", "was",
                "were", "been", "have", "has", "had", "will", "would", "can", "could",
                "about", "into", "through", "over", "after", "other", "some", "such"
            }
            return {w for w in words if w not in stop_words}

        summary_tokens = tokenize(summary)
        if not summary_tokens:
            return 1.0

        source_tokens = tokenize(source_text)
        overlap = summary_tokens.intersection(source_tokens)

        return round(len(overlap) / len(summary_tokens), 4)

    def _summarize_with_llm(self, document: Document, chunks: List[DocumentChunk]) -> PaperSummary:
        """Generate structured summary using configured LLM caller."""
        # Use top informative chunks or first N characters
        context_chunks = chunks[:12] if len(chunks) > 12 else chunks
        context_text = "\n\n".join(f"[{c.section} (Page {c.page_number})]: {c.text}" for c in context_chunks)
        source_chunk_ids = [c.chunk_id for c in context_chunks]

        prompt = OVERALL_PAPER_SUMMARY_PROMPT.format(context=context_text[:8000])
        raw_response = self.llm_caller(prompt, GROUNDEDNESS_SYSTEM_INSTRUCTION)  # type: ignore

        # Try to parse JSON output
        try:
            # Clean possible markdown ```json formatting
            cleaned_json = raw_response.strip()
            if cleaned_json.startswith("```"):
                cleaned_json = re.sub(r"^```(?:json)?\s*", "", cleaned_json)
                cleaned_json = re.sub(r"\s*```$", "", cleaned_json)

            data = json.loads(cleaned_json)
            return PaperSummary(
                document_id=document.document_id,
                summary=data.get("summary", raw_response),
                section=None,
                source_chunks=source_chunk_ids,
                key_contributions=data.get("key_contributions"),
                problem_statement=data.get("problem_statement"),
                methodology=data.get("methodology"),
                findings=data.get("findings"),
                limitations=data.get("limitations"),
                model_used=self.model_name,
            )
        except Exception:
            # Fallback if LLM output was plain text
            return PaperSummary(
                document_id=document.document_id,
                summary=raw_response.strip(),
                section=None,
                source_chunks=source_chunk_ids,
                model_used=self.model_name,
            )

    def _summarize_extractive(self, document: Document, chunks: List[DocumentChunk]) -> PaperSummary:
        """
        Generate grounded structured summary using NLP extractive sentence scoring
        and section heuristics without requiring external APIs.
        """
        source_chunk_ids = [c.chunk_id for c in chunks[:10]]
        full_text = document.full_text or "\n\n".join(p.text for p in document.pages)

        # 1. Overall Executive Summary
        # Extract top sentences from Abstract / Introduction / Conclusion if present
        high_value_sections = ["abstract", "introduction", "conclusion"]
        high_value_chunks = [c for c in chunks if any(s in c.section.lower() for s in high_value_sections)]
        target_text = "\n\n".join(c.text for c in high_value_chunks) if high_value_chunks else full_text

        overall_sentences = self._extract_key_sentences(target_text, num_sentences=5)
        overall_summary = " ".join(overall_sentences) if overall_sentences else full_text[:400]

        # 2. Problem Statement
        problem_statement = self._extract_specific_aspect(
            full_text,
            keywords=["problem", "challenge", "address", "aims to", "focuses on", "limitation of existing", "tackle"],
            section_preference=["abstract", "introduction"],
            chunks=chunks,
        )

        # 3. Methodology
        methodology = self._extract_specific_aspect(
            full_text,
            keywords=["we propose", "method", "architecture", "framework", "approach", "model", "algorithm", "technique"],
            section_preference=["methodology", "methods", "proposed approach", "system architecture"],
            chunks=chunks,
        )

        # 4. Key Contributions
        contributions = self._extract_contributions(full_text, chunks)

        # 5. Findings & Results
        findings = self._extract_specific_aspect(
            full_text,
            keywords=["results show", "outperforms", "achieves", "accuracy", "state-of-the-art", "evaluation shows", "findings indicate"],
            section_preference=["results", "experiments", "evaluation", "experiments and results"],
            chunks=chunks,
        )

        # 6. Limitations
        limitations = self._extract_specific_aspect(
            full_text,
            keywords=["limitation", "trade-off", "constraint", "future work", "bottleneck", "drawback", "ethical"],
            section_preference=["limitations", "discussion", "conclusion"],
            chunks=chunks,
        )

        return PaperSummary(
            document_id=document.document_id,
            summary=overall_summary,
            section=None,
            source_chunks=source_chunk_ids,
            key_contributions=contributions if contributions else None,
            problem_statement=problem_statement or "The paper investigates the stated research objectives.",
            methodology=methodology or "The authors present an experimental and algorithmic formulation.",
            findings=findings or "Empirical evaluations demonstrate the efficacy of the proposed method.",
            limitations=limitations,
            model_used=f"{self.model_name} (extractive)",
        )

    def _extract_key_sentences(self, text: str, num_sentences: int = 5) -> List[str]:
        """Extract highest-scoring sentences based on word frequency and position."""
        if not text:
            return []

        # Split into sentences
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip().replace("\n", " ") for s in raw_sentences if len(s.strip()) > 30]

        if not sentences:
            return []

        if len(sentences) <= num_sentences:
            return sentences

        # Word frequency scoring
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        stop_words = {
            "the", "and", "for", "that", "this", "with", "from", "are", "was",
            "were", "been", "have", "has", "had", "will", "would", "can", "could"
        }
        freq: Dict[str, int] = {}
        for w in words:
            if w not in stop_words:
                freq[w] = freq.get(w, 0) + 1

        max_f = max(freq.values()) if freq else 1

        # Score sentences
        scores: List[float] = []
        for idx, s in enumerate(sentences):
            s_words = re.findall(r"\b[a-zA-Z]{3,}\b", s.lower())
            w_score = sum(freq.get(w, 0) / max_f for w in s_words) / max(1, len(s_words))
            # Boost early sentences (lead bias in scientific abstracts/sections)
            pos_boost = 1.2 if idx < 3 else 1.0
            scores.append(w_score * pos_boost)

        # Select top sentences maintaining original order
        top_indices = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:num_sentences]
        return [sentences[i] for i in sorted(top_indices)]

    def _extract_specific_aspect(
        self,
        full_text: str,
        keywords: List[str],
        section_preference: List[str],
        chunks: List[DocumentChunk],
    ) -> Optional[str]:
        """Extract a coherent sentence or paragraph addressing a specific research aspect."""
        # Check preferred sections first
        preferred_chunks = [
            c for c in chunks if any(p in c.section.lower() for p in section_preference)
        ]
        target_text = "\n\n".join(c.text for c in preferred_chunks) if preferred_chunks else full_text

        raw_sentences = re.split(r"(?<=[.!?])\s+", target_text)
        matched_sentences: List[str] = []

        for s in raw_sentences:
            s_clean = s.strip().replace("\n", " ")
            if len(s_clean) < 30:
                continue
            s_lower = s_clean.lower()
            if any(kw in s_lower for kw in keywords):
                matched_sentences.append(s_clean)
                if len(matched_sentences) >= 2:
                    break

        if matched_sentences:
            return " ".join(matched_sentences)
        return None

    def _extract_contributions(self, full_text: str, chunks: List[DocumentChunk]) -> List[str]:
        """Extract bullet points of key contributions."""
        contributions: List[str] = []

        # Look for bullet points or contribution indicators
        intro_chunks = [c for c in chunks if "intro" in c.section.lower() or "abstract" in c.section.lower()]
        text_to_search = "\n".join(c.text for c in intro_chunks) if intro_chunks else full_text

        # Regex for enumerated contributions e.g. "(1) ... (2) ..." or "First, ... Second, ..."
        bullet_matches = re.findall(
            r"(?:(?:\n\s*[-•*]|\(\d+\)|\b[1-3]\.)\s+)([A-Z][^\n.!?]+(?:[.!?]))",
            text_to_search,
        )
        if bullet_matches:
            for b in bullet_matches[:4]:
                if len(b.strip()) > 20:
                    contributions.append(b.strip())

        if not contributions:
            # Fallback to sentences matching contribution verbs
            verbs = ["we introduce", "we present", "we propose", "our contribution", "our main contributions"]
            for s in re.split(r"(?<=[.!?])\s+", text_to_search):
                s_clean = s.strip().replace("\n", " ")
                if any(v in s_clean.lower() for v in verbs) and len(s_clean) > 30:
                    contributions.append(s_clean)
                    if len(contributions) >= 3:
                        break

        return contributions


def summarize_paper(
    document: Document,
    chunks: Optional[List[DocumentChunk]] = None,
    strategy: str = "auto",
) -> PaperSummary:
    """Convenience helper function to summarize an entire paper."""
    summarizer = PaperSummarizer(strategy=strategy)
    return summarizer.summarize_paper(document, chunks=chunks)


def summarize_section(
    document: Document,
    section_name: str,
    chunks: Optional[List[DocumentChunk]] = None,
    strategy: str = "auto",
) -> PaperSummary:
    """Convenience helper function to summarize a section of a paper."""
    summarizer = PaperSummarizer(strategy=strategy)
    return summarizer.summarize_section(document, section_name, chunks=chunks)
