"""
Section and Heading Detector module for scientific papers.

Identifies logical sections (Abstract, Introduction, Methodology, Results, etc.)
from document pages and computes section page ranges.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from app.models.document import Document, SectionInfo, new_id

logger = logging.getLogger(__name__)


# Canonical academic section categories
CANONICAL_SECTIONS: Dict[str, List[str]] = {
    "Abstract": ["abstract"],
    "Introduction": ["introduction", "intro", "overview"],
    "Related Work": [
        "related work",
        "related works",
        "background",
        "literature review",
        "prior work",
        "state of the art",
    ],
    "Methodology": [
        "methodology",
        "methods",
        "proposed method",
        "proposed approach",
        "method",
        "system architecture",
        "model architecture",
        "framework",
        "approach",
        "formulation",
        "technical approach",
    ],
    "Experiments & Results": [
        "experiments",
        "experimental setup",
        "experimental evaluation",
        "experimental results",
        "experiments and results",
        "results",
        "evaluation",
        "empirical results",
        "findings",
        "performance",
    ],
    "Discussion": [
        "discussion",
        "ablation study",
        "ablation studies",
        "analysis",
        "case study",
    ],
    "Conclusion": [
        "conclusion",
        "conclusions",
        "concluding remarks",
        "conclusion and future work",
        "conclusions and future work",
        "summary and conclusion",
    ],
    "Limitations": [
        "limitations",
        "ethical considerations",
        "broader impact",
    ],
    "References": [
        "references",
        "bibliography",
        "works cited",
    ],
    "Appendix": [
        "appendix",
        "appendices",
        "supplementary material",
        "supplementary materials",
    ],
}


class SectionDetector:
    """
    Detects headings and segments a Document into structured logical sections.
    """

    def __init__(self, canonical_sections: Optional[Dict[str, List[str]]] = None):
        self.canonical_sections = canonical_sections or CANONICAL_SECTIONS

    def detect_sections(self, document: Document) -> List[SectionInfo]:
        """
        Scan all pages of the document, detect section headings, and assign page ranges.

        Args:
            document: Document with extracted pages.

        Returns:
            List[SectionInfo]: Ordered list of detected sections.
        """
        detected_raw: List[Tuple[int, str, str]] = []  # (page_number, raw_heading, canonical_title)

        for page in document.pages:
            page_headings = self._find_headings_in_text(page.text, page.page_number)
            detected_raw.extend(page_headings)

        if not detected_raw:
            # Fallback if no specific academic headings were found
            num_pages = max(1, document.num_pages or len(document.pages))
            default_sec = SectionInfo(
                section_id=new_id("sec"),
                title="Main Content",
                raw_heading="Document Content",
                start_page=1,
                end_page=num_pages,
                order=1,
            )
            document.sections = [default_sec]
            return [default_sec]

        # De-duplicate consecutive identical sections on same or adjacent pages
        unique_headings: List[Tuple[int, str, str]] = []
        for item in detected_raw:
            if unique_headings and unique_headings[-1][2] == item[2]:
                continue
            unique_headings.append(item)

        # Build SectionInfo items with proper start_page and end_page
        sections: List[SectionInfo] = []
        num_pages = max(1, document.num_pages or len(document.pages))

        for idx, (page_num, raw_hd, can_title) in enumerate(unique_headings):
            order = idx + 1
            start_page = page_num

            # Determine end page: either page before next section or last page of document
            if idx + 1 < len(unique_headings):
                next_start = unique_headings[idx + 1][0]
                end_page = max(start_page, next_start if next_start == start_page else next_start - 1)
            else:
                end_page = max(start_page, num_pages)

            sections.append(
                SectionInfo(
                    section_id=new_id("sec"),
                    title=can_title,
                    raw_heading=raw_hd,
                    start_page=start_page,
                    end_page=end_page,
                    order=order,
                )
            )

        document.sections = sections
        return sections

    def _find_headings_in_text(self, text: str, page_number: int) -> List[Tuple[int, str, str]]:
        """Find potential headings in page text."""
        results: List[Tuple[int, str, str]] = []
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        for line in lines:
            # Skip very long lines (unlikely to be pure headings)
            if len(line) > 80:
                continue

            canonical = self._classify_heading(line)
            if canonical:
                results.append((page_number, line, canonical))

        return results

    def _classify_heading(self, line: str) -> Optional[str]:
        """Classify a line as a canonical section heading if it matches patterns."""
        cleaned = line.strip()

        # Remove leading numbers/romans/hashes: "1. ", "1.1 ", "I. ", "## "
        normalized = re.sub(r"^(?:#{1,4}\s+|[0-9IVXLCDM]+(?:\.[0-9]+)*[:.]?\s+|section\s+[0-9]+[:.]?\s+)", "", cleaned, flags=re.IGNORECASE).strip().lower()

        # Check against canonical patterns
        for canonical, keywords in self.canonical_sections.items():
            for kw in keywords:
                # Exact match or starts with keyword followed by colon/dash/space
                if normalized == kw or normalized.startswith(f"{kw}:") or normalized.startswith(f"{kw} -") or normalized.startswith(f"{kw} —"):
                    return canonical
                # Also check all-caps or exact word boundary
                if re.fullmatch(rf"{kw}", normalized, re.IGNORECASE):
                    return canonical

        return None


def detect_sections(document: Document) -> List[SectionInfo]:
    """Convenience helper function to detect sections for a Document."""
    detector = SectionDetector()
    return detector.detect_sections(document)
