"""LLM-assisted paper outline extraction with a safe detector fallback."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

from app.models.document import Document, SectionInfo, new_id

logger = logging.getLogger(__name__)


class SectionOutlineAnalyzer:
    """Extract a hierarchical outline from a paper without trusting raw LLM output."""

    def __init__(self, llm_caller: Optional[Callable[[str, str], str]] = None):
        self.llm_caller = llm_caller

    def analyze(self, document: Document, detected: Iterable[SectionInfo]) -> List[SectionInfo]:
        fallback = self._with_hierarchy(list(detected))
        if not self.llm_caller:
            fallback = [section.model_copy(update={"outline_source": "detector"}) if hasattr(section, "model_copy") else section.copy(update={"outline_source": "detector"}) for section in fallback]
            document.sections = fallback
            return fallback
        try:
            response = self.llm_caller(self._build_prompt(document), self._system_prompt())
            extracted = self._parse_response(response, document.num_pages or len(document.pages))
            if extracted:
                extracted = [section.model_copy(update={"outline_source": "llm"}) if hasattr(section, "model_copy") else section.copy(update={"outline_source": "llm"}) for section in extracted]
                document.sections = extracted
                return extracted
        except Exception as exc:
            logger.warning("Hierarchical section extraction failed: %s", exc)
        fallback = [section.model_copy(update={"outline_source": "detector"}) if hasattr(section, "model_copy") else section.copy(update={"outline_source": "detector"}) for section in fallback]
        document.sections = fallback
        return fallback

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a precise academic paper outline extractor. Return only valid JSON. "
            "Do not invent headings or page numbers. Keep headings in the paper's wording."
        )

    def _build_prompt(self, document: Document) -> str:
        page_blocks = [f"PAGE {page.page_number}\n{page.text[:1800]}" for page in document.pages]
        instructions = """Extract the complete hierarchical outline of this research paper.

Return a JSON array. Each item must have exactly:
{"title":"heading", "level":1, "parent_title":null, "start_page":1, "end_page":1}

Rules:
- Include real top-level headings and nested headings (subsections and sub-subsections).
- Infer hierarchy from numbering, typography cues, and reading order.
- level is 1 for a top-level heading, 2 for a subsection, 3 for a sub-subsection.
- parent_title must be the nearest preceding parent heading, or null for level 1.
- Use inclusive 1-indexed page ranges.
- Deduplicate repeated headers, running headers, and repeated headings from PDF extraction.
- Do not include body sentences, figure captions, author names, or bibliography entries.
- If no nested headings are visible, return the genuine top-level headings only.

"""
        return instructions + "\n\n".join(page_blocks)[:60000]

    def _parse_response(self, response: str, max_page: int) -> List[SectionInfo]:
        match = re.search(r"\[[\s\S]*\]", response.strip())
        if not match:
            return []
        payload = json.loads(match.group(0))
        if not isinstance(payload, list):
            return []

        entries: List[Dict[str, Any]] = []
        seen = set()
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title", "")).strip()
            if not title or len(title) > 120:
                continue
            key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
            if not key or key in seen:
                continue
            try:
                level = max(1, min(8, int(raw.get("level", 1))))
                start = max(1, min(max_page, int(raw.get("start_page", 1))))
                end = max(start, min(max_page, int(raw.get("end_page", start))))
            except (TypeError, ValueError):
                continue
            seen.add(key)
            entries.append({
                "title": title,
                "level": level,
                "parent_title": str(raw.get("parent_title") or "").strip(),
                "start_page": start,
                "end_page": end,
            })

        sections: List[SectionInfo] = []
        stack: List[SectionInfo] = []
        title_to_id: Dict[str, str] = {}
        for index, item in enumerate(entries, start=1):
            level = item["level"]
            while stack and stack[-1].level >= level:
                stack.pop()
            parent_id = stack[-1].section_id if stack else None
            if not parent_id and item["parent_title"]:
                parent_key = re.sub(r"[^a-z0-9]+", " ", item["parent_title"].lower()).strip()
                parent_id = title_to_id.get(parent_key)
            section = SectionInfo(
                section_id=new_id("sec"), title=item["title"], raw_heading=item["title"],
                start_page=item["start_page"], end_page=item["end_page"], order=index,
                level=level, parent_section_id=parent_id,
            )
            sections.append(section)
            stack.append(section)
            title_to_id[re.sub(r"[^a-z0-9]+", " ", section.title.lower()).strip()] = section.section_id
        return sections

    @staticmethod
    def _with_hierarchy(sections: List[SectionInfo]) -> List[SectionInfo]:
        stack: List[SectionInfo] = []
        result: List[SectionInfo] = []
        for section in sections:
            numbered = re.match(r"^\s*(\d+(?:\.\d+)*)[.)]?\s+", section.raw_heading)
            level = len(numbered.group(1).split(".")) if numbered else 1
            while stack and stack[-1].level >= level:
                stack.pop()
            parent_id = stack[-1].section_id if stack else None
            copier = section.model_copy if hasattr(section, "model_copy") else section.copy
            updated = copier(update={"level": level, "parent_section_id": parent_id})
            result.append(updated)
            stack.append(updated)
        return result


__all__ = ["SectionOutlineAnalyzer"]
