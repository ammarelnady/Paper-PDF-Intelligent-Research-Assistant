"""
PDF Text Extractor module.

Extracts text, metadata, and page structure from PDF documents using PyMuPDF (fitz)
with fallback support for pypdf. Normalizes text, cleans artifacts (ligatures, hyphenation),
and produces a structured Document object.
"""

from __future__ import annotations

import io
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple, Union

from app.models.document import Document, PageContent, new_id

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails due to corruption, encryption, or invalid format."""
    pass


class PDFExtractor:
    """
    Extracts text and page structure from PDF files or raw bytes.
    Preserves page boundaries and extracts document metadata.
    """

    def __init__(self, remove_hyphenation: bool = True, normalize_unicode: bool = True):
        self.remove_hyphenation = remove_hyphenation
        self.normalize_unicode = normalize_unicode

    def extract(
        self,
        source: Union[str, Path, bytes, io.BytesIO],
        filename: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Document:
        """
        Extract text from a PDF file path or byte stream.

        Args:
            source: Filepath (str/Path) or raw PDF bytes / BytesIO.
            filename: Original file name.
            document_id: Optional custom document ID.

        Returns:
            Document: Structured document with pages, metadata, and full text.
        """
        resolved_filename = "document.pdf"
        file_bytes: Optional[bytes] = None

        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"PDF file not found at: {path}")
            resolved_filename = filename or path.name
            try:
                with open(path, "rb") as f:
                    file_bytes = f.read()
            except Exception as e:
                raise PDFExtractionError(f"Failed to read file {path}: {e}") from e
        elif isinstance(source, bytes):
            file_bytes = source
            resolved_filename = filename or "uploaded_document.pdf"
        elif isinstance(source, io.BytesIO):
            file_bytes = source.getvalue()
            resolved_filename = filename or "uploaded_document.pdf"
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        if not file_bytes or len(file_bytes.strip()) == 0:
            raise PDFExtractionError("Provided PDF content is empty (0 bytes).")

        # Try PyMuPDF (fitz) first, then fallback to pypdf
        pages, title = self._extract_with_fitz(file_bytes, resolved_filename)
        if pages is None:
            pages, title = self._extract_with_pypdf(file_bytes, resolved_filename)

        if not pages:
            # Create a fallback page if completely blank
            pages = [PageContent(page_number=1, text="")]

        cleaned_pages: List[PageContent] = []
        for p in pages:
            cleaned_text = self._clean_text(p.text)
            cleaned_pages.append(PageContent(page_number=p.page_number, text=cleaned_text))

        full_text = "\n\n".join(p.text for p in cleaned_pages if p.text.strip()).strip()

        doc = Document(
            document_id=document_id or new_id("doc"),
            filename=resolved_filename,
            title=title or self._infer_title_from_text(cleaned_pages, resolved_filename),
            num_pages=len(cleaned_pages),
            pages=cleaned_pages,
            sections=[],
            full_text=full_text,
        )
        return doc

    def _extract_with_fitz(
        self, file_bytes: bytes, filename: str
    ) -> Tuple[Optional[List[PageContent]], Optional[str]]:
        """Extract pages using PyMuPDF (fitz) if available."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.debug("PyMuPDF (fitz) not installed; falling back to alternative extractor.")
            return None, None

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise PDFExtractionError(f"PyMuPDF failed to open PDF document: {e}") from e

        if doc.is_encrypted:
            raise PDFExtractionError("PDF is password protected / encrypted.")

        pages: List[PageContent] = []
        title: Optional[str] = doc.metadata.get("title") if doc.metadata else None
        if title and (title.strip().lower() in ["untitled", filename.lower()] or len(title.strip()) < 3):
            title = None

        first_page_title: Optional[str] = None

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = page.get_text("text") or ""
            pages.append(PageContent(page_number=page_idx + 1, text=page_text))

            # If title is not in metadata, inspect largest font spans on page 1
            if page_idx == 0 and not title:
                try:
                    blocks = page.get_text("dict").get("blocks", [])
                    largest_size = 0.0
                    candidate_title = ""
                    for b in blocks:
                        if "lines" in b:
                            for line in b["lines"]:
                                for span in line.get("spans", []):
                                    span_text = span.get("text", "").strip()
                                    font_size = span.get("size", 0.0)
                                    if font_size > largest_size and len(span_text) > 4:
                                        largest_size = font_size
                                        candidate_title = span_text
                    if candidate_title and len(candidate_title) > 3:
                        first_page_title = candidate_title
                except Exception:
                    pass

        doc.close()
        return pages, title or first_page_title

    def _extract_with_pypdf(
        self, file_bytes: bytes, filename: str
    ) -> Tuple[List[PageContent], Optional[str]]:
        """Extract pages using pypdf fallback."""
        try:
            from pypdf import PdfReader
        except ImportError:
            # Pure text / mock fallback if no PDF library is available
            try:
                # Attempt utf-8 decoding for mock text inputs
                text = file_bytes.decode("utf-8", errors="ignore")
                return [PageContent(page_number=1, text=text)], None
            except Exception as e:
                raise PDFExtractionError(f"No PDF parsing library available (install pymupdf or pypdf): {e}") from e

        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as e:
            raise PDFExtractionError(f"Failed to parse PDF with pypdf: {e}") from e

        if reader.is_encrypted:
            raise PDFExtractionError("PDF is password protected / encrypted.")

        pages: List[PageContent] = []
        title: Optional[str] = None
        if reader.metadata and reader.metadata.title:
            t = reader.metadata.title.strip()
            if len(t) > 3 and t.lower() not in ["untitled", filename.lower()]:
                title = t

        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(PageContent(page_number=idx + 1, text=text))

        return pages, title

    def _clean_text(self, text: str) -> str:
        """Clean extracted PDF text (hyphenation, ligatures, excess whitespace)."""
        if not text:
            return ""

        # Unicode normalization (NFKC fixes ligatures like 'fi', 'fl')
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # Fix de-hyphenation across linebreaks: e.g. "com-\nputer" -> "computer"
        if self.remove_hyphenation:
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

        # Remove standalone page numbers or header/footer artifacts like "Page 1 of 10"
        text = re.sub(r"(?i)\bpage\s+\d+(\s+of\s+\d+)?\b", "", text)

        # Replace carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Replace non-breaking spaces
        text = text.replace("\xa0", " ")

        # Clean consecutive spaces while preserving double newlines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    def _infer_title_from_text(self, pages: List[PageContent], filename: str) -> str:
        """Infer title from the first non-empty lines of page 1 or the filename."""
        if pages and pages[0].text:
            lines = [line.strip() for line in pages[0].text.split("\n") if line.strip()]
            for line in lines[:5]:
                # Exclude obvious non-titles like "arXiv:...", "Journal of...", dates
                if len(line) > 5 and not re.match(r"(?i)^(arxiv|journal|volume|issue|proceedings|page|\d+)", line):
                    return line
        # Fallback to sanitized filename
        base = Path(filename).stem
        return base.replace("_", " ").replace("-", " ").title()


def extract_text_from_pdf(
    source: Union[str, Path, bytes, io.BytesIO],
    filename: Optional[str] = None,
    document_id: Optional[str] = None,
) -> Document:
    """Convenience helper function to extract a Document from PDF."""
    extractor = PDFExtractor()
    return extractor.extract(source, filename=filename, document_id=document_id)
