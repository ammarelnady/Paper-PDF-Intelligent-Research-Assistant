from __future__ import annotations

from pathlib import Path

import pymupdf

from app.core.exceptions import InvalidPdfError
from app.models.document import PageContent


def extract_pdf_pages(pdf_path: Path, document_id: str) -> tuple[PageContent, ...]:
    """Extract normalized text from every page of a readable PDF."""
    try:
        with pymupdf.open(pdf_path) as pdf:
            if pdf.page_count == 0:
                raise InvalidPdfError("The PDF contains no pages")
            pages = tuple(PageContent(document_id, index + 1, _normalize_text(page.get_text("text"))) for index, page in enumerate(pdf))
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError) as error:
        raise InvalidPdfError("The uploaded PDF is corrupted or unreadable") from error
    return pages


def _normalize_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())
