from pathlib import Path

import pymupdf
import pytest

from app.core.exceptions import InvalidPdfError
from app.document_processing.extractor import extract_pdf_pages


def _make_pdf(path: Path) -> None:
    pdf = pymupdf.open()
    first = pdf.new_page()
    first.insert_text((72, 72), "Introduction\nThis is the first page.")
    second = pdf.new_page()
    second.insert_text((72, 72), "Methods\nThis is the second page.")
    pdf.save(path)
    pdf.close()


def test_extract_pdf_pages_preserves_page_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path)
    pages = extract_pdf_pages(pdf_path, "doc-123")
    assert len(pages) == 2
    assert pages[0].document_id == "doc-123"
    assert pages[0].page_number == 1
    assert "first page" in pages[0].text
    assert pages[1].page_number == 2


def test_extract_pdf_pages_rejects_corrupted_file(tmp_path: Path) -> None:
    invalid = tmp_path / "broken.pdf"
    invalid.write_bytes(b"not a PDF")
    with pytest.raises(InvalidPdfError):
        extract_pdf_pages(invalid, "doc-123")
