import pymupdf
import pytest

from app.document_processing.extractor import PDFExtractor
from app.document_processing.section_detector import SectionDetector
from app.document_processing.chunker import MetadataAwareChunker
from app.models.document import PageContent


# ============================================================
# PDF EXTRACTOR TESTS
# ============================================================

def test_pdf_extractor_file_not_found():
    extractor = PDFExtractor()

    with pytest.raises(FileNotFoundError):
        extractor.extract("non_existing_file.pdf")


def test_pdf_extractor_invalid_extension(tmp_path):
    extractor = PDFExtractor()

    file_path = tmp_path / "test.txt"
    file_path.write_text("This is not a PDF.")

    with pytest.raises(ValueError):
        extractor.extract(file_path)


def test_pdf_extractor(tmp_path):

    pdf_path = tmp_path / "test.pdf"

    with pymupdf.open() as pdf:

        page1 = pdf.new_page()
        page1.insert_text(
            (50, 50),
            "Abstract\nThis is a test paper."
        )

        page2 = pdf.new_page()
        page2.insert_text(
            (50, 50),
            "Introduction\nThis is introduction text."
        )

        pdf.save(pdf_path)

    extractor = PDFExtractor()

    pages = extractor.extract(pdf_path)

    assert len(pages) == 2

    assert pages[0].page_number == 1
    assert pages[1].page_number == 2

    assert "Abstract" in pages[0].text
    assert "Introduction" in pages[1].text


# ============================================================
# SECTION DETECTOR TESTS
# ============================================================

def test_section_detector_recognizes_headings():

    detector = SectionDetector()

    assert detector.is_heading("Abstract")
    assert detector.is_heading("Introduction")
    assert detector.is_heading("3 Methods")
    assert detector.is_heading("4.1 Experiments")


def test_section_detector_rejects_false_headings():

    detector = SectionDetector()

    assert not detector.is_heading("123")
    assert not detector.is_heading("2017 English-French dataset")
    assert not detector.is_heading("1.0 · 1020")


def test_section_detector_detects_sections():

    detector = SectionDetector()

    text = """
Abstract
This is the abstract.

Introduction
This is the introduction.

Results
These are the results.
"""

    sections, current_section = detector.detect_sections(text)

    assert len(sections) == 3

    assert sections[0][0] == "Abstract"
    assert sections[1][0] == "Introduction"
    assert sections[2][0] == "Results"

    assert current_section == "Results"


def test_section_continuation_between_pages():

    detector = SectionDetector()

    page1 = """
Introduction
This is page one of the introduction.
"""

    sections1, current_section = detector.detect_sections(
        page1,
        current_section="Unknown"
    )

    page2 = """
This is continuation of the introduction.
More introduction text.
"""

    sections2, current_section = detector.detect_sections(
        page2,
        current_section=current_section
    )

    assert sections1[0][0] == "Introduction"

    assert sections2[0][0] == "Introduction"

    assert current_section == "Introduction"


# ============================================================
# CHUNKER TESTS
# ============================================================

def test_chunker_validation():

    with pytest.raises(ValueError):
        MetadataAwareChunker(chunk_size=0)

    with pytest.raises(ValueError):
        MetadataAwareChunker(
            chunk_size=100,
            overlap=100
        )

    with pytest.raises(ValueError):
        MetadataAwareChunker(
            chunk_size=100,
            overlap=150
        )


def test_chunker_creates_chunks():

    pages = [
        PageContent(
            page_number=1,
            text=(
                "Abstract\n"
                "This is a long piece of text "
                "that should be split into chunks."
            )
        )
    ]

    chunker = MetadataAwareChunker(
        chunk_size=30,
        overlap=5
    )

    chunks = chunker.chunk_pages(
        document_id="test_doc",
        pages=pages
    )

    assert len(chunks) > 0

    for chunk in chunks:

        assert chunk.document_id == "test_doc"

        assert chunk.page_number == 1

        assert chunk.text

        assert chunk.chunk_id.startswith(
            "test_doc_chunk_"
        )

        assert chunk.section == "Abstract"

from pathlib import Path

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


PDF_PATH = Path(
    r"C:\Users\Salma\OneDrive\Desktop"
    r"\paper-research-assistant"
    r"\NIPS-2017-attention-is-all-you-need-Paper.pdf"
)


def test_process_pdf_success():

    with open(PDF_PATH, "rb") as pdf:

        response = client.post(
            "/documents/process",
            files={
                "file": (
                    "test.pdf",
                    pdf,
                    "application/pdf"
                )
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert "document_id" in data
    assert data["filename"] == "test.pdf"
    assert data["total_pages"] == 11

    assert len(data["pages"]) == 11
    assert len(data["chunks"]) > 0


def test_reject_non_pdf():

    response = client.post(
        "/documents/process",
        files={
            "file": (
                "test.txt",
                b"hello world",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400


def test_chunk_metadata():

    with open(PDF_PATH, "rb") as pdf:

        response = client.post(
            "/documents/process",
            files={
                "file": (
                    "test.pdf",
                    pdf,
                    "application/pdf"
                )
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert len(data["chunks"]) > 0

    chunk = data["chunks"][0]

    assert "document_id" in chunk
    assert "chunk_id" in chunk
    assert "page_number" in chunk
    assert "section" in chunk
    assert "text" in chunk

    assert chunk["document_id"] == data["document_id"]
    assert chunk["page_number"] >= 1
    assert chunk["text"]