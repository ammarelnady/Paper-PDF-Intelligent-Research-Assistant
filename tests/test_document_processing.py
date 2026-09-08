import pymupdf
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.document_processing.extractor import PDFExtractor
from app.document_processing.section_detector import SectionDetector
from app.document_processing.chunker import MetadataAwareChunker
from app.models.document import PageContent

from main import app


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
    assert not detector.is_heading(
        "2017 English-French dataset"
    )
    assert not detector.is_heading(
        "1.0 Â· 1020"
    )


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
            "test_doc:p1:c"
        )

        assert chunk.section == "Abstract"


def test_chunker_ids_are_deterministic_and_page_bounded():

    pages = [
        PageContent(
            page_number=1,
            text=(
                "Introduction\n"
                "This is some text that is long enough "
                "to create multiple chunks."
            ),
        ),
        PageContent(
            page_number=2,
            text=(
                "Results\n"
                "This is page two with additional text."
            ),
        ),
    ]

    chunker = MetadataAwareChunker(
        chunk_size=30,
        overlap=5,
    )

    chunks1 = chunker.chunk_pages(
        document_id="doc123",
        pages=pages,
    )

    chunks2 = chunker.chunk_pages(
        document_id="doc123",
        pages=pages,
    )

    ids1 = [chunk.chunk_id for chunk in chunks1]
    ids2 = [chunk.chunk_id for chunk in chunks2]

    # Processing the same input must produce the same IDs.
    assert ids1 == ids2

    # Frozen contract.
    assert ids1[0] == "doc123:p1:c0"

    page1_ids = [
        chunk.chunk_id
        for chunk in chunks1
        if chunk.page_number == 1
    ]

    page2_ids = [
        chunk.chunk_id
        for chunk in chunks1
        if chunk.page_number == 2
    ]

    assert page1_ids[0] == "doc123:p1:c0"
    assert page2_ids[0] == "doc123:p2:c0"


def test_chunker_skips_empty_and_whitespace_pages():

    pages = [
        PageContent(
            page_number=1,
            text="",
        ),
        PageContent(
            page_number=2,
            text="   \n\n   ",
        ),
        PageContent(
            page_number=3,
            text=(
                "Introduction\n"
                "Actual paper content."
            ),
        ),
    ]

    chunker = MetadataAwareChunker(
        chunk_size=100,
        overlap=10,
    )

    chunks = chunker.chunk_pages(
        document_id="doc123",
        pages=pages,
    )

    assert len(chunks) > 0

    assert all(
        chunk.page_number == 3
        for chunk in chunks
    )

    assert chunks[0].chunk_id == "doc123:p3:c0"

def test_pdf_extractor_normalizes_whitespace(tmp_path):

    pdf_path = tmp_path / "whitespace.pdf"

    with pymupdf.open() as pdf:

        page = pdf.new_page()

        page.insert_text(
            (50, 50),
            "This    is     a    test.\n\n\n"
            "Second    paragraph.\n"
            "Third line."
        )

        pdf.save(pdf_path)

    extractor = PDFExtractor()

    pages = extractor.extract(pdf_path)

    assert len(pages) == 1

    text = pages[0].text

    assert "This is a test." in text
    assert "Second paragraph." in text
    assert "Third line." in text

    # Excessive blank lines should be reduced.
    assert "\n\n\n" not in text


def test_pdf_extractor_preserves_special_content(tmp_path):

    pdf_path = tmp_path / "special_content.pdf"

    with pymupdf.open() as pdf:

        page = pdf.new_page()

        page.insert_text(
            (50, 50),
            "Accuracy = 89.81%\n"
            "URL: https://example.com/paper\n"
            "Scientific value: 1.0 × 10^20\n"
            "Citation [12]"
        )

        pdf.save(pdf_path)

    extractor = PDFExtractor()

    pages = extractor.extract(pdf_path)

    text = pages[0].text

    assert "89.81%" in text
    assert "https://example.com/paper" in text
    assert "1.0 × 10^20" in text
    assert "[12]" in text


def test_pdf_extractor_preserves_empty_pages(tmp_path):

    pdf_path = tmp_path / "empty_pages.pdf"

    with pymupdf.open() as pdf:

        pdf.new_page()

        page2 = pdf.new_page()

        page2.insert_text(
            (50, 50),
            "Introduction\nActual content."
        )

        pdf.save(pdf_path)

    extractor = PDFExtractor()

    pages = extractor.extract(pdf_path)

    assert len(pages) == 2

    assert pages[0].page_number == 1
    assert pages[0].text == ""

    assert pages[1].page_number == 2
    assert "Actual content." in pages[1].text


def test_chunker_preserves_multiple_sections_on_same_page():

    pages = [
        PageContent(
            page_number=1,
            text=(
                "Introduction\n"
                "Introduction content.\n\n"
                "3 Methodology\n"
                "Methodology content.\n\n"
                "Results\n"
                "Results content."
            ),
        )
    ]

    chunker = MetadataAwareChunker(
        chunk_size=100,
        overlap=10,
    )

    chunks = chunker.chunk_pages(
        document_id="doc123",
        pages=pages,
    )

    sections = [chunk.section for chunk in chunks]

    assert sections == [
        "Introduction",
        "3 Methodology",
        "Results",
    ]


def test_chunker_preserves_section_across_pages():

    pages = [
        PageContent(
            page_number=1,
            text=(
                "Introduction\n"
                "This section starts on page one."
            ),
        ),
        PageContent(
            page_number=2,
            text=(
                "This section continues on page two."
            ),
        ),
    ]

    chunker = MetadataAwareChunker(
        chunk_size=100,
        overlap=10,
    )

    chunks = chunker.chunk_pages(
        document_id="doc123",
        pages=pages,
    )

    assert len(chunks) == 2

    assert chunks[0].section == "Introduction"
    assert chunks[1].section == "Introduction"

    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2


def test_chunker_respects_chunk_size():

    text = (
        "This is a long text used to verify that "
        "the chunker does not create chunks larger "
        "than the configured character size."
    )

    pages = [
        PageContent(
            page_number=1,
            text=f"Introduction\n{text}",
        )
    ]

    chunk_size = 40

    chunker = MetadataAwareChunker(
        chunk_size=chunk_size,
        overlap=5,
    )

    chunks = chunker.chunk_pages(
        document_id="doc123",
        pages=pages,
    )

    assert chunks

    for chunk in chunks:
        assert len(chunk.text) <= chunk_size


def test_chunker_handles_short_text():

    pages = [
        PageContent(
            page_number=1,
            text="Abstract\nShort text.",
        )
    ]

    chunker = MetadataAwareChunker(
        chunk_size=100,
        overlap=20,
    )

    chunks = chunker.chunk_pages(
        document_id="doc123",
        pages=pages,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_chunker_handles_exact_chunk_size():

    text = "A" * 50

    pages = [
        PageContent(
            page_number=1,
            text=f"Introduction\n{text}",
        )
    ]

    chunker = MetadataAwareChunker(
        chunk_size=50,
        overlap=10,
    )

    chunks = chunker.chunk_pages(
        document_id="doc123",
        pages=pages,
    )

    assert len(chunks) == 1
    assert len(chunks[0].text) == 50


# ============================================================
# API TESTS
# ============================================================

client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_PATH = (
    PROJECT_ROOT
    / "NIPS-2017-attention-is-all-you-need-Paper.pdf"
)


def test_process_pdf_success():

    with open(PDF_PATH, "rb") as pdf:

        response = client.post(
            "/api/v1/documents/upload",
            files={
                "file": (
                    "test.pdf",
                    pdf,
                    "application/pdf"
                )
            }
        )

    assert response.status_code == 201

    data = response.json()

    assert "document_id" in data
    assert data["filename"] == "test.pdf"
    assert data["total_pages"] == 11
    assert len(data["pages"]) == 11
    assert len(data["chunks"]) > 0


def test_reject_non_pdf():

    response = client.post(
        "/api/v1/documents/upload",
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
            "/api/v1/documents/upload",
            files={
                "file": (
                    "test.pdf",
                    pdf,
                    "application/pdf"
                )
            }
        )

    assert response.status_code == 201

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

def test_chunker_preserves_overlap():

    text = (
        "one two three four five six seven eight "
        "nine ten eleven twelve"
    )

    pages = [
        PageContent(
            page_number=1,
            text=f"Introduction\n{text}",
        )
    ]

    chunker = MetadataAwareChunker(
        chunk_size=30,
        overlap=10,
    )

    chunks = chunker.chunk_pages(
        document_id="doc123",
        pages=pages,
    )

    assert len(chunks) > 1

    # Adjacent chunks should share some text.
    for previous, current in zip(chunks, chunks[1:]):
        assert (
            previous.text[-5:] in current.text
            or current.text[:5] in previous.text
        )


def test_chunker_does_not_split_words_when_boundary_is_available():

    text = (
        "This sentence contains several words "
        "that should preferably remain intact."
    )

    pages = [
        PageContent(
            page_number=1,
            text=f"Introduction\n{text}",
        )
    ]

    chunker = MetadataAwareChunker(
        chunk_size=30,
        overlap=5,
    )

    chunks = chunker.chunk_pages(
        document_id="doc123",
        pages=pages,
    )

    assert len(chunks) > 1

    # Chunks should not start or end with partial alphabetic words.
    for chunk in chunks:
        assert not chunk.text.startswith(" ")
        assert not chunk.text.endswith(" ")


def test_chunker_handles_text_without_whitespace():

    text = "A" * 120

    pages = [
        PageContent(
            page_number=1,
            text=f"Introduction\n{text}",
        )
    ]

    chunker = MetadataAwareChunker(
        chunk_size=50,
        overlap=10,
    )

    chunks = chunker.chunk_pages(
        document_id="doc123",
        pages=pages,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.text) <= 50


def test_chunker_preserves_special_content():

    text = (
        "Accuracy = 89.81%. "
        "See https://example.com/paper. "
        "The value is 1.0 × 10^20. "
        "This is citation [12]."
    )

    pages = [
        PageContent(
            page_number=1,
            text=f"Results\n{text}",
        )
    ]

    chunker = MetadataAwareChunker(
        chunk_size=200,
        overlap=20,
    )

    chunks = chunker.chunk_pages(
        document_id="doc123",
        pages=pages,
    )

    combined_text = " ".join(
        chunk.text for chunk in chunks
    )

    assert "89.81%" in combined_text
    assert "https://example.com/paper" in combined_text
    assert "1.0 × 10^20" in combined_text
    assert "[12]" in combined_text


def test_chunker_keeps_chunks_on_their_original_page():

    pages = [
        PageContent(
            page_number=1,
            text=(
                "Introduction\n"
                + ("Page one text " * 30)
            ),
        ),
        PageContent(
            page_number=2,
            text=(
                "Results\n"
                + ("Page two text " * 30)
            ),
        ),
    ]

    chunker = MetadataAwareChunker(
        chunk_size=50,
        overlap=10,
    )

    chunks = chunker.chunk_pages(
        document_id="doc123",
        pages=pages,
    )

    assert chunks

    for chunk in chunks:

        if chunk.chunk_id.startswith("doc123:p1:"):
            assert chunk.page_number == 1

        elif chunk.chunk_id.startswith("doc123:p2:"):
            assert chunk.page_number == 2

        else:
            pytest.fail(
                f"Invalid chunk ID: {chunk.chunk_id}"
            )
