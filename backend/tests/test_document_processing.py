"""
Unit tests for Document Processing module (Salma - Member 1).
Tests PDF extraction, text normalization, section detection, and metadata-aware chunking.
"""

import io
import unittest
from app.document_processing.chunker import DocumentChunker, chunk_document
from app.document_processing.extractor import (
    PDFExtractionError,
    PDFExtractor,
    extract_text_from_pdf,
)
from app.document_processing.section_detector import SectionDetector, detect_sections
from app.models.chunk import DocumentChunk
from app.models.document import Document, PageContent, SectionInfo


class TestDocumentProcessing(unittest.TestCase):
    """Test suite for PDF extraction, Section Detection, and Chunking."""

    def setUp(self):
        # Sample synthetic scientific paper text across 3 pages
        self.page1_text = (
            "Deep Learning for Scientific Paper Understanding\n"
            "Abstract\n"
            "Natural language processing on scientific documents has seen immense growth. "
            "In this paper, we propose a modular architecture for research paper analysis.\n"
            "1. Introduction\n"
            "Research papers contain dense information distributed across specialized sections. "
            "Understanding such structure requires robust extraction and metadata-aware processing."
        )

        self.page2_text = (
            "2. Methodology\n"
            "We propose an end-to-end framework consisting of three core stages: "
            "document ingestion, hierarchical section segmentation, and semantic chunking. "
            "Our model preserves page numbers and section labels for exact citation tracing."
        )

        self.page3_text = (
            "3. Experiments and Results\n"
            "We evaluate our framework on 500 arXiv papers. "
            "The results demonstrate high retrieval recall and faithful summary generation.\n"
            "4. Conclusion\n"
            "In conclusion, our approach bridges the gap between raw PDF documents and structured knowledge."
        )

        self.sample_document = Document(
            document_id="doc_test_123",
            filename="test_paper.pdf",
            title="Deep Learning for Scientific Paper Understanding",
            num_pages=3,
            pages=[
                PageContent(page_number=1, text=self.page1_text),
                PageContent(page_number=2, text=self.page2_text),
                PageContent(page_number=3, text=self.page3_text),
            ],
            full_text=f"{self.page1_text}\n\n{self.page2_text}\n\n{self.page3_text}",
        )

    # -------------------------------------------------------------
    # 1. Extraction & Document Model Tests
    # -------------------------------------------------------------
    def test_page_content_char_count(self):
        page = PageContent(page_number=1, text="Hello world")
        self.assertEqual(page.char_count, 11)
        self.assertEqual(page.page_number, 1)

    def test_extractor_clean_text(self):
        extractor = PDFExtractor()
        raw_text = "This is a com-\nputer science algo-\nrithm with   excessive    spaces."
        cleaned = extractor._clean_text(raw_text)
        self.assertIn("computer", cleaned)
        self.assertIn("algorithm", cleaned)
        self.assertNotIn("   ", cleaned)

    def test_extractor_empty_bytes_raises_error(self):
        extractor = PDFExtractor()
        with self.assertRaises(PDFExtractionError):
            extractor.extract(b"")

    def test_extractor_nonexistent_file_raises_error(self):
        extractor = PDFExtractor()
        with self.assertRaises(FileNotFoundError):
            extractor.extract("non_existent_file_xyz_123.pdf")

    # -------------------------------------------------------------
    # 2. Section Detection Tests
    # -------------------------------------------------------------
    def test_detect_sections_canonical(self):
        detector = SectionDetector()
        sections = detector.detect_sections(self.sample_document)

        self.assertTrue(len(sections) >= 4)
        section_titles = [s.title for s in sections]

        self.assertIn("Abstract", section_titles)
        self.assertIn("Introduction", section_titles)
        self.assertIn("Methodology", section_titles)
        self.assertIn("Experiments & Results", section_titles)
        self.assertIn("Conclusion", section_titles)

        # Verify page ranges
        intro_sec = next(s for s in sections if s.title == "Introduction")
        self.assertEqual(intro_sec.start_page, 1)

        method_sec = next(s for s in sections if s.title == "Methodology")
        self.assertEqual(method_sec.start_page, 2)

    def test_detect_sections_fallback_when_none_found(self):
        plain_doc = Document(
            document_id="doc_plain",
            filename="plain.pdf",
            pages=[PageContent(page_number=1, text="Just a plain text document with no headers.")],
            num_pages=1,
        )
        sections = detect_sections(plain_doc)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].title, "Main Content")
        self.assertEqual(sections[0].start_page, 1)
        self.assertEqual(sections[0].end_page, 1)

    def test_get_section_for_page(self):
        detect_sections(self.sample_document)
        sec_page2 = self.sample_document.get_section_for_page(2)
        self.assertEqual(sec_page2, "Methodology")

    def test_get_text_for_section(self):
        detect_sections(self.sample_document)
        method_text = self.sample_document.get_text_for_section("Methodology")
        self.assertIn("end-to-end framework", method_text)
        self.assertIn("hierarchical section segmentation", method_text)

    # -------------------------------------------------------------
    # 3. Metadata-Aware Chunking Tests
    # -------------------------------------------------------------
    def test_chunking_contract(self):
        detect_sections(self.sample_document)
        chunks = chunk_document(self.sample_document, chunk_size=200, chunk_overlap=30)

        self.assertTrue(len(chunks) > 0)
        for idx, chunk in enumerate(chunks):
            # Verify fixed contract attributes
            self.assertEqual(chunk.document_id, self.sample_document.document_id)
            self.assertTrue(chunk.chunk_id.startswith("chunk_"))
            self.assertIn(chunk.page_number, [1, 2, 3])
            self.assertIsInstance(chunk.section, str)
            self.assertTrue(len(chunk.text) > 0)

            # Verify additive attributes
            self.assertEqual(chunk.chunk_index, idx)
            self.assertTrue(chunk.token_estimate > 0)

    def test_chunking_preserves_section_metadata(self):
        detect_sections(self.sample_document)
        chunks = chunk_document(self.sample_document, chunk_size=300, chunk_overlap=50)

        # Chunks on page 2 should have Methodology section
        page2_chunks = [c for c in chunks if c.page_number == 2]
        self.assertTrue(len(page2_chunks) > 0)
        for c in page2_chunks:
            self.assertEqual(c.section, "Methodology")

    def test_chunking_empty_document(self):
        empty_doc = Document(document_id="empty_doc", filename="empty.pdf", pages=[], num_pages=0)
        chunks = chunk_document(empty_doc)
        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
