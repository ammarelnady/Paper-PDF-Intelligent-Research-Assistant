<<<<<<< HEAD
"""Reserved for Salma's document-processing implementation."""
=======
"""
Document processing pipeline: text extraction, section detection, and metadata-aware chunking.
"""

from app.document_processing.chunker import DocumentChunker, chunk_document
from app.document_processing.extractor import (
    PDFExtractionError,
    PDFExtractor,
    extract_text_from_pdf,
)
from app.document_processing.section_detector import SectionDetector, detect_sections

__all__ = [
    "PDFExtractor",
    "PDFExtractionError",
    "extract_text_from_pdf",
    "SectionDetector",
    "detect_sections",
    "DocumentChunker",
    "chunk_document",
]
>>>>>>> origin/main
