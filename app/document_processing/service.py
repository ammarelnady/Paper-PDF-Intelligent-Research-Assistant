from pathlib import Path

from app.core.config import Settings
from app.document_processing.chunker import chunk_pages
from app.document_processing.extractor import extract_pdf_pages
from app.document_processing.sections import assign_sections
from app.models.document import Document


def process_pdf(document_id: str, filename: str, source_path: Path, settings: Settings) -> Document:
    pages = assign_sections(extract_pdf_pages(source_path, document_id))
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    return Document(
        document_id=document_id,
        filename=filename,
        page_count=len(pages),
        source_path=source_path,
        pages=pages,
        chunks=chunks,
    )
