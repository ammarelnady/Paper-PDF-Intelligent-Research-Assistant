from pathlib import Path
from uuid import uuid4

from app.document_processing.extractor import PDFExtractor
from app.document_processing.chunker import MetadataAwareChunker
from app.models.document import Document


class DocumentProcessor:
    """Main document processing pipeline."""

    def __init__(
        self,
        extractor: PDFExtractor | None = None,
        chunker: MetadataAwareChunker | None = None,
    ):
        self.extractor = extractor or PDFExtractor()
        self.chunker = chunker or MetadataAwareChunker()

    def process(
        self,
        pdf_path: str | Path,
        document_id: str | None = None,
    ) -> Document:

        pdf_path = Path(pdf_path)

        if document_id is None:
            document_id = str(uuid4())

        pages = self.extractor.extract(pdf_path)

        chunks = self.chunker.chunk_pages(
            document_id=document_id,
            pages=pages,
        )

        return Document(
            document_id=document_id,
            filename=pdf_path.name,
            total_pages=len(pages),
            pages=pages,
            chunks=chunks,
        )