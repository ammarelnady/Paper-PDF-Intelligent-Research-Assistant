from app.models.document import DocumentChunk, PageContent
from app.document_processing.section_detector import SectionDetector


class MetadataAwareChunker:
    """Split text into chunks while preserving metadata."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        section_detector: SectionDetector | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be >= 0 and smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.section_detector = (
            section_detector or SectionDetector()
        )

    def _split_text(self, text: str) -> list[str]:
        chunks = []

        start = 0

        while start < len(text):

            end = min(
                start + self.chunk_size,
                len(text)
            )

            # Try to avoid cutting in the middle of a word
            if end < len(text):

                whitespace_pos = text.rfind(" ", start, end)

                if whitespace_pos > start:
                    end = whitespace_pos

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            start = max(0, end - self.overlap)

        return chunks

    def chunk_pages(
        self,
        document_id: str,
        pages: list[PageContent],
    ) -> list[DocumentChunk]:

        chunks = []

        current_section = "Unknown"

        for page in pages:

            if not page.text.strip():
                continue

            sections, current_section = (
                self.section_detector.detect_sections(
                    page.text,
                    current_section=current_section,
                )
            )

            for section_name, section_text in sections:

                text_chunks = self._split_text(
                    section_text
                )

                for text_chunk in text_chunks:

                    chunk_id = (
                        f"{document_id}_chunk_"
                        f"{len(chunks) + 1}"
                    )

                    chunks.append(
                        DocumentChunk(
                            document_id=document_id,
                            chunk_id=chunk_id,
                            page_number=page.page_number,
                            section=section_name,
                            text=text_chunk,
                        )
                    )

        return chunks