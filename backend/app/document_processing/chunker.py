"""
Metadata-aware Document Chunker.

Splits documents into structured, metadata-rich chunks for semantic retrieval,
embeddings, and citations. Preserves document_id, chunk_id, page_number,
section title, and token estimates.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from app.config import settings
from app.models.chunk import DocumentChunk
from app.models.document import Document, new_id

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Chunks extracted Document objects while preserving page and section metadata.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        min_chunk_size: int = settings.MIN_CHUNK_SIZE,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Split a Document into metadata-aware DocumentChunk items.

        Args:
            document: Document with pages and detected sections.

        Returns:
            List[DocumentChunk]: Ordered list of chunks with metadata.
        """
        if not document.pages:
            return []

        chunks: List[DocumentChunk] = []
        global_chunk_index = 0

        for page in document.pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            # Determine section title for this page
            section_title = document.get_section_for_page(page.page_number) or "General"

            # Split page text into natural segments (paragraphs / sentences)
            page_chunks_text = self._split_text_into_chunks(page_text)

            for text_chunk in page_chunks_text:
                if len(text_chunk.strip()) < 10:  # Skip trivial fragments
                    continue

                chunk_obj = DocumentChunk(
                    document_id=document.document_id,
                    chunk_id=new_id("chunk"),
                    page_number=page.page_number,
                    section=section_title,
                    text=text_chunk.strip(),
                    chunk_index=global_chunk_index,
                    token_estimate=max(1, len(text_chunk.strip()) // 4),
                )
                chunks.append(chunk_obj)
                global_chunk_index += 1

        return chunks

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split a block of text into chunks of roughly `chunk_size` characters
        with `chunk_overlap` while preserving sentence boundaries.
        """
        if len(text) <= self.chunk_size:
            return [text]

        # Break into sentences
        sentences = self._split_into_sentences(text)
        if not sentences:
            return [text]

        chunks: List[str] = []
        current_chunk_sentences: List[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence) + 1  # +1 for space

            if current_len + sentence_len > self.chunk_size and current_chunk_sentences:
                # Flush current chunk
                chunk_str = " ".join(current_chunk_sentences).strip()
                if chunk_str:
                    chunks.append(chunk_str)

                # Overlap logic: keep trailing sentences that fit within overlap window
                overlap_sentences: List[str] = []
                overlap_len = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_len + len(s) + 1 <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break

                current_chunk_sentences = list(overlap_sentences)
                current_len = sum(len(s) + 1 for s in current_chunk_sentences)

            current_chunk_sentences.append(sentence)
            current_len += sentence_len

        # Flush any remaining text
        if current_chunk_sentences:
            chunk_str = " ".join(current_chunk_sentences).strip()
            if chunk_str and (not chunks or chunk_str != chunks[-1]):
                chunks.append(chunk_str)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences cleanly."""
        # Normalize double newlines to single space or paragraph marker
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        all_sentences: List[str] = []

        for p in paragraphs:
            # Match standard sentence endings (. ? !) followed by whitespace and capital letter
            raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", p)
            for s in raw_sentences:
                cleaned = s.strip().replace("\n", " ")
                if cleaned:
                    all_sentences.append(cleaned)

        return all_sentences


def chunk_document(
    document: Document,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
    min_chunk_size: int = settings.MIN_CHUNK_SIZE,
) -> List[DocumentChunk]:
    """Convenience helper function to chunk a Document."""
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )
    return chunker.chunk(document)
