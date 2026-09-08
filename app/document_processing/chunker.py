from __future__ import annotations

from app.models.document import DocumentChunk, PageContent, SectionSpan


def chunk_pages(pages: tuple[PageContent, ...], chunk_size: int, chunk_overlap: int) -> tuple[DocumentChunk, ...]:
    """Create character-based, page-bounded chunks at word boundaries.

    ``chunk_size`` and ``chunk_overlap`` are character counts. Section spans are
    chunked independently, so a chunk never contains text from two sections.
    """
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
    chunks: list[DocumentChunk] = []
    for page in pages:
        index = 0
        spans = page.section_spans or (SectionSpan(page.section, 0, len(page.text)),)
        for span in spans:
            for chunk_text in _split_text(page.text[span.start:span.end], chunk_size, chunk_overlap):
                chunks.append(DocumentChunk(
                    document_id=page.document_id,
                    page_number=page.page_number,
                    section=span.section,
                    chunk_id=f"{page.document_id}:p{page.page_number}:c{index}",
                    text=chunk_text,
                ))
                index += 1
    return tuple(chunks)


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> tuple[str, ...]:
    """Split non-empty text with character overlap, preferring word boundaries."""
    text = text.strip()
    if not text:
        return ()
    result: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text) and (boundary := text.rfind(" ", start, end)) > start:
            end = boundary
        chunk_text = text[start:end].strip()
        if chunk_text:
            result.append(chunk_text)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return tuple(result)
