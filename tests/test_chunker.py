from app.document_processing.chunker import chunk_pages
from app.document_processing.sections import assign_sections
from app.models.document import PageContent


def test_section_detection_and_chunks_keep_provenance() -> None:
    pages = (
        PageContent("doc-1", 1, "1. Methodology\n" + "analysis " * 40),
        PageContent("doc-1", 2, "Results\n" + "finding " * 40),
    )
    sectioned = assign_sections(pages)
    chunks = chunk_pages(sectioned, chunk_size=70, chunk_overlap=10)
    assert sectioned[0].section == "Methodology"
    assert sectioned[1].section == "Results"
    assert len(chunks) > 2
    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert all(chunk.document_id == "doc-1" for chunk in chunks)
    assert all(chunk.chunk_id.startswith(f"doc-1:p{chunk.page_number}:c") for chunk in chunks)
    assert all(len(chunk.text) <= 70 for chunk in chunks)


def test_chunker_rejects_invalid_overlap() -> None:
    try:
        chunk_pages((PageContent("doc-1", 1, "short text"),), 10, 10)
    except ValueError as error:
        assert "smaller" in str(error)
    else:
        raise AssertionError("Expected invalid overlap to fail")
