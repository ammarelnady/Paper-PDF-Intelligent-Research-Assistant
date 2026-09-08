import pytest

from app.document_processing.chunker import chunk_pages
from app.document_processing.sections import assign_sections
from app.models.document import PageContent


def test_multiple_numbered_headings_on_one_page_assign_chunk_sections() -> None:
    page = PageContent(
        "paper-1", 1,
        "2 Methodology\nMethod overview text.\n"
        "3.1 Dataset\nDataset details here.\n"
        "3.2 Preprocessing\nPreprocessing details here.",
    )

    chunks = chunk_pages(assign_sections((page,)), chunk_size=1_000, chunk_overlap=0)

    assert [chunk.section for chunk in chunks] == ["Methodology", "Dataset", "Preprocessing"]
    assert all(chunk.page_number == 1 for chunk in chunks)


def test_section_transition_is_preserved_across_pages() -> None:
    pages = (
        PageContent("paper-1", 1, "Introduction\nOpening text."),
        PageContent("paper-1", 2, "Continuation without a heading."),
        PageContent("paper-1", 3, "4 Results\nMeasured outcomes."),
    )

    assigned = assign_sections(pages)

    assert assigned[0].section == "Introduction"
    assert assigned[1].section == "Introduction"
    assert assigned[2].section == "Results"


def test_normal_sentences_are_not_headings() -> None:
    page = PageContent("paper-1", 1, "Introduction\nThis is a normal sentence without a heading.\nMore text.")

    assigned = assign_sections((page,))

    assert len(assigned[0].section_spans) == 1
    assert assigned[0].section_spans[0].section == "Introduction"


def test_empty_and_short_pages_are_safe() -> None:
    pages = (
        PageContent("paper-1", 1, ""),
        PageContent("paper-1", 2, "Short text", "Introduction"),
    )

    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 1
    assert chunks[0].text == "Short text"
    assert chunks[0].section == "Introduction"


def test_long_page_has_overlap_metadata_and_deterministic_ids() -> None:
    text = " ".join(f"word{i}" for i in range(100))
    page = PageContent("paper-1", 7, text, "Methods")

    first_run = chunk_pages((page,), chunk_size=60, chunk_overlap=15)
    second_run = chunk_pages((page,), chunk_size=60, chunk_overlap=15)

    assert len(first_run) > 1
    assert first_run == second_run
    assert "word" in first_run[0].text[-20:]
    assert first_run[0].text.split()[-1] in first_run[1].text
    for index, chunk in enumerate(first_run):
        assert chunk.document_id == "paper-1"
        assert chunk.page_number == 7
        assert chunk.section == "Methods"
        assert chunk.chunk_id == f"paper-1:p7:c{index}"
        assert chunk.text


@pytest.mark.parametrize("chunk_size, overlap", [(0, 0), (10, -1), (10, 10)])
def test_invalid_chunk_configuration_is_rejected(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_pages((PageContent("paper-1", 1, "text"),), chunk_size, overlap)
