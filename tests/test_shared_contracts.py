from app.models import (
    Answer,
    Citation,
    Concept,
    Document,
    DocumentChunk,
    PageContent,
    PaperAnalysis,
    RetrievedChunk,
    RouteDecision,
    SuggestedQuestion,
    Topic,
    WebSource,
)


def test_document_and_page_content_contracts() -> None:
    document = Document("doc-1", "paper.pdf", 2)
    first_page = PageContent("doc-1", 1, "First page")

    assert document.document_id == "doc-1"
    assert document.filename == "paper.pdf"
    assert document.page_count == 2
    assert first_page.page_number == 1
    assert first_page.text == "First page"


def test_document_chunk_contract_preserves_optional_section() -> None:
    chunk = DocumentChunk("doc-1", 2, None, "doc-1:p2:c0", "Chunk text")

    assert chunk.document_id == "doc-1"
    assert chunk.chunk_id == "doc-1:p2:c0"
    assert chunk.page_number == 2
    assert chunk.section is None
    assert chunk.text == "Chunk text"


def test_paper_understanding_contracts_preserve_source_chunks() -> None:
    topic = Topic("topic-1", "Transformer architecture", 0.91, ["doc-1:p1:c0"])
    concept = Concept("attention", 0.85, ["doc-1:p1:c0"])
    analysis = PaperAnalysis("doc-1", [topic], [concept])
    question = SuggestedQuestion(
        "question-1", "doc-1", "What method was used?", ["doc-1:p1:c0"], "methodology"
    )

    assert analysis.topics[0].source_chunk_ids == ["doc-1:p1:c0"]
    assert analysis.concepts[0].name == "attention"
    assert question.category == "methodology"
    assert question.source_chunk_ids == ["doc-1:p1:c0"]


def test_retrieval_and_route_contracts_preserve_multidocument_metadata() -> None:
    result = RetrievedChunk("doc-2", "doc-2:p5:c1", "Evidence", 5, "Results", 0.88)

    assert result.document_id == "doc-2"
    assert result.score == 0.88
    assert result.page_number == 5
    assert result.section == "Results"
    assert [RouteDecision(route, 0.9).route for route in ("RAG", "WEB", "HYBRID")] == ["RAG", "WEB", "HYBRID"]


def test_web_citation_and_answer_contracts() -> None:
    web_source = WebSource("Paper index", "https://example.org/paper", "example.org", "A paper record.")
    paper_citation = Citation("cite-1", "paper", "doc-1:p5:c0", 5, "Methodology", None, None)
    web_citation = Citation("cite-2", "web", "web-1", None, None, web_source.title, web_source.url)
    answer = Answer("Grounded answer.", [paper_citation, web_citation])

    assert web_source.domain == "example.org"
    assert answer.text == "Grounded answer."
    assert answer.citations[0].page_number == 5
    assert answer.citations[1].url == "https://example.org/paper"
