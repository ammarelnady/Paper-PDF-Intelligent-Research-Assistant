from __future__ import annotations

from dataclasses import dataclass
import unittest

from app.questions import QuestionGenerator
from app.topics import (
    ConceptExtractor,
    ImportantSectionIdentifier,
    KeywordExtractor,
    PaperUnderstandingAnalyzer,
    TopicExtractor,
)


@dataclass
class MockChunk:
    document_id: str
    chunk_id: str
    page_number: int
    section: str | None
    text: str


CHUNKS = [
    MockChunk(
        "paper-1",
        "paper-1:p1:c1",
        1,
        "Abstract",
        "We introduce a graph neural network for scientific document classification.",
    ),
    MockChunk(
        "paper-1",
        "paper-1:p2:c1",
        2,
        "3 Methodology",
        "The Graph Neural Network (GNN) uses message passing. The GNN learns document representations.",
    ),
    MockChunk(
        "paper-1",
        "paper-1:p3:c1",
        3,
        "4 Dataset and Evaluation",
        "We evaluate the graph neural network on the PubMed dataset. The dataset contains scientific articles.",
    ),
    MockChunk(
        "paper-1",
        "paper-1:p4:c1",
        4,
        "5 Results",
        "The graph neural network improves classification accuracy over the baseline model.",
    ),
    MockChunk(
        "paper-1",
        "paper-1:p5:c1",
        5,
        "6 Limitations",
        "The main limitation is high computation cost on very large graphs.",
    ),
]


class ChunkBoundaryTests(unittest.TestCase):
    def test_dictionary_chunks_are_supported(self) -> None:
        keywords = KeywordExtractor(max_keywords=3).extract([CHUNKS[0].__dict__])
        self.assertTrue(keywords)

    def test_missing_contract_field_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "page_number"):
            KeywordExtractor().extract([{"document_id": "x", "chunk_id": "c", "section": None, "text": "valid text"}])

    def test_mixed_documents_are_rejected(self) -> None:
        other = MockChunk("paper-2", "c2", 1, None, "Other document text")
        with self.assertRaisesRegex(ValueError, "same document"):
            TopicExtractor().extract([CHUNKS[0], other])


class ExtractionTests(unittest.TestCase):
    def test_keywords_have_scores_and_sources(self) -> None:
        keywords = KeywordExtractor(max_keywords=10).extract(CHUNKS)
        self.assertTrue(all(0 < item.score <= 1 for item in keywords))
        self.assertTrue(all(item.source_chunk_ids for item in keywords))
        self.assertTrue(any("graph neural" in item.text for item in keywords))

    def test_topics_use_sections_and_frozen_contract(self) -> None:
        topics = TopicExtractor(max_topics=5).extract(CHUNKS)
        self.assertEqual(topics[0].topic_id, "topic-001")
        self.assertTrue(any(topic.name == "Methodology" for topic in topics))
        self.assertTrue(all(topic.source_chunk_ids for topic in topics))

    def test_page_labels_are_not_used_as_topics(self) -> None:
        chunk = MockChunk("paper-1", "c1", 1, "Page 1", "transformer attention model transformer attention model")
        topics = TopicExtractor(max_topics=3).extract([chunk])
        self.assertNotIn("Page 1", {topic.name for topic in topics})

    def test_concepts_find_defined_acronym(self) -> None:
        concepts = ConceptExtractor(max_concepts=8).extract(CHUNKS)
        self.assertTrue(any("GNN" in concept.name for concept in concepts))
        self.assertEqual(sum("graph neural network" in concept.name.lower() for concept in concepts), 1)

    def test_important_sections_prioritise_evidence(self) -> None:
        sections = ImportantSectionIdentifier(max_sections=3).identify(CHUNKS)
        names = {item.section for item in sections}
        self.assertIn("3 Methodology", names)
        self.assertIn("5 Results", names)


class QuestionAndPipelineTests(unittest.TestCase):
    def test_llm_questions_are_structured_and_source_linked(self) -> None:
        def mock_llm(_prompt: str, _system: str) -> str:
            return (
                '[{"category":"methodology",'
                '"question":"How does the proposed graph model work?",'
                '"source_chunk_ids":["paper-1:p2:c1"]}]'
            )

        questions = QuestionGenerator(llm_caller=mock_llm).generate(CHUNKS)
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].category, "methodology")
        self.assertEqual(questions[0].source_chunk_ids, ["paper-1:p2:c1"])

    def test_questions_are_grounded_and_diverse(self) -> None:
        topics = TopicExtractor(max_topics=3).extract(CHUNKS)
        concepts = ConceptExtractor(max_concepts=3).extract(CHUNKS)
        questions = QuestionGenerator(max_questions=8).generate(
            CHUNKS, topics=topics, concepts=concepts
        )
        categories = {question.category for question in questions}
        self.assertTrue({"methodology", "results"}.issubset(categories))
        self.assertTrue(all(question.source_chunk_ids for question in questions))
        self.assertEqual(len({question.question for question in questions}), len(questions))

    def test_full_pipeline_preserves_document_id(self) -> None:
        result = PaperUnderstandingAnalyzer(max_questions=6).analyze(CHUNKS)
        self.assertEqual(result.document_id, "paper-1")
        self.assertEqual(result.analysis.document_id, "paper-1")
        self.assertTrue(result.keywords)
        self.assertTrue(result.analysis.topics)
        self.assertTrue(result.analysis.concepts)
        self.assertTrue(result.important_sections)
        self.assertTrue(result.questions)

    def test_outputs_are_deterministic(self) -> None:
        analyzer = PaperUnderstandingAnalyzer()
        first = analyzer.analyze(CHUNKS)
        second = analyzer.analyze(CHUNKS)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
