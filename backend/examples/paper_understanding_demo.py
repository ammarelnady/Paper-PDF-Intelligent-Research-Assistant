"""Small example that can be used during the project discussion."""

from app.topics import PaperUnderstandingAnalyzer


# These dictionaries have the same fields as Salma's DocumentChunk model.
sample_chunks = [
    {
        "document_id": "demo-paper",
        "chunk_id": "demo-paper:p1:c1",
        "page_number": 1,
        "section": "Abstract",
        "text": "We introduce a graph neural network for document classification.",
    },
    {
        "document_id": "demo-paper",
        "chunk_id": "demo-paper:p2:c1",
        "page_number": 2,
        "section": "Methodology",
        "text": "The Graph Neural Network (GNN) uses message passing.",
    },
    {
        "document_id": "demo-paper",
        "chunk_id": "demo-paper:p3:c1",
        "page_number": 3,
        "section": "Results",
        "text": "The model improves document classification accuracy.",
    },
   """ {
    "document_id": "demo-paper",
    "chunk_id": "demo-paper:p4:c1",
    "page_number": 4,
    "section": "Dataset",
    "text": "The researchers trained the model using the PubMed dataset.",
}"""
]


analyzer = PaperUnderstandingAnalyzer()
result = analyzer.analyze(sample_chunks)

print("Keywords:", result.keywords)
print("Topics:", result.analysis.topics)
print("Concepts:", result.analysis.concepts)
print("Important sections:", result.important_sections)
print("Suggested questions:", result.questions)
