# Paper Understanding — Shahd

This feature consumes `DocumentChunk` values and produces grounded paper
topics, keywords, concepts, important sections, and suggested questions.

## Input contract

Each input object (dataclass, Pydantic model, or dictionary) must expose:

```text
document_id, chunk_id, page_number, section, text
```

All chunks in one call must have the same `document_id`; page numbers are
1-based. Empty chunks are ignored and clear `ValueError` messages are raised
for invalid data.

## Example

Run from `backend/`:

```python
from app.topics import PaperUnderstandingAnalyzer

chunks = [
    {
        "document_id": "paper-1",
        "chunk_id": "paper-1:p1:c1",
        "page_number": 1,
        "section": "Abstract",
        "text": "We introduce a graph neural network for document classification.",
    },
    {
        "document_id": "paper-1",
        "chunk_id": "paper-1:p2:c1",
        "page_number": 2,
        "section": "Methodology",
        "text": "The Graph Neural Network (GNN) uses message passing.",
    },
]

result = PaperUnderstandingAnalyzer().analyze(chunks)
print(result.analysis.topics)
print(result.keywords)
print(result.analysis.concepts)
print(result.important_sections)
print(result.questions)
```

## Architecture decisions

- The baseline is deterministic and uses no paid API or model download.
- Chunk-aware TF-IDF scores unigrams and bigrams while preserving sources.
- Section headings and high-scoring phrases provide readable topic labels.
- Concepts include repeated scientific phrases and defined acronyms.
- Section ranking combines semantic heading signals with content coverage.
- Question templates are selected from actual section/content signals; every
  question retains `source_chunk_ids` for later grounding and citations.
- `app/_shared_contracts.py` isolates the temporary dependency on Salma's shared models. It
  automatically imports `app.models.contracts` when that module is available.

## Testing

From `backend/`:

```bash
python -m unittest tests.test_topics_questions -v
```

The tests cover dict/object input, validation, deterministic outputs, frozen
contract fields, extraction quality, source grounding, diversity, and the full
pipeline.

## Integration after Salma's branch is ready

1. Merge/rebase Salma's branch into this feature branch.
2. Confirm the shared models are importable from `app.models.contracts`, or
   update only the import path in `app/_shared_contracts.py`.
3. Run the unit test above with one real `DocumentChunk` list returned by her
   chunker.
4. Remove the fallback dataclasses only after the shared import is stable.

No document-processing or shared-model files are modified by this feature.

## Student-friendly walkthrough

- Run `python examples/paper_understanding_demo.py` from `backend/` for a small
  end-to-end example.
- Read `docs/shahd-discussion-guide-ar.md` for a simple Arabic explanation and
  common discussion questions.
