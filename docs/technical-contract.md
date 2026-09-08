# Technical Contract + Interface Freeze v1

**Technical Contract Version: v1**

**Status: FROZEN**

This document defines the framework-independent data contracts shared by the Paper Research Assistant modules. Breaking changes to frozen contracts require team agreement and corresponding test/documentation updates.

## Shared models

| Model | Purpose |
| --- | --- |
| `Document` | Paper identity and page count. Phase 1 retains optional in-memory `source_path`, `pages`, and `chunks` compatibility fields. |
| `PageContent` | Extracted text for one 1-based PDF page. `SectionSpan` remains an internal section-processing detail. |
| `DocumentChunk` | Retrieval-ready page and section text unit. |
| `Topic`, `Concept`, `PaperAnalysis` | Future paper-understanding outputs. |
| `SuggestedQuestion` | Future question-generation output. |
| `RetrievedChunk` | Future multi-document retrieval result. |
| `RouteDecision`, `WebSource` | Future research-agent outputs. |
| `Citation`, `Answer` | Future grounded-answer output. |

All contracts are plain Python dataclasses in `app.models`; they have no dependency on FastAPI, model providers, vector stores, or feature modules.

## Frozen fields and conventions

`Document` exposes `document_id: str`, `filename: str`, and `page_count: int`.

`PageContent` exposes `document_id: str`, `page_number: int`, and `text: str`. Page numbers are **1-based**: the first page is `1`.

`DocumentChunk` fields are frozen by name:

```text
document_id, chunk_id, page_number, section, text
```

`section` is `str | None`. It is produced by lightweight heuristic heading detection; multiple headings on a page are retained at chunk level.

Chunk IDs are deterministic and must remain:

```text
{document_id}:p{page_number}:c{index}
```

The current chunker is **character-based**, not token-based. `chunk_size` and `chunk_overlap` are character counts. Chunks are page-bounded, section-bounded where headings are detected, and overlap only adjacent chunks within the same section span.

`Topic` has `topic_id`, `name`, `score`, and `source_chunk_ids`. `Concept` has `name`, `score`, and `source_chunk_ids`. `PaperAnalysis` contains `document_id`, `topics`, and `concepts`.

`SuggestedQuestion` contains `question_id`, `document_id`, `question`, `source_chunk_ids`, and `category`. Initial category values are `summary`, `methodology`, `dataset`, `results`, `limitations`, `concept`, and `comparison`; this set is not exhaustive.

`RetrievedChunk` always includes `document_id`, `chunk_id`, `text`, `page_number`, `section`, and `score` so multi-document retrieval will not require a contract change.

`RouteDecision` has `route` and `confidence`; intended route values are `RAG`, `WEB`, and `HYBRID`.

`WebSource` has `title`, `url`, `domain`, and `snippet`. `Citation` has `citation_id`, `source_type`, `source_id`, `page_number`, `section`, `title`, and `url`; expected source types are `paper` and `web`. `Answer` has only `text` and `citations`.

## Module boundaries and dependency direction

```text
app/models  <- document_processing, topics, questions, rag, routing, web_search, llm
```

- Document Processing produces `Document`, `PageContent`, and `DocumentChunk`; it does not depend on future understanding, retrieval, routing, web, or answer contracts.
- Paper Understanding will consume `DocumentChunk` and produce `PaperAnalysis` and `SuggestedQuestion[]`.
- Semantic Retrieval will consume `DocumentChunk[]` and produce `RetrievedChunk[]`.
- Research Agent will produce `RouteDecision` and `WebSource[]`.
- The LLM layer will consume `RetrievedChunk[]`, `WebSource[]`, and `RouteDecision`, then produce `Answer`.

The planned modules are not implemented by this contract.

## Future API contracts — documentation only

Existing endpoint:

```text
POST /api/v1/documents/upload
```

Planned, unimplemented endpoints:

```text
GET  /api/v1/documents/{document_id}
GET  /api/v1/documents/{document_id}/topics
GET  /api/v1/documents/{document_id}/questions
POST /api/v1/retrieval/search
POST /api/v1/query/route
POST /api/v1/chat
```

The intended retrieval request is:

```json
{
  "document_id": "doc123",
  "query": "What dataset was used?",
  "top_k": 5
}
```

The intended response is:

```json
{
  "results": []
}
```
