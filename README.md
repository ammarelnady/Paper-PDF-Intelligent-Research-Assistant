# Paper Research Assistant

This repository follows the Paper Research Assistant architecture. The only
implemented functionality is Ammar's Research Agent routing layer and its
provider-neutral web-search boundary.

## Project structure

```text
Paper-Research-Assistant/
├── backend/
│   ├── app/
│   │   ├── contracts.py             # shared routing technical contracts
│   │   ├── main.py                  # reserved application entry point
│   │   ├── config.py                # reserved configuration module
│   │   ├── models/                  # reserved for document models
│   │   ├── document_processing/     # reserved for PDF/document processing
│   │   ├── summarization/           # reserved for summarization
│   │   ├── topics/                  # reserved for topics and concepts
│   │   ├── questions/               # reserved for suggested questions
│   │   ├── rag/                     # reserved for RAG implementation
│   │   ├── routing/                 # implemented Research Agent routing
│   │   ├── web_search/              # implemented provider-neutral boundary
│   │   ├── llm/                     # reserved for LLM integration
│   │   ├── evaluation/              # reserved for evaluation
│   │   ├── api/                     # reserved for API integration
│   │   └── utils/                   # reserved for utilities
│   ├── data/uploads/
│   ├── tests/test_routing.py
│   ├── requirements.txt
│   ├── .env.example
│   └── .gitignore
└── frontend/
    ├── css/
    └── js/
```

The reserved Python packages contain only `__init__.py` markers and no
implementation from other team members.

## Implemented: Research Agent

The routing flow is deterministic by default and has no external runtime
dependencies:

```text
User query
  → IntentDetector
  → DeterministicQueryClassifier
  → RouteDecision: RAG | WEB | HYBRID
  → ResearchRouter
      → Retriever boundary, WebSearchClient boundary, or both
  → RoutingResult
```

`backend/app/contracts.py` is the authoritative source for the shared routing
contracts:

- `RouteDecision(route, confidence)`
- `RetrievedChunk(document_id, chunk_id, text, page_number, section, score)`
- `WebSource(title, url, domain, snippet)`

Paper-specific queries generally select `RAG`; current or external research
queries select `WEB`; paper-plus-external comparisons select `HYBRID`. Clear
matches have high confidence. Comparison-only and general ambiguous queries
remain low-confidence `HYBRID` decisions and can invoke an injected
`ClassifierFallback` when one is supplied.

### Dependency boundaries

RAG is represented only by the router-owned `Retriever` protocol:

```python
def search(self, query: str) -> Sequence[RetrievedChunk]: ...
```

No retrieval, embeddings, FAISS, reranking, or vector store exists here.

Web search is represented by an injected `SearchProvider` used by
`WebSearchClient`:

```python
def search(self, query: str) -> Sequence[WebSource]: ...
```

`DuckDuckGoHtmlSearchProvider` is the included no-key provider. It uses
DuckDuckGo's public HTML endpoint through Python's standard library, maps
valid results into `WebSource`, and raises `WebSearchProviderError` for
network or response failures. Its transport can be injected for offline
tests. `WEB_SEARCH_PROVIDER` may be set by future application configuration;
the current routing module does not read environment variables directly.

The optional `ClassifierFallback` is an interface only; no Ollama or model
adapter is implemented.

## Testing

Run from the repository root:

```powershell
python -m compileall backend/app backend/tests
python -m unittest discover -s backend/tests -v
```

Routing and provider tests use fake or mocked transports, retrievers,
search-providers, and fallback dependencies. They are completely offline and
do not require internet access, API keys, or Ollama.

## Not yet implemented

- PDF extraction, section detection, chunking, and document models
- Summarization
- Topics, keywords, concepts, and question generation
- Embeddings, FAISS, vector store, retrieval, and reranking
- LLM answer generation and citations
- API integration, frontend, and evaluation
