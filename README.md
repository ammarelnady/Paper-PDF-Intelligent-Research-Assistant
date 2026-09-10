# Paper Research Assistant

An intelligent NLP/LLM research assistant that goes beyond basic "Chat with PDF" — understanding papers, extracting sections, topics, and concepts, summarizing them, performing semantic retrieval, routing queries across the paper and external web search, and generating grounded, cited answers.

## Project Structure

```text
Paper-Research-Assistant/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI application entry point, CORS, health check
│   │   ├── config.py                # Global settings (chunk size, models, API keys)
│   │   ├── models/                  # Salma — Typed data models (Document, DocumentChunk, PaperSummary)
│   │   ├── document_processing/     # Salma — PDF text extraction, section detection, metadata chunking
│   │   ├── summarization/           # Salma — Overall and section-level grounded summarization
│   │   ├── topics/                  # Shahd — Topics, keywords, and concept extraction
│   │   ├── questions/               # Shahd — Grounded suggested questions generation
│   │   ├── rag/                     # Reem — SentenceTransformers, FAISS vector store, BM25, Hybrid RRF
│   │   ├── routing/                 # Ammar — Deterministic query classification (RAG, WEB, HYBRID)
│   │   ├── web_search/              # Ammar — Provider-neutral web search client
│   │   ├── llm/                     # Mohamed — Generation, prompt building, citation formatting
│   │   ├── evaluation/              # Mohamed — Evaluation framework
│   │   ├── api/                     # Mohamed — FastAPI routes and schemas
│   │   └── utils/                   # Shared utilities and logging
│   ├── tests/                       # Unit and integration test suites
│   ├── requirements.txt
│   ├── .env.example
│   └── .gitignore
└── frontend/                        # Vanilla HTML/CSS/JS interface
```

## Architecture & Modules

### 1. Document Processing & Summarization (Salma)
- **PDF Extraction**: `extract_text_from_pdf` using PyMuPDF (`pymupdf`) with automatic noise/watermark stripping and title inference.
- **Section Detection**: `detect_sections` mapping text to canonical academic sections (Abstract, Introduction, Methodology, Results, Conclusion, etc.) with page boundaries.
- **Metadata-Aware Chunking**: `chunk_document` producing chunks tagged with `document_id`, `chunk_id`, `page_number`, `section`, `chunk_index`, and `token_estimate`.
- **Grounded Summarizer**: `PaperSummarizer` producing structured summaries (`summary`, `key_contributions`, `problem_statement`, `methodology`, `findings`, `limitations`, `source_chunks`) and section-level summaries.

### 2. Paper Understanding (Shahd)
- **Topic & Keyword Extraction**: Extracts core topics, domain keywords, and concepts from document chunks.
- **Suggested Questions**: Generates grounded research questions tied to specific paper sections.

### 3. Semantic Retrieval & RAG (Reem)
- **Embeddings**: `SentenceTransformerEmbeddingProvider` (e.g. `all-MiniLM-L6-v2`).
- **Vector Store & Indexing**: `FaissVectorStore` (in-memory cosine similarity search via `IndexFlatIP`) and `Bm25Index`.
- **Hybrid Search**: `SemanticRetriever` supporting FAISS semantic search and `hybrid_rrf` (Reciprocal Rank Fusion) combining FAISS and BM25.

### 4. Research Agent & Query Routing (Ammar)
- **Intent & Query Classification**: Configurable LLM orchestrator routing queries to `RAG` (paper questions), `WEB` (current / external questions), or `HYBRID` (comparative questions), with deterministic fallback.
- **Web Search Client**: Provider-neutral web search abstraction with included no-key `DuckDuckGoHtmlSearchProvider`.

### 5. LLM, API & Evaluation (Mohamed)
- **LLM Grounded Answers**: Prompt construction with retrieved context and web snippets.
- **Citations**: Inline citation formatting referencing exact chunk IDs, pages, and sections.
- **FastAPI Backend & UI**: Endpoints for document upload, sections, summary, questions, and routed chat queries.

## Running Tests

From the `backend/` directory:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Running the application

From `backend/`, create a virtual environment, install `requirements.txt`, and
copy `.env.example` to `.env`. Set `HUGGINGFACE_API_KEY` only in `.env` or in
your deployment secret store; never commit credentials to source control.

```bash
uvicorn app.main:app --reload
```

Then open <http://localhost:8000/app/index.html>. Uploaded document metadata is
stored in `backend/data/documents.json`; generated FAISS indexes are stored in
`backend/data/indices/` and restored when the server starts.

## Current limitations

- OCR for scanned/image-only PDFs is not included.
- The default web search provider depends on DuckDuckGo availability.
- Set `QUERY_CLASSIFIER_MODE=rules` for deterministic offline routing, or
  `QUERY_CLASSIFIER_MODE=llm` to let the configured LLM choose the route.
- Optional bearer-token protection is available through `API_AUTH_TOKEN`, but
  multi-user document isolation is not included; deploy behind an authenticated
  gateway before exposing the service publicly.
