# Paper Research Assistant

Phase 1 of an NTI team project: a modular foundation that accepts a PDF, extracts page-aware text, detects basic section headings, and produces metadata-preserving chunks.

## Features in this phase

- PDF upload API with file type and size validation
- PyMuPDF extraction with page metadata
- Heuristic heading and section detection
- Configurable character-based, overlap-aware chunking that never crosses pages or detected section transitions
- Typed internal models and unit tests

## Setup

Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

## Run the API

```powershell
uvicorn app.main:app --reload
```

Upload a document with:

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/documents/upload -F "file=@paper.pdf"
```

The response contains the generated document ID, pages, and chunks. Interactive API documentation is at `http://127.0.0.1:8000/docs`.

## Test

```powershell
pytest
```

## Configuration

Settings are environment-driven; copy `.env.example` to `.env` and adjust as needed. The current settings control upload location, maximum PDF size, chunk length, and chunk overlap. `PRA_CHUNK_SIZE` and `PRA_CHUNK_OVERLAP` are character counts; overlap applies between adjacent chunks in the same page section.

## Project layout

```text
app/
  api/                    FastAPI endpoints and request/response schemas
  core/                   Settings and shared exceptions
  document_processing/    Storage, extraction, sections, and chunking
  models/                 Clean internal document representations
  topics/ questions/ rag/ routing/ web_search/ llm/  Reserved Phase 2+ modules
tests/                    Unit tests
data/uploads/             Runtime PDF storage (ignored by Git)
frontend/ docs/           Frontend and documentation space
```

## Phase 2 recommendation

Add persistent document/chunk storage and a pluggable embedding + FAISS retrieval layer, then evaluate retrieval quality before adding web routing.

# Paper-PDF-Intelligent-Research-Assistant
