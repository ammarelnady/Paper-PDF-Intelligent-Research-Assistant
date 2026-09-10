# PaperLens AI

PaperLens AI is a local-first research workspace for reading, exploring, and questioning academic PDFs. It combines structured document processing, retrieval-augmented generation (RAG), configurable LLM routing, and optional web search to produce answers grounded in a paper’s content and cited source passages.

## Highlights

- Upload PDF research papers through a lightweight web interface.
- Extract document text, page metadata, academic sections, topics, concepts, and keywords.
- Generate structured summaries covering the research problem, methodology, findings, limitations, and contributions.
- Ask questions with FAISS semantic retrieval, BM25 retrieval, or Hybrid RRF retrieval.
- Route questions to the paper, web search, or a combined evidence path.
- Display page- and section-aware citations for retrieved evidence.
- Generate suggested research questions from the uploaded paper.
- Use an extractive fallback when an LLM provider is unavailable.

## Architecture

```text
PaperLens AI
├── frontend/                  Vanilla HTML, CSS, and JavaScript UI
└── backend/
    ├── app/api/               FastAPI routes and response schemas
    ├── app/document_processing/
    │                           PDF extraction, section detection, chunking
    ├── app/rag/               Embeddings, FAISS, BM25, and Hybrid RRF
    ├── app/routing/           Query intent and route selection
    ├── app/web_search/        Optional provider-neutral web search
    ├── app/llm/               LLM client, prompts, and citations
    ├── app/summarization/     Structured and section-level summaries
    ├── app/topics/            Topics, keywords, and concepts
    ├── app/questions/         Suggested research questions
    └── tests/                 Automated test suite
```

## Requirements

- Python 3.10 or newer
- A modern browser
- Optional Hugging Face API access for LLM-powered summaries, routing, and answers
- Optional FAISS and SentenceTransformers dependencies for semantic retrieval

## Quick start

From the repository root:

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies and configure the environment:

```bash
python -m pip install -r requirements.txt
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
```

Add your own credentials to `.env`. Keep secrets out of source control; the repository’s `.gitignore` excludes local environment files.

Start the API and frontend:

```bash
uvicorn app.main:app --reload
```

Open <http://localhost:8000/app/index.html> in your browser.

## Configuration

The main settings are documented in `backend/.env.example`. Important options include:

- `LLM_PROVIDER`: select Hugging Face, Ollama, or the local fallback.
- `HUGGINGFACE_API_KEY`: API key used for Hugging Face inference.
- `HUGGINGFACE_MODEL`: model used for generation.
- `EMBEDDING_MODEL`: SentenceTransformers model used for semantic retrieval.
- `QUERY_CLASSIFIER_MODE`: choose `llm` for orchestrated routing or `rules` for deterministic routing.
- `WEB_SEARCH_ENABLED`: enable or disable external web retrieval.

## Testing

Run the backend tests from the `backend/` directory:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Data and privacy

Uploaded files and generated local indexes are stored under `backend/data/`. This application is intended for local or controlled deployments. Before exposing it publicly, configure authentication, restrict CORS, protect uploaded documents, and provide separate data isolation for each user.

## Known limitations

- OCR for scanned or image-only PDFs is not included.
- Web search availability depends on the configured provider and network access.
- LLM responses can be incomplete or inaccurate; verify important claims against the cited paper pages.
- The default registry is file-backed and is not intended for concurrent multi-user production workloads.

## License

PaperLens AI is released under the [MIT License](LICENSE). See the license file for the complete terms.

## Acknowledgements

The project builds on FastAPI, PyMuPDF, SentenceTransformers, FAISS, Pydantic, and the Hugging Face inference ecosystem.
