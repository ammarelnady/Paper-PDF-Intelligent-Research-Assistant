# Paper Research Assistant

An NLP/LLM research assistant for extracting evidence from research papers,

answering grounded questions, and preserving source citations.

## Semantic Retrieval / RAG

The `backend/app/rag/` package provides an in-memory semantic retrieval layer:

- `SentenceTransformerEmbeddingProvider` creates normalized `float32` text and
  query embeddings. Its model name is configurable and it loads lazily.
- `FaissVectorStore` indexes normalized vectors with FAISS `IndexFlatIP`, which
  performs cosine-similarity search after normalization.
- `SemanticRetriever` retrieves a configurable candidate pool and returns the
  requested top-K citation-ready chunks. FAISS-only retrieval remains its
  default behavior.
- `ScoreReranker` is the default dependency-free reranker; a cross-encoder may
  replace it later through the `Reranker` interface.
- `Bm25Index` is an optional lexical index, separate from `FaissVectorStore`.
- `reciprocal_rank_fusion` combines independently ranked BM25 and FAISS results
  with the standard `sum(1 / (rrf_k + rank))` formulation.

The RAG module currently accepts any object exposing the stable DocumentChunk
fields: `document_id`, `chunk_id`, `page_number`, `section`, and `text`. Its
results preserve those fields plus `score`, so Salma's eventual `DocumentChunk`
implementation can be used directly without changing the RAG public API.

## Retrieval strategies

`SemanticRetriever.retrieve()` supports multiple retrieval approaches:

- `strategy="faiss"` — semantic FAISS retrieval. An optional `ScoreReranker`
  can rerank its FAISS candidates.
- `ScoreReranker` — can be used with FAISS to combine semantic similarity with
  a lightweight lexical signal. It is dependency-free and replaceable through
  the `Reranker` interface.
- `strategy="hybrid_rrf"` — combines independently ranked FAISS and BM25
  candidates using Reciprocal Rank Fusion (RRF). Supply a populated `Bm25Index`
  to the retriever. The RRF constant defaults to `60` and is configurable
  through `rrf_k`.

## Recommended project strategy

- For the project pipeline, **Hybrid BM25 + FAISS retrieval with RRF is the
  recommended strategy**. It combines semantic and lexical retrieval signals and
  performed best in the current retrieval evaluation.

The other retrieval approaches remain available for comparison, experimentation,

and fallback use. The RAG component intentionally keeps the strategy configurable

rather than forcing a single approach.

To use the recommended Hybrid RRF strategy:

```python
bm25_index = Bm25Index()

bm25_index.add(chunks)

retriever = SemanticRetriever(
    embedding_provider=provider,
    vector_store=faiss_store,
    bm25_index=bm25_index,
    rrf_k=60,
)

results = retriever.retrieve(
    "What optimizer did the authors use?",
    strategy="hybrid_rrf",
)
```

## Evaluation notebook

`notebooks/rag_experiments.ipynb` evaluates the real 38-chunk output from

Salma's *Attention Is All You Need* processing pipeline. It compares FAISS,

FAISS plus `ScoreReranker`, BM25, and BM25 plus FAISS RRF on the same 16

manually labeled questions (18 gold chunk IDs). The notebook reports Recall@3,

Recall@5, Recall@10, MRR, Precision@3, deltas against FAISS, and the optimizer

chunk's rank for FAISS, BM25, and RRF.

## RAG structure

```text
backend/
├── app/
│   └── rag/
│       ├── __init__.py          # Public RAG exports
│       ├── embeddings.py        # EmbeddingProvider and Sentence Transformers
│       ├── vector_store.py      # FAISS index and citation-ready results
│       ├── bm25_store.py        # Optional lexical BM25 index
│       ├── fusion.py             # Reciprocal Rank Fusion helper
│       ├── retriever.py          # FAISS-default and explicit hybrid strategy
│       └── reranker.py           # Optional ScoreReranker interface/implementation
├── tests/
│   ├── test_rag.py               # Deterministic RAG unit tests
│   └── test_rag_end_to_end.py    # Real PDF: processing to FAISS/BM25/RRF test
└── data/uploads/
    └── attention_is_all_you_need.pdf  # Local E2E fixture; gitignored

notebooks/

└── rag_experiments.ipynb         # Real-chunk retrieval evaluation
```

## Real-PDF E2E test

The E2E test uses Salma's extraction, section detection, and chunking before

building FAISS, BM25, and Hybrid RRF retrieval. It verifies result metadata and

prints each query's top chunk ID, page, and section when pytest output capture is

disabled. Put the PDF at `backend/data/uploads/attention_is_all_you_need.pdf`

or set `PAPER_PDF_PATH` to a full PDF path.

## Setup and tests

Install the backend dependencies and run the RAG tests from the repository root:

```bash
pip install -r backend/requirements.txt
python -m pytest backend/tests/test_rag.py
```

Run the E2E test with visible retrieval output:

```bash
PYTHONPATH=backend python -m pytest backend/tests/test_rag_end_to_end.py -s -q
```

In PowerShell, use:

```powershell
$env:PYTHONPATH = (Resolve-Path backend)

python -m pytest backend/tests/test_rag_end_to_end.py -s -q
```
