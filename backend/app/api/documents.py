"""
Document API endpoints: upload, get document, sections, summary, and paper understanding.
"""

from __future__ import annotations

import logging
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.api.schemas import (
    ConceptItem,
    DocumentUploadResponse,
    KeywordItem,
    PaperSummaryResponse,
    PaperUnderstandingResponse,
    SectionResponse,
    SuggestedQuestionResponse,
    TopicItem,
)
from app.config import settings
from app.document_processing import (
    PDFExtractionError,
    chunk_document,
    detect_sections,
    extract_text_from_pdf,
)
from app.document_processing.section_analyzer import SectionOutlineAnalyzer
from app.llm import LLMClient
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.summary import PaperSummary
from app.summarization import PaperSummarizer
from app.topics import PaperUnderstandingAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

# In-memory document storage cache
DOCUMENTS_REGISTRY: Dict[str, Document] = {}
CHUNKS_REGISTRY: Dict[str, List[DocumentChunk]] = {}
SUMMARIES_REGISTRY: Dict[str, PaperSummary] = {}
ANALYSIS_REGISTRY: Dict[str, Any] = {}
VECTOR_STORES_REGISTRY: Dict[str, Any] = {}
BM25_REGISTRY: Dict[str, Any] = {}
REGISTRY_PATH = settings.DATA_DIR / "documents.json"


def _model_dump(value: Any) -> dict:
    """Serialize a Pydantic model across Pydantic v1/v2."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return json.loads(value.json())


def _model_validate(model: Any, data: dict) -> Any:
    """Deserialize a Pydantic model across Pydantic v1/v2."""
    return model.model_validate(data) if hasattr(model, "model_validate") else model.parse_obj(data)


def _save_registry() -> None:
    """Persist document metadata, chunks, and summaries for restart recovery."""
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "documents": [_model_dump(doc) for doc in DOCUMENTS_REGISTRY.values()],
        "chunks": {
            doc_id: [_model_dump(chunk) for chunk in chunks]
            for doc_id, chunks in CHUNKS_REGISTRY.items()
        },
        "summaries": {
            doc_id: _model_dump(summary)
            for doc_id, summary in SUMMARIES_REGISTRY.items()
        },
    }
    temp_path = REGISTRY_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(REGISTRY_PATH)


def load_persisted_documents() -> None:
    """Restore registries and indexes from disk, tolerating missing optional state."""
    if not REGISTRY_PATH.exists():
        return
    try:
        payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        for raw_doc in payload.get("documents", []):
            doc = _model_validate(Document, raw_doc)
            DOCUMENTS_REGISTRY[doc.document_id] = doc
        for doc_id, raw_chunks in payload.get("chunks", {}).items():
            CHUNKS_REGISTRY[doc_id] = [_model_validate(DocumentChunk, item) for item in raw_chunks]
        for doc_id, raw_summary in payload.get("summaries", {}).items():
            SUMMARIES_REGISTRY[doc_id] = _model_validate(PaperSummary, raw_summary)

        from app.rag import Bm25Index, FaissVectorStore, SentenceTransformerEmbeddingProvider
        for doc_id, chunks in CHUNKS_REGISTRY.items():
            index_dir = settings.INDICES_DIR / doc_id
            if chunks and index_dir.exists():
                try:
                    store = FaissVectorStore.load(index_dir)
                    bm25 = Bm25Index()
                    bm25.add(chunks)
                    provider = SentenceTransformerEmbeddingProvider(model_name=settings.EMBEDDING_MODEL)
                    VECTOR_STORES_REGISTRY[doc_id] = (provider, store)
                    BM25_REGISTRY[doc_id] = bm25
                except Exception as exc:
                    logger.warning("Could not restore index for %s: %s", doc_id, exc)
        logger.info("Restored %d persisted documents", len(DOCUMENTS_REGISTRY))
    except Exception as exc:
        logger.warning("Could not restore persisted document state: %s", exc)


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a research paper PDF.
    Extracts text, detects sections, chunks with metadata, and prepares semantic index.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    try:
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        content = await file.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise HTTPException(status_code=413, detail=f"PDF exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB limit.")
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )

        safe_filename = Path(file.filename).name

        # 1. Extract Document (Salma)
        document = extract_text_from_pdf(content, filename=safe_filename)
        if document.num_pages > settings.MAX_PDF_PAGES:
            raise HTTPException(status_code=413, detail=f"PDF exceeds the {settings.MAX_PDF_PAGES} page limit.")

        # 2. Save validated PDF to data/uploads
        save_path = settings.UPLOAD_DIR / safe_filename
        with open(save_path, "wb") as f:
            f.write(content)

        # 3. Detect and hierarchically organize sections with the LLM when available
        sections = detect_sections(document)

        llm = LLMClient()
        llm_caller = lambda prompt, sys_prompt: llm.generate(prompt=prompt, system_instruction=sys_prompt)
        sections = SectionOutlineAnalyzer(
            llm_caller=llm_caller if settings.HUGGINGFACE_API_KEY else None,
        ).analyze(document, sections)

        # 4. Chunk with Metadata (Salma)
        chunks = chunk_document(document, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

        # 5. Summarize (Salma) - wire LLM when API key is available
        summarizer = PaperSummarizer(
            strategy="llm" if settings.HUGGINGFACE_API_KEY else "extractive",
            llm_caller=llm_caller if settings.HUGGINGFACE_API_KEY else None,
            model_name=settings.HUGGINGFACE_MODEL,
        )
        summary = summarizer.summarize_paper(document, chunks=chunks)

        # 6. Paper Understanding (Shahd)
        analyzer = PaperUnderstandingAnalyzer(
            llm_caller=llm_caller if settings.HUGGINGFACE_API_KEY else None,
        )
        understanding = analyzer.analyze(chunks)

        # 7. RAG Semantic Indexing (Reem)
        try:
            from app.rag import (
                Bm25Index,
                FaissVectorStore,
                SentenceTransformerEmbeddingProvider,
            )
            provider = SentenceTransformerEmbeddingProvider(model_name=settings.EMBEDDING_MODEL)
            embeddings = provider.embed_texts([c.text for c in chunks])
            store = FaissVectorStore(embedding_dimension=embeddings.shape[1])
            store.add(chunks, embeddings)

            # Persist FAISS index to disk
            store.save(settings.INDICES_DIR / document.document_id)

            bm25 = Bm25Index()
            bm25.add(chunks)

            VECTOR_STORES_REGISTRY[document.document_id] = (provider, store)
            BM25_REGISTRY[document.document_id] = bm25
        except Exception as e:
            logger.warning(f"RAG indexing skipped or failed ({e}); basic retrieval active.")

        # Cache in registries
        DOCUMENTS_REGISTRY[document.document_id] = document
        CHUNKS_REGISTRY[document.document_id] = chunks
        SUMMARIES_REGISTRY[document.document_id] = summary
        ANALYSIS_REGISTRY[document.document_id] = understanding
        _save_registry()

        return DocumentUploadResponse(
            document_id=document.document_id,
            filename=document.filename,
            title=document.title,
            num_pages=document.num_pages,
            num_chunks=len(chunks),
            num_sections=len(sections),
            message="Document uploaded, structured, and indexed successfully.",
        )

    except HTTPException:
        raise
    except PDFExtractionError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during PDF ingestion")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Processing failed: {e}")


@router.get("/", response_model=List[DocumentUploadResponse])
def list_documents(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List all currently processed documents."""
    documents = [
        DocumentUploadResponse(
            document_id=doc.document_id,
            filename=doc.filename,
            title=doc.title,
            num_pages=doc.num_pages,
            num_chunks=len(CHUNKS_REGISTRY.get(doc.document_id, [])),
            num_sections=len(doc.sections),
            message="Active document in registry.",
        )
        for doc in DOCUMENTS_REGISTRY.values()
    ]
    return documents[offset:offset + limit]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str) -> None:
    """Delete a document and its derived local artifacts."""
    document = DOCUMENTS_REGISTRY.pop(document_id, None)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    CHUNKS_REGISTRY.pop(document_id, None)
    SUMMARIES_REGISTRY.pop(document_id, None)
    ANALYSIS_REGISTRY.pop(document_id, None)
    VECTOR_STORES_REGISTRY.pop(document_id, None)
    BM25_REGISTRY.pop(document_id, None)

    upload_path = settings.UPLOAD_DIR / Path(document.filename).name
    if upload_path.exists():
        upload_path.unlink()
    index_path = settings.INDICES_DIR / document_id
    if index_path.exists():
        shutil.rmtree(index_path)
    _save_registry()


@router.get("/{document_id}/summary", response_model=PaperSummaryResponse)
def get_paper_summary(document_id: str):
    """Get the structured overall summary for a document."""
    summary = SUMMARIES_REGISTRY.get(document_id)
    required_fields = (
        summary.problem_statement if summary else None,
        summary.methodology if summary else None,
        summary.findings if summary else None,
        summary.limitations if summary else None,
    )
    if summary and any(
        not value or (isinstance(value, str) and value.strip().lower() in {"n/a", "none", "null"})
        for value in required_fields
    ):
        # Regenerate summaries created by older versions that returned raw LLM
        # fallback text without the structured fields.
        summary = None
    if not summary:
        doc = DOCUMENTS_REGISTRY.get(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        chunks = CHUNKS_REGISTRY.get(document_id, [])
        llm = LLMClient()
        llm_caller = lambda prompt, sys_prompt: llm.generate(prompt=prompt, system_instruction=sys_prompt)
        summarizer = PaperSummarizer(
            strategy="llm" if settings.HUGGINGFACE_API_KEY else "extractive",
            llm_caller=llm_caller if settings.HUGGINGFACE_API_KEY else None,
            model_name=settings.HUGGINGFACE_MODEL,
        )
        summary = summarizer.summarize_paper(doc, chunks=chunks)
        SUMMARIES_REGISTRY[document_id] = summary

    return PaperSummaryResponse(
        document_id=summary.document_id,
        summary=summary.summary,
        section=summary.section,
        key_contributions=summary.key_contributions,
        problem_statement=summary.problem_statement,
        methodology=summary.methodology,
        findings=summary.findings,
        limitations=summary.limitations,
        source_chunks=summary.source_chunks,
        model_used=summary.model_used,
    )


@router.get("/{document_id}/sections", response_model=List[SectionResponse])
def get_sections(document_id: str):
    """Get detected sections for a document."""
    doc = DOCUMENTS_REGISTRY.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    # Upgrade documents created before hierarchical outlines were introduced.
    if doc.sections and all(section.outline_source == "detector" for section in doc.sections) and settings.HUGGINGFACE_API_KEY:
        llm = LLMClient()
        llm_caller = lambda prompt, sys_prompt: llm.generate(prompt=prompt, system_instruction=sys_prompt)
        sections = SectionOutlineAnalyzer(llm_caller=llm_caller).analyze(doc, doc.sections)
        if sections:
            _save_registry()
    return [
        SectionResponse(
            section_id=s.section_id,
            title=s.title,
            raw_heading=s.raw_heading,
            start_page=s.start_page,
            end_page=s.end_page,
            order=s.order,
            level=s.level,
            parent_section_id=s.parent_section_id,
        )
        for s in doc.sections
    ]


@router.get("/{document_id}/understanding", response_model=PaperUnderstandingResponse)
def get_understanding(document_id: str):
    """Get topics, keywords, and concepts for a document."""
    analysis = ANALYSIS_REGISTRY.get(document_id)
    if not analysis:
        doc = DOCUMENTS_REGISTRY.get(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
        chunks = CHUNKS_REGISTRY.get(document_id, [])
        llm = LLMClient()
        llm_caller = lambda prompt, sys_prompt: llm.generate(
            prompt=prompt, system_instruction=sys_prompt
        )
        analyzer = PaperUnderstandingAnalyzer(
            llm_caller=llm_caller if settings.HUGGINGFACE_API_KEY else None,
        )
        analysis = analyzer.analyze(chunks)
        ANALYSIS_REGISTRY[document_id] = analysis

    return PaperUnderstandingResponse(
        document_id=document_id,
        topics=[
            TopicItem(
                topic_id=t.topic_id,
                name=t.name,
                score=float(t.score),
                source_chunk_ids=list(t.source_chunk_ids),
            )
            for t in analysis.analysis.topics
        ],
        keywords=[
            KeywordItem(
                text=k.text,
                score=float(k.score),
                source_chunk_ids=list(k.source_chunk_ids),
            )
            for k in analysis.keywords
        ],
        concepts=[
            ConceptItem(
                concept_id=f"concept-{index:03d}",
                name=c.name,
                definition=None,
                source_chunk_ids=list(c.source_chunk_ids),
            )
            for index, c in enumerate(analysis.analysis.concepts, start=1)
        ],
    )
