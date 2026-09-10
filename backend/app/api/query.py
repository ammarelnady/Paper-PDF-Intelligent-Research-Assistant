"""
Interactive Research Query API endpoint.

Classifies query intent (RAG, WEB, HYBRID), performs semantic/lexical retrieval
and web search, and generates grounded answers with inline citations using LLM.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from app.api.documents import (
    BM25_REGISTRY,
    CHUNKS_REGISTRY,
    DOCUMENTS_REGISTRY,
    VECTOR_STORES_REGISTRY,
)
from app.api.schemas import PaperCitation, QueryRequest, QueryResponse, WebCitation
from app.config import settings
from app.contracts import RetrievedChunk as ContractRetrievedChunk, RouteDecision, WebSource
from app.llm import CitationFormatter, LLMClient, PromptBuilder
from app.rag.vector_store import RetrievedChunk
from app.routing import DeterministicQueryClassifier, LLMQueryClassifier, ResearchRouter
from app.web_search import LLMSearchQueryRewriter, SearchQueryPreparer, WebSearchClient, build_search_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query & Research"])

# Initialize Shared Components
prompt_builder = PromptBuilder()
llm_client = LLMClient()
web_client = WebSearchClient(
    provider=build_search_provider(settings.WEB_SEARCH_PROVIDER),
    query_preparer=SearchQueryPreparer(LLMSearchQueryRewriter(llm_client)),
)
classifier = (
    LLMQueryClassifier(llm_client=llm_client)
    if settings.QUERY_CLASSIFIER_MODE.lower() in {"llm", "auto"}
    else DeterministicQueryClassifier()
)


@router.post("/", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """
    Process a research question:
    1. Route to RAG, WEB, or HYBRID
    2. Retrieve relevant evidence from paper and/or web
    3. Construct grounded prompt
    4. Generate LLM answer with strict citation tracing
    """
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not re.search(r"[^\W_]", query_text):
        raise HTTPException(status_code=400, detail="Query must contain letters or digits.")

    # 1. Retrieve RAG chunks if document_id is provided
    retrieved_chunks: List[RetrievedChunk] = []
    if request.document_id:
        doc = DOCUMENTS_REGISTRY.get(request.document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Specified document_id was not found.")

        # Check vector store
        vs_entry = VECTOR_STORES_REGISTRY.get(request.document_id)
        if vs_entry:
            provider, store = vs_entry
            bm25_index = BM25_REGISTRY.get(request.document_id)
            try:
                from app.rag import SemanticRetriever
                retriever = SemanticRetriever(
                    embedding_provider=provider,
                    vector_store=store,
                    bm25_index=bm25_index,
                    rrf_k=settings.RRF_K,
                )
                strategy = request.strategy or ("hybrid_rrf" if bm25_index else "faiss")
                retrieved_chunks = list(
                    retriever.retrieve(query_text, top_k=request.top_k, strategy=strategy)
                )
            except Exception as e:
                logger.warning(f"RAG retrieval error: {e}; falling back to text search.")

        if not retrieved_chunks:
            # Fallback simple lexical match over document chunks
            chunks = CHUNKS_REGISTRY.get(request.document_id, [])
            words = set(query_text.lower().split())
            scored = []
            for c in chunks:
                score = sum(1 for w in words if w in c.text.lower())
                if score > 0:
                    scored.append((score, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            for score, c in scored[: request.top_k]:
                retrieved_chunks.append(
                    RetrievedChunk(
                        document_id=c.document_id,
                        chunk_id=c.chunk_id,
                        page_number=c.page_number,
                        section=c.section,
                        text=c.text,
                        score=float(score),
                    )
                )

    # 2. Setup Router and Decide Routing
    class AdapterRetriever:
        def search(self, q: str):
            return [
                ContractRetrievedChunk(
                    document_id=c.document_id,
                    chunk_id=c.chunk_id,
                    page_number=c.page_number,
                    section=c.section,
                    text=c.text,
                    score=c.score,
                )
                for c in retrieved_chunks
            ]

    route_classifier = classifier
    if request.document_id and request.strategy:
        # The strategy selector controls paper retrieval, so an explicit paper
        # strategy must not be overridden by the LLM into a WEB route.
        class PaperOnlyClassifier:
            def classify(self, _query: str) -> RouteDecision:
                return RouteDecision(route="RAG", confidence=1.0)

        route_classifier = PaperOnlyClassifier()

    research_router = ResearchRouter(
        classifier=route_classifier,
        retriever=AdapterRetriever() if retrieved_chunks else None,
        web_search=web_client,
    )

    routing_result = research_router.route(query_text)
    route = routing_result.decision.route.upper()
    confidence = routing_result.decision.confidence
    web_sources: List[WebSource] = list(routing_result.web_sources)
    if routing_result.web_search_error:
        logger.warning("External search unavailable; continuing without web sources: %s", routing_result.web_search_error)

    # 3. Build Grounded Prompt
    prompt = prompt_builder.build_prompt(
        query=query_text,
        retrieved_chunks=retrieved_chunks if route in ["RAG", "HYBRID"] else [],
        web_sources=web_sources if route in ["WEB", "HYBRID"] else [],
        route=route,
    )

    # 4. Generate Answer via LLM
    if route == "WEB" and not web_sources:
        raw_answer = (
            "I could not retrieve live web sources for this question right now. "
            "Please retry the search, or configure an academic/web search provider."
        )
    else:
        raw_answer = llm_client.generate(
            prompt=prompt,
            system_instruction=prompt_builder.system_prompt,
            temperature=0.2,
        )

    # 5. Format Citations
    citation_data = CitationFormatter.format_citations(
        answer=raw_answer,
        chunks=retrieved_chunks if route in ["RAG", "HYBRID"] else [],
        web_sources=web_sources if route in ["WEB", "HYBRID"] else [],
    )

    return QueryResponse(
        query=query_text,
        route=route,
        confidence=round(confidence, 4),
        answer=citation_data["answer"],
        paper_citations=[
            PaperCitation(
                chunk_id=c["chunk_id"],
                document_id=c["document_id"],
                page_number=c["page_number"],
                section=c["section"],
                score=c["score"],
                snippet=c["snippet"],
            )
            for c in citation_data["paper_citations"]
        ],
        web_citations=[
            WebCitation(
                title=w["title"],
                url=w["url"],
                domain=w["domain"],
                snippet=w["snippet"],
            )
            for w in citation_data["web_citations"]
        ],
        has_citations=citation_data["has_citations"],
    )
