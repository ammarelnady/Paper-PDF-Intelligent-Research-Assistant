"""
FastAPI Application Entry Point for Paper PDF Intelligent Research Assistant.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import documents_router, query_router, questions_router
from app.api.documents import load_persisted_documents
from app.config import settings
from app.middleware import ApiProtectionMiddleware

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Intelligent NLP/LLM research assistant for PDF understanding, summarization, semantic retrieval, and grounded citation generation.",
)

app.add_middleware(ApiProtectionMiddleware)

# Enable CORS for local development and frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(documents_router, prefix=settings.API_V1_STR)
app.include_router(questions_router, prefix=settings.API_V1_STR)
app.include_router(query_router, prefix=settings.API_V1_STR)

# Restore persisted document metadata and indexes after every restart.
load_persisted_documents()

# Serve Frontend static directory if present
frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.HUGGINGFACE_MODEL,
    }


@app.get("/", tags=["Root"])
def root():
    """Root redirect / information endpoint."""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API v{settings.VERSION}",
        "docs": "/docs",
        "health": "/health",
        "frontend": "/app/index.html",
    }
