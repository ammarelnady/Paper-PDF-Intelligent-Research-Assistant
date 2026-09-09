"""
FastAPI Application Entry Point for Paper PDF Intelligent Research Assistant.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Intelligent NLP/LLM research assistant for PDF understanding, summarization, semantic retrieval, and grounded citation generation.",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "project": settings.PROJECT_NAME, "version": settings.VERSION}
