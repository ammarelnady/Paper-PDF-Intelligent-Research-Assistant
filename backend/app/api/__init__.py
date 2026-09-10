"""FastAPI API routers for documents, questions, and query routing."""

from app.api.documents import router as documents_router
from app.api.query import router as query_router
from app.api.questions import router as questions_router

__all__ = ["documents_router", "questions_router", "query_router"]
