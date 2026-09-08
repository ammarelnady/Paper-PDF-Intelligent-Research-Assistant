from fastapi import FastAPI

from app.api.routes.documents import router as documents_router


def create_app() -> FastAPI:
    """Create the HTTP application without side effects at import time."""
    app = FastAPI(title="Paper Research Assistant", version="0.1.0")
    app.include_router(documents_router, prefix="/api/v1")
    return app


app = create_app()
