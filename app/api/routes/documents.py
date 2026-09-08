import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from app.api.schemas import ChunkResponse, DocumentResponse
from app.core.config import Settings
from app.core.exceptions import InvalidPdfError
from app.document_processing.service import process_pdf
from app.document_processing.storage import save_pdf_upload

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


def get_settings() -> Settings:
    return Settings.from_env()


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile, settings: Settings = Depends(get_settings)) -> DocumentResponse:
    """Store and process one PDF, returning page-aware chunks."""
    try:
        document_id, path = await save_pdf_upload(file, settings)
        document = process_pdf(document_id, file.filename or path.name, path, settings)
    except InvalidPdfError as error:
        if "path" in locals():
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Unable to remove failed PDF upload: %s", path)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    return DocumentResponse(
        document_id=document.document_id,
        filename=document.filename,
        page_count=document.page_count,
        chunks=[ChunkResponse(
            document_id=chunk.document_id, page_number=chunk.page_number,
            section=chunk.section, chunk_id=chunk.chunk_id, text=chunk.text,
        ) for chunk in document.chunks],
    )
