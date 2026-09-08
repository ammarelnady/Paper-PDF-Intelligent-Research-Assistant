import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.document_processing.processor import DocumentProcessor


router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"]
)


processor = DocumentProcessor()


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED
)
async def upload_document(
    file: UploadFile = File(...)
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    temp_path = None

    try:
        file_content = await file.read()

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False
        ) as temp_file:

            temp_file.write(file_content)
            temp_path = Path(temp_file.name)

        document = processor.process(temp_path)

        # Keep the original uploaded filename
        document.filename = file.filename

        return document.model_dump()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
