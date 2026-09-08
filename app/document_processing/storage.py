from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import Settings
from app.core.exceptions import InvalidPdfError


async def save_pdf_upload(upload: UploadFile, settings: Settings) -> tuple[str, Path]:
    """Persist an upload while enforcing extension, signature, and size limits."""
    filename = Path(upload.filename or "document.pdf").name
    if Path(filename).suffix.lower() != ".pdf":
        raise InvalidPdfError("Only .pdf files are accepted")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    document_id = str(uuid4())
    destination = settings.upload_dir / f"{document_id}.pdf"
    size = 0
    try:
        with destination.open("wb") as output:
            while data := await upload.read(1024 * 1024):
                size += len(data)
                if size > settings.max_upload_bytes:
                    raise InvalidPdfError("Uploaded file exceeds the configured size limit")
                output.write(data)
        with destination.open("rb") as saved_file:
            signature = saved_file.read(5)
        if size == 0 or signature != b"%PDF-":
            raise InvalidPdfError("The uploaded file is not a valid PDF")
        return document_id, destination
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
