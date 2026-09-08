from io import BytesIO
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from app.api.routes.documents import get_settings
from app.core.config import Settings
from app.main import app


def _pdf_bytes(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def _client(tmp_path: Path, max_upload_bytes: int = 50_000_000) -> TestClient:
    settings = Settings(
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "data" / "uploads",
        max_upload_bytes=max_upload_bytes,
        chunk_size=1_000,
        chunk_overlap=150,
        embedding_model="unused-in-phase-1",
        llm_model=None,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_valid_pdf_upload_returns_page_aware_chunks(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("paper.pdf", _pdf_bytes("1 Introduction\nPaper text."), "application/pdf")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "paper.pdf"
    assert body["page_count"] == 1
    assert body["chunks"]
    chunk = body["chunks"][0]
    assert set(chunk) == {"document_id", "page_number", "section", "chunk_id", "text"}
    assert chunk["page_number"] == 1
    assert chunk["section"] == "Introduction"
    assert chunk["chunk_id"].startswith(f"{body['document_id']}:p1:c")


def test_invalid_and_empty_uploads_return_422(tmp_path: Path) -> None:
    client = _client(tmp_path)
    try:
        invalid = client.post("/api/v1/documents/upload", files={"file": ("notes.txt", b"text", "text/plain")})
        empty = client.post("/api/v1/documents/upload", files={"file": ("empty.pdf", b"", "application/pdf")})
    finally:
        app.dependency_overrides.clear()

    assert invalid.status_code == 422
    assert empty.status_code == 422


def test_corrupted_pdf_and_oversized_upload_return_422(tmp_path: Path) -> None:
    client = _client(tmp_path, max_upload_bytes=20)
    try:
        corrupted = client.post(
            "/api/v1/documents/upload",
            files={"file": ("corrupted.pdf", b"%PDF-not-readable", "application/pdf")},
        )
        oversized = client.post(
            "/api/v1/documents/upload",
            files={"file": ("large.pdf", b"%PDF-" + b"x" * 100, "application/pdf")},
        )
    finally:
        app.dependency_overrides.clear()

    assert corrupted.status_code == 422
    assert oversized.status_code == 422
