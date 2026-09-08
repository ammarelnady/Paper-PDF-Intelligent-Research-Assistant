from pydantic import BaseModel


class ChunkResponse(BaseModel):
    document_id: str
    page_number: int
    section: str | None
    chunk_id: str
    text: str


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunks: list[ChunkResponse]
