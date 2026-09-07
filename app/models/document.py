from pydantic import BaseModel, Field

class PageContent(BaseModel):
    page_number: int = Field(..., ge=1)
    text: str

class DocumentChunk(BaseModel):
    document_id: str
    chunk_id: str
    page_number: int = Field(..., ge=1)
    section: str
    text: str = Field(..., min_length=1)

class Document(BaseModel):
    document_id: str
    filename: str
    total_pages: int = Field(..., ge=0)
    pages: list[PageContent]
    chunks: list[DocumentChunk]