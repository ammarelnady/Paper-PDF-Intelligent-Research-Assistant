from fastapi import FastAPI

from app.api.documents import router as documents_router


app = FastAPI(
    title="Paper Research Assistant",
    description="Document processing API for research papers",
    version="1.0.0"
)


app.include_router(documents_router)


@app.get("/")
def root():
    return {
        "message": "Paper Research Assistant API is running"
    }