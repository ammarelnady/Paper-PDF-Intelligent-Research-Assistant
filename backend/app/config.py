"""
Application configuration and environment settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # Fallback for basic pydantic
    from pydantic import BaseModel as BaseSettings  # type: ignore
    SettingsConfigDict = None  # type: ignore


class Settings(BaseSettings):
    """Global configuration settings for the PaperLens AI backend."""

    # Project metadata
    PROJECT_NAME: str = "Paper PDF Intelligent Research Assistant"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api"

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    INDICES_DIR: Path = DATA_DIR / "indices"

    # Chunking configurations (Salma)
    CHUNK_SIZE: int = 800          # Target character length per chunk
    CHUNK_OVERLAP: int = 150       # Character overlap between consecutive chunks
    MIN_CHUNK_SIZE: int = 100      # Minimum characters for a valid chunk

    # Summarization configurations (Salma)
    DEFAULT_SUMMARY_STRATEGY: str = "auto"  # 'auto', 'extractive' or 'llm'
    SUMMARY_MODEL_NAME: str = "Qwen/Qwen2.5-72B-Instruct"
    MAX_SUMMARY_SENTENCES: int = 7
    SUMMARY_MIN_LENGTH: int = 100
    SUMMARY_MAX_LENGTH: int = 500

    # LLM Configurations (Mohamed)
    LLM_PROVIDER: str = "huggingface"  # 'huggingface', 'ollama', or 'extractive'
    LLM_MODEL: str = "Qwen/Qwen2.5-72B-Instruct"
    HUGGINGFACE_API_KEY: Optional[str] = None
    HUGGINGFACE_MODEL: str = "Qwen/Qwen2.5-72B-Instruct"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    API_AUTH_TOKEN: Optional[str] = None

    # Embeddings & Vector Search (Reem)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_TOP_K: int = 5
    RRF_K: int = 60

    # Web Search & Routing (Ammar)
    WEB_SEARCH_PROVIDER: str = "duckduckgo"
    QUERY_CLASSIFIER_MODE: str = "llm"
    TAVILY_API_KEY: Optional[str] = None
    SERPER_API_KEY: Optional[str] = None

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    MAX_UPLOAD_SIZE_MB: int = 25
    MAX_PDF_PAGES: int = 500
    RATE_LIMIT_PER_MINUTE: int = 60

    if SettingsConfigDict:
        model_config = SettingsConfigDict(env_file=str(Path(__file__).resolve().parent.parent / ".env"), extra="ignore")
    else:
        class Config:
            env_file = str(Path(__file__).resolve().parent.parent / ".env")
            extra = "ignore"


# Ensure directories exist
settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.INDICES_DIR.mkdir(parents=True, exist_ok=True)
