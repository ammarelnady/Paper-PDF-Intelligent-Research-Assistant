<<<<<<< HEAD
"""Reserved project configuration module; routing uses no runtime configuration yet."""
=======
"""
Application configuration and environment settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
try:
    from pydantic_settings import BaseSettings
except ImportError:  # Fallback for basic pydantic
    from pydantic import BaseModel as BaseSettings  # type: ignore


class Settings(BaseSettings):
    """Global configuration settings for the Paper Research Assistant backend."""

    # Project metadata
    PROJECT_NAME: str = "Paper PDF Intelligent Research Assistant"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api"

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"

    # Chunking configurations (Salma)
    CHUNK_SIZE: int = 800          # Target character length per chunk
    CHUNK_OVERLAP: int = 150       # Character overlap between consecutive chunks
    MIN_CHUNK_SIZE: int = 100      # Minimum characters for a valid chunk

    # Summarization configurations (Salma)
    DEFAULT_SUMMARY_STRATEGY: str = "extractive"  # 'extractive' or 'llm'
    SUMMARY_MODEL_NAME: str = "t5-base"           # Configurable summarization model
    MAX_SUMMARY_SENTENCES: int = 7
    SUMMARY_MIN_LENGTH: int = 100
    SUMMARY_MAX_LENGTH: int = 500

    # LLM & External API Keys (Configurable)
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
    SERPER_API_KEY: Optional[str] = None

    # Embeddings & Vector Search (Reem)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_TOP_K: int = 5

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"


# Ensure directories exist
settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
>>>>>>> origin/main
