from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Environment-backed settings for the application."""

    data_dir: Path
    upload_dir: Path
    max_upload_bytes: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    llm_model: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(__file__).resolve().parents[2]
        data_dir = _resolve_path(os.getenv("PRA_DATA_DIR", "data"), project_root)
        upload_dir = _resolve_path(os.getenv("PRA_UPLOAD_DIR", "data/uploads"), project_root)
        chunk_size = _positive_int("PRA_CHUNK_SIZE", 1000)
        overlap = _non_negative_int("PRA_CHUNK_OVERLAP", 150)
        if overlap >= chunk_size:
            raise ValueError("PRA_CHUNK_OVERLAP must be smaller than PRA_CHUNK_SIZE")
        return cls(
            data_dir=data_dir,
            upload_dir=upload_dir,
            max_upload_bytes=_positive_int("PRA_MAX_UPLOAD_BYTES", 52_428_800),
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            embedding_model=os.getenv("PRA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            llm_model=os.getenv("PRA_LLM_MODEL") or None,
        )


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _non_negative_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value
