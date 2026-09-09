"""Prepare Salma's chunks before passing them to the NLP modules.

The function in this file accepts either dictionaries or Python objects. This
lets us test Shahd's part now and connect it to Salma's final model later.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{1,}")
STOP_WORDS = {
    "about", "above", "after", "again", "against", "all", "also", "among",
    "an", "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "between", "both", "but", "by", "can", "could", "did", "do",
    "does", "doing", "during", "each", "for", "from", "further", "had", "has",
    "have", "having", "here", "how", "however", "if", "in", "into", "is",
    "it", "its", "may", "more", "most", "no", "not", "of", "on", "only",
    "or", "other", "our", "out", "over", "paper", "results", "same", "should",
    "show", "shown", "shows", "such", "than", "that", "the", "their", "these",
    "they", "this", "those", "through", "to", "under", "using", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "will",
    "with", "within", "would", "you",
}


@dataclass(frozen=True)
class ChunkView:
    """A small, validated copy of the fields needed by Shahd's code."""

    document_id: str
    chunk_id: str
    page_number: int
    section: str | None
    text: str


def tokens(text: str) -> list[str]:
    return [token.lower().strip("'-") for token in TOKEN_RE.findall(text)]


def content_tokens(text: str) -> list[str]:
    return [token for token in tokens(text) if token not in STOP_WORDS and len(token) > 2]


def _read_field(chunk: Any, field_name: str) -> Any:
    """Read one field from a dictionary or an object."""

    if isinstance(chunk, Mapping):
        return chunk[field_name]
    return getattr(chunk, field_name)


def normalize_chunks(chunks: Iterable[Any]) -> list[ChunkView]:
    """Validate input chunks and return one consistent representation."""

    normalized_chunks: list[ChunkView] = []
    required_fields = ("document_id", "chunk_id", "page_number", "section", "text")

    for index, chunk in enumerate(chunks):
        if isinstance(chunk, Mapping):
            missing_fields = [name for name in required_fields if name not in chunk]
        else:
            missing_fields = [name for name in required_fields if not hasattr(chunk, name)]

        if missing_fields:
            fields = ", ".join(missing_fields)
            raise ValueError(f"Chunk {index} is missing fields: {fields}")

        text = str(_read_field(chunk, "text")).strip()
        if not text:
            continue

        page_number = _read_field(chunk, "page_number")
        if not isinstance(page_number, int) or page_number < 1:
            raise ValueError(f"Chunk {index} has an invalid 1-based page_number")

        section = _read_field(chunk, "section")
        normalized_chunks.append(
            ChunkView(
                document_id=str(_read_field(chunk, "document_id")),
                chunk_id=str(_read_field(chunk, "chunk_id")),
                page_number=page_number,
                section=str(section).strip() if section else None,
                text=text,
            )
        )

    if not normalized_chunks:
        raise ValueError("At least one non-empty chunk is required")

    document_ids = {chunk.document_id for chunk in normalized_chunks}
    if len(document_ids) != 1:
        raise ValueError("All chunks must belong to the same document")

    return normalized_chunks


def remove_duplicates(values: Iterable[str]) -> list[str]:
    """Remove duplicates while keeping the original order."""

    return list(dict.fromkeys(values))


__all__ = [
    "ChunkView",
    "content_tokens",
    "normalize_chunks",
    "remove_duplicates",
    "tokens",
]
