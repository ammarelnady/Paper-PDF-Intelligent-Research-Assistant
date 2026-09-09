"""The router's minimal boundary to a separately implemented retrieval system."""

from __future__ import annotations

from typing import Protocol, Sequence

from app.contracts import RetrievedChunk


class Retriever(Protocol):
    """A retrieval implementation supplies paper chunks with provenance."""

    def search(self, query: str) -> Sequence[RetrievedChunk]:
        """Retrieve chunks relevant to the query."""
