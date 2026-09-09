"""Embedding providers used by the semantic-retrieval component."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np


class EmbeddingError(RuntimeError):
    """Raised when text cannot be converted into usable embeddings."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """A replaceable source of normalized, float32 text embeddings."""

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of non-empty text strings as a two-dimensional array."""

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one non-empty query as a one-dimensional array."""


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Return row-normalized float32 vectors suitable for cosine similarity."""

    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] == 0:
        raise EmbeddingError("Embeddings must be a non-empty two-dimensional array.")
    if not np.isfinite(array).all():
        raise EmbeddingError("Embeddings must contain only finite numeric values.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise EmbeddingError("Cannot normalize a zero-vector embedding.")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


class SentenceTransformerEmbeddingProvider:
    """Sentence Transformers embedding provider with lazy model loading."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", model: Any | None = None) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        self.model_name = model_name
        self._model = model

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed and normalize a batch of text strings."""

        text_list = list(texts)
        if not text_list:
            raise EmbeddingError("At least one text is required to create embeddings.")
        if any(not isinstance(text, str) or not text.strip() for text in text_list):
            raise EmbeddingError("Every text to embed must be a non-empty string.")
        try:
            vectors = self._get_model().encode(text_list, convert_to_numpy=True, show_progress_bar=False)
        except EmbeddingError:
            raise
        except Exception as exc:  # pragma: no cover - model/backend dependent
            raise EmbeddingError("Sentence Transformer failed to encode the provided texts.") from exc
        return normalize_embeddings(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one query as a normalized float32 vector."""

        if not isinstance(query, str) or not query.strip():
            raise EmbeddingError("query must be a non-empty string.")
        return self.embed_texts([query])[0]

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise EmbeddingError(
                    "sentence-transformers is required for SentenceTransformerEmbeddingProvider. "
                    "Install it before using the production embedding provider."
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # pragma: no cover - model/backend dependent
                raise EmbeddingError(f"Unable to load Sentence Transformer model '{self.model_name}'.") from exc
        return self._model
