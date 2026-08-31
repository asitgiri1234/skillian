"""Ollama-backed :class:`~app.providers.embeddings.EmbeddingProvider`."""

from __future__ import annotations

import logging

import httpx
import ollama

from app.config import Settings, get_settings
from app.providers.embeddings import (
    EmbeddingDimensionError,
    EmbeddingProvider,
    EmbeddingUnavailableError,
)

logger = logging.getLogger(__name__)

# nomic-embed-text emits 768 dimensions. Kept here rather than probed at startup
# so a misconfigured model is caught on the first call, not silently written into
# a column of the wrong width.
NOMIC_EMBED_DIM = 768

_MODEL_DIMENSIONS: dict[str, int] = {
    "nomic-embed-text": NOMIC_EMBED_DIM,
    "nomic-embed-text:latest": NOMIC_EMBED_DIM,
    "nomic-embed-text:v1.5": NOMIC_EMBED_DIM,
}


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embeddings via a locally running Ollama daemon."""

    name = "ollama"

    def __init__(
        self,
        settings: Settings | None = None,
        client: ollama.Client | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.model = self._settings.ollama_embed_model
        # Unknown model: assume the configured schema width rather than refusing
        # to start. The check in _validate still catches a genuine mismatch on
        # the first call, so a new model works without editing this table.
        self.dimension = _MODEL_DIMENSIONS.get(self.model, NOMIC_EMBED_DIM)
        self._client = client or ollama.Client(
            host=self._settings.ollama_host,
            timeout=self._settings.ollama_timeout_seconds,
        )

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embed(model=self.model, input=texts)
        except ollama.ResponseError as exc:
            if getattr(exc, "status_code", None) == 404:
                raise EmbeddingUnavailableError(
                    f"Ollama has no model {self.model!r}. Run: ollama pull {self.model}"
                ) from exc
            raise EmbeddingUnavailableError(f"Ollama error: {exc}") from exc
        except httpx.TimeoutException as exc:
            # Before TransportError, which it subclasses — see the note in
            # ollama_llm._chat. Embedding is fast, so a timeout here usually
            # means an oversized batch rather than a slow machine.
            raise EmbeddingUnavailableError(
                f"Ollama did not respond within "
                f"{self._settings.ollama_timeout_seconds:g}s while embedding "
                f"{len(texts)} text(s). The daemon is reachable; it is just "
                "slow. Raise OLLAMA_TIMEOUT_SECONDS in .env, or reduce "
                "EMBED_BATCH_SIZE."
            ) from exc
        except (httpx.TransportError, ConnectionError) as exc:
            raise EmbeddingUnavailableError(
                f"Cannot reach the Ollama daemon at {self._settings.ollama_host}. "
                "Is it running? Try: ollama serve"
            ) from exc

        vectors = [list(vector) for vector in response.embeddings]
        if len(vectors) != len(texts):
            raise EmbeddingDimensionError(
                f"Asked for {len(texts)} embeddings, got {len(vectors)}"
            )
        for vector in vectors:
            self._validate(vector)
        return vectors

    def _validate(self, vector: list[float]) -> None:
        if len(vector) != self.dimension:
            raise EmbeddingDimensionError(
                f"{self.model!r} returned {len(vector)} dimensions, expected "
                f"{self.dimension}. The vector columns are sized for "
                f"{self.dimension}; a different model needs a migration."
            )

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            # An all-zero vector would be a valid write but is meaningless under
            # cosine distance (undefined), so refuse rather than poison the table.
            raise ValueError("Cannot embed empty text")
        return self._embed_many([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Override: Ollama's /api/embed accepts a list, so one round-trip does
        the whole batch instead of N."""
        if not texts:
            return []
        if any(not text or not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")
        return self._embed_many(texts)
