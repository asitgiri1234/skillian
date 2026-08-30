"""The embedding provider contract."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingError(RuntimeError):
    """Base class for embedding provider failures."""


class EmbeddingUnavailableError(EmbeddingError):
    """Backend unreachable or embedding model not available."""


class EmbeddingDimensionError(EmbeddingError):
    """The backend returned a vector of unexpected width.

    Worth its own type: a dimension mismatch fails at INSERT with an opaque
    pgvector error, far from the cause. Catching it at the provider boundary
    names the real problem (wrong model configured).
    """


class EmbeddingProvider(ABC):
    """Turns text into a dense vector."""

    #: Stable identifier, matching the key in the provider registry.
    name: str

    #: Output width. Must equal models.EMBEDDING_DIM or writes will fail.
    dimension: int

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return the embedding for a single string.

        Raises:
            EmbeddingUnavailableError: backend unreachable or model missing.
            EmbeddingDimensionError: returned vector width != ``dimension``.
        """
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed several strings, returned in input order.

        Concrete, not abstract: the default loops over :meth:`embed` so a new
        provider only has to implement one method. Providers with a real batch
        endpoint should override this — the round-trip per item dominates.
        """
        return [self.embed(text) for text in texts]
